"""Train the supplied model on sharded read-site tensors.

BED-derived labels are weak site labels, not verified single-molecule truth.
This driver trains one named classification task; ratio heads are disabled.
Sequence encoder is the model's randomly initialized local Transformer.
MFE remains metadata: the supplied model has no energy input.
"""
import argparse
import importlib.util
import json
import random
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


def load_model_file(path):
    spec = importlib.util.spec_from_file_location('user_rna_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def masked_signal_forward(self, signal, signal_mask):
    # Preserve interior holes instead of interpreting valid-count as prefix length.
    x = signal * signal_mask.unsqueeze(-1)
    x = x.transpose(1, 2)
    mask = signal_mask[:, None].float()
    for layer in self.input_projection:
        x = layer(x)
        if isinstance(layer, nn.Conv1d):
            mask = F.max_pool1d(mask, layer.kernel_size, layer.stride, layer.padding)
        x = x * mask
    x = x.transpose(1, 2)
    valid = mask[:, 0].bool()
    z = x
    for block in self.mamba:
        z = block(z, valid)
    recurrent, _ = self.lstm(x)
    weights = valid.unsqueeze(-1).to(x.dtype)
    pool = lambda v: (v * weights).sum(1) / weights.sum(1).clamp_min(1)
    a, b = pool(z), pool(recurrent)
    gate = self.fusion_gate(torch.cat([a, b], -1))
    return self.out_norm(gate * a + (1 - gate) * b)


def prepare_sites(path):
    table = pd.read_pickle(path) if path.suffix == '.pkl' else pd.read_csv(path)
    if table.site_id.duplicated().any():
        raise ValueError('Duplicate site_id in static table')
    cache = {}
    vocab = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'T': 4, 'N': 5}
    for r in table.to_dict('records'):
        sequence = r['sequence_101_rna'].upper()
        n = len(sequence)
        if n != 101 or sequence[50] != 'C':
            raise ValueError('Expected 101 nt reference window centred on C')
        valid = np.ones(n, dtype=bool)
        left, right = int(r['left_padding_nt']), int(r['right_padding_nt'])
        valid[:left] = False
        if right:
            valid[-right:] = False
        tokens = np.array([vocab[c] for c in sequence], dtype=np.int64)
        tokens[~valid] = 0
        adj = np.eye(n, dtype=np.float32)
        for i in range(n - 1):
            adj[i, i+1] = adj[i+1, i] = 1
        for i, j in json.loads(r['base_pair_edges_0based_json']):
            adj[i, j] = adj[j, i] = 1
        adj *= valid[:, None] * valid[None, :]
        cache[r['site_id']] = (tokens, valid, adj, int(r['groundtruth']))
    return cache


def batches(args, sites, split, epoch):
    files = sorted((args.data / 'metadata').glob('part-*.csv.gz'))
    if not files:
        raise ValueError('No CSV.GZ metadata shards found')
    rng = np.random.default_rng(args.seed + epoch)
    if split == 'train':
        rng.shuffle(files)
    for path in files:
        frame = pd.read_csv(path)
        stem = path.name.removesuffix('.csv.gz')
        signal = np.load(args.data / 'tensors' / f'signal-{stem}.npy', mmap_mode='r')
        mask = np.load(args.data / 'tensors' / f'mask-{stem}.npy', mmap_mode='r')
        if len(frame) != len(signal) or mask.shape != signal.shape[:2]:
            raise ValueError(f'Shard shape mismatch: {path}')
        selected = frame.loc[frame['split'].eq(split)]
        order = np.arange(len(selected))
        if split == 'train':
            rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            rows = selected.iloc[order[start:start+args.batch_size]]
            ids = rows.tensor_row.to_numpy(dtype=np.int64)
            features = [sites[s] for s in rows.site_id]
            labels = rows.groundtruth.to_numpy(dtype=np.float32)
            if not np.array_equal(labels, [f[3] for f in features]):
                raise ValueError('Metadata/static labels disagree')
            arrays = [np.array(signal[ids], copy=True), np.array(mask[ids], dtype=bool),
                      np.stack([f[0] for f in features]), np.stack([f[1] for f in features]),
                      np.stack([f[2] for f in features]), labels]
            yield [torch.from_numpy(v).to(args.device) for v in arrays]


def run_epoch(model, args, sites, split, epoch, optimizer=None):
    model.train(optimizer is not None)
    total, n, tp, fp, fn, tn = 0., 0, 0, 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(batches(args, sites, split, epoch)):
            if args.max_batches and step >= args.max_batches:
                break
            signal, mask, tokens, seq_mask, graph, y = batch
            out = model(signal, mask, tokens, seq_mask, graph)
            logits = out['read_logits'][:, 0]
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            p = logits.detach() >= 0
            truth = y.bool()
            tp += int((p & truth).sum()); fp += int((p & ~truth).sum())
            fn += int((~p & truth).sum()); tn += int((~p & ~truth).sum())
            total += loss.item() * len(y); n += len(y)
            if step % 100 == 0:
                print(f'{split} epoch={epoch} batch={step} loss={loss.item():.5f}', flush=True)
    if not n:
        raise ValueError(f'Empty {split} split')
    return {'loss': total/n, 'n': n, 'accuracy': (tp+tn)/n,
            'precision': tp/max(tp+fp, 1), 'recall': tp/max(tp+fn, 1),
            'f1': 2*tp/max(2*tp+fp+fn, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-file', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--sites', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--task', default='m5C', choices=['m5C'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-batches', type=int, default=0, help='0=full epoch; positive=smoke test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory to preserve existing checkpoints')
    if args.d_model % 4 or args.batch_size < 1 or args.epochs < 1:
        parser.error('d-model must be divisible by 4; epochs/batch-size must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    sites = prepare_sites(args.sites)
    # Verify reference groups never appear in more than one supplied split.
    groups = {}
    for path in sorted((args.data/'metadata').glob('part-*.csv.gz')):
        pairs = pd.read_csv(path, usecols=['reference_name', 'split']).drop_duplicates()
        for ref, split in pairs.itertuples(index=False, name=None):
            if split not in ('train', 'validation', 'test') or groups.get(ref, split) != split:
                raise ValueError('Invalid split or reference leakage')
            groups[ref] = split
    if set(groups.values()) != {'train', 'validation', 'test'}:
        raise ValueError('Need nonempty train, validation and test groups')
    first = next((args.data/'tensors').glob('signal-*.npy'))
    points = np.load(first, mmap_mode='r').shape[-1]
    module = load_model_file(args.model_file)
    cfg = module.ModelConfig(signal_features=points, d_model=args.d_model)
    model = module.RNAModificationModel(cfg)
    model.signal_encoder.forward = types.MethodType(masked_signal_forward, model.signal_encoder)
    model.read_classifier = nn.Linear(cfg.d_model, 1)
    model.modification_order = (args.task,)
    for parameter in model.ratio_regressor.parameters():
        parameter.requires_grad = False
    model.to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'args': vars(args), 'model_config': asdict(cfg), 'label_semantics': 'BED site-derived weak labels',
              'mask_adapter': 'masked_signal_forward', 'ratio_training': False,
              'reference_splits': groups, 'sequence_encoder': 'local untrained Transformer'}
    (args.output/'config.json').write_text(json.dumps(config, default=str, indent=2))
    best = float('inf')
    for epoch in range(1, args.epochs+1):
        train = run_epoch(model, args, sites, 'train', epoch, optimizer)
        valid = run_epoch(model, args, sites, 'validation', epoch)
        record = {'epoch': epoch, 'train': train, 'validation': valid}
        with (args.output/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(record, flush=True)
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'config': config}
        torch.save(checkpoint, args.output/'last.pt')
        if valid['loss'] < best:
            best = valid['loss']; torch.save(checkpoint, args.output/'best.pt')
    checkpoint = torch.load(args.output/'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    test = run_epoch(model, args, sites, 'test', 0)
    (args.output/'test_metrics.json').write_text(json.dumps(test, indent=2))
    print('test', test)


if __name__ == '__main__':
    main()
