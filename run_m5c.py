"""One-command WSL pipeline. Defaults: 100k-read pilot, 10 training epochs.
Install once: python -m pip install numpy pandas torch pysam pod5 ViennaRNA
Requires dorado on PATH if an aligned moves BAM is not supplied.
"""
import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


def site_table(reference, bed, output):
    import pandas as pd
    import RNA
    refs = {}
    for line in reference.read_text().splitlines():
        if line.startswith('>'):
            name = line[1:].split()[0]; refs[name] = ''
        elif line.strip():
            refs[name] += line.strip().upper().replace('T', 'U')
    truth = set()
    for line in bed.read_text().splitlines():
        if not line or line.startswith(('#', 'track', 'browser')):
            continue
        fields = line.split()
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])
        if end != start+1 or refs[chrom][start] != 'C':
            raise ValueError('BED must contain single-base C positions in 0-based coordinates')
        if len(fields) >= 6 and fields[5] != '+':
            raise ValueError('This synthetic-template pipeline expects plus-strand BED')
        truth.add((chrom, start))
    rows = []
    for chrom, sequence in refs.items():
        for pos, base in enumerate(sequence):
            if base != 'C': continue
            lo, hi = max(0, pos-50), min(len(sequence), pos+51)
            left, right = lo-(pos-50), pos+51-hi
            actual = sequence[lo:hi]
            structure, energy = RNA.fold(actual)
            structure = '.'*left + structure + '.'*right
            stack, edges = [], []
            for i, symbol in enumerate(structure):
                if symbol == '(': stack.append(i)
                elif symbol == ')': edges.append([stack.pop(), i])
            rows.append(dict(site_id=f'{chrom}:{pos+1}:C', reference_name=chrom,
                position_0based=pos, position_1based=pos+1, reference_base='C', modification='m5C',
                sequence_101_rna='N'*left+actual+'N'*right,
                left_padding_nt=left, right_padding_nt=right,
                secondary_structure_dot_bracket_101=structure,
                base_pair_edges_0based_json=json.dumps(edges),
                minimum_free_energy_kcal_per_mol=float(energy),
                groundtruth=int((chrom,pos) in truth)))
    frame = pd.DataFrame(rows)
    frame.to_pickle(output)
    print('m5C sites:', len(frame), 'BED positive:', frame.groundtruth.sum(), flush=True)


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/g/rna-mod-validation-all5mer-2026.07'))
    p.add_argument('--output', type=Path, default=here/'m5c_output')
    p.add_argument('--base-model', default='/mnt/g/rna004_hac@v6.0.0')
    p.add_argument('--dorado', default='dorado')
    p.add_argument('--bam', type=Path, help='Existing aligned m5C BAM with mv tags')
    p.add_argument('--signal-store', type=Path, help='Existing m5C ADC store; never use the Inosine store')
    p.add_argument('--model-file', type=Path, default=here/'multimodal_rna_modification_model.py')
    p.add_argument('--max-reads', type=int, default=0, help='pilot input reads, 0=all')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--signal-radius', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    for dependency in ('numpy','pandas','torch','pysam','pod5','RNA'):
        importlib.import_module(dependency)
    import pysam
    import pod5
    pod = a.root/'subset_20m/m5C/subset.pod5'
    ref = a.root/'references/C_reference.fasta'
    bed = a.root/'references/m5C_positions.bed'
    for file in (pod, ref, bed, a.model_file):
        if not file.exists(): p.error(f'Missing file: {file}')
    a.output.mkdir(parents=True, exist_ok=True)
    run_config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config_path = a.output/'pipeline_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        p.error('Parameters changed: use a new --output directory')
    config_path.write_text(json.dumps(run_config, indent=2))
    def run(script, *args):
        subprocess.run([sys.executable, str(here/script), *map(str,args)], check=True)
    bam = a.bam or a.output/'m5C.with_moves.bam'
    if not bam.exists():
        if a.bam: p.error(f'Missing supplied BAM: {bam}')
        command = [a.dorado,'basecaller',a.base_model,str(pod),'--reference',str(ref),
                   '--mm2-opts','-x sr','--emit-moves']
        if a.max_reads:
            ids = a.output/'pilot_read_ids.txt'
            with pod5.Reader(pod) as reader, ids.open('w') as f:
                for i, record in enumerate(reader.reads()):
                    if i >= a.max_reads: break
                    f.write(str(record.read_id)+'\n')
            command += ['--read-ids',str(ids)]
        temporary = bam.with_suffix('.bam.partial')
        with temporary.open('wb') as f:
            subprocess.run(command, stdout=f, check=True)
        temporary.replace(bam)
    with pysam.AlignmentFile(str(bam),'rb') as f:
        found = False
        for i, read in enumerate(f.fetch(until_eof=True)):
            if not read.is_unmapped and read.has_tag('mv'):
                found = True; break
            if i >= 100000: break
        if not found: p.error('BAM lacks mapped reads with mv in first 100001 records')
        if not all(name.startswith('5mers_C_') for name in f.references):
            p.error('BAM is not aligned to expected synthetic C references')
    sites = a.output/'m5c_candidate_sites.pkl'
    if not sites.exists(): site_table(ref, bed, sites)
    store = a.signal_store or a.output/'raw_signal_store'
    if not (store/'manifest.json').exists():
        if a.signal_store: p.error('Provided signal store is incomplete')
        if store.exists() and any(store.iterdir()):
            p.error('Partial raw store exists; use a new output directory')
        options = ['--pod5',pod,'--output-dir',store]
        if a.max_reads: options += ['--max-reads',a.max_reads]
        run('extract_pod5_raw_signals.py',*options)
    dataset = a.output/'read_site_dataset'
    if not (dataset/'dataset_manifest.json').exists():
        run('build_read_site_dataset.py','--bam',bam,'--site-table',sites,
            '--signal-index',store/'read_signal_index.csv.gz',
            '--signal-bin',store/'raw_signal_adc.int16.bin','--output-dir',dataset,
            '--max-reads',a.max_reads,'--signal-radius',a.signal_radius,'--metadata-format','csv')
    if not a.prepare_only:
        training = a.output/'training'
        if (training/'test_metrics.json').exists():
            print('Training already complete:', training); return
        run('train_m5c.py','--model-file',a.model_file,'--data',dataset,'--sites',sites,
            '--output',training,'--task','m5C','--epochs',a.epochs,
            '--batch-size',a.batch_size,'--device',a.device)


if __name__ == '__main__': main()
