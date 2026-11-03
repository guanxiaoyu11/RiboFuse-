# RiboFuse

**Multimodal RNA modification analysis from nanopore direct RNA sequencing.**

RiboFuse combines nanopore current, nucleotide sequence and predicted RNA secondary structure in a PyTorch model. Parallel Mamba-inspired selective state-space and bidirectional LSTM encoders capture signal context, while sequence and graph-attention branches represent nucleotide context and structural connectivity. Hierarchical gating integrates these representations for modification prediction.

This repository contains a research implementation and a runnable **m5C classification pipeline for synthetic reference templates**. The broader framework is designed to support modification detection and site-level fraction estimation, but the supplied training driver does not yet train a site-level quantitative model.

## Implementation status

| Component | Current implementation |
| --- | --- |
| Raw-current input | POD5 extraction, ADC storage and read-specific pA calibration |
| Signal-to-base mapping | Basecaller move tags joined with reference alignments |
| Signal encoder | Custom Mamba-inspired selective state-space blocks and BiLSTM |
| Sequence encoder | Local Transformer by default; optional pretrained-model interface |
| Structure encoder | Edge-weighted graph attention over backbone and pairing edges |
| Fusion | Feature-wise signal gate followed by a three-way modality gate |
| Training workflow | Binary m5C classification using BED-derived site labels |
| Fraction estimation | Regression module exists; frozen in the supplied m5C driver; site-level aggregation and supervision are not implemented |
| Folding energy | Stored as metadata; not consumed by the model |

The selective state-space branch is a pure-PyTorch implementation, **not the official `mamba_ssm.Mamba` layer**. A pretrained genomic language model is optional, not enabled by default. The default sequence Transformer is initialized from scratch and trained with the classifier.

## Model overview

```text
POD5 current + aligned basecaller moves
                 │
       Base-aligned current windows
                 │
          Convolutional projection
                 │
        ┌────────┴─────────┐
   Selective SSM        BiLSTM
        └────────┬─────────┘
          Feature-wise gate ──────────────────┐
                                             │
101-nt reference window → Sequence encoder ────┼→ Modality gate → Classifier
                               │             │
                  Predicted pairing graph    │
                               │             │
                      Graph attention ────────┘
```

The current pipeline extracts reference-centred sequence windows, rather than using the read's error-containing sequence as its nucleotide input. Secondary structure is predicted from the available sequence, with padding masked out. Consequently, the graph is a sequence-derived connectivity constraint, not an independent structural measurement.

Default model settings are a hidden width of 128, two selective state-space blocks, a two-layer BiLSTM, three sequence-Transformer layers, three graph-attention layers, four attention heads and dropout of 0.15. The training driver infers the signal feature width from the stored tensors and replaces the model's two-output classifier with a single m5C logit.

## Repository layout

The commands below follow the supplied local code layout:

```text
README.md
outputs/
└── m5c_pipeline/
    ├── run_m5c.py                          # Pipeline entry point
    ├── extract_pod5_raw_signals.py          # POD5 → ADC store and read index
    ├── build_read_site_dataset.py          # Aligned read–site examples
    ├── multimodal_rna_modification_model.py # PyTorch architecture
    └── train_m5c.py                        # Classification training
```

If these files are placed at the repository root in your checkout, remove `outputs/m5c_pipeline/` from the example script paths.

## Installation

Use a dedicated Python environment. The pipeline is intended for Linux or WSL; a CUDA-compatible PyTorch installation is needed for GPU training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas torch pysam pod5 ViennaRNA
```

Install PyTorch with the CUDA build appropriate for your machine when using a GPU. `ViennaRNA` provides the Python module `RNA`. The default CSV metadata workflow does not require PyArrow.

Dorado must be available on `PATH` if the pipeline needs to generate basecalls. It is not required by the wrapper when a suitable existing aligned BAM is supplied. An optional Hugging Face encoder additionally requires its compatible dependencies and checkpoint; the standard m5C driver uses the local Transformer.

Exact dependency versions and hardware should be recorded for each experiment. No validated environment lockfile is supplied here.

## Input data

The wrapper expects this dataset layout under `--root`:

```text
DATA_ROOT/
├── subset_20m/
│   └── m5C/
│       └── subset.pod5
└── references/
    ├── C_reference.fasta
    └── m5C_positions.bed
```

- **POD5:** raw current with read identifiers and calibration parameters.
- **Reference FASTA:** synthetic C-template references. The wrapper checks that BAM reference names start with `5mers_C_`.
- **Ground-truth BED:** single-nucleotide C positions in **0-based, half-open coordinates**. If a strand column is present, this wrapper accepts only `+`.
- **Aligned BAM:** reads aligned to the same reference, retaining Dorado `mv` tags and matching POD5 read identifiers. FASTQ or an aligned BAM without move information is insufficient for this workflow.

This entry point is specific to synthetic templates. Do not apply its reference checks, strand assumptions or negative-label interpretation unchanged to spliced biological RNA alignments. Existing raw-signal stores must come from the matching **m5C** POD5, not an inosine or other sample.

## Quick start: process all reads

From the repository root, with an existing aligned BAM containing move tags:

```bash
python outputs/m5c_pipeline/run_m5c.py \
  --root /path/to/DATA_ROOT \
  --bam /path/to/m5C.aligned.with_moves.bam \
  --output /path/to/results/m5c_all_reads \
  --max-reads 0 \
  --epochs 10 \
  --batch-size 32 \
  --device cuda
```

Use `--device cpu` for CPU execution. In WSL, a Windows drive path such as `G:\data` is usually accessed as `/mnt/g/data`.

**The wrapper defaults to a 100,000-read pilot. Set `--max-reads 0` to remove this limit.** This processes all eligible reads in the supplied POD5/BAM; it does not locate additional files outside those inputs. Unmapped, secondary, supplementary or low-MAPQ alignments are excluded, and reads without usable move information cannot yield aligned examples.

If an aligned BAM is unavailable, provide the Dorado model location instead:

```bash
python outputs/m5c_pipeline/run_m5c.py \
  --root /path/to/DATA_ROOT \
  --base-model /path/to/dorado_model \
  --output /path/to/results/m5c_from_pod5 \
  --max-reads 0 \
  --device cuda
```

The wrapper calls Dorado with reference alignment and move emission. Choose a basecalling model compatible with the sequencing chemistry and check the generated alignments before training.

### Reuse an extracted current store

```bash
python outputs/m5c_pipeline/run_m5c.py \
  --root /path/to/DATA_ROOT \
  --bam /path/to/m5C.aligned.with_moves.bam \
  --signal-store /path/to/m5c_raw_signal_store \
  --output /path/to/results/m5c_reuse_store \
  --max-reads 0 \
  --device cuda
```

The store must include `raw_signal_adc.int16.bin`, `read_signal_index.csv.gz` and `manifest.json`. Adding `--prepare-only` constructs the dataset without starting training. Use a new output directory when changing wrapper parameters; the wrapper checks its saved configuration and reuses completed stages.

## Dataset construction

For each candidate reference C, the pipeline records its coordinates, a 101-nt window, predicted secondary structure, pairing edges, minimum free energy and BED membership. Read-linked signal extraction then:

1. Joins BAM and POD5 records by read identifier.
2. Uses move information and reference alignment to map signal events to bases.
3. Converts ADC samples using `pA = (ADC + calibration_offset) × calibration_scale`.
4. Normalizes current using the median and scaled median absolute deviation.
5. Resamples each base-associated event to 16 values and builds a 21-base signal window by default.
6. Writes sharded metadata and NumPy tensors, retaining masks for unavailable positions.

The signal window and sequence window have different lengths by design: the default signal tensor is `[21, 16]`, whereas sequence and structure inputs cover 101 nucleotide positions.

### All reads versus all negative sites

Removing the read limit does **not** disable negative-site sampling. The builder defaults to `--negative-ratio 1.0`. To retain all negative candidate examples that pass its other filters, first prepare the candidate table and raw store with the wrapper, then run the builder into a fresh dataset directory:

```bash
python outputs/m5c_pipeline/build_read_site_dataset.py \
  --bam /path/to/m5C.aligned.with_moves.bam \
  --site-table /path/to/prepared/m5c_candidate_sites.pkl \
  --signal-index /path/to/m5c_raw_signal_store/read_signal_index.csv.gz \
  --signal-bin /path/to/m5c_raw_signal_store/raw_signal_adc.int16.bin \
  --output-dir /path/to/datasets/m5c_all_candidates \
  --max-reads 0 \
  --all-negatives \
  --metadata-format csv
```

The wrapper does not currently expose `--all-negatives` directly. For biological samples, absence from a BED file must not automatically be interpreted as a verified unmodified state.

## Train on a prepared dataset

```bash
python outputs/m5c_pipeline/train_m5c.py \
  --model-file outputs/m5c_pipeline/multimodal_rna_modification_model.py \
  --data /path/to/read_site_dataset \
  --sites /path/to/m5c_candidate_sites.pkl \
  --output /path/to/training/new_run \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --seed 42 \
  --device cuda
```

The trainer requires CSV.GZ metadata shards and uses memory-mapped NumPy signal arrays. Defaults include AdamW, a learning rate of `1e-4`, gradient clipping at 1.0 and binary cross-entropy with logits. It selects the best checkpoint by validation loss and evaluates that checkpoint on the test partition.

Splits are assigned deterministically by reference-name hashing, with nominal proportions of **80% training, 10% validation and 10% test**. Realized counts can differ. All reads and sites sharing one reference name remain together, and the trainer checks this constraint. Different names for identical or homologous sequences are not automatically clustered; a sequence-aware split audit is still necessary. All three partitions must be nonempty.

### Output files

```text
RESULTS/
├── pipeline_config.json
├── m5c_candidate_sites.pkl
├── raw_signal_store/                  # When extracted by the wrapper
│   ├── raw_signal_adc.int16.bin
│   ├── read_signal_index.csv.gz
│   └── manifest.json
├── read_site_dataset/
│   ├── dataset_manifest.json
│   ├── metadata/part-*.csv.gz
│   └── tensors/
│       ├── signal-part-*.npy
│       ├── mask-part-*.npy
│       └── dwell-part-*.npy
└── training/
    ├── config.json
    ├── history.jsonl
    ├── best.pt
    ├── last.pt
    └── test_metrics.json
```

The current trainer reports loss, sample count, accuracy, precision, recall and F1. AUROC, AUPRC and quantitative metrics are not calculated by this driver. Do not treat manuscript figure-generation scripts as a substitute for held-out prediction evaluation.

## Interpretation and limitations

- BED membership supplies site-derived labels, not verified modification states for every covering molecule. This distinction matters for partially modified preparations and biological samples.
- The present runnable workflow is m5C-specific. The base model contains m6A/m5C outputs, but other modification types require explicit task-specific preprocessing, labels and training adaptations.
- A sigmoid regression output computed for one read is not, by itself, a validated site-level modification fraction. Quantification requires cross-read aggregation, calibration and independent fraction labels.
- Local folding may miss long-range pairing and protein-bound conformations. Minimum free energy is saved for analysis, not used as an additional input branch.
- Co-occurrence between predicted modification states is not evidence of biochemical cooperation or competition without appropriate controls and independent validation.
- Manuscript plotting assets marked **illustrative** contain placeholder or simulated values. They must not be presented as results from completed model evaluation.

## Reproducibility

Retain the exact code revision, software environment, POD5/BAM provenance, Dorado model identifier, reference and annotation versions, split assignments, training configuration and checkpoint for every run. The example commands document the inspected implementation; they are not evidence of a completed end-to-end training run on your uploaded repository.

Only load trusted pickle files and PyTorch checkpoints. The current workflow uses pickle site tables and loads its own full training checkpoint.

## Citation and license

Publication metadata and a citation entry will be added when available. Refer to the repository's `LICENSE` file for reuse terms, if one is provided; this README does not assign a license.
