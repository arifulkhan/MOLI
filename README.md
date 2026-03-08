# MOLI

**M**ultilayer **O**rchestration for **L**ayered **I**nference (MOLI)

MOLI is a lightweight, PyTorch-first framework for layered and multiplex graph experiments. It provides:

- Core structural operators on multilayer graphs (supra-adjacency, Laplacian, random-walk transitions)
- Learning and evaluation baselines (link prediction, percolation)
- Additional methods:
  - `temporal_structural`
  - `rl_interventions`
- Consistent benchmarking wrapper (runtime + peak memory + payload)
- Reproducible dataset ingestion from simple text files

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Main Use Cases](#main-use-cases)
- [Expected Outcomes](#expected-outcomes)
- [Artifacts](#artifacts)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Git (if cloning)

## Installation

From the project root:

```bash
cd /Users/arifkhan/Documents/Projects/AI/graph/multilayer
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy torch scipy
```

Optional (recommended for exploration):

```bash
pip install scikit-learn matplotlib
```

## Project Structure

```text
AI/graph/multilayer/
├─ src/
│  └─ multilayer_framework/
│     ├─ cli/runner.py          # CLI entry and method orchestration
│     ├─ core/                  # dataset loading + graph core + registry/config
│     ├─ methods/               # algorithm implementations
│     ├─ bench/runner.py        # benchmark wrapper
│     ├─ metrics.py             # evaluation helpers
│     └─ main.py                # module entrypoint
├─ data/                       # dataset folders
├─ configs/                    # optional experiment configs
├─ artifacts/                  # benchmark outputs
└─ .gitignore
```

## How to Run

Run the CLI from repository root:

```bash
cd /Users/arifkhan/Documents/Projects/AI/graph/multilayer
python -m src.multilayer_framework.main \
  --dataset data/GATNE-Example \
  --method supracentrality \
  --epochs 3 \
  --out artifacts/moli_supracentrality.json
```

The framework supports options:

- `--dataset`: path containing dataset split files
- `--method`: one of
  - `supra_adjacency`
  - `supra_laplacian`
  - `random_walk`
  - `supracentrality`
  - `percolation`
  - `link_prediction`
  - `temporal_structural`
  - `rl_interventions`
- `--epochs`: used by `link_prediction`
- `--episodes`: used by `rl_interventions`
- `--max-steps`: used by `rl_interventions`
- `--epsilon`: exploration ratio for `rl_interventions`
- `--out`: JSON output file for report

### Data format

`train.txt`, `valid.txt`, and `test.txt` contain one edge per line:

- Train: `layer source target`
- Valid/Test (if used for link prediction): `layer source target label`

Use datasets under `data/`, for example `data/GATNE-Example`.

## Main Use Cases

All runs print a completion line (`Run finished: <method>`) and timing summary in the terminal.

### 1) Structural methods

```bash
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method supra_adjacency --out artifacts/demo_supra_adjacency.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method supra_laplacian --out artifacts/demo_supra_laplacian.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method random_walk --out artifacts/demo_random_walk.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method supracentrality --epochs 3 --out artifacts/demo_supracentrality.json
```

### 2) Robustness and link prediction

```bash
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method percolation --out artifacts/demo_percolation.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method link_prediction --epochs 2 --out artifacts/demo_link_pred.json
```

### 3) Temporal structure and intervention methods

```bash
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method temporal_structural --out artifacts/demo_temporal_structural.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method rl_interventions --episodes 20 --max-steps 10 --out artifacts/demo_rl_interventions.json
```

If you want two saved run packs, use `m1`/`m2` suffixes when passing `--out`:

```bash
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method supracentrality --out artifacts/m1.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method rl_interventions --episodes 20 --out artifacts/m2.json
```

## Expected Outcomes

Every command writes JSON like:

```json
{
  "method": "supracentrality",
  "runtime_s": 0.12,
  "peak_mem_mb": 40.2,
  "payload": {
    "method": "supracentrality",
    "values": [...],
    "meta": {
      "alpha": 0.85,
      "iter": 50
    }
  }
}
```

Notes per method:

- `supra_adjacency` / `supra_laplacian`
  - `payload.values` contains adjacency/diagonal summary arrays
  - `meta.shape` and summary metrics are included
- `random_walk`
  - returns a probability-like vector and `meta.n_steps`
- `percolation`
  - returns one value: largest connected component fraction after node removal
- `link_prediction`
  - returns embedding matrix in `payload.values`
  - returns validation AUC in `payload.meta.val_auc` when `valid.txt` is available
- `temporal_structural`
  - returns a single scalar regularization score in `payload.values`
  - meta includes `lambda_tv`, `state`, `num_nodes`, `layers`
- `rl_interventions`
  - returns per-episode returns in `payload.values`
  - meta includes `mean_return`, `episodes`, `max_steps`, `epsilon`, `layer_visits`

A successful run prints:

```text
Run finished: <method>
runtime_s: <value>
peak_mem_mb: <value>
```

## Troubleshooting

- If method-specific arguments are missing, use `--help`:

```bash
python -m src.multilayer_framework.main --help
```

- If a dataset path is wrong, you’ll get `FileNotFoundError`
- If artifacts are not generated, check write permissions for `artifacts/`

## License

Internal/experimental project; add your preferred license file if you plan public distribution.
