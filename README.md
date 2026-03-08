# MOLI

**M**ultilayer **O**rchestration for **L**ayered **I**nference (MOLI)

MOLI is a lightweight, PyTorch-first experimental framework for layered/multiplex graph analysis. It provides:

- Core structural operators on multilayer graphs (supra-adjacency, Laplacian, random-walk transitions)
- Supervised and unsupervised-style baselines (link prediction, percolation)
- Additional milestone methods:
  - `temporal_structural` (M3)
  - `rl_interventions` (M4)
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

Optional (recommended for notebooks / exploration):

```bash
pip install scikit-learn matplotlib
```

## Project Structure

```text
AI/graph/multilayer/
├─ README_M1.md                 # M1 milestone quick-run examples
├─ implementation.md            # milestone notes and what was completed
├─ implementation_plan.md        # roadmap
├─ src/
│  └─ multilayer_framework/
│     ├─ cli/runner.py          # CLI entry and method orchestration
│     ├─ core/                  # dataset loading + graph core + registry/config
│     ├─ methods/               # algo implementations
│     ├─ bench/runner.py        # benchmark wrapper
│     ├─ metrics.py             # simple evaluation helpers
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
  --out artifacts/m1_supracentrality.json
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

`train.txt`, `valid.txt`, and `test.txt` are expected to contain one edge per line:

- Train: `layer source target`
- Valid/Test (if used for link prediction): `layer source target label`

You can use the included datasets under `data/`, e.g. `data/GATNE-Example`.

## Main Use Cases

All runs print a simple status line (`M1 run finished: ...`) and metrics summary in the terminal.

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

### 3) M3 and M4

```bash
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method temporal_structural --out artifacts/demo_temporal_structural.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method rl_interventions --episodes 20 --max-steps 10 --out artifacts/demo_rl_interventions.json
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
  - `meta.shape` and sparsity/traces are included
- `random_walk`
  - returns a probability-like vector after a few steps and `meta.n_steps`
- `percolation`
  - returns one value: largest connected component fraction after node removal
- `link_prediction`
  - returns embedding matrix under `payload.values`
  - returns validation AUC in `payload.meta.val_auc` when `valid.txt` is available
- `temporal_structural`
  - returns a single scalar regularization score in `payload.values`
  - meta includes `lambda_tv`, `state`, `num_nodes`, `layers`
- `rl_interventions`
  - returns per-episode returns in `payload.values`
  - meta includes `mean_return`, `episodes`, `max_steps`, `epsilon`, `layer_visits`

A successful run prints:

```text
M1 run finished: <method>
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
