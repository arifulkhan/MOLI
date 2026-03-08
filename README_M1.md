# Multilayer Framework M1

Run M1 methods from repo root:

```bash
cd /Users/arifkhan/Documents/Projects/AI/graph/multilayer
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method supracentrality --epochs 3 --out artifacts/m1_supracentrality.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method link_prediction --epochs 2 --out artifacts/m1_link_pred.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method percolation --out artifacts/m1_percolation.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method random_walk --out artifacts/m1_randomwalk.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method temporal_structural --out artifacts/m1_temporal.json
python -m src.multilayer_framework.main --dataset data/GATNE-Example --method rl_interventions --episodes 20 --max-steps 10 --out artifacts/m1_rl.json
```

Artifacts are emitted as JSON under `artifacts/` and include runtime and peak memory plus method payload.
