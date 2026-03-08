# Multilayer Dataset Inventory (v1)

## Datasets collected for initial implementation

| Dataset | Domain | Source repo | Files pulled | Notes |
|---|---|---|---|---|
| GATNE-Amazon | E-commerce review-derived heterogeneous multiplex graph | `THUDM/GATNE` | `train.txt`, `valid.txt`, `test.txt`, `feature.txt` | Sampled multiplex dataset with explicit GATNE train/eval format. |
| GATNE-Twitter | Social interaction multiplex graph | `THUDM/GATNE` | `train.txt`, `valid.txt`, `test.txt` | Publicly documented in repo as source-sampled network. |
| GATNE-YouTube | Social interaction multiplex graph | `THUDM/GATNE` | `train.txt`, `valid.txt`, `test.txt` | Publicly documented in repo as source-sampled network. |
| GATNE-Example | Tiny template dataset | `THUDM/GATNE` | `train.txt`, `valid.txt`, `test.txt`, `feature.txt` | Minimal example for quick smoke tests. |
| DMGI-IMDb | Attributed multiplex graph | `pcy1302/DMGI` | `imdb.pkl` | Exact source link provided by repo README (dropbox). |
| DMGI-preprocessed | Attributed multiplex bundle | `pcy1302/DMGI` | `data.tar.gz` (+ extracted `imdb.pkl`, `dblp`, `amazon` preprocessed files expected) | Link provided by repo README. |
| MultiVERSE-CKM Physicians | Multiplex graph benchmark | `Lpiol/MultiVERSE` | `CKM-Physicians-Innovation_multiplex.edges` | Directly included in repository dataset folder. |
| MultiVERSE-MH | Multiplex-heterogeneous benchmark | `Lpiol/MultiVERSE` | `Multiplex_1.txt`, `Multiplex_2.txt`, `bipartite.txt` | Directly included in repository dataset folder. |
| MultiVERSE-MH-Toy | Synthetic toy benchmark | `Lpiol/MultiVERSE` | `M1_toy.txt`, `M2_toy.txt`, `bipartite_toy.txt` | Directly included in repository dataset folder. |

## Notes
- Some repositories (e.g., BraneMF, MGN-Net, muxViz) are method-centric and do not ship fixed standard datasets in-repo, only data formats/preprocessing instructions.
- These can be added later as adapter datasets when a concrete source is selected.
