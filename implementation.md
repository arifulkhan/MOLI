# implementation

## M1 Milestone (fresh)
- Core engine in `src/multilayer_framework` with PyTorch-first abstractions.
- Structural methods: supra-adjacency, supra-Laplacian, random-walk flow, supracentrality, percolation.
- ML methods: link prediction baseline model with layer-aware validation pipeline.
- Benchmark harness: runtime/memory + core metrics.
- RL extension point with `env.py` stubs.
- Dataset validation path driven by files in `data/`.

## M3 and M4 additions
- Added `temporal_structural` method to measure inter-layer smoothness on layer embeddings.
- Added `rl_interventions` method with a tiny environment and epsilon-greedy policy for intervention-style traversal.
