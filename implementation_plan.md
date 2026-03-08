# Multilayer Network Framework Plan (Execution-Focused)

## Do now
1. Build a PyTorch-first plugin architecture with method registry.
2. Implement core graph operators: supra-adjacency construction, supra-Laplacian, random-walk transition, and layer coupling.
3. Add ML/AI modules: multilayer embeddings, link prediction, and temporal-structural models.
4. Add benchmark harness with consistent train/val/test splits and layer-aware metrics.
5. Add baseline RL interface stubs (environment/state, action, reward) so RL can be integrated cleanly.

## Do with effort
1. Add advanced structural modules: motif mining, multislice modularity, percolation and cascade simulators.
2. Add causal inference and intervention evaluation utilities.
3. Add dataset adapters for external/legacy formats (edge lists, temporal snapshots, bipartite multiplex inputs).
4. Add optimization/performance layer: sparse kernel improvements, mixed precision, and scaling benchmarks.

## Notes
- This aligns with the framework goal: `single parameterized code base` with a shared execution API.
- All dataset ingestion should be reproducible and path-driven from `multilayer/data` manifests.
