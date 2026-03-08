from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import connected_components


@dataclass
class PercolationResult:
    method: str
    values: np.ndarray
    meta: dict


class PercolationMethod:
    def __init__(self, graph):
        self.graph = graph

    def run(self, p_remove: float = 0.02):
        A = self.graph.supra_adjacency()
        if A.shape[0] == 0:
            return PercolationResult("percolation", np.array([]), {"removed": 0})
        rng = np.random.default_rng(0)
        mask = rng.random(A.shape[0]) > p_remove
        active = np.where(mask)[0]
        if len(active) == 0:
            return PercolationResult("percolation", np.array([0.0]), {"removed": int(A.shape[0])})

        A2 = A[active][:, active]
        _, labels = connected_components(csgraph=A2, directed=False, return_labels=True)
        if labels.size == 0:
            gsize = 0
        else:
            counts = np.bincount(labels)
            gsize = counts.max()
        return PercolationResult(
            method="percolation",
            values=np.array([float(gsize / len(active))], dtype=np.float32),
            meta={"removed": int(A.shape[0] - len(active)), "largest_component_fraction": float(gsize / len(active))},
        )
