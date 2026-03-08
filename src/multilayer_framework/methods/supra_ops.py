from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SupraOpsResult:
    method: str
    values: np.ndarray
    meta: dict


class SupraAdjacencyMethod:
    def __init__(self, graph):
        self.graph = graph

    def run(self):
        A = self.graph.supra_adjacency()
        nnz = A.nnz
        return SupraOpsResult(
            method="supra_adjacency",
            values=np.asarray(A.sum(axis=0)).ravel(),
            meta={"shape": tuple(A.shape), "nnz": int(nnz)},
        )


class SupraLaplacianMethod:
    def __init__(self, graph):
        self.graph = graph

    def run(self):
        L = self.graph.supra_laplacian()
        diag = np.asarray(L.diagonal()).ravel()
        return SupraOpsResult(
            method="supra_laplacian",
            values=diag,
            meta={"shape": tuple(L.shape), "trace": float(diag.sum())},
        )


class RandomWalkMethod:
    def __init__(self, graph):
        self.graph = graph

    def run(self, n_steps: int = 5):
        P = self.graph.transition()
        if P.shape[0] == 0:
            return SupraOpsResult(method="random_walk", values=np.array([]), meta={"n_steps": n_steps})
        v = np.ones(P.shape[0], dtype=np.float32)
        v /= v.sum()
        Q = P.copy()
        for _ in range(max(1, n_steps)):
            v = np.asarray(v @ Q).ravel()
            if v.sum() > 0:
                v = v / v.sum()
        return SupraOpsResult(
            method="random_walk",
            values=v,
            meta={"n_steps": n_steps, "l1": float(v.sum())},
        )


class CentralityMethod:
    def __init__(self, graph, alpha: float = 0.85, max_iter: int = 50):
        self.graph = graph
        self.alpha = alpha
        self.max_iter = max_iter

    def run(self):
        A = self.graph.transition()
        n = A.shape[0]
        if n == 0:
            return SupraOpsResult(method="supracentrality", values=np.array([]), meta={"alpha": self.alpha})
        v = np.ones(n, dtype=np.float32) / n
        p = np.ones(n, dtype=np.float32) / n
        for _ in range(self.max_iter):
            v = self.alpha * (v @ A).ravel() + (1.0 - self.alpha) * p
            s = v.sum()
            if s > 0:
                v = v / s
        return SupraOpsResult(method="supracentrality", values=v, meta={"alpha": self.alpha, "iter": self.max_iter})


def register_methods(reg):
    reg.register("supra_adjacency", SupraAdjacencyMethod, "Build supra-adjacency block operator")
    reg.register("supra_laplacian", SupraLaplacianMethod, "Compute supra-Laplacian diagonal")
    reg.register("random_walk", RandomWalkMethod, "Run short random-walk aggregation")
    reg.register("supracentrality", CentralityMethod, "Supracentrality via damped power iteration")
