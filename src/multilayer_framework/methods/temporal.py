from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TemporalCouplingResult:
    method: str
    values: np.ndarray
    meta: dict


class TemporalStructuralRegularizer:
    def __init__(self, lambda_tv: float = 0.1):
        self.lambda_tv = lambda_tv

    def run(self, layer_states: list[np.ndarray]):
        if not layer_states or len(layer_states) == 1:
            return TemporalCouplingResult("temporal_structural", np.array([0.0], dtype=np.float32), {"lambda_tv": self.lambda_tv})
        losses = []
        for i in range(1, len(layer_states)):
            diff = layer_states[i] - layer_states[i - 1]
            losses.append(float(np.linalg.norm(diff, ord=2)))
        val = float(self.lambda_tv * np.mean(losses))
        return TemporalCouplingResult("temporal_structural", np.array([val], dtype=np.float32), {"lambda_tv": self.lambda_tv})


class TemporalStructuralMethod:
    def __init__(self, graph, state: str = "degree", lambda_tv: float = 0.1):
        self.graph = graph
        self.state = state
        self.reg = TemporalStructuralRegularizer(lambda_tv=lambda_tv)

    def _layer_state(self, layer: int) -> np.ndarray:
        adj = self.graph._layer_adj(layer)
        if adj.shape[0] == 0:
            return np.array([], dtype=np.float32)
        degrees = np.asarray(adj.sum(axis=1)).ravel()
        if self.state == "strength":
            total = float(degrees.sum())
            if total > 0:
                return degrees / total
            return np.zeros(self.graph.num_nodes, dtype=np.float32)
        return np.asarray(degrees, dtype=np.float32)

    def run(self):
        if self.graph.num_layers < 2 or self.graph.num_nodes == 0:
            return TemporalCouplingResult(
                method="temporal_structural",
                values=np.array([0.0], dtype=np.float32),
                meta={
                    "lambda_tv": self.reg.lambda_tv,
                    "state": self.state,
                    "num_layers": int(self.graph.num_layers),
                },
            )

        layer_states = [self._layer_state(layer_id) for layer_id in self.graph.layer_ids]
        result = self.reg.run(layer_states)
        result.meta.update(
            {
                "state": self.state,
                "num_nodes": int(self.graph.num_nodes),
                "layers": list(self.graph.layer_ids),
            }
        )
        return result


def register_methods(reg):
    reg.register("temporal_structural", TemporalStructuralMethod, "Measure temporal smoothness across layer embeddings")
