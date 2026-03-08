from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import sparse


@dataclass
class LayeredGraph:
    num_nodes: int
    num_layers: int
    layer_ids: List[int]
    edges_by_layer: Dict[int, np.ndarray]
    edge_weights_by_layer: Dict[int, np.ndarray] | None = None
    directed: bool = False
    coupling_weight: float = 0.2

    def _layer_adj(self, layer: int) -> sparse.csr_matrix:
        e = self.edges_by_layer.get(layer)
        if e is None or len(e) == 0:
            return sparse.identity(self.num_nodes, format="csr")
        rows = e[:, 1].astype(np.int64)
        cols = e[:, 2].astype(np.int64)
        if self.edge_weights_by_layer and layer in self.edge_weights_by_layer:
            data = self.edge_weights_by_layer[layer]
        else:
            data = np.ones(len(rows), dtype=np.float32)
        return sparse.coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes)).tocsr()

    @classmethod
    def from_edges(cls, num_nodes: int, edges: np.ndarray, directed: bool = False, coupling_weight: float = 0.2):
        layers = sorted(np.unique(edges[:, 0]).tolist())
        edges_by_layer: Dict[int, np.ndarray] = {}
        for l in layers:
            mask = edges[:, 0] == l
            edges_by_layer[l] = edges[mask][:, :3]
        return cls(
            num_nodes=num_nodes,
            num_layers=len(layers),
            layer_ids=layers,
            edges_by_layer=edges_by_layer,
            directed=directed,
            coupling_weight=coupling_weight,
        )

    def supra_adjacency(self) -> sparse.csr_matrix:
        if self.num_layers == 0:
            return sparse.csr_matrix((self.num_nodes * 0, self.num_nodes * 0))

        layers = self.layer_ids
        blocks: List[List[sparse.csr_matrix]] = []
        for i in layers:
            row: List[sparse.csr_matrix] = []
            for j in layers:
                if i == j:
                    row.append(self._layer_adj(i))
                else:
                    row.append(sparse.eye(self.num_nodes, format="csr", dtype=np.float32) * self.coupling_weight)
            blocks.append(row)
        return sparse.bmat(blocks, format="csr")

    def supra_laplacian(self) -> sparse.csr_matrix:
        A = self.supra_adjacency()
        deg = np.asarray(A.sum(axis=1)).ravel()
        return sparse.diags(deg) - A

    def transition(self) -> sparse.csr_matrix:
        A = self.supra_adjacency()
        deg = np.asarray(A.sum(axis=1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
        Dinv = sparse.diags(inv)
        return Dinv @ A

    def layer_from_node_state(self, state_id: int):
        if self.num_nodes == 0:
            return None, None
        layer_index = state_id // self.num_nodes
        if layer_index < 0 or layer_index >= len(self.layer_ids):
            return None, None
        return self.layer_ids[layer_index], state_id % self.num_nodes

    def layer_index_from_node_state(self, state_id: int):
        if self.num_nodes == 0:
            return None
        layer_index = state_id // self.num_nodes
        if layer_index < 0 or layer_index >= len(self.layer_ids):
            return None
        return int(layer_index)

    def state_count(self):
        return self.num_nodes * self.num_layers
