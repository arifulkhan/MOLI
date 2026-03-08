from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..metrics import auc_roc


@dataclass
class LinkPredictionResult:
    method: str
    values: np.ndarray
    meta: dict


class BilinearLinkPredictionModel(nn.Module):
    def __init__(self, num_nodes: int, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, src: torch.Tensor, dst: torch.Tensor):
        hs = self.emb(src)
        ht = self.emb(dst)
        return (hs * ht).sum(dim=-1)


class LinkPredictionMethod:
    def __init__(self, graph, features=None, epochs: int = 5, lr: float = 0.05, neg_ratio: int = 1, dim: int = 16):
        self.graph = graph
        self.epochs = epochs
        self.lr = lr
        self.neg_ratio = neg_ratio
        self.dim = dim
        self.num_nodes = graph.num_nodes

    def _build_dataset(self, edges, labels):
        src = torch.as_tensor(edges[:, 1], dtype=torch.long)
        dst = torch.as_tensor(edges[:, 2], dtype=torch.long)
        y = torch.as_tensor(labels, dtype=torch.float32)
        return TensorDataset(src, dst, y)

    def _sample_negatives(self, n, device):
        u = torch.randint(low=0, high=self.num_nodes, size=(n,), device=device)
        v = torch.randint(low=0, high=self.num_nodes, size=(n,), device=device)
        return u, v

    def run(self, train_data, valid_data=None):
        tr_edges, tr_labels = train_data
        tr_edges = np.asarray(tr_edges)
        if tr_labels is None:
            tr_labels = np.ones(len(tr_edges), dtype=np.float32)

        model = BilinearLinkPredictionModel(self.num_nodes, self.dim)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()

        ds = self._build_dataset(tr_edges, tr_labels)
        loader = DataLoader(ds, batch_size=512, shuffle=True)
        model.train()

        for _ in range(self.epochs):
            for src, dst, y in loader:
                neg_u, neg_v = self._sample_negatives(len(src), src.device)
                neg_y = torch.zeros_like(y)
                all_src = torch.cat([src, neg_u], dim=0)
                all_dst = torch.cat([dst, neg_v], dim=0)
                all_y = torch.cat([y, neg_y], dim=0)

                logits = model(all_src, all_dst)
                loss = bce(logits, all_y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        def eval_split(split):
            if split is None:
                return None
            edges, labels = split
            if edges is None:
                return None
            s = torch.as_tensor(edges[:, 1], dtype=torch.long)
            d = torch.as_tensor(edges[:, 2], dtype=torch.long)
            y = torch.as_tensor(labels, dtype=torch.float32)
            with torch.no_grad():
                pred = torch.sigmoid(model(s, d)).cpu().numpy()
            score = auc_roc(y.numpy(), pred)
            return score

        metrics = {
            "val_auc": eval_split(valid_data),
            "epochs": int(self.epochs),
            "dim": int(self.dim),
        }

        with torch.no_grad():
            emb = model.emb.weight.detach().cpu().numpy()
        return LinkPredictionResult("link_prediction", emb, metrics)
