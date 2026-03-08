from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import os

import numpy as np


@dataclass
class MultiLayerSplit:
    edges: np.ndarray
    labels: np.ndarray | None = None


@dataclass
class MultiLayerDataset:
    path: Path
    directed: bool = False

    def _read_edges(self, fn: Path) -> np.ndarray:
        if not fn.exists():
            raise FileNotFoundError(fn)
        rows: List[Tuple[int, int, int]] = []
        with fn.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                l, u, v = map(int, parts[:3])
                rows.append((l, u, v))
                if not self.directed and u != v:
                    rows.append((l, v, u))
        return np.asarray(rows, dtype=np.int64)

    def _read_labeled(self, fn: Path) -> tuple[np.ndarray, np.ndarray]:
        edges: List[Tuple[int, int, int]] = []
        y: List[int] = []
        with fn.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                l, u, v, lbl = map(int, parts[:4])
                edges.append((l, u, v))
                y.append(lbl)
                if not self.directed and u != v:
                    edges.append((l, v, u))
                    y.append(lbl)
        return np.asarray(edges, dtype=np.int64), np.asarray(y, dtype=np.int64)

    @staticmethod
    def _read_features(fn: Path, max_node: int):
        with fn.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split()
            if len(header) < 2:
                return None
            n_nodes, dim = int(header[0]), int(header[1])
            x = np.zeros((n_nodes, dim), dtype=np.float32)
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                idx = int(parts[0])
                if idx >= max_node:
                    continue
                if idx >= n_nodes:
                    continue
                vals = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                x[idx] = vals
            return x

    def load(self) -> Dict[str, object]:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(path)
        train = path / "train.txt"
        valid = path / "valid.txt"
        test = path / "test.txt"

        tr_edges = self._read_edges(train)
        va_edges, va_labels = self._read_labeled(valid) if valid.exists() else (None, None)
        te_edges, te_labels = self._read_labeled(test) if test.exists() else (None, None)

        all_ids = set()
        for arr in (tr_edges, va_edges, te_edges):
            if arr is not None and arr.size:
                all_ids.update(arr[:, 1].tolist())
                all_ids.update(arr[:, 2].tolist())
                all_ids.update(arr[:, 0].tolist())

        num_nodes = max(all_ids) + 1 if all_ids else 0
        feat_path = path / "feature.txt"
        features = self._read_features(feat_path, num_nodes) if feat_path.exists() else None

        layer_ids = sorted(np.unique(tr_edges[:, 0]).tolist()) if tr_edges.size else []
        return {
            "train": MultiLayerSplit(edges=tr_edges, labels=None),
            "valid": MultiLayerSplit(edges=va_edges, labels=va_labels) if va_edges is not None else None,
            "test": MultiLayerSplit(edges=te_edges, labels=te_labels) if te_edges is not None else None,
            "num_nodes": num_nodes,
            "layers": layer_ids,
            "features": features,
            "data_path": str(path),
        }
