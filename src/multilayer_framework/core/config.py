from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DatasetConfig:
    path: str
    name: str = "dataset"
    node_count: Optional[int] = None
    directed: bool = False
    coupling_weight: float = 0.2


@dataclass
class MethodConfig:
    method: str
    params: Dict[str, float | int | str | bool] | None = None


@dataclass
class FrameworkConfig:
    dataset: DatasetConfig
    method: str
    seed: int = 42
    device: str = "cpu"
    batch_size: int = 256
    epochs: int = 5
    eval_ratio: float = 0.2
    log_dir: str = "artifacts"
