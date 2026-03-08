from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import time
import tracemalloc
from pathlib import Path


@dataclass
class BenchmarkResult:
    method: str
    runtime_s: float
    peak_mem_mb: float
    payload: dict


def run_with_benchmark(name: str, fn, *args, **kwargs) -> BenchmarkResult:
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = out
    if hasattr(out, "__dict__") and not isinstance(out, (list, tuple, dict, int, float, str, bool)):
        payload = out
    return BenchmarkResult(method=name, runtime_s=dt, peak_mem_mb=peak / (1024 * 1024), payload=payload)


def write_report(path: str | Path, result: BenchmarkResult):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, default=str)
