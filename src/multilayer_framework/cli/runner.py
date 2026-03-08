from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import DatasetConfig, FrameworkConfig
from ..core.loader import MultiLayerDataset
from ..core.graph import LayeredGraph
from ..core.registry import MethodRegistry
from ..methods.supra_ops import register_methods as register_supra
from ..methods.percolation import PercolationMethod
from ..methods.link_prediction import LinkPredictionMethod
from ..methods.temporal import TemporalStructuralMethod
from ..methods.rl_stub import RLInterventionMethod
from ..bench.runner import run_with_benchmark, write_report


def build_framework_config(args):
    return FrameworkConfig(
        dataset=DatasetConfig(path=args.dataset),
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


def build_graph(cfg, data):
    return LayeredGraph.from_edges(
        num_nodes=data["num_nodes"],
        edges=data["train"].edges,
        directed=False,
        coupling_weight=0.2,
    )


def build_methods(reg):
    register_supra(reg)
    reg.register("percolation", PercolationMethod, "simulate cascade robustness")
    reg.register("link_prediction", LinkPredictionMethod, "ML bilinear link predictor")
    reg.register("temporal_structural", TemporalStructuralMethod, "temporal structural regularizer")
    reg.register("rl_interventions", RLInterventionMethod, "simple intervention policy over layers")


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to dataset folder")
    p.add_argument("--method", default="supracentrality", choices=[
        "supra_adjacency",
        "supra_laplacian",
        "random_walk",
        "supracentrality",
        "percolation",
        "link_prediction",
        "temporal_structural",
        "rl_interventions",
    ])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="artifacts/moli_report.json")
    args = p.parse_args()

    cfg = build_framework_config(args)
    dataset = MultiLayerDataset(Path(cfg.dataset.path))
    data = dataset.load()
    graph = build_graph(cfg, data)

    reg = MethodRegistry()
    build_methods(reg)

    if args.method in {"supra_adjacency", "supra_laplacian", "random_walk", "supracentrality"}:
        method = reg.create(args.method, graph)
        if args.method == "random_walk":
            result = run_with_benchmark(args.method, method.run, n_steps=5)
        else:
            result = run_with_benchmark(args.method, method.run)
    elif args.method == "percolation":
        method = reg.create(args.method, graph)
        result = run_with_benchmark(args.method, method.run, p_remove=0.02)
    elif args.method == "temporal_structural":
        method = reg.create(args.method, graph)
        result = run_with_benchmark(args.method, method.run)
    elif args.method == "rl_interventions":
        method = reg.create(
            args.method,
            graph,
            episodes=args.episodes,
            max_steps=args.max_steps,
            epsilon=args.epsilon,
        )
        result = run_with_benchmark(args.method, method.run)
    else:
        method = reg.create(args.method, graph, epochs=cfg.epochs)
        train = (data["train"].edges, data["train"].labels if data["train"].labels is not None else None)
        valid = None
        if data["valid"] is not None:
            valid = (data["valid"].edges, data["valid"].labels)
        result = run_with_benchmark(args.method, method.run, train, valid)

    write_report(args.out, result)
    print(f"Run finished: {args.method}")
    print(f"runtime_s: {result.runtime_s:.4f}")
    print(f"peak_mem_mb: {result.peak_mem_mb:.3f}")


if __name__ == "__main__":
    run()
