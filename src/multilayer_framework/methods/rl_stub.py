from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Tuple

import numpy as np


@dataclass
class RLState:
    node: int
    layer: int
    t: int


class InterventionsEnv:
    """Simple RL environment for interventional sampling on a multiplex graph."""

    def __init__(self, graph, max_steps: int = 10, seed: int = 0):
        self.graph = graph
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.t = 0
        self.state = RLState(0, 0, 0)
        self._action_space = list(range(graph.num_layers)) if graph.num_layers else [0]

    def reset(self):
        self.t = 0
        if self.graph.num_nodes == 0:
            self.state = RLState(-1, 0, self.t)
            return self.state

        layer = self.rng.choice(self._action_space)
        node = self.rng.randrange(self.graph.num_nodes)
        self.state = RLState(node=node, layer=layer, t=self.t)
        return self.state

    def step(self, action):
        self.t += 1
        done = self.t >= self.max_steps or self.graph.num_nodes == 0
        if done:
            return self.state, 0.0, True, {"layer": int(self.state.layer)}

        next_layer = int(action) % len(self._action_space) if self._action_space else 0
        next_layer = self._action_space[next_layer]
        from_node = self.state.node

        next_node = from_node
        if self.graph.num_nodes and self.graph.num_nodes > 0:
            adj = self.graph._layer_adj(next_layer)
            row = adj[from_node].toarray().ravel() if adj.shape[0] > 0 else np.array([])
            candidates = np.where(row > 0)[0].astype(np.int64).tolist()
            if candidates:
                next_node = self.rng.choice(candidates)

        reward = self._reward(from_node, next_node, next_layer)
        self.state = RLState(node=next_node, layer=next_layer, t=self.t)
        return self.state, reward, done, {"layer": next_layer}

    def _reward(self, from_node: int, to_node: int, layer: int) -> float:
        if self.graph.num_nodes == 0 or to_node < 0:
            return 0.0
        adj = self.graph._layer_adj(layer)
        if adj.shape[0] == 0:
            return 0.0
        degrees = np.asarray(adj.sum(axis=1)).ravel()
        if degrees.max() <= 0:
            return 0.0
        return float(degrees[to_node] / float(degrees.max()))


@dataclass
class RLInterventionResult:
    method: str
    values: np.ndarray
    meta: dict


class RLInterventionMethod:
    def __init__(self, graph, episodes: int = 20, max_steps: int = 10, epsilon: float = 0.2, seed: int = 0):
        self.graph = graph
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def run(self):
        if self.graph.num_layers == 0 or self.graph.num_nodes == 0:
            return RLInterventionResult(
                method="rl_interventions",
                values=np.array([0.0], dtype=np.float32),
                meta={
                    "episodes": int(self.episodes),
                    "max_steps": int(self.max_steps),
                    "epsilon": float(self.epsilon),
                },
            )

        env = InterventionsEnv(self.graph, max_steps=self.max_steps, seed=123)
        episode_returns: list[float] = []
        layer_visits = np.zeros(self.graph.num_layers, dtype=np.float32)

        for _ in range(max(1, self.episodes)):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                if self.rng.random() < self.epsilon:
                    action = self.rng.randrange(self.graph.num_layers)
                else:
                    ranked: list[Tuple[int, float]] = []
                    for a in range(self.graph.num_layers):
                        score = env._reward(state.node, state.node, a)
                        ranked.append((a, score))
                    action, _ = max(ranked, key=lambda x: x[1])

                state, reward, done, info = env.step(action)
                layer = int(info.get("layer", 0))
                if 0 <= layer < len(layer_visits):
                    layer_visits[layer] += 1.0
                total_reward += reward

            episode_returns.append(total_reward)

        returns = np.asarray(episode_returns, dtype=np.float32)
        return RLInterventionResult(
            method="rl_interventions",
            values=returns,
            meta={
                "episodes": int(len(returns)),
                "max_steps": int(self.max_steps),
                "epsilon": float(self.epsilon),
                "mean_return": float(returns.mean()) if returns.size else 0.0,
                "layer_visits": layer_visits.astype(float).tolist(),
                "layer_ids": list(self.graph.layer_ids),
            },
        )


def register_methods(reg):
    reg.register("rl_interventions", RLInterventionMethod, "Evaluate a simple intervention policy over layer actions")
