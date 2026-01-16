# search.py — Monte Carlo Tree Search (AlphaZero-style) for portfolio reweighting

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Optional

import copy
import numpy as np


# ----------------------------
# Config (frozen)
# ----------------------------
@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 100
    max_depth: int = 3
    c_puct: float = 1.5
    discount: float = 0.99


# ----------------------------
# Tree Node
# ----------------------------
class TreeNode:
    """
    Node in the MCTS tree.
    Stores statistics for a (state, action) edge.
    """
    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, TreeNode] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ----------------------------
# MCTS
# ----------------------------
class MCTS:
    """
    AlphaZero-style MCTS.
    Uses:
      - policy priors from the actor
      - state value from the critic
      - environment transition via env.step()
    """

    def __init__(
        self,
        model,
        env,
        action_generator: Callable[[int], np.ndarray],
        cfg: Optional[MCTSConfig] = None,
    ):
        self.model = model
        self.env = env
        self.action_generator = action_generator
        self.cfg = cfg or MCTSConfig()

    def run(self) -> np.ndarray:
        """
        Run MCTS simulations and return selected target weights.
        """
        root = TreeNode(prior=1.0)

        obs = self.env.get_observation()
        priors, _, _ = self.model.infer(obs["market"], obs["portfolio"])
        for a, p in enumerate(priors):
            root.children[a] = TreeNode(prior=p)

        for _ in range(self.cfg.num_simulations):
            env_copy = copy.deepcopy(self.env)
            self._simulate(env_copy, root, depth=0)

        best_action = max(
            root.children.items(),
            key=lambda kv: kv[1].visit_count,
        )[0]
        return self.action_generator(best_action)

    def _simulate(self, env, node: TreeNode, depth: int) -> float:
        if depth >= self.cfg.max_depth:
            obs = env.get_observation()
            _, value, _ = self.model.infer(obs["market"], obs["portfolio"])
            return float(value)

        action, child = self._select(node)
        w_target = self.action_generator(action)
        _, reward, done, _ = env.step(w_target)

        if done:
            v = 0.0
        else:
            if child.visit_count == 0:
                obs = env.get_observation()
                priors, v, _ = self.model.infer(obs["market"], obs["portfolio"])
                for a, p in enumerate(priors):
                    child.children[a] = TreeNode(prior=p)
                v = float(v)
            else:
                v = self._simulate(env, child, depth + 1)

        total_value = reward + self.cfg.discount * v
        child.value_sum += total_value
        child.visit_count += 1
        node.visit_count += 1
        return total_value

    def _select(self, node: TreeNode):
        """
        PUCT selection rule.
        """
        best_score = -1e18
        best_action = None
        best_child = None

        total_visits = max(1, node.visit_count)

        for action, child in node.children.items():
            u = (
                self.cfg.c_puct
                * child.prior
                * np.sqrt(total_visits)
                / (1 + child.visit_count)
            )
            q = child.value
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child
