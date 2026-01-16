# train.py — training loop (search-guided policy & value learning)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable

import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from data import MarketFeatureBuilder
from spaces import PortfolioEnv
from models import (
    ModelConfig,
    AlphaZeroPortfolioModel,
    actor_loss_from_mcts,
    critic_loss,
)
from search import MCTSConfig, TreeNode


# ----------------------------
# Candidate Action Generator
# ----------------------------
@dataclass(frozen=True)
class CandidateActionConfig:
    num_actions: int = 32
    max_tilt: float = 0.02
    top_k: int = 5
    seed: int = 7


class CandidateActionGenerator:
    """
    Maps discrete action_id -> target portfolio weights.
    Actions are simple, interpretable reweighting moves.
    """
    def __init__(self, env: PortfolioEnv, cfg: CandidateActionConfig):
        self.env = env
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def __call__(self, action_id: int) -> np.ndarray:
        w = self.env.weights.copy()

        # Basic actions
        if action_id == 0:
            return w
        if action_id == 1:
            return np.ones(self.env.N) / self.env.N
        if action_id == 2:
            return 0.8 * w
        if action_id == 3:
            return 1.2 * w

        # Momentum-based tilts
        t = self.env.t
        L = 20
        if t - L < 0:
            mom = np.zeros(self.env.N)
        else:
            mom = self.env.logp.iloc[t].to_numpy() - self.env.logp.iloc[t - L].to_numpy()

        idx = np.argsort(mom)
        bottom = idx[: self.cfg.top_k]
        top = idx[-self.cfg.top_k :]

        tilt = self.cfg.max_tilt * self.rng.uniform(0.25, 1.0)
        w_new = w.copy()
        w_new[bottom] = np.maximum(0.0, w_new[bottom] - tilt / len(bottom))
        w_new[top] += tilt / len(top)

        return w_new


# ----------------------------
# MCTS with policy targets
# ----------------------------
class MCTSWithPolicy:
    def __init__(
        self,
        model: AlphaZeroPortfolioModel,
        env: PortfolioEnv,
        action_generator: Callable[[int], np.ndarray],
        cfg: MCTSConfig,
    ):
        self.model = model
        self.env = env
        self.action_generator = action_generator
        self.cfg = cfg

    def run(self):
        root = TreeNode(prior=1.0)
        obs = self.env.get_observation()
        priors, root_value, _ = self.model.infer(obs["market"], obs["portfolio"])

        for a, p in enumerate(priors):
            root.children[a] = TreeNode(prior=float(p))

        for _ in range(self.cfg.num_simulations):
            env_copy = copy.deepcopy(self.env)
            self._simulate(env_copy, root, depth=0)

        visits = np.array(
            [root.children[a].visit_count for a in range(len(root.children))],
            dtype=float,
        )
        pi = visits / visits.sum() if visits.sum() > 0 else np.ones_like(visits) / len(visits)
        best_action = int(np.argmax(visits))
        return self.action_generator(best_action), pi, root_value

    def _select(self, node: TreeNode):
        total = max(1, node.visit_count)
        best_score = -1e18
        best_a, best_child = 0, None
        for a, child in node.children.items():
            u = self.cfg.c_puct * child.prior * np.sqrt(total) / (1 + child.visit_count)
            score = child.value + u
            if score > best_score:
                best_score, best_a, best_child = score, a, child
        return best_a, best_child

    def _simulate(self, env: PortfolioEnv, node: TreeNode, depth: int) -> float:
        if depth >= self.cfg.max_depth:
            obs = env.get_observation()
            _, v, _ = self.model.infer(obs["market"], obs["portfolio"])
            return float(v)

        a, child = self._select(node)
        w_target = self.action_generator(a)
        _, r, done, _ = env.step(w_target)

        if done:
            v = 0.0
        else:
            if child.visit_count == 0:
                obs = env.get_observation()
                priors, v, _ = self.model.infer(obs["market"], obs["portfolio"])
                for aa, p in enumerate(priors):
                    child.children[aa] = TreeNode(prior=float(p))
                v = float(v)
            else:
                v = self._simulate(env, child, depth + 1)

        total = r + self.cfg.discount * v
        child.value_sum += total
        child.visit_count += 1
        node.visit_count += 1
        return total


# ----------------------------
# Training Config
# ----------------------------
@dataclass(frozen=True)
class TrainConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 64
    max_steps: int = 2_000
    mcts_sims: int = 80
    mcts_depth: int = 3
    save_path: str = "checkpoint.pt"


# ----------------------------
# Training Loop
# ----------------------------
def train(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
):
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    builder = MarketFeatureBuilder(prices, volumes)
    market_features = builder.batch_features()

    env = PortfolioEnv(prices, market_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroPortfolioModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    action_gen = CandidateActionGenerator(
        env, CandidateActionConfig(num_actions=model_cfg.num_actions)
    )
    mcts = MCTSWithPolicy(
        model,
        env,
        action_gen,
        MCTSConfig(
            num_simulations=train_cfg.mcts_sims,
            max_depth=train_cfg.mcts_depth,
        ),
    )

    buf_market, buf_port, buf_pi, buf_v = [], [], [], []

    obs = env.reset()
    for step in range(train_cfg.max_steps):
        w_target, pi, _ = mcts.run()
        buf_market.append(obs["market"])
        buf_port.append(obs["portfolio"])
        buf_pi.append(pi)

        obs_next, r, done, _ = env.step(w_target)

        with torch.no_grad():
            m = torch.tensor(obs_next["market"], dtype=torch.float32, device=device).unsqueeze(0)
            p = torch.tensor(obs_next["portfolio"], dtype=torch.float32, device=device).unsqueeze(0)
            _, v_next, _ = model(m, p)
            v_target = r + train_cfg.gamma * float(v_next.item()) * (0.0 if done else 1.0)
        buf_v.append(v_target)

        obs = obs_next
        if done:
            obs = env.reset()

        if len(buf_market) >= train_cfg.batch_size:
            idx = np.random.choice(len(buf_market), train_cfg.batch_size, replace=False)
            market_b = torch.tensor(np.array([buf_market[i] for i in idx]), dtype=torch.float32, device=device)
            port_b = torch.tensor(np.array([buf_port[i] for i in idx]), dtype=torch.float32, device=device)
            pi_b = torch.tensor(np.array([buf_pi[i] for i in idx]), dtype=torch.float32, device=device)
            v_b = torch.tensor(np.array([buf_v[i] for i in idx]).reshape(-1, 1), dtype=torch.float32, device=device)

            logits, value, _ = model(market_b, port_b)
            loss = actor_loss_from_mcts(logits, pi_b) + critic_loss(value, v_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    torch.save(
        {"model_state": model.state_dict(), "model_cfg": model_cfg.__dict__},
        train_cfg.save_path,
    )
    return model
