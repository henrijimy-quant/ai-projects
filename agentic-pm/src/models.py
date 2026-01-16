# models.py — encoder + actor–critic networks

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config (frozen)
# ----------------------------
@dataclass(frozen=True)
class ModelConfig:
    market_dim: int = 5
    portfolio_dim: int = 4
    latent_dim: int = 32
    num_actions: int = 32
    encoder_hidden: Tuple[int, ...] = (64, 64)
    head_hidden: Tuple[int, ...] = (64,)
    dropout: float = 0.0


# ----------------------------
# Helpers
# ----------------------------
def _mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ----------------------------
# Encoder
# ----------------------------
class Encoder(nn.Module):
    """
    Encodes (market_features + portfolio_features) -> latent state z
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = _mlp(
            cfg.market_dim + cfg.portfolio_dim,
            cfg.encoder_hidden,
            cfg.latent_dim,
            cfg.dropout,
        )

    def forward(self, market: torch.Tensor, portfolio: torch.Tensor) -> torch.Tensor:
        x = torch.cat([market, portfolio], dim=-1)
        return self.net(x)


# ----------------------------
# Actor
# ----------------------------
class ActorHead(nn.Module):
    """
    Outputs action logits over a discrete candidate action set.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = _mlp(
            cfg.latent_dim + cfg.portfolio_dim,
            cfg.head_hidden,
            cfg.num_actions,
            cfg.dropout,
        )

    def forward(self, z: torch.Tensor, portfolio: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, portfolio], dim=-1)
        return self.net(x)


# ----------------------------
# Critic
# ----------------------------
class CriticHead(nn.Module):
    """
    Outputs scalar value estimate for (state, portfolio).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = _mlp(
            cfg.latent_dim + cfg.portfolio_dim,
            cfg.head_hidden,
            1,
            cfg.dropout,
        )

    def forward(self, z: torch.Tensor, portfolio: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, portfolio], dim=-1)
        return self.net(x)


# ----------------------------
# Full Model
# ----------------------------
class AlphaZeroPortfolioModel(nn.Module):
    """
    Encoder + Actor + Critic
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.actor = ActorHead(cfg)
        self.critic = CriticHead(cfg)

    def forward(self, market: torch.Tensor, portfolio: torch.Tensor):
        z = self.encoder(market, portfolio)
        logits = self.actor(z, portfolio)
        value = self.critic(z, portfolio)
        return logits, value, z

    @torch.no_grad()
    def infer(
        self,
        market_np: np.ndarray,
        portfolio_np: np.ndarray,
        device: Optional[torch.device] = None,
    ):
        device = device or next(self.parameters()).device
        market = torch.tensor(market_np, dtype=torch.float32, device=device).unsqueeze(0)
        portfolio = torch.tensor(portfolio_np, dtype=torch.float32, device=device).unsqueeze(0)

        z = self.encoder(market, portfolio)
        logits = self.actor(z, portfolio)
        priors = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        value = self.critic(z, portfolio).squeeze(0).item()
        return priors, value, z.squeeze(0).cpu().numpy()


# ----------------------------
# Loss helpers (used in training)
# ----------------------------
def actor_loss_from_mcts(logits: torch.Tensor, target_pi: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(target_pi * logp).sum(dim=-1).mean()


def critic_loss(value: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(value, target_v)
