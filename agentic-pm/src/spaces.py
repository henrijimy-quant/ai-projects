# spaces.py — portfolio environment, state/action definitions, and transitions

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Configs (frozen)
# ----------------------------
@dataclass(frozen=True)
class PortfolioConstraintConfig:
    long_only: bool = True
    max_weight: float = 0.05
    min_weight: float = 0.0
    turnover_cap: float = 0.20
    allow_cash: bool = True
    cash_weight_cap: float = 0.30
    eps: float = 1e-12


@dataclass(frozen=True)
class CostModelConfig:
    tc_bps: float = 5.0
    slippage_bps: float = 0.0


@dataclass(frozen=True)
class PortfolioFeatureConfig:
    lookback_port_ret: int = 5
    lookback_port_vol: int = 20
    eps: float = 1e-12


# ----------------------------
# Environment
# ----------------------------
class PortfolioEnv:
    """
    Portfolio environment with constrained reweighting actions.
    Market features are provided externally (from data.py).
    Portfolio features are computed here.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        market_features: pd.DataFrame,
        initial_value: float = 2_000_000.0,
        initial_weights: Optional[np.ndarray] = None,
        constraints: Optional[PortfolioConstraintConfig] = None,
        costs: Optional[CostModelConfig] = None,
        feat_cfg: Optional[PortfolioFeatureConfig] = None,
    ):
        if not prices.index.equals(market_features.index):
            market_features = market_features.reindex(prices.index)

        self.prices = prices.astype(float)
        self.market_features = market_features.astype(float)
        self.tickers = list(self.prices.columns)
        self.T, self.N = self.prices.shape

        self.constraints = constraints or PortfolioConstraintConfig()
        self.costs = costs or CostModelConfig()
        self.feat_cfg = feat_cfg or PortfolioFeatureConfig()

        self.logp = np.log(self.prices + self.constraints.eps)
        self.r1 = self.logp.diff().fillna(0.0)

        self.initial_value = float(initial_value)
        self.initial_weights = (
            self._project_weights(initial_weights)
            if initial_weights is not None
            else self._default_init_weights()
        )

        self.reset()

    # ----------------------------
    # Init / Reset
    # ----------------------------
    def _default_init_weights(self) -> np.ndarray:
        w = np.ones(self.N) / self.N
        return self._project_weights(w)

    def reset(self, start_t: Optional[int] = None) -> Dict[str, np.ndarray]:
        min_t = self.min_valid_t()
        self.t = int(min_t if start_t is None else start_t)
        if not (min_t <= self.t < self.T - 1):
            raise ValueError("Invalid start_t")

        self.value = self.initial_value
        self.peak_value = self.initial_value
        self.weights = self.initial_weights.copy()
        self.cash_weight = (
            max(0.0, 1.0 - self.weights.sum()) if self.constraints.allow_cash else 0.0
        )
        self.port_log_returns = np.zeros(self.T, dtype=float)
        return self.get_observation()

    def min_valid_t(self) -> int:
        return max(
            self.feat_cfg.lookback_port_vol,
            self.feat_cfg.lookback_port_ret,
            1,
        )

    # ----------------------------
    # Observations
    # ----------------------------
    def _portfolio_features_at(self, t: int) -> Dict[str, float]:
        cfg = self.feat_cfg

        r_start = max(0, t - cfg.lookback_port_ret + 1)
        r_log = self.port_log_returns[r_start : t + 1].sum()
        port_ret_5 = float(np.expm1(r_log))

        v_start = max(0, t - cfg.lookback_port_vol + 1)
        window = self.port_log_returns[v_start : t + 1]
        port_vol = float(np.std(window)) if window.size > 1 else 0.0

        dd = (self.value - self.peak_value) / max(self.peak_value, cfg.eps)
        drawdown = -float(dd)

        hhi = float(np.sum(self.weights ** 2))

        return {
            "port_ret_5": port_ret_5,
            "port_vol": port_vol,
            "drawdown": drawdown,
            "concentration_hhi": hhi,
        }

    def get_observation(self) -> Dict[str, np.ndarray]:
        mf = self.market_features.iloc[self.t]
        market_vec = np.array(
            [
                mf.get("momentum_20", np.nan),
                mf.get("vol_5", np.nan),
                mf.get("vol_of_vol", np.nan),
                mf.get("avg_corr_20", np.nan),
                mf.get("dispersion_1", np.nan),
            ],
            dtype=float,
        )

        pf = self._portfolio_features_at(self.t)
        portfolio_vec = np.array(
            [
                pf["port_ret_5"],
                pf["port_vol"],
                pf["drawdown"],
                pf["concentration_hhi"],
            ],
            dtype=float,
        )
        return {"market": market_vec, "portfolio": portfolio_vec}

    # ----------------------------
    # Action projection / costs
    # ----------------------------
    def _project_weights(self, w: np.ndarray) -> np.ndarray:
        c = self.constraints
        w = np.asarray(w, dtype=float).copy()

        if c.long_only:
            w = np.clip(w, c.min_weight, c.max_weight)
        else:
            w = np.clip(w, -c.max_weight, c.max_weight)

        s = w.sum()
        if c.allow_cash:
            if s > 1.0:
                w /= max(s, c.eps)
            if c.cash_weight_cap is not None:
                min_risky = 1.0 - c.cash_weight_cap
                if w.sum() < min_risky:
                    scale = min_risky / max(w.sum(), c.eps)
                    w = np.clip(w * scale, c.min_weight, c.max_weight)
                    if w.sum() > 1.0:
                        w /= max(w.sum(), c.eps)
        else:
            if s <= c.eps:
                w = np.ones_like(w) / w.size
            else:
                w /= s
        return w

    def _apply_turnover_cap(self, w_target: np.ndarray) -> Tuple[np.ndarray, float]:
        c = self.constraints
        dw = w_target - self.weights
        l1 = float(np.sum(np.abs(dw)))
        if l1 > c.turnover_cap:
            dw *= c.turnover_cap / max(l1, c.eps)
            w_new = self._project_weights(self.weights + dw)
            l1 = float(np.sum(np.abs(w_new - self.weights)))
            return w_new, l1
        return w_target, l1

    def _transaction_cost(self, turnover_l1: float) -> float:
        bps = (self.costs.tc_bps + self.costs.slippage_bps) / 10_000.0
        return float(bps * turnover_l1 * self.value)

    # ----------------------------
    # Step
    # ----------------------------
    def step(self, w_target: np.ndarray):
        if self.t >= self.T - 1:
            raise RuntimeError("Episode ended")

        w_target = self._project_weights(w_target)
        w_target, turnover = self._apply_turnover_cap(w_target)
        tc = self._transaction_cost(turnover)

        self.weights = w_target
        self.cash_weight = (
            max(0.0, 1.0 - self.weights.sum()) if self.constraints.allow_cash else 0.0
        )

        t0, t1 = self.t, self.t + 1
        r_assets = self.r1.iloc[t1].to_numpy()
        port_log_r = float(np.dot(self.weights, r_assets))
        self.port_log_returns[t1] = port_log_r

        prev_value = self.value
        self.value = max(prev_value * np.exp(port_log_r) - tc, self.feat_cfg.eps)
        self.peak_value = max(self.peak_value, self.value)
        self.t = t1

        reward = (self.value - prev_value) / max(prev_value, self.feat_cfg.eps)
        done = self.t >= self.T - 1

        info = {
            "t": self.t,
            "weights": self.weights.copy(),
            "cash_weight": self.cash_weight,
            "value": self.value,
            "peak_value": self.peak_value,
            "turnover_l1": turnover,
            "tc_dollars": tc,
            "port_log_return": port_log_r,
        }
        return self.get_observation(), float(reward), done, info
