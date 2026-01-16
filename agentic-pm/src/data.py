# data.py — raw daily price/volume -> MARKET features per timestamp (for 60 stocks)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketFeatureConfig:
    lookback_momentum: int = 20   # window for momentum + correlation
    lookback_vol: int = 5         # window for realized volatility
    lookback_vol_of_vol: int = 5  # shift size to compare vol windows
    eps: float = 1e-12

class MarketFeatureBuilder:
    """
    Builds market features from raw daily price and volume.

    Inputs:
      prices  : pd.DataFrame (T x N) adjusted close, index=timestamps, columns=tickers
      volumes : pd.DataFrame (T x N) volume, same shape/index/columns

    Outputs at each timestep t (scalars):
      - momentum_20     : cross-sectional mean of 20-day log returns
      - vol_5           : cross-sectional mean of 5-day realized vol (std of daily log returns)
      - vol_of_vol      : cross-sectional mean of (vol_5 - vol_5 shifted by lookback_vol_of_vol)
      - avg_corr_20     : average pairwise correlation of daily log returns over 20 days
      - dispersion_1    : cross-sectional std of 1-day log returns
      - volume_shock_20 : (optional extra) cross-sectional mean z-score of volume vs 20D rolling stats
    """
    def __init__(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        cfg: Optional[MarketFeatureConfig] = None,
        tickers: Optional[list[str]] = None,
    ):
        self.cfg = cfg or MarketFeatureConfig()

        if tickers is not None:
            prices = prices[tickers]
            volumes = volumes[tickers]

        if not prices.index.equals(volumes.index):
            raise ValueError("prices and volumes must have identical timestamp index.")
        if not prices.columns.equals(volumes.columns):
            raise ValueError("prices and volumes must have identical ticker columns.")

        self.prices = prices.astype(float).copy()
        self.volumes = volumes.astype(float).copy()
        self.tickers = list(self.prices.columns)

        # Precompute log-prices and 1-step log returns
        self.logp = np.log(self.prices.replace(0, np.nan))
        self.r1 = self.logp.diff()

        # Precompute rolling ingredients
        self._prep_rolling()

    def _prep_rolling(self) -> None:
        c = self.cfg

        # Momentum (L-day log return)
        self.mom_L = self.logp.diff(c.lookback_momentum)

        # Realized vol over S-day window (std of daily returns)
        self.vol_S = self.r1.rolling(window=c.lookback_vol).std()

        # Prior vol window for "vol of vol"
        self.vol_S_prev = self.vol_S.shift(c.lookback_vol_of_vol)

        # Optional: volume z-score vs 20-day rolling mean/std
        vol_mean = self.volumes.rolling(window=c.lookback_momentum).mean()
        vol_std = self.volumes.rolling(window=c.lookback_momentum).std().replace(0, np.nan)
        self.vol_z = (self.volumes - vol_mean) / (vol_std + c.eps)

    def min_valid_t(self) -> int:
        """
        Earliest integer row index t where the core features are defined.
        """
        c = self.cfg
        # Need:
        # - r1 defined: t >= 1
        # - mom_L defined: t >= lookback_momentum
        # - vol_S defined: t >= lookback_vol
        # - vol_of_vol uses shift: t >= lookback_vol + lookback_vol_of_vol
        return max(1, c.lookback_momentum, c.lookback_vol, c.lookback_vol + c.lookback_vol_of_vol)

    def _avg_pairwise_corr(self, t: int) -> float:
        """
        Average pairwise correlation of daily log returns over last lookback_momentum days ending at t.
        """
        c = self.cfg
        window = self.r1.iloc[t - c.lookback_momentum + 1 : t + 1].to_numpy()

        # Drop rows with NaNs (e.g., around missing data)
        if np.isnan(window).any():
            window = window[~np.isnan(window).any(axis=1)]
        if window.shape[0] < 2:
            return float("nan")

        corr = np.corrcoef(window, rowvar=False)  # (N x N)
        if np.isnan(corr).all():
            return float("nan")

        n = corr.shape[0]
        off_diag_sum = np.nansum(corr) - np.nansum(np.diag(corr))
        denom = n * (n - 1)
        return float(off_diag_sum / max(denom, 1))

    def features_at(self, t: int) -> Dict[str, float]:
        """
        Compute market features at integer location t.
        """
        if t < self.min_valid_t():
            raise ValueError(f"t={t} too early; need t >= {self.min_valid_t()}")

        mom20 = float(np.nanmean(self.mom_L.iloc[t].to_numpy()))
        vol5 = float(np.nanmean(self.vol_S.iloc[t].to_numpy()))
        vol_of_vol = float(np.nanmean((self.vol_S.iloc[t] - self.vol_S_prev.iloc[t]).to_numpy()))
        avg_corr20 = self._avg_pairwise_corr(t)
        disp1 = float(np.nanstd(self.r1.iloc[t].to_numpy()))
        vol_shock = float(np.nanmean(self.vol_z.iloc[t].to_numpy()))

        return {
            "momentum_20": mom20,
            "vol_5": vol5,
            "vol_of_vol": vol_of_vol,
            "avg_corr_20": float(avg_corr20),
            "dispersion_1": disp1,
            "volume_shock_20": vol_shock,  # optional; ignore if you want exactly the core 5
        }

    def batch_features(self, start_t: Optional[int] = None, end_t: Optional[int] = None) -> pd.DataFrame:
        """
        Compute features for a range of integer locations [start_t, end_t] inclusive.
        Returns a DataFrame indexed by timestamps.
        """
        if start_t is None:
            start_t = self.min_valid_t()
        if end_t is None:
            end_t = len(self.prices) - 1

        start_t = max(start_t, self.min_valid_t())
        end_t = min(end_t, len(self.prices) - 1)

        rows = []
        idx = []
        for t in range(start_t, end_t + 1):
            rows.append(self.features_at(t))
            idx.append(self.prices.index[t])

        return pd.DataFrame(rows, index=pd.Index(idx, name=self.prices.index.name))


if __name__ == "__main__":
    # Example usage (you supply prices_df and volumes_df):
    # builder = MarketFeatureBuilder(prices_df, volumes_df)
    # mf_df = builder.batch_features()
    # print(mf_df.head())
    pass
