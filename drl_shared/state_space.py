"""Global shared state-space, feature engineering, and Eq.4 reward helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BP, EWMA_SPAN, MACD_PAIRS, MACD_VOL_WINDOW
from drl_shared.spec import (
    CONTINUOUS_ACTION_RANGE,
    DISCRETE_ACTION_VALUES,
    FEATURE_DIM,
    HORIZONS,
    RSI_WINDOW,
    SEQ_LEN,
    SIGMA_TGT_DEFAULT,
    WARMUP,
)


def compute_additive_returns(prices: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(prices), dtype=float)
    returns[1:] = prices[1:] - prices[:-1]
    return returns


def compute_ewma_sigma(returns: np.ndarray) -> np.ndarray:
    return pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().to_numpy(dtype=float)


def action_id_to_position(action_id: int) -> float:
    return float(DISCRETE_ACTION_VALUES[int(action_id)])


def position_to_action_id(position: float) -> int:
    for idx, value in enumerate(DISCRETE_ACTION_VALUES):
        if np.isclose(position, value):
            return idx
    raise ValueError(f"Unsupported position value: {position}")


def continuous_action_to_position(action: float) -> float:
    lo, hi = CONTINUOUS_ACTION_RANGE
    return float(np.clip(action, lo, hi))


def build_feature_matrix(
    prices: np.ndarray,
    returns: np.ndarray,
    sigma: np.ndarray,
    feature_spec_override: dict | None = None,
) -> np.ndarray:
    """Build the shared 7-dimensional state matrix (paper Section 3.1).

    Features:
    0: Normalized close price (60-day rolling std)
    1-4: Returns over 21/42/63/252 days, normalized by sigma*sqrt(H)
    5: MACD (averaged multi-scale) normalized by 63-day price volatility
    6: RSI(30) normalized to [-1, 1]
    """
    n = len(prices)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    # Feature 0: Normalized close price series (z-score)
    rolling_mean = pd.Series(prices).rolling(window=60, min_periods=5).mean().to_numpy(dtype=float)
    rolling_std = pd.Series(prices).rolling(window=60, min_periods=5).std().to_numpy(dtype=float)
    feats[:, 0] = (prices - rolling_mean) / (rolling_std + 1e-10)

    # Features 1-4: Returns over horizons, normalized by sigma*sqrt(H)
    for idx, horizon in enumerate(HORIZONS):
        col = np.zeros(n, dtype=float)
        for i in range(horizon, n):
            col[i] = (prices[i] - prices[i - horizon]) / (sigma[i] * np.sqrt(horizon) + 1e-10)
        feats[:, idx + 1] = col

    # Feature 5: MACD (averaged multi-scale, Eq.3)
    # Inner: q_t = (m(S) - m(L)) / std(p_{t-63:t})
    # Outer: MACD_t = q_t / std(q_{t-252:t})
    macd_total = np.zeros(n, dtype=float)
    for short_span, long_span in MACD_PAIRS:
        ema_s = pd.Series(prices).ewm(span=short_span, adjust=False).mean().to_numpy(dtype=float)
        ema_l = pd.Series(prices).ewm(span=long_span, adjust=False).mean().to_numpy(dtype=float)
        macd_total += (ema_s - ema_l)
    macd_raw = macd_total / max(len(MACD_PAIRS), 1)
    macd_vol = (
        pd.Series(prices)
        .rolling(window=MACD_VOL_WINDOW, min_periods=5)
        .std()
        .to_numpy(dtype=float)
    )
    q_t = macd_raw / (macd_vol + 1e-10)
    q_std_252 = pd.Series(q_t).rolling(window=252, min_periods=21).std().to_numpy(dtype=float)
    feats[:, 5] = q_t / (q_std_252 + 1e-10)

    # Feature 6: RSI(30) — Wilder smoothing (α=1/n EMA), normalized to [-1, 1]
    delta = np.diff(prices, prepend=prices[0])
    alpha = 1.0 / RSI_WINDOW
    gain = pd.Series(np.where(delta > 0, delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float) + 1e-10
    rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    feats[:, 6] = (rsi - 50.0) / 50.0

    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def get_feature_window(features: np.ndarray, idx: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    if idx < seq_len:
        pad = np.zeros((seq_len - idx, FEATURE_DIM), dtype=np.float32)
        return np.vstack([pad, features[:idx]])
    return features[idx - seq_len:idx]


def compute_eq4_reward(
    prices: np.ndarray,
    returns: np.ndarray,
    sigma: np.ndarray,
    idx: int,
    action: float,
    prev_action: float,
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    bp: float = BP,
    prev_sigma: float | None = None,
) -> tuple[float, float, float, float]:
    """Paper Eq.4 reward for one step.

    TC uses σ_{t-1} for current position and σ_{t-2} for previous position.
    At time t, action A_{t-1} was decided based on r_t, so:
      vol_scale_current = σ_tgt / σ_{t-1}
      vol_scale_prev    = σ_tgt / σ_{t-2}  (from prev_sigma)
    """
    if idx <= 0 or idx >= len(returns):
        return 0.0, 0.0, 0.0, 0.0
    sig_t_1 = sigma[idx - 1]
    if not np.isfinite(sig_t_1) or sig_t_1 <= 0:
        return 0.0, 0.0, 0.0, 0.0
    vol_scale = sigma_tgt / sig_t_1
    gross = action * vol_scale * returns[idx]

    # TC: use σ_{t-2} for previous position's vol_scale
    if prev_sigma is not None and np.isfinite(prev_sigma) and prev_sigma > 0:
        vol_scale_prev = sigma_tgt / prev_sigma
    else:
        vol_scale_prev = vol_scale
    tc = bp * prices[idx - 1] * abs(action * vol_scale - prev_action * vol_scale_prev)
    return gross - tc, gross, tc, vol_scale


@dataclass
class ContractArrays:
    ticker: str
    prices: np.ndarray
    returns: np.ndarray
    sigma: np.ndarray
    features: np.ndarray
    dates: np.ndarray
    source: str


def build_contract_arrays(
    ticker: str,
    prices: np.ndarray,
    dates: np.ndarray,
    source: str,
    feature_spec_override: dict | None = None,
) -> ContractArrays:
    returns = compute_additive_returns(prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma, feature_spec_override=feature_spec_override)
    return ContractArrays(
        ticker=ticker,
        prices=np.asarray(prices, dtype=float),
        returns=returns,
        sigma=sigma,
        features=features,
        dates=np.asarray(dates),
        source=source,
    )


class ContractEnv:
    """Single-contract environment using the shared state/reward pipeline."""

    def __init__(
        self,
        contract: ContractArrays,
        sigma_tgt: float = SIGMA_TGT_DEFAULT,
        start_idx: int = WARMUP,
        max_idx: int | None = None,
    ):
        self.contract = contract
        self.prices = contract.prices
        self.returns = contract.returns
        self.sigma = contract.sigma
        self.features = contract.features
        self.sigma_tgt = sigma_tgt
        self.start_idx = max(WARMUP, int(start_idx))
        self.max_idx = min(len(self.prices) - 1, int(max_idx)) if max_idx is not None else len(self.prices) - 1
        self.idx = self.start_idx
        self.last_position = 0.0
        self.last_sigma = self.sigma[self.start_idx - 1] if self.start_idx >= 1 else self.sigma[0]

    def reset(self) -> np.ndarray:
        self.idx = self.start_idx
        self.last_position = 0.0
        self.last_sigma = self.sigma[self.start_idx - 1] if self.start_idx >= 1 else self.sigma[0]
        return get_feature_window(self.features, self.idx)

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool]:
        position = action_id_to_position(action_id)
        self.idx += 1
        if self.idx >= self.max_idx:
            return get_feature_window(self.features, min(self.idx, self.max_idx)), 0.0, True

        reward, _, _, _ = compute_eq4_reward(
            self.prices,
            self.returns,
            self.sigma,
            self.idx,
            position,
            self.last_position,
            sigma_tgt=self.sigma_tgt,
            bp=BP,
            prev_sigma=self.last_sigma,
        )
        self.last_sigma = self.sigma[self.idx - 1]
        self.last_position = position
        done = self.idx >= self.max_idx - 1
        return get_feature_window(self.features, self.idx), float(reward), done
