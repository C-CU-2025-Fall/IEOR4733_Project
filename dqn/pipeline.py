"""Shared DQN state, reward, and environment pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BP, EWMA_SPAN, MACD_PAIRS, MACD_STD_WINDOW, MACD_VOL_WINDOW
from dqn.spec import ACTION_VALUES, FEATURE_DIM, HORIZONS, RSI_WINDOW, SEQ_LEN, SIGMA_TGT, WARMUP


def compute_additive_returns(prices: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(prices), dtype=float)
    returns[1:] = prices[1:] - prices[:-1]
    return returns


def compute_ewma_sigma(returns: np.ndarray) -> np.ndarray:
    return pd.Series(returns).ewm(span=EWMA_SPAN, adjust=False).std().to_numpy(dtype=float)


def action_id_to_position(action_id: int) -> float:
    return float(ACTION_VALUES[int(action_id)])


def position_to_action_id(position: float) -> int:
    for idx, value in enumerate(ACTION_VALUES):
        if np.isclose(position, value):
            return idx
    raise ValueError(f"Unsupported position value: {position}")


def build_feature_matrix(prices: np.ndarray, returns: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Build the paper-faithful 8-dimensional state feature matrix."""
    n = len(prices)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    p_mean = prices.mean()
    p_std = prices.std() + 1e-10
    feats[:, 0] = (prices - p_mean) / p_std

    for idx, horizon in enumerate(HORIZONS):
        col = np.zeros(n, dtype=float)
        for i in range(horizon, n):
            col[i] = (prices[i] - prices[i - horizon]) / (sigma[i] * np.sqrt(horizon) + 1e-10)
        feats[:, idx + 1] = col

    price_series = pd.Series(prices)
    std63 = price_series.rolling(MACD_VOL_WINDOW, min_periods=1).std().to_numpy(dtype=float) + 1e-10
    macd_total = np.zeros(n, dtype=float)
    for short_span, long_span in MACD_PAIRS:
        ema_s = price_series.ewm(span=short_span, adjust=False).mean().to_numpy(dtype=float)
        ema_l = price_series.ewm(span=long_span, adjust=False).mean().to_numpy(dtype=float)
        q = (ema_s - ema_l) / std63
        std_q = pd.Series(q).rolling(MACD_STD_WINDOW, min_periods=1).std().to_numpy(dtype=float) + 1e-10
        macd_total += q / std_q
    feats[:, 5] = macd_total / len(MACD_PAIRS)

    delta = np.diff(prices, prepend=prices[0])
    gain = pd.Series(np.where(delta > 0, delta, 0.0)).rolling(RSI_WINDOW, min_periods=1).mean().to_numpy(dtype=float)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0)).rolling(RSI_WINDOW, min_periods=1).mean().to_numpy(dtype=float) + 1e-10
    feats[:, 6] = (50.0 - 50.0 / (1.0 + gain / loss)) / 50.0

    feats[:, 7] = sigma / (sigma.mean() + 1e-10)
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
    sigma_tgt: float = SIGMA_TGT,
    bp: float = BP,
) -> tuple[float, float, float, float]:
    """Paper Eq.4 reward for one step."""
    if idx <= 0 or idx >= len(returns):
        return 0.0, 0.0, 0.0, 0.0
    sig_prev = sigma[idx - 1]
    if not np.isfinite(sig_prev) or sig_prev <= 0:
        return 0.0, 0.0, 0.0, 0.0
    vol_scale = sigma_tgt / sig_prev
    gross = action * vol_scale * returns[idx]
    tc = bp * prices[idx - 1] * abs(action * vol_scale - prev_action * vol_scale)
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


class ContractEnv:
    """Single-contract environment using the shared DQN pipeline."""

    def __init__(self, contract: ContractArrays, sigma_tgt: float = SIGMA_TGT):
        self.contract = contract
        self.prices = contract.prices
        self.returns = contract.returns
        self.sigma = contract.sigma
        self.features = contract.features
        self.sigma_tgt = sigma_tgt
        self.max_idx = len(self.prices) - 1
        self.idx = WARMUP
        self.last_position = 0.0

    def reset(self) -> np.ndarray:
        self.idx = WARMUP
        self.last_position = 0.0
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
        )
        self.last_position = position
        done = self.idx >= self.max_idx - 1
        return get_feature_window(self.features, self.idx), float(reward), done

