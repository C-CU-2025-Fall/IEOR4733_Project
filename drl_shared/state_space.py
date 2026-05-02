"""Global shared state-space, feature engineering, and Eq.4 reward helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BP, EWMA_SPAN, MACD_VOL_WINDOW
from drl_shared.spec import (
    CONTINUOUS_ACTION_RANGE,
    DISCRETE_ACTION_VALUES,
    FEATURE_DIM,
    HORIZONS,
    MACD_PAIRS_ACTIVE,
    MARKET_FEATURE_DIM,
    RSI_WINDOW,
    RSI_WINDOWS,
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
    """Build the 5-dimensional market feature matrix (pruned v2).

    Features:
    0: r_t / sigma_t — 1-day vol-normalized return (replaces non-stationary p_t/rollstd)
    1: Return over 21 days, normalized by sigma*sqrt(21)
    2: MACD (8, 24) normalized
    3: MACD (16, 48) normalized
    4: RSI(30) normalized to [-1, 1]

    Previous action (6th channel) is appended at runtime by ContractEnv.
    """
    n = len(prices)
    feats = np.zeros((n, MARKET_FEATURE_DIM), dtype=np.float32)

    # Feature 0: r_t / sigma_t — 1-day vol-normalized return
    feats[:, 0] = returns / (sigma + 1e-10)

    # Feature 1: Return over 21 days, normalized by sigma*sqrt(H)
    horizon = HORIZONS[0]  # 21
    col = np.zeros(n, dtype=float)
    for i in range(horizon, n):
        col[i] = (prices[i] - prices[i - horizon]) / (sigma[i] * np.sqrt(horizon) + 1e-10)
    feats[:, 1] = col

    # Features 2-3: MACD pairs (8,24) and (16,48), each independently normalized
    macd_vol = (
        pd.Series(prices)
        .rolling(window=MACD_VOL_WINDOW, min_periods=5)
        .std()
        .to_numpy(dtype=float)
    )
    for macd_idx, (short_span, long_span) in enumerate(MACD_PAIRS_ACTIVE):
        ema_s = pd.Series(prices).ewm(span=short_span, adjust=False).mean().to_numpy(dtype=float)
        ema_l = pd.Series(prices).ewm(span=long_span, adjust=False).mean().to_numpy(dtype=float)
        q_t = (ema_s - ema_l) / (macd_vol + 1e-10)
        q_std_252 = pd.Series(q_t).rolling(window=252, min_periods=21).std().to_numpy(dtype=float)
        feats[:, 2 + macd_idx] = q_t / (q_std_252 + 1e-10)

    # Feature 4: RSI(30) — Wilder smoothing (α=1/n EMA), normalized to [-1, 1]
    delta = np.diff(prices, prepend=prices[0])
    alpha = 1.0 / RSI_WINDOW
    gain = pd.Series(np.where(delta > 0, delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float) + 1e-10
    rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    feats[:, 4] = (rsi - 50.0) / 50.0

    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def build_feature_matrix_v3(
    prices: np.ndarray,
    returns: np.ndarray,
    sigma: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    oi: np.ndarray | None = None,
    feature_spec_override: dict | None = None,
) -> np.ndarray:
    """Build the 12-dimensional market feature matrix (enhanced).

    Features:
    0: r_t / sigma_t                   - ultra-short momentum
    1: ret_5d / (sigma * sqrt(5))      - short-term momentum
    2: ret_21d / (sigma * sqrt(21))    - medium-term momentum
    3: ret_126d / (sigma * sqrt(126))  - long-term trend
    4: MACD(8,24) normalized           - trend (paper Eq.3)
    5: RSI_5 normalized                - ultra-short oscillator
    6: RSI_30 normalized               - medium-term oscillator
    7: ATR(20) / ATR_MA(20)            - volatility regime (OHLC)
    8: Volume / Volume_MA(20)          - liquidity / activity
    9: ΔOI / |OI_{t-1}|, clipped       - positioning flow
    10: (p - max_126d) / max_126d       - drawdown from 6M high
    11: (O_t - C_{t-1}) / sigma_t      - overnight gap (vol-normalized)
    """
    n = len(prices)
    feats = np.zeros((n, MARKET_FEATURE_DIM), dtype=np.float32)

    # Feature 0: r_t / sigma_t
    feats[:, 0] = returns / (sigma + 1e-10)

    # Features 1-3: Return horizons (5d, 21d, 126d)
    for feat_idx, horizon in enumerate(HORIZONS):
        col = np.zeros(n, dtype=float)
        for i in range(horizon, n):
            col[i] = (prices[i] - prices[i - horizon]) / (sigma[i] * np.sqrt(horizon) + 1e-10)
        feats[:, 1 + feat_idx] = col

    # Feature 4: MACD(8,24) (single pair, paper Eq.3)
    macd_vol = (
        pd.Series(prices)
        .rolling(window=MACD_VOL_WINDOW, min_periods=5)
        .std()
        .to_numpy(dtype=float)
    )
    for macd_idx, (short_span, long_span) in enumerate(MACD_PAIRS_ACTIVE):
        ema_s = pd.Series(prices).ewm(span=short_span, adjust=False).mean().to_numpy(dtype=float)
        ema_l = pd.Series(prices).ewm(span=long_span, adjust=False).mean().to_numpy(dtype=float)
        q_t = (ema_s - ema_l) / (macd_vol + 1e-10)
        q_std_252 = pd.Series(q_t).rolling(window=252, min_periods=21).std().to_numpy(dtype=float)
        feats[:, 4 + macd_idx] = q_t / (q_std_252 + 1e-10)

    # Features 5-6: RSI at two windows (5, 30)
    delta = np.diff(prices, prepend=prices[0])
    for rsi_idx, window in enumerate(RSI_WINDOWS):
        alpha = 1.0 / window
        gain = pd.Series(np.where(delta > 0, delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)
        loss = pd.Series(np.where(delta < 0, -delta, 0.0)).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float) + 1e-10
        rsi = 100.0 - 100.0 / (1.0 + gain / loss)
        feats[:, 5 + rsi_idx] = (rsi - 50.0) / 50.0

    # Feature 7: ATR norm (True Range / ATR_MA(20))
    if high is not None and low is not None:
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - prices[i - 1]), abs(low[i] - prices[i - 1]))
        atr_ma = pd.Series(tr).rolling(20).mean().to_numpy(dtype=float)
        feats[:, 7] = tr / (atr_ma + 1e-10)
    else:
        feats[:, 7] = 1.0  # neutral default

    # Feature 8: Volume norm (Volume / Volume_MA(20))
    if volume is not None:
        vol_ma = pd.Series(volume).rolling(20).mean().to_numpy(dtype=float)
        feats[:, 8] = volume / (vol_ma + 1e-10)
    else:
        feats[:, 8] = 1.0  # neutral default

    # Feature 9: OI change (ΔOI / |OI_{t-1}|, clipped)
    if oi is not None:
        oi_chg = np.zeros(n)
        oi_chg[1:] = (oi[1:] - oi[:-1]) / (np.abs(oi[:-1]) + 1e-10)
        feats[:, 9] = np.clip(oi_chg, -5, 5)
    else:
        feats[:, 9] = 0.0

    # Feature 10: Drawdown from 126-day high
    rolling_max = pd.Series(prices).rolling(126, min_periods=20).max().to_numpy(dtype=float)
    feats[:, 10] = (prices - rolling_max) / (rolling_max + 1e-10)

    # Feature 11: Overnight gap — (O_t - C_{t-1}) / sigma_t
    if open_ is not None:
        gap = np.zeros(n, dtype=float)
        gap[1:] = (open_[1:] - prices[:-1]) / (sigma[1:] + 1e-10)
        feats[:, 11] = gap
    else:
        feats[:, 11] = 0.0

    return np.nan_to_num(feats, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)


def get_feature_window(features: np.ndarray, idx: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Return (seq_len, MARKET_FEATURE_DIM) window of market features."""
    if idx < seq_len:
        pad = np.zeros((seq_len - idx, MARKET_FEATURE_DIM), dtype=np.float32)
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
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    open_: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    oi: np.ndarray | None = None,
    feature_spec_override: dict | None = None,
) -> ContractArrays:
    returns = compute_additive_returns(prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix_v3(
        prices, returns, sigma,
        high=high, low=low, open_=open_, volume=volume, oi=oi,
        feature_spec_override=feature_spec_override,
    )
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
    """Single-contract environment using the shared state/reward pipeline.

    Returns states of shape (SEQ_LEN, FEATURE_DIM).
    """

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

    def _make_state(self, idx: int) -> np.ndarray:
        """Build state (SEQ_LEN, FEATURE_DIM) — market features only."""
        return get_feature_window(self.features, idx)  # (SEQ_LEN, FEATURE_DIM)

    def reset(self) -> np.ndarray:
        self.idx = self.start_idx
        self.last_position = 0.0
        self.last_sigma = self.sigma[self.start_idx - 1] if self.start_idx >= 1 else self.sigma[0]
        return self._make_state(self.idx)

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool]:
        position = action_id_to_position(action_id)
        self.idx += 1
        if self.idx >= self.max_idx:
            self.last_position = position
            return self._make_state(min(self.idx, self.max_idx)), 0.0, True

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
        return self._make_state(self.idx), float(reward), done
