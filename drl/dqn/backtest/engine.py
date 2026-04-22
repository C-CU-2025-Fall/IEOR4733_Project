"""DQN backtest adapter: model inference only, baseline engine for metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import EXCLUDED_CONTRACTS, SOURCE_OVERRIDES
from drl.dqn.model import DQNAgent
from drl.dqn.spec import RETRAIN_ROUNDS, SIGMA_TGT, WARMUP, contract_model_path
from drl_shared.state_space import action_id_to_position, build_feature_matrix, get_feature_window
from strategy_backtester import backtest_strategy_metrics, paper_table3_reference

STRATEGY_LABELS = {
    "long": "Long",
    "sign_r": "Sign(R)",
    "sign(r)": "Sign(R)",
    "macd": "MACD",
    "dqn": "DQN",
}


def canonical_strategy_name(strategy: str) -> str:
    key = strategy.strip().lower()
    if key not in STRATEGY_LABELS:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return STRATEGY_LABELS[key]


def _load_dqn_agent(round_num: int, ticker: str, checkpoint: str | None = None) -> DQNAgent:
    ckpt_path = Path(checkpoint) if checkpoint else contract_model_path(round_num, ticker, algorithm="dqn")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing DQN checkpoint for {ticker}: {ckpt_path}")
    agent = DQNAgent()
    agent.load(ckpt_path)
    agent.q_net.eval()
    return agent


def _resolve_rounds_for_dates(
    dates: pd.DatetimeIndex,
    round_num: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    if round_num is not None:
        return [(np.ones(len(dates), dtype=bool), round_num)]
    masks = []
    for rn in sorted(RETRAIN_ROUNDS):
        info = RETRAIN_ROUNDS[rn]
        mask = (dates >= pd.Timestamp(info["test_start"])) & (dates <= pd.Timestamp(info["test_end"]))
        if np.any(mask):
            masks.append((np.asarray(mask, dtype=bool), rn))
    return masks


def dqn_position_provider(round_num: int | None = None, checkpoint: str | None = None):
    """Return a provider(rd)->positions for baseline_run.compute_strategy_metrics."""
    cache: dict[tuple[str, int], DQNAgent] = {}

    def provider(rd) -> np.ndarray:
        ticker = rd["tk"]
        prices = np.asarray(rd["prices"], dtype=float)
        returns = np.asarray(rd["rt"], dtype=float)
        sigma = np.asarray(rd["sigma"], dtype=float)
        start, t1 = int(rd["start"]), int(rd["t1"])
        eval_dates = pd.to_datetime(rd["dates"])
        eval_len = min(len(eval_dates), max(0, t1 - start + 1))

        features = build_feature_matrix(prices, returns, sigma)
        positions = np.zeros(len(prices), dtype=float)
        if eval_len <= 0:
            return positions

        round_masks = _resolve_rounds_for_dates(eval_dates[:eval_len], round_num=round_num)
        if not round_masks:
            return positions

        for mask, rn in round_masks:
            agent_key = (ticker, rn)
            if agent_key not in cache:
                cache[agent_key] = _load_dqn_agent(
                    rn,
                    ticker=ticker,
                    checkpoint=checkpoint if round_num is not None else None,
                )
            agent = cache[agent_key]
            idx_local = np.where(mask)[0]
            for local_idx in idx_local:
                full_idx = start + int(local_idx)
                if full_idx < WARMUP or full_idx >= len(prices):
                    continue
                state = get_feature_window(features, full_idx)
                action_id = agent.predict_action_id(state)
                positions[full_idx] = action_id_to_position(action_id)
        return positions

    return provider


def portfolio_metrics(
    asset_name: str,
    strategy: str,
    round_num: int | None = None,
    checkpoint: str | None = None,
    sigma_tgt: float = SIGMA_TGT,
    excluded_contracts: list[str] | None = None,
    source_overrides: dict[str, str] | None = None,
) -> dict[str, float]:
    strategy_name = canonical_strategy_name(strategy)
    excluded = EXCLUDED_CONTRACTS if excluded_contracts is None else excluded_contracts
    overrides = SOURCE_OVERRIDES if source_overrides is None else source_overrides
    provider = dqn_position_provider(round_num=round_num, checkpoint=checkpoint) if strategy_name == "DQN" else None
    return backtest_strategy_metrics(
        asset_name=asset_name,
        strategy=strategy_name,
        sigma_tgt=sigma_tgt,
        position_provider=provider,
        excluded_contracts=excluded,
        source_overrides=overrides,
    )


def paper_reference(asset_name: str, strategy: str) -> dict[str, float] | None:
    strategy_name = canonical_strategy_name(strategy)
    return paper_table3_reference(asset_name, strategy_name)
