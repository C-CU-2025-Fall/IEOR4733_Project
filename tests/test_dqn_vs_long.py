#!/usr/bin/env python3
"""Unit test: per-contract DQN vs Long-only backtest comparison.

Usage:
    pytest tests/test_dqn_vs_long.py -v
    pytest tests/test_dqn_vs_long.py -v -k "SN"          # single contract
    pytest tests/test_dqn_vs_long.py -v -k "Forex"        # all Forex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl.dqn.model import DQNAgent
from drl.dqn.spec import SEQ_LEN, MODEL_ROOT, contract_data_path
from drl_shared.spec import universe_tickers, current_source_policy
from drl_shared.state_space import ContractArrays, ContractEnv
from strategies import strategy_long_only


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_contract(ticker: str, round_num: int) -> ContractArrays:
    path = contract_data_path(round_num, ticker)
    assert path.exists(), f"No features for {ticker} r{round_num}: {path}"
    data = np.load(path, allow_pickle=True)
    return ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(data.get("source", "")),
    )


def _latest_model_dir(ticker: str, round_num: int) -> Path | None:
    base = MODEL_ROOT / ticker / f"r{round_num}"
    if not base.exists():
        return None
    per_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("per_")])
    return per_dirs[-1] if per_dirs else None


def _backtest_dqn(ticker: str, round_num: int, sigma_tgt: float = 0.058) -> dict:
    """Run greedy DQN backtest, return reward array + metrics."""
    model_dir = _latest_model_dir(ticker, round_num)
    assert model_dir is not None, f"No per-contract model for {ticker} r{round_num}"

    checkpoint = model_dir / "checkpoint.pt"
    assert checkpoint.exists(), f"No checkpoint: {checkpoint}"

    manifest = {}
    mpath = model_dir / "manifest.json"
    if mpath.exists():
        with open(mpath) as f:
            manifest = json.load(f)
    sigma_tgt = manifest.get("sigma_tgt", sigma_tgt)

    agent = DQNAgent(device="cpu")
    agent.load(checkpoint)

    contract = _load_contract(ticker, round_num)
    env = ContractEnv(contract, sigma_tgt=sigma_tgt)
    state = env.reset()
    rewards = []
    positions = []
    done = False
    while not done:
        action_id = agent.act(state, eps=0.0)
        positions.append(float(action_id) - 1.0)  # action_id 0,1,2 → -1,0,+1
        next_state, reward, done = env.step(action_id)
        rewards.append(reward)
        state = next_state

    rewards = np.array(rewards)
    return {
        "rewards": rewards,
        "positions": np.array(positions),
        "cum_return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
        "mdd": _compute_mdd(rewards),
        "steps": len(rewards),
        "sigma_tgt": sigma_tgt,
    }


def _backtest_long_only(ticker: str, round_num: int, sigma_tgt: float = 0.058) -> dict:
    """Run long-only backtest on same data using ContractEnv."""
    contract = _load_contract(ticker, round_num)
    env = ContractEnv(contract, sigma_tgt=sigma_tgt)
    state = env.reset()
    rewards = []
    # Long-only: always action_id=2 (position +1)
    done = False
    while not done:
        _, reward, done = env.step(2)  # action 2 = long
        rewards.append(reward)
    rewards = np.array(rewards)
    return {
        "rewards": rewards,
        "cum_return": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-10)),
        "mdd": _compute_mdd(rewards),
        "steps": len(rewards),
    }


def _compute_mdd(rewards: np.ndarray) -> float:
    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum - peak))


# ── test fixtures ────────────────────────────────────────────────────────────

FOREX_TICKERS = ["AN", "BN", "CN", "DX", "FN", "JN", "MP", "NK", "SN"]


def _available_contracts(tickers: list[str], round_num: int = 1) -> list[str]:
    """Return tickers that have both features and a trained model."""
    avail = []
    for t in tickers:
        if contract_data_path(round_num, t).exists() and _latest_model_dir(t, round_num):
            avail.append(t)
    return avail


# ── tests ────────────────────────────────────────────────────────────────────

class TestDQNvsLong:
    """Compare per-contract DQN against long-only baseline."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self, request):
        """Skip if no trained model available for the ticker."""
        # check happens per-test via parametrize skip below
        pass

    @pytest.mark.parametrize("ticker", FOREX_TICKERS)
    @pytest.mark.parametrize("round_num", [1, 2])
    def test_dqn_completes(self, ticker, round_num):
        """DQN backtest runs without error and produces valid output."""
        model_dir = _latest_model_dir(ticker, round_num)
        if model_dir is None:
            pytest.skip(f"No model for {ticker} r{round_num}")

        res = _backtest_dqn(ticker, round_num)
        assert res["steps"] > 0, "DQN produced 0 steps"
        assert np.isfinite(res["cum_return"]), "DQN cum_return is NaN/Inf"
        assert np.isfinite(res["sharpe"]), "DQN sharpe is NaN/Inf"

    @pytest.mark.parametrize("ticker", FOREX_TICKERS)
    @pytest.mark.parametrize("round_num", [1, 2])
    def test_long_only_completes(self, ticker, round_num):
        """Long-only backtest runs and produces valid output."""
        if not contract_data_path(round_num, ticker).exists():
            pytest.skip(f"No features for {ticker} r{round_num}")

        res = _backtest_long_only(ticker, round_num)
        assert res["steps"] > 0
        assert np.isfinite(res["cum_return"])

    @pytest.mark.parametrize("ticker", FOREX_TICKERS)
    @pytest.mark.parametrize("round_num", [1, 2])
    def test_dqn_vs_long_comparison(self, ticker, round_num):
        """Core comparison: DQN vs Long-only on same contract + round."""
        model_dir = _latest_model_dir(ticker, round_num)
        if model_dir is None:
            pytest.skip(f"No model for {ticker} r{round_num}")

        dqn = _backtest_dqn(ticker, round_num)
        lon = _backtest_long_only(ticker, round_num)

        # Same data → same number of steps
        assert dqn["steps"] == lon["steps"], \
            f"Step mismatch: DQN={dqn['steps']} vs Long={lon['steps']}"

        # Log comparison
        diff_pct = ((dqn["cum_return"] - lon["cum_return"]) / (abs(lon["cum_return"]) + 1e-10)) * 100
        print(f"\n  {ticker} r{round_num}: DQN={dqn['cum_return']:+.4f} Long={lon['cum_return']:+.4f} "
              f"diff={diff_pct:+.1f}% sharpe_dqn={dqn['sharpe']:+.3f} sharpe_long={lon['sharpe']:+.3f} "
              f"mdd_dqn={dqn['mdd']:+.4f} mdd_long={lon['mdd']:+.4f}")

        # DQN should not crash (no assertion on beating long — that's empirical)

    @pytest.mark.parametrize("ticker", FOREX_TICKERS)
    @pytest.mark.parametrize("round_num", [1, 2])
    def test_dqn_position_diversity(self, ticker, round_num):
        """DQN should use more than just one position (not degenerate)."""
        model_dir = _latest_model_dir(ticker, round_num)
        if model_dir is None:
            pytest.skip(f"No model for {ticker} r{round_num}")

        res = _backtest_dqn(ticker, round_num)
        unique_pos = np.unique(res["positions"])
        assert len(unique_pos) > 1, \
            f"DQN degenerate: only uses {len(unique_pos)} position(s) for {ticker}"

    def test_forex_summary(self):
        """Print summary table for all available Forex contracts."""
        results = []
        for ticker in FOREX_TICKERS:
            for r in [1, 2]:
                model_dir = _latest_model_dir(ticker, r)
                if model_dir is None:
                    continue
                dqn = _backtest_dqn(ticker, r)
                lon = _backtest_long_only(ticker, r)
                results.append({
                    "ticker": ticker, "round": r,
                    "dqn_cum": dqn["cum_return"], "long_cum": lon["cum_return"],
                    "dqn_sharpe": dqn["sharpe"], "long_sharpe": lon["sharpe"],
                    "dqn_mdd": dqn["mdd"], "long_mdd": lon["mdd"],
                })

        if not results:
            pytest.skip("No trained Forex models available")

        print(f"\n{'Ticker':>6s} {'Rd':>3s} {'DQN_cum':>9s} {'Long_cum':>9s} {'DQN_Shp':>8s} {'Long_Shp':>8s} {'DQN_MDD':>8s} {'Long_MDD':>8s}")
        print("-" * 70)
        for r in results:
            print(f"{r['ticker']:>6s} r{r['round']:>2d} {r['dqn_cum']:>+9.4f} {r['long_cum']:>+9.4f} "
                  f"{r['dqn_sharpe']:>+8.3f} {r['long_sharpe']:>+8.3f} "
                  f"{r['dqn_mdd']:>+8.4f} {r['long_mdd']:>+8.4f}")

        dqn_wins = sum(1 for r in results if r["dqn_cum"] > r["long_cum"])
        print(f"\nDQN beats Long-only: {dqn_wins}/{len(results)} ({dqn_wins/len(results)*100:.0f}%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
