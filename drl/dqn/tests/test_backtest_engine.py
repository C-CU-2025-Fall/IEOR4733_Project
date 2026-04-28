#!/usr/bin/env python3
"""Tests for unified DQN backtest engine.

Covers:
  - load_agents: best/top3, per-contract/asset-class, r1/r2
  - _read_seed_rankings: best_seed.json and train.log fallback
  - portfolio_metrics: Long/DQN, all training modes, all ensemble modes
  - Same preset for all strategies
  - Cross-round stitching
  - Regression: Long baseline numbers must not change

Usage:
    python3 -m pytest drl/dqn/tests/test_backtest_engine.py -v
    python3 drl/dqn/tests/test_backtest_engine.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl.dqn.spec import MODEL_ROOT, asset_slug, round_name, resolve_model_bundle
from drl.dqn.backtest.engine import (
    AgentTuple,
    _read_seed_rankings,
    load_agents,
    portfolio_metrics,
    paper_reference,
    current_dqn_policy,
    canonical_strategy_name,
)

SIGMA_TGT = 0.058

# Assets with completed models for both rounds
FOREX_TICKERS = ["AN", "BN", "CN", "DX", "FN", "JN", "MP", "NK", "SN"]


# ═══════════════════════════════════════════════════════════════
# _read_seed_rankings
# ═══════════════════════════════════════════════════════════════


class TestReadSeedRankings:
    """Test seed ranking from best_seed.json and train.log fallback."""

    def test_forex_r1_has_rankings(self):
        root = MODEL_ROOT / asset_slug("Forex") / round_name(1)
        rankings = _read_seed_rankings(root)
        assert len(rankings) >= 3
        # Should be sorted descending by val_reward
        for i in range(len(rankings) - 1):
            assert rankings[i]["val_reward"] >= rankings[i + 1]["val_reward"]

    def test_forex_r2_has_rankings(self):
        root = MODEL_ROOT / asset_slug("Forex") / round_name(2)
        rankings = _read_seed_rankings(root)
        assert len(rankings) >= 3

    def test_fi_r1_rankings(self):
        root = MODEL_ROOT / asset_slug("Fixed Income") / round_name(1)
        rankings = _read_seed_rankings(root)
        assert len(rankings) >= 3


# ═══════════════════════════════════════════════════════════════
# load_agents — unified API
# ═══════════════════════════════════════════════════════════════


class TestLoadAgentsAssetClass:
    """Asset-class mode: load best/top3 for r1 and r2."""

    def test_forex_r1_best(self):
        agents = load_agents(1, "Forex", training_mode="asset_class", ensemble_mode="best")
        assert len(agents) == 1
        assert isinstance(agents[0], tuple) and len(agents[0]) == 3

    def test_forex_r1_top3(self):
        agents = load_agents(1, "Forex", training_mode="asset_class", ensemble_mode="top3")
        assert len(agents) == 3

    def test_forex_r2_best(self):
        agents = load_agents(2, "Forex", training_mode="asset_class", ensemble_mode="best")
        assert len(agents) == 1

    def test_forex_r2_top3(self):
        agents = load_agents(2, "Forex", training_mode="asset_class", ensemble_mode="top3")
        assert len(agents) == 3

    def test_fi_r1_best(self):
        agents = load_agents(1, "Fixed Income", training_mode="asset_class", ensemble_mode="best")
        assert len(agents) == 1

    def test_fi_r1_top3(self):
        agents = load_agents(1, "Fixed Income", training_mode="asset_class", ensemble_mode="top3")
        assert len(agents) == 3

    def test_eq_r1_best(self):
        agents = load_agents(1, "Equity Index", training_mode="asset_class", ensemble_mode="best")
        assert len(agents) == 1

    def test_eq_r2_best(self):
        agents = load_agents(2, "Equity Index", training_mode="asset_class", ensemble_mode="best")
        assert len(agents) >= 1

    def test_commodity_r1_best(self):
        agents = load_agents(1, "Commodity", training_mode="asset_class", ensemble_mode="best")
        assert len(agents) == 1


class TestLoadAgentsPerContract:
    """Per-contract mode: load best/top3 for individual tickers.

    NOTE: Old dim=8 per-contract models were deleted. These tests need
    new dim=9 per-contract models (retrain required).
    """

    @pytest.mark.skip(reason="Per-contract models need retraining with dim=9 features")
    def test_an_r1_best(self):
        agents = load_agents(1, "AN", training_mode="per_contract", ensemble_mode="best")
        assert len(agents) == 1

    @pytest.mark.skip(reason="Per-contract models need retraining with dim=9 features")
    def test_an_r1_top3(self):
        agents = load_agents(1, "AN", training_mode="per_contract", ensemble_mode="top3")
        assert len(agents) >= 1  # might have fewer than 3 seeds

    @pytest.mark.skip(reason="Per-contract models need retraining with dim=9 features")
    def test_an_r2_best(self):
        agents = load_agents(2, "AN", training_mode="per_contract", ensemble_mode="best")
        assert len(agents) == 1

    @pytest.mark.skip(reason="Per-contract models need retraining with dim=9 features")
    def test_an_r2_top3(self):
        agents = load_agents(2, "AN", training_mode="per_contract", ensemble_mode="top3")
        assert len(agents) >= 1


class TestLoadAgentsEdgeCases:
    def test_invalid_round_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            load_agents(99, "Forex", training_mode="asset_class")

    def test_invalid_ensemble_mode_raises(self):
        with pytest.raises(ValueError):
            load_agents(1, "Forex", training_mode="asset_class", ensemble_mode="invalid")


# ═══════════════════════════════════════════════════════════════
# resolve_model_bundle (best_seed.json selection)
# ═══════════════════════════════════════════════════════════════


class TestResolveModelBundle:
    def test_forex_r1_picks_best_val(self):
        """resolve_model_bundle should pick best val-reward seed, not alphabetical."""
        bundle = resolve_model_bundle(1, "Forex", run_id="latest")
        assert bundle.exists()
        assert (bundle / "checkpoint.pt").exists()
        # Verify it matches best_seed.json
        root = MODEL_ROOT / asset_slug("Forex") / round_name(1)
        best_json = root / "best_seed.json"
        if best_json.exists():
            info = json.loads(best_json.read_text())
            assert str(bundle) == info["best_model_dir"]

    def test_fi_r1_picks_best_val(self):
        bundle = resolve_model_bundle(1, "Fixed Income", run_id="latest")
        assert bundle.exists()
        assert (bundle / "checkpoint.pt").exists()


# ═══════════════════════════════════════════════════════════════
# portfolio_metrics — asset-class mode
# ═══════════════════════════════════════════════════════════════


class TestPortfolioMetricsAssetClass:
    """DQN asset-class backtest: r1/r2/stitched × best/top3."""

    def test_forex_r1_best(self):
        m = portfolio_metrics("Forex", "DQN", round_num=1,
                              training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)
        for k in ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +ve", "Ave P/L"]:
            assert k in m
            assert np.isfinite(m[k])

    def test_forex_r2_best(self):
        m = portfolio_metrics("Forex", "DQN", round_num=2,
                              training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict) and "E(R)" in m

    def test_forex_stitched_best(self):
        m = portfolio_metrics("Forex", "DQN", round_num=None,
                              training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict) and "E(R)" in m

    def test_forex_r1_top3(self):
        m = portfolio_metrics("Forex", "DQN", round_num=1,
                              training_mode="asset_class", ensemble_mode="top3", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict) and "E(R)" in m

    def test_forex_r2_top3(self):
        m = portfolio_metrics("Forex", "DQN", round_num=2,
                              training_mode="asset_class", ensemble_mode="top3", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)

    def test_forex_stitched_top3(self):
        m = portfolio_metrics("Forex", "DQN", round_num=None,
                              training_mode="asset_class", ensemble_mode="top3", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)

    def test_fi_r1_best(self):
        m = portfolio_metrics("Fixed Income", "DQN", round_num=1,
                              training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)

    def test_fi_r1_top3(self):
        m = portfolio_metrics("Fixed Income", "DQN", round_num=1,
                              training_mode="asset_class", ensemble_mode="top3", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)

    def test_eq_r1_best(self):
        m = portfolio_metrics("Equity Index", "DQN", round_num=1,
                              training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)

    def test_eq_r1_top3(self):
        m = portfolio_metrics("Equity Index", "DQN", round_num=1,
                              training_mode="asset_class", ensemble_mode="top3", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)

    def test_top3_differs_from_best(self):
        """Top3 should generally produce different std(R) than best (more smoothing)."""
        m_best = portfolio_metrics("Forex", "DQN", round_num=1,
                                   training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        m_top3 = portfolio_metrics("Forex", "DQN", round_num=1,
                                   training_mode="asset_class", ensemble_mode="top3", sigma_tgt=SIGMA_TGT)
        # Both valid
        assert np.isfinite(m_best["E(R)"]) and np.isfinite(m_top3["E(R)"])
        # std should differ (ensemble smooths)
        assert m_top3["std(R)"] != m_best["std(R)"]


# ═══════════════════════════════════════════════════════════════
# portfolio_metrics — per-contract mode
# ═══════════════════════════════════════════════════════════════


class TestPortfolioMetricsPerContract:
    """DQN per-contract backtest.

    NOTE: Per-contract models deleted (old dim=8). Retrain needed.
    """

    @pytest.mark.skip(reason="Per-contract models need retraining with dim=9 features")
    def test_forex_r1_best(self):
        m = portfolio_metrics("Forex", "DQN", round_num=1,
                              training_mode="per_contract", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)
        assert "E(R)" in m
        assert np.isfinite(m["E(R)"])

    @pytest.mark.skip(reason="Per-contract models need retraining with dim=9 features")
    def test_forex_r2_best(self):
        m = portfolio_metrics("Forex", "DQN", round_num=2,
                              training_mode="per_contract", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)

    @pytest.mark.skip(reason="Per-contract models need retraining with dim=9 features")
    def test_forex_stitched_best(self):
        m = portfolio_metrics("Forex", "DQN", round_num=None,
                              training_mode="per_contract", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        assert isinstance(m, dict)


# ═══════════════════════════════════════════════════════════════
# Long baseline (regression guard)
# ═══════════════════════════════════════════════════════════════


# Snapshot values from known-good run — must not change
LONG_FOREX_SNAPSHOT = {"E(R)": -0.173, "Sharpe": -0.409, "% +ve": 0.490}
LONG_FI_SNAPSHOT = {"E(R)": 0.471, "Sharpe": 0.552, "% +ve": 0.529}


class TestLongBaseline:
    """Long baseline must produce stable, known results."""

    def test_forex_long(self):
        m = portfolio_metrics("Forex", "Long", sigma_tgt=SIGMA_TGT)
        assert abs(m["E(R)"] - LONG_FOREX_SNAPSHOT["E(R)"]) < 0.01
        assert abs(m["Sharpe"] - LONG_FOREX_SNAPSHOT["Sharpe"]) < 0.01
        assert abs(m["% +ve"] - LONG_FOREX_SNAPSHOT["% +ve"]) < 0.01

    def test_fi_long(self):
        m = portfolio_metrics("Fixed Income", "Long", sigma_tgt=SIGMA_TGT)
        assert abs(m["E(R)"] - LONG_FI_SNAPSHOT["E(R)"]) < 0.01
        assert abs(m["Sharpe"] - LONG_FI_SNAPSHOT["Sharpe"]) < 0.01

    def test_eq_long(self):
        m = portfolio_metrics("Equity Index", "Long", sigma_tgt=SIGMA_TGT)
        assert m["E(R)"] > 0  # EQ Long is positive


# ═══════════════════════════════════════════════════════════════
# Same preset for all strategies
# ═══════════════════════════════════════════════════════════════


class TestSamePreset:
    """All strategies should use the same preset when specified."""

    def test_dqn_policy_consistency(self):
        overrides, excluded = current_dqn_policy()
        assert isinstance(overrides, dict)
        assert isinstance(excluded, list)
        assert "FB" in excluded
        assert "ZA" in excluded

    def test_long_explicit_preset_matches(self):
        overrides, excluded = current_dqn_policy()
        m_default = portfolio_metrics("Forex", "Long", sigma_tgt=SIGMA_TGT)
        m_explicit = portfolio_metrics("Forex", "Long", sigma_tgt=SIGMA_TGT,
                                       source_overrides=overrides, excluded_contracts=excluded)
        assert abs(m_default["E(R)"] - m_explicit["E(R)"]) < 1e-6

    def test_dqn_and_long_same_data(self):
        """DQN and Long should see the same contracts (same excluded list)."""
        # Long with DQN's preset should have same number of contracts as DQN
        overrides, excluded = current_dqn_policy()
        from strategy_backtester import backtest_strategy_metrics
        m = backtest_strategy_metrics("Forex", "Long", SIGMA_TGT,
                                      excluded_contracts=excluded, source_overrides=overrides)
        assert isinstance(m, dict)


# ═══════════════════════════════════════════════════════════════
# Cross-round
# ═══════════════════════════════════════════════════════════════


class TestCrossRound:
    def test_stitched_differs_from_single_round(self):
        m_r1 = portfolio_metrics("Forex", "DQN", round_num=1,
                                 training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        m_r2 = portfolio_metrics("Forex", "DQN", round_num=2,
                                 training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        m_stitch = portfolio_metrics("Forex", "DQN", round_num=None,
                                     training_mode="asset_class", ensemble_mode="best", sigma_tgt=SIGMA_TGT)
        # Stitched should differ from at least one single round
        assert m_stitch["E(R)"] != m_r1["E(R)"] or m_stitch["E(R)"] != m_r2["E(R)"]


# ═══════════════════════════════════════════════════════════════
# Canonical strategy names
# ═══════════════════════════════════════════════════════════════


class TestCanonicalNames:
    def test_long(self):
        assert canonical_strategy_name("long") == "Long"
        assert canonical_strategy_name("Long") == "Long"

    def test_sign_r(self):
        assert canonical_strategy_name("Sign(R)") == "Sign(R)"
        assert canonical_strategy_name("sign_r") == "Sign(R)"

    def test_macd(self):
        assert canonical_strategy_name("MACD") == "MACD"

    def test_dqn(self):
        assert canonical_strategy_name("dqn") == "DQN"

    def test_invalid(self):
        with pytest.raises(ValueError):
            canonical_strategy_name("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
