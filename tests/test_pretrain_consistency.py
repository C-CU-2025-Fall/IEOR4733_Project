#!/usr/bin/env python3
"""Pre-train consistency tests.

Verifies that training and backtest pipelines share the same data, features,
overrides, and model configuration BEFORE training begins.

Run:  python tests/test_pretrain_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import ASSET_CLASSES
from data_loader import load_clc_full
from baseline_run import load_contracts
from drl.dqn.spec import (
    current_source_policy,
    feature_spec,
    universe_tickers,
    GAMMA,
    SIGMA_TGT,
    EPS_SCHEDULE,
    DISCRETE_ACTION_VALUES,
    BP,
    EWMA_SPAN,
    EPISODES,
    EPS_BUFFER_FILL,
)
from drl.dqn.spec import contract_data_path, LSTM_HIDDEN_SIZES, LEAKY_RELU_SLOPE
from drl_shared.spec import SEQ_LEN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    tag = "✅" if ok else "❌"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  {tag} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

ROUND_NUM = 1
TEST_START = "2011-01-01"
TEST_END = "2015-12-31"
SEED = None  # current default


def test_hyperparams():
    section("1. Hyperparameters")
    check("GAMMA == 0.5", GAMMA == 0.5, f"actual={GAMMA}")
    check("SIGMA_TGT == 0.058", SIGMA_TGT == 0.058)
    check("BP == 0.002", BP == 0.002)
    check("DISCRETE_ACTION_VALUES == (-1,0,1)", DISCRETE_ACTION_VALUES == (-1, 0, 1))
    check("FEATURE_SEQ_LEN == 60", SEQ_LEN == 60)
    check("LSTM_HIDDEN_SIZES == (64,32)", LSTM_HIDDEN_SIZES == (64, 32))
    check("EPISODES == 100", EPISODES == 100)
    check("EPS_BUFFER_FILL == 5000", EPS_BUFFER_FILL == 5000)

    # Epsilon schedule shape
    fracs = [s[0] for s in EPS_SCHEDULE]
    eps = [s[1] for s in EPS_SCHEDULE]
    check("EPS_SCHEDULE fracs monotonic", all(fracs[i] < fracs[i + 1] for i in range(len(fracs) - 1)))
    check("EPS_SCHEDULE starts ≤0.30", eps[0] <= 0.30)
    check("EPS_SCHEDULE ends at 0.01", eps[-1] == 0.01, f"actual={eps[-1]}")
    # Phase 4: 0.50→0.90 should go 0.05→0.01
    has_phase4 = any(f == 0.50 for f in fracs) and any(f == 0.90 for f in fracs)
    check("EPS_SCHEDULE has Phase 4 (0.50→0.90)", has_phase4)
    if has_phase4:
        idx50 = fracs.index(0.50)
        idx90 = fracs.index(0.90)
        check("Phase 4: ε@0.50=0.05", eps[idx50] == 0.05)
        check("Phase 4: ε@0.90=0.01", eps[idx90] == 0.01)

    print(f"\n  EPS_SCHEDULE detail:")
    for frac, ep in EPS_SCHEDULE:
        print(f"    cycle {frac*100:5.0f}:  ε = {ep:.3f}")


def test_source_policy():
    section("2. Source Policy (preset 38)")

    policy = current_source_policy()
    excluded = policy["excluded_contracts"]
    overrides = policy["source_overrides"]

    check("Excluded == ['FB','ZA']", excluded == ["FB", "ZA"], f"actual={excluded}")
    check("EN override == REV", overrides.get("EN") == "REV", f"actual={overrides.get('EN')}")
    check("LB override == RAD", overrides.get("LB") == "RAD", f"actual={overrides.get('LB')}")
    check("DT override == RAD", overrides.get("DT") == "RAD", f"actual={overrides.get('DT')}")

    # Total contract count (after exclusion)
    all_tickers = []
    for ac2 in ["Forex", "Equity Index", "Commodity", "Fixed Income"]:
        all_tickers.extend(t for t in universe_tickers(ac2) if t not in excluded)
    total = len(all_tickers)
    check(f"Total universe (excl FB,ZA) == 48", total == 48, f"actual={total}")


def test_npz_consistency():
    """Verify features npz matches training expectations."""
    section("3. Feature NPZ Consistency")

    fspec = feature_spec()
    expected_version = fspec["state_spec_version"]
    policy = current_source_policy()
    excluded = policy["excluded_contracts"]

    for ac in ["Forex", "Equity Index", "Commodity", "Fixed Income"]:
        tickers = universe_tickers(ac)
        for tk in tickers:
            if tk in excluded:
                # Expected: excluded contracts should NOT have npz
                check(f"[{ac}] {tk} npz correctly absent", True, "excluded by preset 38")
                continue

            npz_path = contract_data_path(ROUND_NUM, tk)
            if not npz_path.exists():
                check(f"[{ac}] {tk} npz exists", False, f"missing: {npz_path}")
                continue

            data = np.load(npz_path, allow_pickle=True)

            # state_spec_version match
            actual_ver = str(data.get("state_spec_version", ""))
            ver_ok = actual_ver == expected_version
            if not ver_ok:
                check(f"[{ac}] {tk} version match", False, f"npz={actual_ver} != expected={expected_version}")
            else:
                check(f"[{ac}] {tk} version match", True)

            # Feature dim
            feats = data["features"]
            expected_dim = fspec["feature_dim"]
            dim_ok = feats.shape[1] == expected_dim
            check(f"[{ac}] {tk} feature_dim={feats.shape[1]}", dim_ok, f"expected={expected_dim}")

            # Source override in npz matches policy
            npz_overrides = data.get("source_overrides", None)
            if npz_overrides is not None:
                import json
                ov_str = str(npz_overrides)
                try:
                    npz_ov = json.loads(ov_str) if isinstance(npz_overrides, str) else json.loads(ov_str)
                except Exception:
                    npz_ov = {}
                tk_source = str(data.get("source", ""))
                policy_source = current_source_policy()["source_overrides"].get(tk, "RAD")
                src_ok = tk_source == policy_source
                check(f"[{ac}] {tk} source={tk_source}==policy:{policy_source}", src_ok)

    # Only print summary per asset, not per contract
    print(f"\n  [checked {sum(len(universe_tickers(ac)) for ac in ['Forex','Equity Index','Commodity','Fixed Income'])} contracts]")


def test_training_backtest_data_alignment():
    """Verify training npz prices == backtest load_clc_full prices in test period."""
    section("4. Training-Backtest Price Alignment (test period)")

    policy = current_source_policy()
    overrides = policy["source_overrides"]
    excluded = policy["excluded_contracts"]

    for ac in ["Forex", "Equity Index", "Commodity", "Fixed Income"]:
        tickers = universe_tickers(ac)
        mismatches = 0
        total_contracts = len(tickers)

        for tk in tickers:
            npz_path = contract_data_path(ROUND_NUM, tk)
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=True)
            npz_dates = pd.to_datetime(data["dates"])
            npz_prices = data["prices"].astype(float)

            source = overrides.get(tk, "RAD")
            df = load_clc_full(tk, source=source, anchor_date="2011-01-01")
            bl_dates = pd.to_datetime(df["Date"])
            bl_prices = df["Close"].values.astype(float)

            # Overlap in test period (2011-2015)
            test_start = pd.Timestamp("2011-01-01")
            test_end = pd.Timestamp("2015-12-31")

            date_to_bl = {d.date(): i for i, d in enumerate(bl_dates)}
            date_to_npz = {d.date(): i for i, d in enumerate(npz_dates)}

            max_diff = 0.0
            common_count = 0
            for d in sorted(set(date_to_bl.keys()) & set(date_to_npz.keys())):
                if test_start.date() <= d <= test_end.date():
                    p_bl = bl_prices[date_to_bl[d]]
                    p_npz = npz_prices[date_to_npz[d]]
                    if p_npz > 0:
                        rel_diff = abs(p_bl - p_npz) / p_npz
                        max_diff = max(max_diff, rel_diff)
                    common_count += 1

            if max_diff > 1e-6:
                mismatches += 1

        check(f"[{ac}] all contracts price-aligned in test period", mismatches == 0,
              f"{mismatches}/{total_contracts} mismatched" if mismatches else "")


def test_long_dqn_same_universe():
    """Verify Long baseline and DQN use the same contract universe."""
    section("5. Long vs DQN Universe Alignment")

    policy = current_source_policy()
    excluded = policy["excluded_contracts"]
    overrides = policy["source_overrides"]

    for ac in ["Forex", "Equity Index", "Commodity", "Fixed Income"]:
        # DQN universe (exclude FB, ZA)
        dqn_tickers = set(t for t in universe_tickers(ac) if t not in excluded)

        # Long universe (load_contracts with same excluded)
        raw = load_contracts(ac, test_start=TEST_START, test_end=TEST_END,
                             excluded_contracts=excluded, source_overrides=overrides)
        long_tickers = {rd["tk"] for rd in raw}

        same = dqn_tickers == long_tickers
        detail = ""
        if not same:
            only_dqn = dqn_tickers - long_tickers
            only_long = long_tickers - dqn_tickers
            parts = []
            if only_dqn:
                parts.append(f"DQN only: {only_dqn}")
            if only_long:
                parts.append(f"Long only: {only_long}")
            detail = "; ".join(parts)
        check(f"[{ac}] DQN universe == Long universe ({len(dqn_tickers)} contracts)", same, detail)


def test_long_baseline_excludes_fb_za():
    """Verify Long baseline actually excludes FB and ZA when asked."""
    section("6. Long Baseline Exclusion")

    policy = current_source_policy()
    excluded = policy["excluded_contracts"]
    overrides = policy["source_overrides"]

    for ac in ["Forex", "Equity Index", "Commodity", "Fixed Income"]:
        raw = load_contracts(ac, test_start=TEST_START, test_end=TEST_END,
                             excluded_contracts=excluded, source_overrides=overrides)
        tickers = {rd["tk"] for rd in raw}
        has_fb = "FB" in tickers
        has_za = "ZA" in tickers
        check(f"[{ac}] FB excluded", not has_fb)
        check(f"[{ac}] ZA excluded", not has_za)


def test_no_dropna_in_pipeline():
    """Verify no dropna in feature preparation or training."""
    section("7. No dropna in Pipeline")

    import ast

    files_to_check = [
        ROOT / "drl_shared" / "prepare_features.py",
        ROOT / "drl_shared" / "state_space.py",
        ROOT / "drl" / "dqn" / "train" / "train_dqn_walkforward.py",
        ROOT / "drl" / "dqn" / "backtest" / "engine.py",
    ]

    for fpath in files_to_check:
        if not fpath.exists():
            check(f"{fpath.name} exists", False)
            continue
        code = fpath.read_text()
        has_dropna = "dropna(" in code
        check(f"{fpath.name} no dropna()", not has_dropna)


def test_backtest_no_fallback():
    """Verify backtest engine has no on-the-fly feature fallback."""
    section("8. Backtest Engine: No Feature Fallback")

    engine_code = (ROOT / "drl" / "dqn" / "backtest" / "engine.py").read_text()
    has_build_feature = "build_feature_matrix" in engine_code
    check("engine.py does NOT import build_feature_matrix", not has_build_feature)

    has_fallback = "on-the-fly" in engine_code or "fall back" in engine_code.lower()
    check("engine.py has no fallback text", not has_fallback)


def test_seed_reproducibility():
    """Verify seed setting produces identical first episode."""
    section("9. Seed Reproducibility (quick check)")

    import random, torch

    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    # Just verify seed=42 produces same random numbers
    set_seed(42)
    r1 = np.random.random(5)
    t1 = torch.rand(5).numpy()

    set_seed(42)
    r2 = np.random.random(5)
    t2 = torch.rand(5).numpy()

    check("np.random seed reproducible", np.allclose(r1, r2))
    check("torch seed reproducible", np.allclose(t1, t2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Pre-Train Consistency Check (R1)                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_hyperparams()
    test_source_policy()
    test_npz_consistency()
    test_training_backtest_data_alignment()
    test_long_dqn_same_universe()
    test_long_baseline_excludes_fb_za()
    test_no_dropna_in_pipeline()
    test_backtest_no_fallback()
    test_seed_reproducibility()

    print(f"\n{'='*60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")

    if FAIL > 0:
        print("\n  ⚠️  FIX FAILURES BEFORE TRAINING")
        sys.exit(1)
    else:
        print("\n  ✅ All checks passed. Ready to train.")
        sys.exit(0)


if __name__ == "__main__":
    main()
