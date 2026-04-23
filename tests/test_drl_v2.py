import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import baseline_run
import drl.dqn.spec as dqn_spec
import drl_shared.spec as shared_spec
from drl.dqn.model import DQNAgent
from drl.dqn.train.train_dqn_walkforward import train_contract_round
from drl_shared.state_space import build_feature_matrix, compute_additive_returns, compute_ewma_sigma


class DRLV2FeatureTests(unittest.TestCase):
    def test_close_feature_is_causal_and_matches_formula(self):
        prices = 100.0 + np.cumsum(np.sin(np.arange(360) / 9.0) + 0.1)
        returns = compute_additive_returns(prices)
        sigma = compute_ewma_sigma(returns)

        full = build_feature_matrix(prices, returns, sigma)
        trunc_n = 300
        truncated = build_feature_matrix(prices[:trunc_n], returns[:trunc_n], sigma[:trunc_n])

        self.assertEqual(full.shape, (360, 8))
        np.testing.assert_allclose(full[:trunc_n], truncated, rtol=0, atol=1e-6)

        ema_price = pd.Series(prices).ewm(span=60, adjust=False).mean().to_numpy(dtype=float)
        manual = (prices - ema_price) / (sigma * np.sqrt(60) + 1e-10)
        manual = np.nan_to_num(manual, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        np.testing.assert_allclose(full[:, 0], manual, rtol=0, atol=1e-6)
        self.assertTrue(np.isfinite(full).all())


class DRLV2BacktestTests(unittest.TestCase):
    def test_vectorized_eq4_matches_loop_reference(self):
        n = 320
        prices = 80.0 + np.cumsum(np.cos(np.arange(n) / 13.0) + 0.05)
        rt = compute_additive_returns(prices)
        sigma = compute_ewma_sigma(rt)
        sigma[:5] = np.nan
        positions = np.where(np.arange(n) % 3 == 0, 1.0, np.where(np.arange(n) % 3 == 1, -1.0, 0.0))
        rd = {
            "tk": "ZZ",
            "prices": prices,
            "rt": rt,
            "sigma": sigma,
        }

        loop = baseline_run.compute_contract_returns_from_positions_loop(rd, positions, 0.058, detail=True)
        fast = baseline_run.compute_contract_returns_from_positions(rd, positions, 0.058, detail=True)
        for key in ("Rt", "scaled_pos", "gross_pnl", "tc_cost"):
            np.testing.assert_allclose(fast[key], loop[key], rtol=0, atol=1e-12)


class DRLV2BundleTests(unittest.TestCase):
    def test_training_smoke_creates_versioned_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_root = tmp_path / "features"
            model_root = tmp_path / "models"

            n = 270
            prices = 120.0 + np.cumsum(np.sin(np.arange(n) / 11.0) + 0.03)
            returns = compute_additive_returns(prices)
            sigma = compute_ewma_sigma(returns)
            features = build_feature_matrix(prices, returns, sigma)
            f_spec = shared_spec.feature_spec("v2")

            with patch.object(shared_spec, "FEATURE_ROOT", feature_root), patch.object(dqn_spec, "MODEL_ROOT", model_root):
                feature_path = shared_spec.feature_data_path(1, "AN", model_version="v2")
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    feature_path,
                    ticker="AN",
                    prices=prices,
                    returns=returns,
                    sigma=sigma,
                    features=features,
                    dates=np.array(pd.date_range("2005-01-03", periods=n, freq="B")),
                    source="TEST",
                    round="r1",
                    model_version="v2",
                    state_spec_version=f_spec["state_spec_version"],
                    feature_spec=json.dumps(f_spec, sort_keys=True),
                    train_start="2005-01-01",
                    train_end="2010-12-31",
                    test_start="2011-01-01",
                    test_end="2015-12-31",
                )

                checkpoint, bundle = train_contract_round(
                    "AN",
                    1,
                    episodes=1,
                    model_version="v2",
                    device="cpu",
                    seed=7,
                )

                self.assertTrue(checkpoint.exists())
                self.assertTrue((bundle / "manifest.json").exists())
                self.assertTrue((bundle / "train_config.json").exists())
                self.assertTrue((bundle / "feature_spec.json").exists())
                self.assertTrue((bundle / "episode_metrics.csv").exists())
                self.assertTrue((bundle / "train.log").exists())

                with (bundle / "manifest.json").open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                self.assertEqual(manifest["model_version"], "v2")
                self.assertEqual(manifest["state_spec_version"], "v2_ewma60_close_deviation")
                self.assertEqual(manifest["sigma_tgt"], 0.058)

    def test_dqn_batch_predict_returns_valid_action_ids(self):
        agent = DQNAgent(device="cpu")
        states = np.zeros((5, 60, 8), dtype=np.float32)
        action_ids = agent.predict_action_ids(states)
        self.assertEqual(action_ids.shape, (5,))
        self.assertTrue(set(action_ids.tolist()).issubset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
