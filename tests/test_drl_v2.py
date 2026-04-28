import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import baseline_run
import drl.dqn.backtest.engine as backtest_engine
import drl.dqn.spec as dqn_spec
import drl.dqn.train.train_dqn_walkforward as train_mod
import drl_shared.spec as shared_spec
from config import PAPER_TABLE3
from drl.dqn.model import DQNAgent
from drl.dqn.train.train_dqn_walkforward import parse_rounds, train_asset_round
from drl_shared.state_space import build_feature_matrix, compute_additive_returns, compute_ewma_sigma


def _write_feature_npz(path: Path, prices: np.ndarray, source: str = "TEST"):
    returns = compute_additive_returns(prices)
    sigma = compute_ewma_sigma(returns)
    features = build_feature_matrix(prices, returns, sigma)
    f_spec = shared_spec.feature_spec()
    policy = shared_spec.current_source_policy()
    dates = np.array(pd.date_range("2005-01-03", periods=len(prices), freq="B"))
    dt = pd.to_datetime(dates)
    train_end_idx = int(np.where(dt <= pd.Timestamp("2010-12-31"))[0][-1])
    test_start_idx = int(np.where(dt >= pd.Timestamp("2011-01-01"))[0][0])
    test_end_idx = int(np.where(dt <= pd.Timestamp("2015-12-31"))[0][-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ticker=path.parent.name,
        prices=prices,
        returns=returns,
        sigma=sigma,
        features=features,
        dates=dates,
        source=source,
        round="r1",
        feature_line=f_spec["feature_line"],
        state_spec_version=f_spec["state_spec_version"],
        feature_spec=json.dumps(f_spec, sort_keys=True),
        preset=policy["preset"],
        excluded_contracts=json.dumps(policy["excluded_contracts"]),
        source_overrides=json.dumps(policy["source_overrides"]),
        train_start="2005-01-01",
        train_end="2010-12-31",
        test_start="2011-01-01",
        test_end="2015-12-31",
        train_start_idx=0,
        train_end_idx=train_end_idx,
        test_start_idx=test_start_idx,
        test_end_idx=test_end_idx,
    )
    return returns, sigma, features


class DRLMainlineFeatureTests(unittest.TestCase):
    def test_feature_spec_is_single_mainline(self):
        spec = shared_spec.feature_spec()
        self.assertEqual(spec["feature_line"], shared_spec.ACTIVE_FEATURE_LINE)
        self.assertEqual(spec["state_spec_version"], shared_spec.STATE_SPEC_VERSION)
        self.assertEqual(spec["preset"], "structural_38")
        self.assertEqual(spec["close_feature"]["name"], "normalized_close_price_60d_rolling_std")
        self.assertEqual(spec["feature_dim"], 9)

    def test_close_feature_matches_formula(self):
        prices = 100.0 + np.cumsum(np.sin(np.arange(360) / 9.0) + 0.1)
        returns = compute_additive_returns(prices)
        sigma = compute_ewma_sigma(returns)
        rolling_std = pd.Series(prices).rolling(window=60, min_periods=5).std().to_numpy(dtype=float)

        feats = build_feature_matrix(prices, returns, sigma)
        manual = prices / (rolling_std + 1e-10)
        manual = np.nan_to_num(manual, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        self.assertEqual(feats.shape[1], shared_spec.FEATURE_DIM)
        np.testing.assert_allclose(feats[:, 0], manual, rtol=0, atol=1e-6)

    def test_close_feature_is_causal(self):
        prices = 100.0 + np.cumsum(np.sin(np.arange(360) / 9.0) + 0.1)
        returns = compute_additive_returns(prices)
        sigma = compute_ewma_sigma(returns)
        full = build_feature_matrix(prices, returns, sigma)
        trunc_n = 300
        truncated = build_feature_matrix(prices[:trunc_n], returns[:trunc_n], sigma[:trunc_n])
        np.testing.assert_allclose(full[:trunc_n], truncated, rtol=0, atol=1e-6)


class DRLMainlineBacktestTests(unittest.TestCase):
    def test_vectorized_eq4_matches_loop_reference(self):
        n = 320
        prices = 80.0 + np.cumsum(np.cos(np.arange(n) / 13.0) + 0.05)
        rt = compute_additive_returns(prices)
        sigma = compute_ewma_sigma(rt)
        sigma[:5] = np.nan
        positions = np.where(np.arange(n) % 3 == 0, 1.0, np.where(np.arange(n) % 3 == 1, -1.0, 0.0))
        rd = {"tk": "ZZ", "prices": prices, "rt": rt, "sigma": sigma}

        loop = baseline_run.compute_contract_returns_from_positions_loop(rd, positions, 0.058, detail=True)
        fast = baseline_run.compute_contract_returns_from_positions(rd, positions, 0.058, detail=True)
        for key in ("Rt", "scaled_pos", "gross_pnl", "tc_cost"):
            np.testing.assert_allclose(fast[key], loop[key], rtol=0, atol=1e-12)

    def test_current_dqn_policy_is_structural38(self):
        overrides, excluded = backtest_engine.current_dqn_policy()
        policy = shared_spec.current_source_policy()
        self.assertEqual(overrides, policy["source_overrides"])
        self.assertEqual(sorted(excluded), sorted(policy["excluded_contracts"]))

    def test_portfolio_metrics_routes_dqn_through_structural38_policy(self):
        captured = {}

        def fake_backtest(**kwargs):
            captured.update(kwargs)
            return {name: 0.0 for name in ["E(R)", "std(R)", "DD", "Sharpe", "Sortino", "MDD", "Calmar", "% +ve", "Ave P/L"]}

        with patch.object(backtest_engine, "backtest_strategy_metrics", side_effect=fake_backtest):
            backtest_engine.portfolio_metrics(
                asset_name="Forex",
                strategy="DQN",
                round_num=1,
                checkpoint="/tmp/missing.pt",
                sigma_tgt=0.058,
            )

        policy = shared_spec.current_source_policy()
        self.assertEqual(captured["source_overrides"], policy["source_overrides"])
        self.assertEqual(sorted(captured["excluded_contracts"]), sorted(policy["excluded_contracts"]))

    def test_paper_reference_uses_dqn_row(self):
        self.assertEqual(backtest_engine.paper_reference("Forex", "DQN"), PAPER_TABLE3["Forex"]["DQN"])


class DRLMainlineBundleTests(unittest.TestCase):
    def test_feature_artifact_persists_explicit_round_split_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "AN" / "r1.npz"
            prices = 120.0 + np.cumsum(np.sin(np.arange(3200) / 17.0) + 0.02)
            _write_feature_npz(path, prices)
            data = np.load(path, allow_pickle=True)
            self.assertEqual(int(data["train_start_idx"]), 0)
            self.assertLess(int(data["train_end_idx"]), int(data["test_start_idx"]))
            dates = pd.to_datetime(data["dates"])
            self.assertLessEqual(dates[int(data["train_end_idx"])], pd.Timestamp("2010-12-31"))
            self.assertGreaterEqual(dates[int(data["test_start_idx"])], pd.Timestamp("2011-01-01"))

    def test_training_smoke_creates_asset_class_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_root = tmp_path / "features"
            model_root = tmp_path / "models"

            n = 3200
            prices = 120.0 + np.cumsum(np.sin(np.arange(n) / 11.0) + 0.03)
            f_spec = shared_spec.feature_spec()
            policy = shared_spec.current_source_policy()

            with patch.object(shared_spec, "FEATURE_ROOT", feature_root), patch.object(dqn_spec, "MODEL_ROOT", model_root):
                for ticker in ("AN", "BN"):
                    feature_path = shared_spec.feature_data_path(1, ticker)
                    _write_feature_npz(feature_path, prices)
                index_path = shared_spec.asset_index_path("Forex", 1)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                with index_path.open("w", encoding="utf-8") as fh:
                    json.dump({
                        "asset_class": "Forex",
                        "round": "r1",
                        "member_tickers": ["AN", "BN"],
                        "excluded_contracts": policy["excluded_contracts"],
                        "source_overrides": policy["source_overrides"],
                        "state_spec_version": f_spec["state_spec_version"],
                    }, fh)

                with patch.object(train_mod, "MODEL_ROOT", model_root):
                    checkpoint, bundle = train_asset_round(
                        "Forex",
                        1,
                        episodes=1,
                        sigma_tgt=0.058,
                        device="cpu",
                        seed=7,
                    )

                self.assertTrue(checkpoint.exists())
                self.assertTrue((bundle / "manifest.json").exists())
                self.assertTrue((bundle / "train_config.json").exists())
                self.assertTrue((bundle / "feature_spec.json").exists())
                self.assertTrue((bundle / "episode_metrics.csv").exists())
                self.assertTrue((bundle / "contract_metrics.csv").exists())
                self.assertTrue((bundle / "validation_metrics.csv").exists())
                self.assertTrue((bundle / "train.log").exists())
                self.assertEqual(bundle.parent.name, "r1")
                self.assertEqual(bundle.parent.parent.name, "Forex")

                with (bundle / "manifest.json").open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                self.assertEqual(manifest["feature_line"], shared_spec.ACTIVE_FEATURE_LINE)
                self.assertEqual(manifest["state_spec_version"], shared_spec.STATE_SPEC_VERSION)
                self.assertEqual(manifest["preset"], "structural_38")
                self.assertEqual(manifest["sigma_tgt"], 0.058)
                self.assertEqual(manifest["training_mode"], "asset_class_shared")
                self.assertEqual(manifest["asset_class"], "Forex")
                self.assertEqual(manifest["member_tickers"], ["AN", "BN"])
                self.assertEqual(manifest["architecture"]["paper_reference_ids"], [49, 18, 50])
                self.assertTrue(manifest["architecture"]["dueling_dqn"])
                self.assertTrue(manifest["architecture"]["double_dqn"])
                self.assertTrue(manifest["architecture"]["fixed_q_targets"])
                self.assertEqual(manifest["hyperparameters"]["epsilon_mode"], "constant")
                self.assertEqual(manifest["hyperparameters"]["eps_start"], 0.3)
                self.assertEqual(manifest["hyperparameters"]["eps_end"], 0.3)
                self.assertIsNone(manifest["hyperparameters"]["eps_decay_steps"])
                self.assertGreater(manifest["hyperparameters"]["max_steps_per_ep"], 1500)
                splits = manifest["contract_round_splits"]["AN"]
                self.assertEqual(splits["train_start"], "2005-01-01")
                self.assertEqual(splits["train_end"], "2010-12-31")
                self.assertEqual(splits["test_start"], "2011-01-01")
                self.assertEqual(splits["test_end"], "2015-12-31")

    def test_train_asset_round_passes_sigma_tgt_to_env(self):
        prices = 120.0 + np.cumsum(np.sin(np.arange(3200) / 11.0) + 0.03)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_root = tmp_path / "features"
            model_root = tmp_path / "models"
            feature_path = feature_root / "AN" / "r1.npz"
            _write_feature_npz(feature_path, prices)
            index_path = feature_root / "Forex" / "r1" / "index.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("w", encoding="utf-8") as fh:
                json.dump({"member_tickers": ["AN"]}, fh)

            captured = {}

            class FakeEnv:
                def __init__(self, contract, sigma_tgt=0.058, **kwargs):
                    _ = kwargs
                    captured["sigma_tgt"] = sigma_tgt
                    self.start_idx = 252
                    self.max_idx = len(contract.prices) - 1

                def reset(self):
                    return np.zeros((shared_spec.SEQ_LEN, shared_spec.FEATURE_DIM), dtype=np.float32)

                def step(self, action_id):
                    _ = action_id
                    return np.zeros((shared_spec.SEQ_LEN, shared_spec.FEATURE_DIM), dtype=np.float32), 0.0, True

            with patch.object(shared_spec, "FEATURE_ROOT", feature_root), patch.object(dqn_spec, "MODEL_ROOT", model_root), patch.object(train_mod, "MODEL_ROOT", model_root), patch.object(train_mod, "ContractEnv", FakeEnv):
                train_asset_round("Forex", 1, episodes=1, sigma_tgt=0.061, device="cpu", seed=11)

            self.assertEqual(captured["sigma_tgt"], 0.061)

    def test_train_loader_excludes_post_2010_rows_from_r1_train_contract(self):
        prices = 120.0 + np.cumsum(np.sin(np.arange(3200) / 11.0) + 0.03)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_root = tmp_path / "features"
            model_root = tmp_path / "models"
            feature_path = feature_root / "AN" / "r1.npz"
            _write_feature_npz(feature_path, prices)
            index_path = feature_root / "Forex" / "r1" / "index.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("w", encoding="utf-8") as fh:
                json.dump({"member_tickers": ["AN"]}, fh)

            captured = {}

            class FakeEnv:
                def __init__(self, contract, sigma_tgt=0.058, **kwargs):
                    _ = sigma_tgt
                    _ = kwargs
                    captured["last_train_date"] = str(pd.to_datetime(contract.dates[-1]).date())
                    captured["n_train_rows"] = len(contract.prices)
                    self.start_idx = 252
                    self.max_idx = len(contract.prices) - 1

                def reset(self):
                    return np.zeros((shared_spec.SEQ_LEN, shared_spec.FEATURE_DIM), dtype=np.float32)

                def step(self, action_id):
                    _ = action_id
                    return np.zeros((shared_spec.SEQ_LEN, shared_spec.FEATURE_DIM), dtype=np.float32), 0.0, True

            with patch.object(shared_spec, "FEATURE_ROOT", feature_root), patch.object(dqn_spec, "MODEL_ROOT", model_root), patch.object(train_mod, "MODEL_ROOT", model_root), patch.object(train_mod, "ContractEnv", FakeEnv):
                train_asset_round("Forex", 1, episodes=1, sigma_tgt=0.058, device="cpu", seed=13)

            self.assertEqual(captured["last_train_date"], "2010-12-31")
            self.assertLess(captured["n_train_rows"], 1600)

    def test_default_checkpoint_resolution_does_not_fall_back_to_archive_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_root = tmp_path / "models"
            (model_root / "v2.1" / "AN" / "r1" / "20260423T000000").mkdir(parents=True, exist_ok=True)
            (model_root / "walkforward").mkdir(parents=True, exist_ok=True)
            with patch.object(dqn_spec, "MODEL_ROOT", model_root):
                resolved = dqn_spec.resolve_checkpoint_path(1, "Forex")
            self.assertEqual(resolved, model_root / "Forex" / "r1" / "latest" / "checkpoint.pt")

    def test_default_round_parser_trains_both_rounds(self):
        self.assertEqual(parse_rounds(None), [1, 2])
        self.assertEqual(parse_rounds("both"), [1, 2])
        self.assertEqual(parse_rounds("1"), [1])


class DQNInferenceTests(unittest.TestCase):
    def test_epsilon_is_constant(self):
        agent = DQNAgent(device="cpu")
        self.assertEqual(agent.epsilon_for_step(0), 0.3)
        self.assertEqual(agent.epsilon_for_step(50000), 0.3)
        self.assertEqual(agent.epsilon_for_step(500000), 0.3)

    def test_predict_action_ids_returns_valid_action_ids(self):
        agent = DQNAgent(device="cpu")
        states = np.zeros((8, shared_spec.SEQ_LEN, shared_spec.FEATURE_DIM), dtype=np.float32)
        action_ids = agent.predict_action_ids(states)
        self.assertEqual(action_ids.shape, (8,))
        self.assertTrue(np.isin(action_ids, [0, 1, 2]).all())

    def test_dqn_stabilizers_are_active(self):
        agent = DQNAgent(device="cpu")
        self.assertTrue(hasattr(agent.q_net, "value"))
        self.assertTrue(hasattr(agent.q_net, "advantage"))
        state = np.zeros((shared_spec.SEQ_LEN, shared_spec.FEATURE_DIM), dtype=np.float32)
        for idx in range(dqn_spec.BATCH_SIZE):
            agent.push(state, idx % 3, 0.01, state, 0.0)
        agent.train_steps = dqn_spec.TAU - 1
        loss = agent.learn()
        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(agent.target_updates, 1)


if __name__ == "__main__":
    unittest.main()
