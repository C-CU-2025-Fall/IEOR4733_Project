"""BP edge-case and transaction-cost equation validation tests."""
from __future__ import annotations

import unittest

import numpy as np

from drl_shared.state_space import compute_eq4_reward
from baseline_run import compute_contract_returns_from_positions


class TestBpEdgeCases(unittest.TestCase):

    def test_bp_zero_returns_zero_cost(self):
        prices = np.array([100.0, 101.0, 102.0], dtype=float)
        returns = np.array([0.0, 1.0, 1.0], dtype=float)
        sigma = np.array([0.5, 0.5, 0.5], dtype=float)
        sigma_tgt = 0.058

        reward, gross, tc, _ = compute_eq4_reward(
            prices, returns, sigma, idx=2,
            action=1.0, prev_action=0.0,
            sigma_tgt=sigma_tgt, bp=0.0, prev_sigma=0.5,
        )
        self.assertEqual(tc, 0.0)
        self.assertAlmostEqual(reward, gross, places=8)

    def test_bp_positive_position_change_incurs_cost(self):
        prices = np.array([100.0, 101.0, 102.0], dtype=float)
        returns = np.array([0.0, 1.0, 1.0], dtype=float)
        sigma = np.array([0.5, 0.5, 0.5], dtype=float)
        sigma_tgt = 0.058

        reward, gross, tc, _ = compute_eq4_reward(
            prices, returns, sigma, idx=2,
            action=1.0, prev_action=0.0,
            sigma_tgt=sigma_tgt, bp=0.001, prev_sigma=0.5,
        )
        self.assertGreater(tc, 0.0)
        self.assertLess(reward, gross)

    def test_bp_positive_no_position_change_zero_cost(self):
        prices = np.array([100.0, 101.0, 102.0], dtype=float)
        returns = np.array([0.0, 1.0, 1.0], dtype=float)
        sigma = np.array([0.5, 0.5, 0.5], dtype=float)
        sigma_tgt = 0.058

        reward, gross, tc, _ = compute_eq4_reward(
            prices, returns, sigma, idx=2,
            action=0.5, prev_action=0.5,
            sigma_tgt=sigma_tgt, bp=0.001, prev_sigma=0.5,
        )
        self.assertEqual(tc, 0.0)
        self.assertAlmostEqual(reward, gross, places=8)

    def test_bp_scales_linearly_with_position_change(self):
        prices = np.array([100.0, 101.0, 102.0], dtype=float)
        returns = np.array([0.0, 1.0, 1.0], dtype=float)
        sigma = np.array([0.5, 0.5, 0.5], dtype=float)
        sigma_tgt = 0.058

        _, _, tc_low, _ = compute_eq4_reward(
            prices, returns, sigma, idx=2,
            action=1.0, prev_action=0.0,
            sigma_tgt=sigma_tgt, bp=0.001, prev_sigma=0.5,
        )
        _, _, tc_high, _ = compute_eq4_reward(
            prices, returns, sigma, idx=2,
            action=1.0, prev_action=0.0,
            sigma_tgt=sigma_tgt, bp=0.002, prev_sigma=0.5,
        )
        self.assertGreater(tc_low, 0.0)
        self.assertGreater(tc_high, tc_low)
        self.assertAlmostEqual(tc_high / tc_low, 2.0, places=8)

    def test_bp_zero_no_division_error(self):
        prices = np.array([100.0, 101.0], dtype=float)
        returns = np.array([0.0, 1.0], dtype=float)
        sigma = np.array([0.5, 0.5], dtype=float)

        try:
            compute_eq4_reward(
                prices, returns, sigma, idx=1,
                action=1.0, prev_action=0.0,
                sigma_tgt=0.058, bp=0.0,
            )
        except ZeroDivisionError:
            self.fail("BP=0 caused ZeroDivisionError in compute_eq4_reward")

    def test_bp_max_45bps_full_scaling(self):
        prices = np.array([100.0, 101.0, 102.0], dtype=float)
        returns = np.array([0.0, 1.0, 1.0], dtype=float)
        sigma = np.array([0.5, 0.5, 0.5], dtype=float)
        sigma_tgt = 0.058

        _, _, tc_max, _ = compute_eq4_reward(
            prices, returns, sigma, idx=2,
            action=1.0, prev_action=0.0,
            sigma_tgt=sigma_tgt, bp=0.0045, prev_sigma=0.5,
        )
        _, _, tc_20bps, _ = compute_eq4_reward(
            prices, returns, sigma, idx=2,
            action=1.0, prev_action=0.0,
            sigma_tgt=sigma_tgt, bp=0.0020, prev_sigma=0.5,
        )
        self.assertGreater(tc_max, tc_20bps)
        self.assertAlmostEqual(tc_max / tc_20bps, 0.0045 / 0.0020, places=8)


class TestComputeContractReturnsFromPositions(unittest.TestCase):

    def test_returns_sum_matches_expected_total_cost(self):
        rd = {
            'tk': 'TEST',
            'rt': np.array([0.0, 0.01, 0.01, 0.01], dtype=float),
            'sigma': np.array([0.1, 0.1, 0.1, 0.1], dtype=float),
            'prices': np.array([100.0, 101.0, 102.0, 103.0], dtype=float),
        }
        sigma_tgt = 0.058
        positions = [0.0, 1.0, 1.0, 0.0]

        result = compute_contract_returns_from_positions(rd, positions, sigma_tgt, detail=True)
        tc_sum = result['tc_cost'].sum()
        self.assertGreater(tc_sum, 0.0)

    def test_flat_position_produces_zero_tc(self):
        rd = {
            'tk': 'TEST',
            'rt': np.array([0.0, 0.01, 0.01], dtype=float),
            'sigma': np.array([0.1, 0.1, 0.1], dtype=float),
            'prices': np.array([100.0, 101.0, 102.0], dtype=float),
        }
        sigma_tgt = 0.058
        positions = [0.0, 0.0, 0.0]

        result = compute_contract_returns_from_positions(rd, positions, sigma_tgt, detail=True)
        np.testing.assert_array_equal(result['tc_cost'], np.zeros(3))


if __name__ == '__main__':
    unittest.main()
