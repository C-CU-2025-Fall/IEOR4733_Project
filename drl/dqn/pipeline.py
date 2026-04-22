"""Compatibility re-export for shared DRL state-space helpers."""
from drl_shared.state_space import (
    ContractArrays,
    ContractEnv,
    action_id_to_position,
    build_contract_arrays,
    build_feature_matrix,
    compute_additive_returns,
    compute_eq4_reward,
    compute_ewma_sigma,
    continuous_action_to_position,
    get_feature_window,
    position_to_action_id,
)

__all__ = [
    "ContractArrays",
    "ContractEnv",
    "action_id_to_position",
    "build_contract_arrays",
    "build_feature_matrix",
    "compute_additive_returns",
    "compute_eq4_reward",
    "compute_ewma_sigma",
    "continuous_action_to_position",
    "get_feature_window",
    "position_to_action_id",
]

