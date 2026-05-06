"""Global strategy backtesting entrypoint on top of baseline_run metrics stack."""
from __future__ import annotations

from baseline_run import compute_strategy_metrics, load_contracts
from config import EXCLUDED_CONTRACTS, PAPER_TABLE3, SOURCE_OVERRIDES
from metrics import METRIC_NAMES


def backtest_strategy_metrics(
    asset_name: str,
    strategy: str,
    sigma_tgt: float,
    position_provider=None,
    excluded_contracts: list[str] | None = None,
    source_overrides: dict[str, str] | None = None,
    aggregation_mode: str = "variable_n",
    port_vol_target: float | None = None,
    port_bridge: str = "constant_posthoc",
    round_output: bool = True,
    save_audit_to: str | None = None,
) -> dict[str, float]:
    excluded = EXCLUDED_CONTRACTS if excluded_contracts is None else excluded_contracts
    overrides = SOURCE_OVERRIDES if source_overrides is None else source_overrides
    raw = load_contracts(asset_name, excluded_contracts=excluded, source_overrides=overrides)
    metric_map = compute_strategy_metrics(
        raw_data=raw,
        strat=strategy,
        sigma_tgt=sigma_tgt,
        aggregation_mode=aggregation_mode,
        port_vol_target=port_vol_target,
        port_bridge=port_bridge,
        position_provider=position_provider,
        save_audit_to=save_audit_to,
    )
    if round_output:
        return {name: round(metric_map[name], 3) for name in METRIC_NAMES}
    return {name: float(metric_map[name]) for name in METRIC_NAMES}


def paper_table3_reference(asset_name: str, strategy: str) -> dict[str, float] | None:
    return PAPER_TABLE3.get(asset_name, {}).get(strategy)


def contract_count(
    asset_name: str,
    excluded_contracts: list[str] | None = None,
    source_overrides: dict[str, str] | None = None,
) -> int:
    excluded = EXCLUDED_CONTRACTS if excluded_contracts is None else excluded_contracts
    overrides = SOURCE_OVERRIDES if source_overrides is None else source_overrides
    raw = load_contracts(asset_name, excluded_contracts=excluded, source_overrides=overrides)
    return len(raw)
