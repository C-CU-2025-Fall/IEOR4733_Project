"""
Utils Module - General-purpose utility functions

Contents:
    - metrics.py: Risk-return metric calculation
    - indicators.py: Technical indicator calculation
    - visualization.py: Plotting and visualization
    - helpers.py: General helper functions

Main functions:
    - calculate_sharpe_ratio()
    - calculate_max_drawdown()
    - calculate_returns_metrics()
    - plot_strategy_comparison()
    - compute_sma(), compute_ema()
    - compute_macd()

Usage example:
    from src.core.utils import metrics, indicators
    
    sharpe = metrics.calculate_sharpe_ratio(returns)
    macd = indicators.compute_macd(prices)
"""

__all__ = [
    "metrics",
    "indicators",
    "visualization",
    "helpers",
]
