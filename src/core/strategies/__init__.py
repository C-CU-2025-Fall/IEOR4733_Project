"""
Strategies Module - Trading strategy implementations

Included strategies:
    - LongOnlyStrategy: Long-only benchmark
    - SignRStrategy: Sign-of-returns signal strategy
    - MACDStrategy: MACD technical indicator strategy
    - RouteB: A2C + regime detection strategy
    - DQNPaperStrategy: DQN strategy from the paper
"""

__all__ = [
    "LongOnlyStrategy",
    "SignRStrategy", 
    "MACDStrategy",
    "RouteB",
    "DQNPaperStrategy",
]
