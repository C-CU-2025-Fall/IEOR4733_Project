"""
DRL Trading Strategies - Modular Python Framework

A complete trading strategy framework including:
- Traditional strategies: Long Only, Sign(R), MACD
- Deep reinforcement learning: DQN (paper replication), A2C (pending upload), Route B
- Full data management, indicator calculation, and visualization tools

Usage example:
    from src.core.strategies import LongOnlyStrategy, SignRStrategy, MACDStrategy
    from src.core.models.dqn import DQNEvaluator
    
    long_strategy = LongOnlyStrategy(config)
    dqn_evaluator = DQNEvaluator(model_path)
"""

__version__ = "1.0.0"
__author__ = "IEOR 4733 Project Team"

from src.core import strategies, models, data, utils

__all__ = [
    "strategies",
    "models", 
    "data",
    "utils",
]
