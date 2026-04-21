"""
RL Models Package

包含深度强化学习模型的实现：
- DQN: Deep Q-Network with Fixed Q-targets
- PG: Policy Gradient with Monte Carlo
- A2C: Advantage Actor-Critic

使用方法：
  python rl_models/train_all_rl_models.py      # 训练所有模型
  python rl_models/train_dqn_paper_aligned.py  # 仅训练 DQN
  python rl_models/evaluate_rl_models.py       # 评估已训练的模型
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# 导出模块
__all__ = [
    'train_dqn_paper_aligned',
    'train_pg_paper_aligned',
    'train_a2c_paper_aligned',
    'evaluate_rl_models'
]
