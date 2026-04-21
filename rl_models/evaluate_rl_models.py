#!/usr/bin/env python3
"""
RL 模型回测和评估脚本

- 加载训练好的 DQN/PG/A2C 模型
- 在测试期间 (2016-2019) 进行回测
- 计算与规则策略相同的指标
- 生成性能报告
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
import torch
import torch.nn.functional as F

# 导入现有项目模块（使用 .. 引用父目录）
from ..baseline_run import (
    SIGMA, EWMA_SPAN, BP, TRADING_DAYS, WARM_START,
    compute_contract_returns, compute_portfolio_returns
)
from ..data_loader import load_clc_full
from ..metrics import compute_metrics
from ..config import PAPER_TABLE3, EXCLUDED_CONTRACTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# LSTM 网络定义 (与训练脚本一致)
# =============================================================================

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(torch.nn.LeakyReLU(0.01))
        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        
        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.mlp(lstm_out[:, -1, :])

# =============================================================================
# 特征提取 (与训练脚本一致)
# =============================================================================

class FeatureEngineer:
    """特征工程"""
    def __init__(self):
        pass
    
    def compute_features(self, returns):
        """从收益率计算特征"""
        features = np.zeros(8)
        
        if len(returns) < 2:
            return features
        
        # 1. 平均收益率
        features[0] = np.mean(returns)
        
        # 2. 波动率
        features[1] = np.std(returns)
        
        # 3. 偏度
        if len(returns) >= 3:
            features[2] = (np.mean((returns - np.mean(returns))**3) / 
                          (np.std(returns)**3 + 1e-8))
        
        # 4. 峰度
        if len(returns) >= 4:
            features[3] = (np.mean((returns - np.mean(returns))**4) / 
                          (np.std(returns)**4 + 1e-8))
        
        # 5. 最大回撤
        cum_returns = np.cumprod(1 + returns) - 1
        if len(cum_returns) > 0:
            running_max = np.maximum.accumulate(cum_returns)
            features[4] = np.min(cum_returns - running_max)
        
        # 6. 夏普比 (简化版，假设无风险利率为 0)
        if features[1] > 0:
            features[5] = np.mean(returns) / features[1] * np.sqrt(252)
        
        # 7. Sortino 比 (下行波动)
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            if downside_std > 0:
                features[6] = np.mean(returns) / downside_std * np.sqrt(252)
        
        # 8. 正收益率比例
        features[7] = np.sum(returns > 0) / len(returns)
        
        return features.astype(np.float32)

# =============================================================================
# 回测环境
# =============================================================================

class BacktestEnv:
    def __init__(self, returns, start_date=None, end_date=None):
        self.returns = returns
        self.n = len(returns)
        self.t = 0
        self.position = 0
        self.wealth = 1.0
        self.feature_eng = FeatureEngineer()
        self.history = []
    
    def reset(self, start_idx=100):
        self.t = start_idx
        self.position = 0
        self.wealth = 1.0
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        if self.t < 100:
            return np.zeros(8, dtype=np.float32)
        ret_window = self.returns[self.t-100:self.t]
        features = self.feature_eng.compute_features(ret_window)
        return features
    
    def step(self, action):
        if self.t >= self.n - 1:
            return self._get_state(), 0, True
        
        # action: 0 (short), 1 (hold), 2 (long)
        action_map = {0: -1, 1: 0, 2: 1}
        pos = action_map.get(action, 0)
        
        pnl = self.position * self.returns[self.t+1] - BP * abs(pos - self.position)
        self.wealth *= (1 + pnl)
        self.position = pos
        self.t += 1
        reward = pnl
        
        done = self.t >= self.n - 1
        
        # 记录历史
        self.history.append({
            'wealth': self.wealth,
            'position': self.position,
            'pnl': pnl
        })
        
        return self._get_state(), reward, done

# =============================================================================
# 模型推理函数
# =============================================================================

def infer_dqn(model, state):
    """DQN 推理：选择最大 Q 值的动作"""
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = model(s)
        action = q_values.argmax(1).item()
        return action - 1  # 转换为 -1, 0, 1

def infer_pg(model, state):
    """PG 推理：从策略采样动作"""
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        logits = model(s)
        probs = F.softmax(logits, dim=1)[0]
        action = torch.multinomial(probs, 1).item()
        return action - 1  # 转换为 -1, 0, 1

def infer_a2c_actor(model, state):
    """A2C Actor 推理：从策略采样动作"""
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        logits = model(s)
        probs = F.softmax(logits, dim=1)[0]
        action = torch.multinomial(probs, 1).item()
        return action - 1  # 转换为 -1, 0, 1

# =============================================================================
# 回测函数
# =============================================================================

def backtest_model(model, returns, model_type='dqn', start_idx=WARM_START):
    """
    运行模型回测
    Returns:
        wealth_path: 财富路径数组
    """
    env = BacktestEnv(returns)
    state = env.reset(start_idx)
    
    wealth_path = [1.0]
    
    infer_func = {
        'dqn': infer_dqn,
        'pg': infer_pg,
        'a2c': infer_a2c_actor
    }.get(model_type, infer_dqn)
    
    while not True:
        action = infer_func(model, state)
        next_state, reward, done = env.step(action)
        wealth_path.append(env.wealth)
        state = next_state
        
        if done:
            break
    
    return np.array(wealth_path)

# =============================================================================
# 加载和回测
# =============================================================================

def load_model_file(model_path, model_type):
    """加载保存的模型"""
    try:
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        return models
    except Exception as e:
        print(f"  ❌ 加载失败：{e}")
        return None

def evaluate_all_models():
    """评估所有已训练的 RL 模型"""
    print("="*80)
    print("🔥 RL 模型回测和评估")
    print("="*80)
    
    model_patterns = {
        'dqn': 'models_dqn_paper_*.pkl',
        'pg': 'models_pg_paper_*.pkl',
        'a2c': 'models_a2c_paper_*.pkl'
    }
    
    results = {}
    
    for model_type, pattern in model_patterns.items():
        print(f"\n📊 搜索 {model_type.upper()} 模型：{pattern}")
        
        # 查找最新的模型文件
        import glob
        files = glob.glob(pattern)
        if not files:
            print(f"  ❌ 未找到 {model_type} 模型文件")
            continue
        
        latest_file = max(files, key=os.path.getctime)
        print(f"  📁 找到：{latest_file}")
        
        models = load_model_file(latest_file, model_type)
        if models is None:
            continue
        
        results[model_type] = models
    
    if not results:
        print("\n❌ 没有找到任何模型文件")
        print("   请先运行: python rl_models/train_all_rl_models.py")
        return
    
    print("\n" + "="*80)
    print("✅ 模型加载完成，现在可以进行回测")
    print("="*80)
    
    # TODO: 集成与现有的 baseline_run 系统
    print("\n💡 下一步：")
    print("   1. 加载测试期数据 (2016-01-01 to 2019-12-31)")
    print("   2. 运行 RL 模型推理获得交易头寸")
    print("   3. 计算投资组合回报")
    print("   4. 使用 metrics.py 计算 9 个指标")
    print("   5. 生成 Table 3 扩展版本 (Long + MACD + Sign(R) + DQN + PG + A2C)")
    print("="*80)

if __name__ == "__main__":
    evaluate_all_models()
