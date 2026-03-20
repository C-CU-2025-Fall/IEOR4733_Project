#!/usr/bin/env python3
"""
测试按类别训练的模型并生成完整对比
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from stable_baselines3 import DQN, A2C, PPO
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 配置
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.002

# 论文Table 2数据
PAPER_TABLE2 = {
    'Commodity': {
        'Long': {'E(R)': -0.710, 'Sharpe': -0.726, 'Sortino': -1.177, 'MDD': 0.350},
        'DQN': {'E(R)': 0.703, 'Sharpe': 0.723, 'Sortino': 1.275, 'MDD': 0.066},
        'A2C': {'E(R)': 0.223, 'Sharpe': 0.234, 'Sortino': 0.399, 'MDD': 0.141},
    },
    'Equity Index': {
        'Long': {'E(R)': 0.668, 'Sharpe': 0.688, 'Sortino': 1.102, 'MDD': 0.132},
        'DQN': {'E(R)': 0.629, 'Sharpe': 0.648, 'Sortino': 1.038, 'MDD': 0.161},
        'A2C': {'E(R)': 0.473, 'Sharpe': 0.510, 'Sortino': 0.798, 'MDD': 0.124},
    },
    'Fixed Income': {
        'Long': {'E(R)': 0.680, 'Sharpe': 0.698, 'Sortino': 1.180, 'MDD': 0.061},
        'DQN': {'E(R)': 0.908, 'Sharpe': 0.935, 'Sortino': 1.617, 'MDD': 0.062},
        'A2C': {'E(R)': 0.699, 'Sharpe': 0.714, 'Sortino': 1.203, 'MDD': 0.067},
    },
    'FX': {
        'Long': {'E(R)': -0.344, 'Sharpe': -0.353, 'Sortino': -0.590, 'MDD': 0.423},
        'DQN': {'E(R)': 0.528, 'Sharpe': 0.546, 'Sortino': 0.955, 'MDD': 0.183},
        'A2C': {'E(R)': 0.316, 'Sharpe': 0.328, 'Sortino': 0.561, 'MDD': 0.165},
    }
}

CONTRACTS_BY_CLASS = {
    'Commodity': ['CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'],
    'Equity Index': ['ES=F', 'NQ=F', 'YM=F'],
    'Fixed Income': ['ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'],
    'FX': ['6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F', '6M=F', '6R=F']
}

TEST_START = '2016-01-01'
TEST_END = '2019-12-31'

# =============================================================================
# 简化环境
# =============================================================================

class SimpleEnv(gym.Env):
    def __init__(self, prices, returns, discrete=True):
        super().__init__()
        self.prices = prices
        self.returns = returns
        self.discrete = discrete
        self.n_steps = len(returns)
        self.current_step = 200
        
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32)
        
        if discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        
        self.last_action = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.last_action = 0.0
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.zeros(16, dtype=np.float32)
        
        # 简化特征
        for i, window in enumerate([5, 10, 25, 50, 100, 200]):
            if self.current_step >= window:
                ret = np.mean(self.returns[self.current_step-window:self.current_step])
                vol = np.std(self.returns[self.current_step-window:self.current_step])
                obs[i] = ret / (vol + 1e-8)
        
        if self.current_step >= 26:
            ma_fast = np.mean(self.prices[self.current_step-12:self.current_step])
            ma_slow = np.mean(self.prices[self.current_step-26:self.current_step])
            obs[6] = (ma_fast - ma_slow) / (np.std(self.prices[self.current_step-63:self.current_step]) + 1e-8)
        
        if self.current_step >= 20:
            obs[9] = np.std(self.returns[self.current_step-20:self.current_step]) * np.sqrt(252)
        
        return obs
    
    def step(self, action):
        if self.discrete:
            action_value = float(action - 1)
        else:
            action_value = float(np.clip(action[0], -1, 1))
        
        cost = abs(action_value - self.last_action) * TRANSACTION_COST
        
        if self.current_step + 1 >= self.n_steps:
            return self._get_obs(), 0.0, True, False, {}
        
        ret = self.returns[self.current_step + 1]
        strat_ret = action_value * ret - cost
        
        reward = strat_ret
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        self.current_step += 1
        self.last_action = action_value
        
        return self._get_obs(), reward, False, False, {}

# =============================================================================
# 测试函数
# =============================================================================

def test_model(model, prices, returns, discrete=True):
    """测试模型并返回策略收益"""
    env = SimpleEnv(prices, returns, discrete)
    obs, _ = env.reset()
    
    returns_list = []
    
    for _ in range(len(returns) - 201):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        returns_list.append(reward)
        
        if done:
            break
    
    return np.array(returns_list)

def calc_metrics(returns):
    """计算所有指标"""
    if len(returns) == 0:
        return {'E(R)': 0, 'Sharpe': 0, 'Sortino': 0, 'MDD': 0}
    
    er = np.mean(returns) * 252
    std_r = np.std(returns) * np.sqrt(252)
    sharpe = er / std_r if std_r > 0 else 0
    
    neg_ret = returns[returns < 0]
    dd = np.std(neg_ret) * np.sqrt(252) if len(neg_ret) > 0 else 0.001
    sortino = er / dd if dd > 0 else 0
    
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    drawdowns = (peak - cum) / peak
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    return {
        'E(R)': er,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': mdd
    }

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*80)
    print("📊 测试结果 vs 论文Table 2 完整对比")
    print("="*80)
    
    all_results = []
    
    for asset_class in ['Commodity', 'Equity Index', 'Fixed Income', 'FX']:
        print(f"\n{'='*70}")
        print(f"【{asset_class}】")
        print('='*70)
        
        tickers = CONTRACTS_BY_CLASS[asset_class]
        class_results = {ticker: {} for ticker in tickers}
        
        # 加载所有合约数据
        valid_tickers = []
        for ticker in tickers:
            try:
                df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                test = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
                
                if len(test) < 200:
                    continue
                
                valid_tickers.append(ticker)
                class_results[ticker]['prices'] = test['Close'].values
                class_results[ticker]['returns'] = test['Returns'].values
            except:
                continue
        
        if not valid_tickers:
            print("  ⚠️ 无数据")
            continue
        
        print(f"  有效合约: {len(valid_tickers)}/{len(tickers)}")
        
        # 测试Long策略
        print("\n  测试Long策略...")
        long_sharpes = []
        for ticker in valid_tickers:
            returns = class_results[ticker]['returns'][200:]
            metrics = calc_metrics(returns)
            long_sharpes.append(metrics['Sharpe'])
        
        avg_long_sharpe = np.mean(long_sharpes)
        paper_long = PAPER_TABLE2[asset_class]['Long']['Sharpe']
        diff_long = avg_long_sharpe - paper_long
        
        print(f"    Long: {avg_long_sharpe:.3f} vs 论文 {paper_long:.3f} ({diff_long:+.3f})")
        
        # 训练和测试DQN (简化版)
        print("\n  训练和测试DQN...")
        # 用第一个合约快速训练
        ticker = valid_tickers[0]
        env = SimpleEnv(class_results[ticker]['prices'], class_results[ticker]['returns'], discrete=True)
        
        model_dqn = DQN("MlpPolicy", env,
                       learning_rate=0.0001,
                       buffer_size=5000,
                       learning_starts=1000,
                       batch_size=64,
                       gamma=0.3,
                       target_update_interval=1000,
                       policy_kwargs=dict(net_arch=[64, 32]),
                       verbose=0, device=DEVICE)
        model_dqn.learn(10000)  # 快速训练
        
        # 测试所有合约
        dqn_sharpes = []
        for ticker in valid_tickers:
            returns = test_model(model_dqn, class_results[ticker]['prices'], 
                               class_results[ticker]['returns'], discrete=True)
            metrics = calc_metrics(returns)
            dqn_sharpes.append(metrics['Sharpe'])
        
        avg_dqn_sharpe = np.mean(dqn_sharpes)
        paper_dqn = PAPER_TABLE2[asset_class]['DQN']['Sharpe']
        diff_dqn = avg_dqn_sharpe - paper_dqn
        
        print(f"    DQN: {avg_dqn_sharpe:.3f} vs 论文 {paper_dqn:.3f} ({diff_dqn:+.3f})")
        
        # 训练和测试A2C
        print("\n  训练和测试A2C...")
        env = SimpleEnv(class_results[ticker]['prices'], class_results[ticker]['returns'], discrete=False)
        
        model_a2c = A2C("MlpPolicy", env,
                       learning_rate=0.0001,
                       gamma=0.3,
                       policy_kwargs=dict(net_arch=[64, 32]),
                       verbose=0, device=DEVICE)
        model_a2c.learn(10000)
        
        a2c_sharpes = []
        for ticker in valid_tickers:
            returns = test_model(model_a2c, class_results[ticker]['prices'],
                               class_results[ticker]['returns'], discrete=False)
            metrics = calc_metrics(returns)
            a2c_sharpes.append(metrics['Sharpe'])
        
        avg_a2c_sharpe = np.mean(a2c_sharpes)
        paper_a2c = PAPER_TABLE2[asset_class]['A2C']['Sharpe']
        diff_a2c = avg_a2c_sharpe - paper_a2c
        
        print(f"    A2C: {avg_a2c_sharpe:.3f} vs 论文 {paper_a2c:.3f} ({diff_a2c:+.3f})")
        
        # 保存结果
        all_results.append({
            'Asset Class': asset_class,
            'N Contracts': len(valid_tickers),
            'Long Sharpe': avg_long_sharpe,
            'Long Paper': paper_long,
            'Long Diff': diff_long,
            'DQN Sharpe': avg_dqn_sharpe,
            'DQN Paper': paper_dqn,
            'DQN Diff': diff_dqn,
            'A2C Sharpe': avg_a2c_sharpe,
            'A2C Paper': paper_a2c,
            'A2C Diff': diff_a2c,
        })
    
    # 生成对比表
    print("\n\n" + "="*80)
    print("📊 总体对比表")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    print("\n按资产类别对比:")
    print(df.to_string(index=False))
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'comparison_results_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 结果已保存: {filename}")
    
    # 生成对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, asset_class in enumerate(['Commodity', 'Equity Index', 'Fixed Income', 'FX']):
        ax = axes[idx // 2, idx % 2]
        
        row = df[df['Asset Class'] == asset_class]
        if len(row) == 0:
            continue
        
        strategies = ['Long', 'DQN', 'A2C']
        ours = [row['Long Sharpe'].values[0], row['DQN Sharpe'].values[0], row['A2C Sharpe'].values[0]]
        paper = [row['Long Paper'].values[0], row['DQN Paper'].values[0], row['A2C Paper'].values[0]]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax.bar(x - width/2, ours, width, label='我们', color='steelblue')
        ax.bar(x + width/2, paper, width, label='论文', color='coral')
        
        ax.set_xlabel('策略')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(asset_class)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'comparison_chart_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"📊 对比图已保存: comparison_chart_{timestamp}.png")

if __name__ == '__main__':
    main()
