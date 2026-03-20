#!/usr/bin/env python3
"""
完整复现：40个合约 + 论文Table 2对比 + Figure 1对比
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from paper_components import (
    DifferentialSharpeRatio,
    MultiTimeScaleState,
    VolatilityScaler
)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from stable_baselines3 import DQN, A2C
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 论文配置
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.001

# 论文超参数
GAMMA = 0.3
BUFFER_SIZE = 5000
BATCH_SIZE_DQN = 64
BATCH_SIZE_A2C = 128
LEARNING_RATE = 0.0001
TARGET_UPDATE = 1000
TOTAL_TIMESTEPS = 50000

# 训练/测试期
TRAIN_START = '2011-01-03'
TRAIN_END = '2015-12-31'
TEST_START = '2016-01-01'
TEST_END = '2019-12-31'

# 40个合约按资产类别分组 (论文 Appendix A)
CONTRACTS_BY_CLASS = {
    'Commodity': [
        'CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F',
        'KC=F', 'CC=F', 'SB=F', 'CT=F', 'LC=F', 'LBS=F', 'OJ=F'
    ],
    'Equity Index': [
        'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'EMD=F', 'VA=F'
    ],
    'Fixed Income': [
        'ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F', 'FGBL.F', 'FGBM.F'
    ],
    'FX': [
        '6E=F', '6J=F', '6B=F', '6A=F', '6C=F', '6S=F', '6N=F'
    ]
}

# =============================================================================
# 简化环境
# =============================================================================

class SimpleEnv(gym.Env):
    def __init__(self, prices, returns, use_dsr=True):
        super().__init__()
        self.prices = prices
        self.returns = returns
        self.use_dsr = use_dsr
        self.n_steps = len(returns)
        
        self.observation_space = spaces.Box(-np.inf, np.inf, (16,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        
        self.state_builder = MultiTimeScaleState()
        self.dsr = DifferentialSharpeRatio(eta=0.01)
        self.scaler = VolatilityScaler(target_vol=0.10)
        
        self.step_idx = 200
        self.last_action = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 200
        self.last_action = 0.0
        self.dsr.reset()
        return self._obs(), {}
    
    def _obs(self):
        return self.state_builder.compute(
            self.prices[:self.step_idx+1],
            self.returns[:self.step_idx+1],
            self.step_idx
        ).astype(np.float32)
    
    def step(self, action):
        action = float(np.clip(action[0] if hasattr(action, '__len__') else action, -1, 1))
        cost = abs(action - self.last_action) * TRANSACTION_COST
        
        if self.step_idx + 1 >= self.n_steps:
            return self._obs(), 0, True, False, {}
        
        ret = self.returns[self.step_idx + 1]
        strat_ret = action * ret - cost
        
        scaled = self.scaler.scale(1.0, self.returns[:self.step_idx+1], self.step_idx)
        strat_ret *= scaled
        
        reward = self.dsr.update(strat_ret) if self.use_dsr else strat_ret
        
        self.step_idx += 1
        self.last_action = action
        
        return self._obs(), reward, self.step_idx >= self.n_steps - 1, False, {}

class DiscreteEnv(SimpleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Discrete(3)
    
    def step(self, action):
        return super().step(np.array([float(action - 1)]))

# =============================================================================
# 指标计算
# =============================================================================

def calc_metrics(returns, positions):
    """计算论文Table 2的所有指标"""
    sr = returns[:len(positions)] * positions
    sr = sr[np.isfinite(sr)]
    if len(sr) == 0:
        return {'E(R)': 0, 'Std(R)': 0, 'DD': 0, 'Sharpe': 0, 
                'Sortino': 0, 'MDD': 0, 'Calmar': 0}
    
    annual_factor = np.sqrt(252)
    
    # E(R) - 年化收益
    er = np.mean(sr) * 252
    
    # Std(R) - 年化标准差
    std_r = np.std(sr) * annual_factor
    
    # Downside Deviation (DD) - 只计算负收益
    neg_ret = sr[sr < 0]
    dd = np.std(neg_ret) * annual_factor if len(neg_ret) > 0 else 0.001
    
    # Sharpe Ratio
    risk_free = 0.02
    sharpe = (er - risk_free) / std_r if std_r > 0 else 0
    
    # Sortino Ratio
    sortino = (er - risk_free) / dd if dd > 0 else 0
    
    # Maximum Drawdown
    cum = np.cumprod(1 + sr)
    peak = np.maximum.accumulate(cum)
    drawdowns = (peak - cum) / peak
    mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Calmar Ratio
    calmar = er / mdd if mdd > 0 else 0
    
    return {
        'E(R)': er,
        'Std(R)': std_r,
        'DD': dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': -mdd,
        'Calmar': calmar
    }

# =============================================================================
# 基线策略
# =============================================================================

def baseline_long(n): 
    return np.ones(n)

def baseline_sign(returns, window=252):
    """Sign(R) strategy"""
    signals = np.zeros(len(returns))
    for i in range(window, len(returns)):
        cum_ret = np.sum(returns[i-window:i])
        signals[i] = np.sign(cum_ret)
    return signals

def baseline_macd(prices):
    """MACD strategy"""
    p = pd.Series(prices)
    macd = p.ewm(12).mean() - p.ewm(26).mean()
    sig = macd / (0.89 * np.exp(-macd**2 / 4))
    return np.tanh(sig.values)

# =============================================================================
# 训练单个合约
# =============================================================================

def train_contract(ticker, quiet=False):
    """训练单个合约，返回所有策略结果"""
    try:
        df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
    except:
        if not quiet:
            print(f"  ⚠️ {ticker} 数据不存在")
        return None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
    test = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
    
    if len(train) < 500 or len(test) < 100:
        if not quiet:
            print(f"  ⚠️ {ticker} 数据不足")
        return None
    
    train_p = train['Close'].values
    train_r = train['Returns'].values
    test_p = test['Close'].values
    test_r = test['Returns'].values
    
    results = []
    
    # Long
    pos = baseline_long(len(test_r)-200)
    m = calc_metrics(test_r[200:], pos)
    m['Ticker'] = ticker
    m['Strategy'] = 'Long'
    results.append(m)
    
    # Sign(R)
    pos = baseline_sign(test_r)[200:]
    pos = pos[:len(test_r)-200]
    if len(pos) > 0:
        m = calc_metrics(test_r[200:200+len(pos)], pos)
        m['Ticker'] = ticker
        m['Strategy'] = 'Sign(R)'
        results.append(m)
    
    # MACD
    pos = baseline_macd(test_p)[200:]
    pos = pos[:len(test_r)-200]
    if len(pos) > 0:
        m = calc_metrics(test_r[200:200+len(pos)], pos)
        m['Ticker'] = ticker
        m['Strategy'] = 'MACD'
        results.append(m)
    
    # DQN
    try:
        env_train = DiscreteEnv(train_p, train_r, use_dsr=True)
        env_test = DiscreteEnv(test_p, test_r, use_dsr=False)
        
        model = DQN("MlpPolicy", env_train,
                   learning_rate=LEARNING_RATE,
                   buffer_size=BUFFER_SIZE,
                   learning_starts=1000,
                   batch_size=BATCH_SIZE_DQN,
                   gamma=GAMMA,
                   target_update_interval=TARGET_UPDATE,
                   policy_kwargs=dict(net_arch=[64, 32]),
                   verbose=0, device=DEVICE)
        
        model.learn(TOTAL_TIMESTEPS, progress_bar=False)
        
        obs, _ = env_test.reset()
        pos = []
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            pos.append(float(a - 1))
            obs, _, done, _, _ = env_test.step(a)
        
        m = calc_metrics(test_r[200:200+len(pos)], np.array(pos))
        m['Ticker'] = ticker
        m['Strategy'] = 'DQN'
        results.append(m)
    except:
        pass
    
    # A2C
    try:
        env_train = SimpleEnv(train_p, train_r, use_dsr=True)
        env_test = SimpleEnv(test_p, test_r, use_dsr=False)
        
        model = A2C("MlpPolicy", env_train,
                   learning_rate=LEARNING_RATE,
                   gamma=GAMMA,
                   policy_kwargs=dict(net_arch=[64, 32]),
                   verbose=0, device=DEVICE)
        
        model.learn(TOTAL_TIMESTEPS, progress_bar=False)
        
        obs, _ = env_test.reset()
        pos = []
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            pos.append(float(a[0]))
            obs, _, done, _, _ = env_test.step(a)
        
        m = calc_metrics(test_r[200:200+len(pos)], np.array(pos))
        m['Ticker'] = ticker
        m['Strategy'] = 'A2C'
        results.append(m)
    except:
        pass
    
    return results

# =============================================================================
# 训练所有合约
# =============================================================================

def train_all_contracts():
    """训练所有40个合约"""
    
    print("="*80)
    print("🤖 完整复现：40个合约")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"γ: {GAMMA}")
    print(f"步数: {TOTAL_TIMESTEPS:,}")
    print()
    
    all_results = []
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        print(f"\n{'='*60}")
        print(f"📊 {asset_class} ({len(tickers)} 合约)")
        print('='*60)
        
        class_results = []
        
        for ticker in tickers:
            if not os.path.exists(f'{DATA_DIR}/{ticker}.csv'):
                print(f"  ⚠️ {ticker} 跳过 (无数据)")
                continue
            
            print(f"  🔄 {ticker}...", end=' ', flush=True)
            
            results = train_contract(ticker, quiet=True)
            if results:
                all_results.extend(results)
                class_results.extend(results)
                
                # 打印Sharpe
                for r in results:
                    if r['Strategy'] in ['Long', 'DQN']:
                        print(f"{r['Strategy']}: {r['Sharpe']:.2f}", end=' ', flush=True)
            print()
        
        # 计算该资产类别的平均
        if class_results:
            class_df = pd.DataFrame(class_results)
            print(f"\n  📈 {asset_class} 平均结果:")
            for strategy in ['Long', 'Sign(R)', 'MACD', 'DQN', 'A2C']:
                strat_df = class_df[class_df['Strategy'] == strategy]
                if len(strat_df) > 0:
                    mean_sharpe = strat_df['Sharpe'].mean()
                    print(f"    {strategy:<8} | Sharpe: {mean_sharpe:>7.3f} | N={len(strat_df)}")
    
    # 保存结果
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results_all40_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 结果已保存: {filename}")
    
    return df

# =============================================================================
# 生成论文Table 2对比
# =============================================================================

def create_table2_comparison(df):
    """生成与论文Table 2的对比"""
    
    print("\n" + "="*80)
    print("📊 Table 2 对比 (Portfolio-level Volatility Targeting)")
    print("="*80)
    
    # 论文Table 2数据
    paper_table2 = {
        'Commodity': {
            'Long': {'Sharpe': -0.726, 'Sortino': -1.177, 'Calmar': -0.140},
            'Sign(R)': {'Sharpe': 0.354, 'Sortino': 0.606, 'Calmar': 0.119},
            'MACD': {'Sharpe': -0.175, 'Sortino': -0.293, 'Calmar': -0.060},
            'DQN': {'Sharpe': 0.723, 'Sortino': 1.275, 'Calmar': 0.501},
            'PG': {'Sharpe': 0.063, 'Sortino': 0.106, 'Calmar': 0.023},
            'A2C': {'Sharpe': 0.234, 'Sortino': 0.399, 'Calmar': 0.091},
        },
        'Equity Index': {
            'Long': {'Sharpe': 0.688, 'Sortino': 1.102, 'Calmar': 0.509},
            'Sign(R)': {'Sharpe': 0.236, 'Sortino': 0.374, 'Calmar': 0.077},
            'MACD': {'Sharpe': 0.017, 'Sortino': 0.027, 'Calmar': 0.006},
            'DQN': {'Sharpe': 0.648, 'Sortino': 1.038, 'Calmar': 0.381},
            'PG': {'Sharpe': 0.447, 'Sortino': 0.714, 'Calmar': 0.185},
            'A2C': {'Sharpe': 0.510, 'Sortino': 0.798, 'Calmar': 0.328},
        },
        'All': {
            'Long': {'Sharpe': 0.058, 'Sortino': 0.092, 'Calmar': 0.013},
            'Sign(R)': {'Sharpe': 0.441, 'Sortino': 0.737, 'Calmar': 0.201},
            'MACD': {'Sharpe': 0.091, 'Sortino': 0.153, 'Calmar': 0.035},
            'DQN': {'Sharpe': 1.288, 'Sortino': 2.220, 'Calmar': 1.025},
            'PG': {'Sharpe': 0.754, 'Sortino': 1.247, 'Calmar': 0.480},
            'A2C': {'Sharpe': 1.050, 'Sortino': 1.785, 'Calmar': 0.685},
        }
    }
    
    # 按资产类别计算我们的结果
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        class_df = df[df['Ticker'].isin(tickers)]
        
        if len(class_df) == 0:
            continue
        
        print(f"\n【{asset_class}】")
        print(f"{'策略':<10} | {'论文Sharpe':>10} | {'我们Sharpe':>10} | {'差距':>10}")
        print("-" * 50)
        
        for strategy in ['Long', 'Sign(R)', 'MACD', 'DQN', 'A2C']:
            strat_df = class_df[class_df['Strategy'] == strategy]
            
            if len(strat_df) > 0:
                our_sharpe = strat_df['Sharpe'].mean()
                
                # 找论文对应值
                paper_sharpe = None
                for ac in paper_table2:
                    if asset_class in ['Commodity', 'Equity Index'] and ac == asset_class:
                        if strategy in paper_table2[ac]:
                            paper_sharpe = paper_table2[ac][strategy]['Sharpe']
                    elif asset_class not in ['Commodity', 'Equity Index'] and ac == 'All':
                        if strategy in paper_table2[ac]:
                            paper_sharpe = paper_table2[ac][strategy]['Sharpe']
                
                if paper_sharpe is not None:
                    diff = our_sharpe - paper_sharpe
                    print(f"{strategy:<10} | {paper_sharpe:>10.3f} | {our_sharpe:>10.3f} | {diff:>+10.3f}")
                else:
                    print(f"{strategy:<10} | {'-':>10} | {our_sharpe:>10.3f} | {'-':>10}")

# =============================================================================
# 生成Figure 1对比
# =============================================================================

def create_figure1_comparison(df):
    """生成与论文Figure 1的对比 - 累积收益曲线"""
    
    print("\n" + "="*80)
    print("📊 Figure 1 对比 - 累积收益曲线")
    print("="*80)
    print("⚠️ 需要保存每日仓位才能生成累积收益曲线")
    print("当前结果文件只包含最终指标，无法回溯生成Figure 1")
    print()
    print("如需生成Figure 1，需要:")
    print("1. 在训练时保存每日仓位")
    print("2. 计算每日策略收益")
    print("3. 累积得到曲线")
    
    # 简化版：用最终收益估算
    print("\n简化版对比 (使用E(R)估算):")
    print()
    
    for asset_class in ['Commodity', 'Equity Index']:
        tickers = CONTRACTS_BY_CLASS.get(asset_class, [])
        class_df = df[df['Ticker'].isin(tickers)]
        
        if len(class_df) == 0:
            continue
        
        print(f"【{asset_class}】")
        for strategy in ['Long', 'DQN', 'A2C']:
            strat_df = class_df[class_df['Strategy'] == strategy]
            if len(strat_df) > 0:
                mean_ret = strat_df['E(R)'].mean()
                print(f"  {strategy:<8} | 年化收益: {mean_ret:>7.2%}")
        print()

# =============================================================================
# 主函数
# =============================================================================

def main():
    # 训练所有合约
    df = train_all_contracts()
    
    # 生成对比
    create_table2_comparison(df)
    create_figure1_comparison(df)
    
    # 汇总统计
    print("\n" + "="*80)
    print("📊 总体汇总")
    print("="*80)
    
    summary = df.groupby('Strategy').agg({
        'Sharpe': ['mean', 'std'],
        'E(R)': 'mean',
        'MDD': 'mean'
    }).round(3)
    
    print(summary)

if __name__ == '__main__':
    main()
