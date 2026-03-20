#!/usr/bin/env python3
"""
完整复现：33个可用合约
- 论文超参数完全对齐
- 按资产类别分组
- 生成Table 2对比
- 生成Figure 1对比
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

# 论文超参数 (Table 1)
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

# 33个可用合约按资产类别分组
CONTRACTS_BY_CLASS = {
    'Commodity': [
        'CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F',
        'ZC=F', 'ZS=F', 'ZW=F',
        'KC=F', 'CC=F', 'SB=F', 'CT=F', 'OJ=F'
    ],
    'Equity Index': [
        'ES=F', 'NQ=F', 'YM=F', 'RTY=F'
    ],
    'Fixed Income': [
        'ZN=F', 'ZB=F', 'ZF=F', 'ZT=F', 'GE=F'
    ],
    'FX': [
        '6E=F', '6J=F', '6B=F', '6A=F', '6C=F',
        '6S=F', '6N=F', '6M=F', '6R=F'
    ]
}

# =============================================================================
# 环境
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
        self.returns_history = []
        self.positions_history = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 200
        self.last_action = 0.0
        self.dsr.reset()
        self.returns_history = []
        self.positions_history = []
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
        
        self.returns_history.append(strat_ret)
        self.positions_history.append(action)
        
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
# 指标计算 (完全对齐论文Table 2)
# =============================================================================

def calc_metrics_paper(returns):
    """计算论文Table 2的所有指标"""
    if len(returns) == 0:
        return {'E(R)': 0, 'Std(R)': 0, 'DD': 0, 'Sharpe': 0, 
                'Sortino': 0, 'MDD': 0, 'Calmar': 0}
    
    annual_factor = np.sqrt(252)
    
    # E(R) - Annualized expected return
    er = np.mean(returns) * 252
    
    # Std(R) - Annualized standard deviation
    std_r = np.std(returns) * annual_factor
    
    # Downside Deviation (DD) - only negative returns
    neg_ret = returns[returns < 0]
    dd = np.std(neg_ret) * annual_factor if len(neg_ret) > 0 else 0.001
    
    # Sharpe Ratio (assume risk-free rate = 0 for simplicity, as in paper)
    sharpe = er / std_r if std_r > 0 else 0
    
    # Sortino Ratio
    sortino = er / dd if dd > 0 else 0
    
    # Maximum Drawdown
    cum = np.cumprod(1 + returns)
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
        'MDD': -mdd,  # Negative to show drawdown
        'Calmar': calmar
    }

# =============================================================================
# 基线策略
# =============================================================================

def baseline_long(n): 
    return np.ones(n)

def baseline_sign(returns, window=252):
    """Sign(R) strategy from paper"""
    signals = np.zeros(len(returns))
    for i in range(window, len(returns)):
        cum_ret = np.sum(returns[i-window:i])
        signals[i] = np.sign(cum_ret)
    return signals

def baseline_macd(prices):
    """MACD strategy from paper Eq. 11"""
    p = pd.Series(prices)
    macd = p.ewm(span=12).mean() - p.ewm(span=26).mean()
    # Paper's MACD signal formula
    sig = macd / (0.89 * np.exp(-macd**2 / 4))
    return np.tanh(sig.values)

# =============================================================================
# 训练单个合约
# =============================================================================

def train_contract(ticker, quiet=False):
    """训练单个合约，返回所有策略结果和每日收益"""
    try:
        df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
    except:
        if not quiet:
            print(f"  ⚠️ {ticker} 数据不存在")
        return None, None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
    test = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
    
    if len(train) < 500 or len(test) < 200:
        if not quiet:
            print(f"  ⚠️ {ticker} 数据不足 (训练{len(train)}, 测试{len(test)})")
        return None, None
    
    train_p = train['Close'].values
    train_r = train['Returns'].values
    test_p = test['Close'].values
    test_r = test['Returns'].values
    
    results = []
    daily_returns = {}
    
    # ===== Long =====
    pos = baseline_long(len(test_r)-200)
    strat_ret = test_r[200:] * pos
    m = calc_metrics_paper(strat_ret)
    m['Ticker'] = ticker
    m['Strategy'] = 'Long'
    results.append(m)
    daily_returns['Long'] = strat_ret
    
    # ===== Sign(R) =====
    pos = baseline_sign(test_r)[200:]
    pos = pos[:len(test_r)-200]
    if len(pos) > 0:
        strat_ret = test_r[200:200+len(pos)] * pos
        m = calc_metrics_paper(strat_ret)
        m['Ticker'] = ticker
        m['Strategy'] = 'Sign(R)'
        results.append(m)
        daily_returns['Sign(R)'] = strat_ret
    
    # ===== MACD =====
    pos = baseline_macd(test_p)[200:]
    pos = pos[:len(test_r)-200]
    if len(pos) > 0:
        strat_ret = test_r[200:200+len(pos)] * pos
        m = calc_metrics_paper(strat_ret)
        m['Ticker'] = ticker
        m['Strategy'] = 'MACD'
        results.append(m)
        daily_returns['MACD'] = strat_ret
    
    # ===== DQN =====
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
        
        # Test and collect daily returns
        obs, _ = env_test.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_test.step(a)
        
        strat_ret = np.array(env_test.returns_history)
        m = calc_metrics_paper(strat_ret)
        m['Ticker'] = ticker
        m['Strategy'] = 'DQN'
        results.append(m)
        daily_returns['DQN'] = strat_ret
    except Exception as e:
        if not quiet:
            print(f"    ⚠️ {ticker} DQN失败: {e}")
    
    # ===== A2C =====
    try:
        env_train = SimpleEnv(train_p, train_r, use_dsr=True)
        env_test = SimpleEnv(test_p, test_r, use_dsr=False)
        
        model = A2C("MlpPolicy", env_train,
                   learning_rate=LEARNING_RATE,
                   gamma=GAMMA,
                   policy_kwargs=dict(net_arch=[64, 32]),
                   verbose=0, device=DEVICE)
        
        model.learn(TOTAL_TIMESTEPS, progress_bar=False)
        
        # Test and collect daily returns
        obs, _ = env_test.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_test.step(a)
        
        strat_ret = np.array(env_test.returns_history)
        m = calc_metrics_paper(strat_ret)
        m['Ticker'] = ticker
        m['Strategy'] = 'A2C'
        results.append(m)
        daily_returns['A2C'] = strat_ret
    except Exception as e:
        if not quiet:
            print(f"    ⚠️ {ticker} A2C失败: {e}")
    
    return results, daily_returns

# =============================================================================
# 训练所有合约
# =============================================================================

def train_all_contracts():
    """训练所有33个合约"""
    
    print("="*80)
    print("🤖 完整复现：33个合约")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"γ: {GAMMA}")
    print(f"Buffer: {BUFFER_SIZE}")
    print(f"Batch: DQN={BATCH_SIZE_DQN}, A2C={BATCH_SIZE_A2C}")
    print(f"网络: [64, 32]")
    print(f"步数: {TOTAL_TIMESTEPS:,}")
    print()
    
    all_results = []
    all_daily_returns = {}
    
    total_contracts = sum(len(v) for v in CONTRACTS_BY_CLASS.values())
    trained = 0
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        print(f"\n{'='*70}")
        print(f"📊 {asset_class} ({len(tickers)} 合约)")
        print('='*70)
        
        class_results = []
        
        for ticker in tickers:
            trained += 1
            print(f"  [{trained}/{total_contracts}] {ticker:<8} | ", end='', flush=True)
            
            results, daily_returns = train_contract(ticker, quiet=True)
            
            if results:
                all_results.extend(results)
                class_results.extend(results)
                all_daily_returns[ticker] = daily_returns
                
                # Print Sharpe for each strategy
                for r in results:
                    if r['Strategy'] in ['Long', 'MACD', 'DQN']:
                        print(f"{r['Strategy']}: {r['Sharpe']:>6.2f} | ", end='', flush=True)
                print("✅")
            else:
                print("❌")
        
        # 计算该资产类别的平均
        if class_results:
            class_df = pd.DataFrame(class_results)
            print(f"\n  📈 {asset_class} 平均结果:")
            for strategy in ['Long', 'Sign(R)', 'MACD', 'DQN', 'A2C']:
                strat_df = class_df[class_df['Strategy'] == strategy]
                if len(strat_df) > 0:
                    mean_sharpe = strat_df['Sharpe'].mean()
                    mean_ret = strat_df['E(R)'].mean()
                    print(f"    {strategy:<8} | Sharpe: {mean_sharpe:>7.3f} | Return: {mean_ret:>7.2%} | N={len(strat_df)}")
    
    # 保存结果
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results_all33_{timestamp}.csv'
    df.to_csv(filename, index=False)
    print(f"\n💾 结果已保存: {filename}")
    
    # 保存每日收益（用于生成Figure 1）
    import pickle
    with open(f'daily_returns_{timestamp}.pkl', 'wb') as f:
        pickle.dump(all_daily_returns, f)
    print(f"💾 每日收益已保存: daily_returns_{timestamp}.pkl")
    
    return df, all_daily_returns

# =============================================================================
# 生成论文对比
# =============================================================================

def create_paper_comparison(df, daily_returns):
    """生成完整的论文对比"""
    
    # ===== Table 2 对比 =====
    print("\n" + "="*80)
    print("📊 Table 2 对比 (Portfolio-level Volatility Targeting)")
    print("="*80)
    
    # 论文Table 2数据
    paper_table2 = {
        'Commodity': {
            'Long':   {'E(R)': -0.710, 'Sharpe': -0.726, 'Sortino': -1.177, 'MDD': 0.350},
            'Sign(R)': {'E(R)': 0.347, 'Sharpe': 0.354, 'Sortino': 0.606, 'MDD': 0.116},
            'MACD':   {'E(R)': -0.171, 'Sharpe': -0.175, 'Sortino': -0.293, 'MDD': 0.190},
            'DQN':    {'E(R)': 0.703, 'Sharpe': 0.723, 'Sortino': 1.275, 'MDD': 0.066},
            'A2C':    {'E(R)': 0.223, 'Sharpe': 0.234, 'Sortino': 0.399, 'MDD': 0.141},
        },
        'Equity Index': {
            'Long':   {'E(R)': 0.668, 'Sharpe': 0.688, 'Sortino': 1.102, 'MDD': 0.132},
            'Sign(R)': {'E(R)': 0.228, 'Sharpe': 0.236, 'Sortino': 0.374, 'MDD': 0.344},
            'MACD':   {'E(R)': 0.016, 'Sharpe': 0.017, 'Sortino': 0.027, 'MDD': 0.311},
            'DQN':    {'E(R)': 0.629, 'Sharpe': 0.648, 'Sortino': 1.038, 'MDD': 0.161},
            'A2C':    {'E(R)': 0.473, 'Sharpe': 0.510, 'Sortino': 0.798, 'MDD': 0.124},
        },
        'Fixed Income': {
            'Long':   {'E(R)': 0.680, 'Sharpe': 0.698, 'Sortino': 1.180, 'MDD': 0.061},
            'Sign(R)': {'E(R)': 0.214, 'Sharpe': 0.221, 'Sortino': 0.363, 'MDD': 0.080},
            'MACD':   {'E(R)': 0.219, 'Sharpe': 0.228, 'Sortino': 0.380, 'MDD': 0.065},
            'DQN':    {'E(R)': 0.908, 'Sharpe': 0.935, 'Sortino': 1.617, 'MDD': 0.062},
            'A2C':    {'E(R)': 0.699, 'Sharpe': 0.714, 'Sortino': 1.203, 'MDD': 0.067},
        },
        'FX': {
            'Long':   {'E(R)': -0.344, 'Sharpe': -0.353, 'Sortino': -0.590, 'MDD': 0.423},
            'Sign(R)': {'E(R)': -0.297, 'Sharpe': -0.306, 'Sortino': -0.502, 'MDD': 0.434},
            'MACD':   {'E(R)': 0.006, 'Sharpe': 0.007, 'Sortino': 0.011, 'MDD': 0.329},
            'DQN':    {'E(R)': 0.528, 'Sharpe': 0.546, 'Sortino': 0.955, 'MDD': 0.183},
            'A2C':    {'E(R)': 0.316, 'Sharpe': 0.328, 'Sortino': 0.561, 'MDD': 0.165},
        },
        'All': {
            'Long':   {'E(R)': 0.055, 'Sharpe': 0.058, 'Sortino': 0.092, 'MDD': 0.071},
            'Sign(R)': {'E(R)': 0.429, 'Sharpe': 0.441, 'Sortino': 0.737, 'MDD': 0.038},
            'MACD':   {'E(R)': 0.089, 'Sharpe': 0.091, 'Sortino': 0.153, 'MDD': 0.008},
            'DQN':    {'E(R)': 1.258, 'Sharpe': 1.288, 'Sortino': 2.220, 'MDD': 0.002},
            'A2C':    {'E(R)': 1.024, 'Sharpe': 1.050, 'Sortino': 1.785, 'MDD': 0.007},
        }
    }
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        class_df = df[df['Ticker'].isin(tickers)]
        
        if len(class_df) == 0:
            continue
        
        print(f"\n【{asset_class}】")
        print(f"{'策略':<10} | {'论文E(R)':>8} | {'我们E(R)':>8} | {'论文Sharpe':>10} | {'我们Sharpe':>10} | {'差距':>8}")
        print("-" * 70)
        
        for strategy in ['Long', 'Sign(R)', 'MACD', 'DQN', 'A2C']:
            strat_df = class_df[class_df['Strategy'] == strategy]
            
            if len(strat_df) > 0:
                our_er = strat_df['E(R)'].mean()
                our_sharpe = strat_df['Sharpe'].mean()
                
                # 获取论文数据
                paper_data = paper_table2.get(asset_class, {}).get(strategy, {})
                paper_er = paper_data.get('E(R)', 0)
                paper_sharpe = paper_data.get('Sharpe', 0)
                
                diff = our_sharpe - paper_sharpe
                
                print(f"{strategy:<10} | {paper_er:>8.3f} | {our_er:>8.3f} | {paper_sharpe:>10.3f} | {our_sharpe:>10.3f} | {diff:>+8.3f}")
    
    # ===== Figure 1 对比 =====
    print("\n" + "="*80)
    print("📊 Figure 1 对比 - 累积收益曲线")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Paper vs Our Results: Cumulative Returns', fontsize=14, fontweight='bold')
    
    plot_idx = 0
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        if plot_idx >= 5:
            break
        
        ax = axes[plot_idx // 3, plot_idx % 3]
        
        # 为每个策略画累积收益
        for strategy in ['Long', 'MACD', 'DQN', 'A2C']:
            # 合并该类别所有合约的收益
            all_returns = []
            for ticker in tickers:
                if ticker in daily_returns and strategy in daily_returns[ticker]:
                    all_returns.append(daily_returns[ticker][strategy])
            
            if all_returns:
                # 取平均
                min_len = min(len(r) for r in all_returns)
                avg_returns = np.mean([r[:min_len] for r in all_returns], axis=0)
                
                # 计算累积收益
                cum_ret = np.cumprod(1 + avg_returns)
                
                ax.plot(cum_ret, label=strategy, linewidth=2)
        
        ax.set_title(asset_class, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Return')
        
        plot_idx += 1
    
    # 最后一个图：All Portfolio
    ax = axes[1, 2]
    for strategy in ['Long', 'MACD', 'DQN', 'A2C']:
        all_returns = []
        for asset_class, tickers in CONTRACTS_BY_CLASS.items():
            for ticker in tickers:
                if ticker in daily_returns and strategy in daily_returns[ticker]:
                    all_returns.append(daily_returns[ticker][strategy])
        
        if all_returns:
            min_len = min(len(r) for r in all_returns)
            avg_returns = np.mean([r[:min_len] for r in all_returns], axis=0)
            cum_ret = np.cumprod(1 + avg_returns)
            ax.plot(cum_ret, label=strategy, linewidth=2)
    
    ax.set_title('All Contracts', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative Return')
    
    plt.tight_layout()
    plt.savefig('figure1_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Figure 1 已保存: figure1_comparison.png")
    
    # ===== Table 3 对比 (Raw Signal) =====
    print("\n" + "="*80)
    print("📊 Table 3 对比 (Raw Signal - 无Volatility Scaling)")
    print("="*80)
    print("⚠️ 我们实现了Volatility Scaling，所以结果对应Table 2")
    print("   Table 3是未scaled版本，论文中表现更差")

# =============================================================================
# 主函数
# =============================================================================

def main():
    # 训练所有合约
    df, daily_returns = train_all_contracts()
    
    # 生成论文对比
    create_paper_comparison(df, daily_returns)
    
    # 保存汇总
    print("\n" + "="*80)
    print("📊 总体汇总")
    print("="*80)
    
    summary = df.groupby('Strategy').agg({
        'E(R)': 'mean',
        'Std(R)': 'mean',
        'Sharpe': ['mean', 'std'],
        'Sortino': 'mean',
        'MDD': 'mean',
        'Calmar': 'mean'
    }).round(3)
    
    print(summary)
    
    # 按资产类别汇总
    print("\n" + "="*80)
    print("📊 按资产类别汇总 (Sharpe Ratio)")
    print("="*80)
    
    for asset_class, tickers in CONTRACTS_BY_CLASS.items():
        class_df = df[df['Ticker'].isin(tickers)]
        if len(class_df) > 0:
            print(f"\n【{asset_class}】")
            for strategy in ['Long', 'Sign(R)', 'MACD', 'DQN', 'A2C']:
                strat_df = class_df[class_df['Strategy'] == strategy]
                if len(strat_df) > 0:
                    mean_sharpe = strat_df['Sharpe'].mean()
                    std_sharpe = strat_df['Sharpe'].std()
                    print(f"  {strategy:<8} | Sharpe: {mean_sharpe:>7.3f} ± {std_sharpe:.3f} | N={len(strat_df)}")

if __name__ == '__main__':
    main()
