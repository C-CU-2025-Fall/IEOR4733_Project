#!/usr/bin/env python3
"""
论文超参数对齐版
- MLP架构 (stable-baselines3兼容)
- γ = 0.3 (论文超参)
- Buffer = 5000 (论文超参)
- Batch = 64/128 (论文超参)
- 40个合约
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from paper_components import (
    DifferentialSharpeRatio,
    MultiTimeScaleState,
    VolatilityScaler
)

import torch
print("=" * 80)
print("🤖 Deep RL Trading - 论文超参数对齐版")
print("=" * 80)
print(f"PyTorch: {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

from stable_baselines3 import DQN, A2C
import gymnasium as gym
from gymnasium import spaces

# =============================================================================
# 论文超参数 (Table 1)
# =============================================================================

DATA_DIR = 'data/futures_processed'
TRANSACTION_COST = 0.001

# 论文超参数
GAMMA = 0.3  # ⭐ 论文用0.3，不是0.99!
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

# 40个合约
ALL_TICKERS = [
    'CL=F', 'GC=F', 'SI=F', 'HG=F', 'NG=F', 'ZC=F', 'ZS=F', 'ZW=F', 
    'KC=F', 'CC=F', 'SB=F', 'CT=F', 'LC=F', 'LBS=F', 'OJ=F',
    'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'ZN=F', 'ZB=F', 'ZF=F', 'ZT=F',
    '6E=F', '6J=F', '6B=F', '6A=F', '6C=F'
]

# =============================================================================
# 环境 (简化版)
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
    sr = returns[:len(positions)] * positions
    sr = sr[np.isfinite(sr)]
    if len(sr) == 0:
        return {'Sharpe': 0, 'Return': 0, 'MDD': 0}
    
    er = np.mean(sr) * 252
    std = np.std(sr) * np.sqrt(252)
    sharpe = (er - 0.02) / std if std > 0 else 0
    
    cum = np.cumprod(1 + sr)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    mdd = np.max(dd) if len(dd) > 0 else 0
    
    return {'Sharpe': sharpe, 'Return': er, 'MDD': -mdd}

# =============================================================================
# 基线
# =============================================================================

def baseline_long(n): return np.ones(n)

def baseline_macd(prices):
    p = pd.Series(prices)
    macd = p.ewm(12).mean() - p.ewm(26).mean()
    sig = macd / (0.89 * np.exp(-macd**2 / 4))
    return np.tanh(sig.values)

# =============================================================================
# 训练单个合约
# =============================================================================

def train_one(ticker):
    try:
        df = pd.read_csv(f'{DATA_DIR}/{ticker}.csv')
    except:
        return None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
    test = df[(df['Date'] >= TEST_START) & (df['Date'] <= TEST_END)]
    
    if len(train) < 500 or len(test) < 100:
        return None
    
    train_p = train['Close'].values
    train_r = train['Returns'].values
    test_p = test['Close'].values
    test_r = test['Returns'].values
    
    print(f"  📊 {ticker}: 训练{len(train)}天, 测试{len(test)}天", flush=True)
    
    results = []
    
    # 基线
    pos = baseline_long(len(test_r)-200)
    m = calc_metrics(test_r[200:], pos)
    results.append({'Ticker': ticker, 'Strategy': 'Long', **m})
    print(f"    Long  | Sharpe: {m['Sharpe']:>6.3f}", flush=True)
    
    pos = baseline_macd(test_p)[200:200+len(test_r)-200]
    m = calc_metrics(test_r[200:200+len(pos)], pos)
    results.append({'Ticker': ticker, 'Strategy': 'MACD', **m})
    print(f"    MACD  | Sharpe: {m['Sharpe']:>6.3f}", flush=True)
    
    # DQN (γ=0.3)
    try:
        env_train = DiscreteEnv(train_p, train_r, use_dsr=True)
        env_test = DiscreteEnv(test_p, test_r, use_dsr=False)
        
        model = DQN("MlpPolicy", env_train,
                   learning_rate=LEARNING_RATE,
                   buffer_size=BUFFER_SIZE,
                   learning_starts=1000,
                   batch_size=BATCH_SIZE_DQN,
                   gamma=GAMMA,  # 0.3
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
        results.append({'Ticker': ticker, 'Strategy': 'DQN', **m})
        print(f"    DQN   | Sharpe: {m['Sharpe']:>6.3f} ✅", flush=True)
    except Exception as e:
        print(f"    DQN   | ⚠️ {e}", flush=True)
    
    # A2C (γ=0.3)
    try:
        env_train = SimpleEnv(train_p, train_r, use_dsr=True)
        env_test = SimpleEnv(test_p, test_r, use_dsr=False)
        
        model = A2C("MlpPolicy", env_train,
                   learning_rate=LEARNING_RATE,
                   gamma=GAMMA,  # 0.3
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
        results.append({'Ticker': ticker, 'Strategy': 'A2C', **m})
        print(f"    A2C   | Sharpe: {m['Sharpe']:>6.3f} ✅", flush=True)
    except Exception as e:
        print(f"    A2C   | ⚠️ {e}", flush=True)
    
    return results

# =============================================================================
# 主函数
# =============================================================================

def main():
    print(f"\n📋 配置:")
    print(f"  γ = {GAMMA} (论文超参)")
    print(f"  Buffer = {BUFFER_SIZE}")
    print(f"  Batch = DQN {BATCH_SIZE_DQN}, A2C {BATCH_SIZE_A2C}")
    print(f"  网络 = [64, 32]")
    print(f"  步数 = {TOTAL_TIMESTEPS:,}")
    
    # 检查可用合约
    available = [t for t in ALL_TICKERS if os.path.exists(f'{DATA_DIR}/{t}.csv')]
    print(f"\n可用合约: {len(available)}/{len(ALL_TICKERS)}")
    
    # 测试模式
    test_mode = True
    if test_mode:
        test_tickers = ['ES=F', 'CL=F', 'GC=F']
        available = [t for t in test_tickers if t in available]
        print(f"🧪 测试模式: {len(available)} 个合约\n")
    
    all_results = []
    for idx, ticker in enumerate(available):
        print(f"\n[{idx+1}/{len(available)}] {ticker}")
        print('='*50)
        r = train_one(ticker)
        if r:
            all_results.extend(r)
    
    # 保存
    df = pd.DataFrame(all_results)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = f'results_gamma03_{ts}.csv'
    df.to_csv(fn, index=False)
    print(f"\n💾 保存: {fn}")
    
    # 汇总
    print("\n" + "="*60)
    print("📊 汇总 (γ=0.3)")
    print("="*60)
    for s in ['Long', 'MACD', 'DQN', 'A2C']:
        d = df[df['Strategy'] == s]
        if len(d) > 0:
            print(f"{s:<8} | Sharpe: {d['Sharpe'].mean():>6.3f} ± {d['Sharpe'].std():.3f}")

if __name__ == '__main__':
    main()
