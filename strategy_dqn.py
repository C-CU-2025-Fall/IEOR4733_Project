#!/usr/bin/env python3
"""
strategy_dqn.py — DQN strategy for IEOR4733 paper reproduction

Architecture: LSTM [64, 32] + Leaky-ReLU + Double DQN + Fixed Q-targets
Paper Table 1 hyperparameters
Train per-contract on full data, produce continuous position signal

Usage:
    python strategy_dqn.py train --all                    # Train all 50 contracts
    python strategy_dqn.py train --asset "Fixed Income"   # One asset class
    python strategy_dqn.py train --ticker ES              # Single contract
    python strategy_dqn.py status                         # Show trained models
"""
import os, sys, time, argparse, pickle, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import ASSET_CLASSES, EWMA_SPAN, BP
from data_loader import load_clc_full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.path.join(ROOT, "models", "dqn")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Paper Table 1 Hyperparameters ────────────────────────────────
LR = 0.0001
GAMMA = 0.3
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TAU = 1000
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY_STEPS = 50000  # linear decay over this many env steps
EPISODES = 200
WARMUP = 252
FEATURE_DIM = 8
SEQ_LEN = 60
ACTION_DIM = 3
MAX_STEPS_PER_EP = 1500


# ─── Feature Engineering (paper state space) ──────────────────────
def _compute_features_chunk(prices, returns, sigma):
    """Compute all 8 features for a price/return/sigma chunk."""
    n = len(prices)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    # 1. normalized price
    p_mean, p_std = prices.mean(), prices.std() + 1e-10
    feats[:, 0] = (prices - p_mean) / p_std

    # 2-5. multi-horizon returns (vol-adjusted)
    for idx, h in enumerate([21, 42, 63, 252]):
        col = np.zeros(n)
        for i in range(h, n):
            col[i] = (prices[i] - prices[i - h]) / (sigma[i] * np.sqrt(h) + 1e-10)
        feats[:, idx + 1] = col

    # 6. Multi-scale MACD (paper: 3 time-scales averaged)
    # Scale 1: (8, 24)
    ema8 = pd.Series(prices).ewm(span=8, adjust=False).mean().values
    ema24 = pd.Series(prices).ewm(span=24, adjust=False).mean().values
    std63 = pd.Series(prices).rolling(63, min_periods=1).std().values + 1e-10
    q1 = (ema8 - ema24) / std63
    std252_q1 = pd.Series(q1).rolling(252, min_periods=1).std().values + 1e-10
    macd1 = q1 / std252_q1
    
    # Scale 2: (16, 48)
    ema16 = pd.Series(prices).ewm(span=16, adjust=False).mean().values
    ema48 = pd.Series(prices).ewm(span=48, adjust=False).mean().values
    q2 = (ema16 - ema48) / std63
    std252_q2 = pd.Series(q2).rolling(252, min_periods=1).std().values + 1e-10
    macd2 = q2 / std252_q2
    
    # Scale 3: (32, 96)
    ema32 = pd.Series(prices).ewm(span=32, adjust=False).mean().values
    ema96 = pd.Series(prices).ewm(span=96, adjust=False).mean().values
    q3 = (ema32 - ema96) / std63
    std252_q3 = pd.Series(q3).rolling(252, min_periods=1).std().values + 1e-10
    macd3 = q3 / std252_q3
    
    # Average across scales (paper Eq 12)
    feats[:, 5] = (macd1 + macd2 + macd3) / 3

    # 7. RSI normalized
    delta = np.diff(prices, prepend=prices[0])
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(30, min_periods=1).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(30, min_periods=1).mean().values + 1e-10
    feats[:, 6] = (50 - 50 / (1 + gain / loss)) / 50

    # 8. vol ratio
    feats[:, 7] = sigma / (sigma.mean() + 1e-10)

    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def build_all_features(prices, returns, sigma):
    """Precompute features for entire series. Shape: (T, FEATURE_DIM)."""
    return _compute_features_chunk(prices, returns, sigma)


def get_feature_window(features, idx, seq_len=SEQ_LEN):
    """Get (seq_len, FEATURE_DIM) window ending at idx."""
    if idx < seq_len:
        # pad with zeros at the beginning
        pad = np.zeros((seq_len - idx, FEATURE_DIM), dtype=np.float32)
        return np.vstack([pad, features[:idx]])
    return features[idx - seq_len:idx]


# ─── Environment ──────────────────────────────────────────────────
class ContractEnv:
    """Single-contract trading environment."""

    def __init__(self, prices, returns, sigma, features, sigma_tgt=0.063):
        self.prices = prices
        self.returns = returns
        self.sigma = sigma
        self.features = features  # precomputed
        self.sigma_tgt = sigma_tgt
        self.max_idx = len(prices) - 1
        self.idx = WARMUP
        self.last_action = 0.0
        self.last_sig = sigma[WARMUP - 1] if WARMUP >= 1 else sigma[0]

    def reset(self):
        self.idx = WARMUP
        self.last_action = 0.0
        self.last_sig = self.sigma[WARMUP - 1] if WARMUP >= 1 else self.sigma[0]
        return get_feature_window(self.features, self.idx)

    def step(self, action_id):
        action = action_id - 1.0  # {0,1,2} → {-1,0,+1}
        self.idx += 1
        if self.idx >= self.max_idx:
            return get_feature_window(self.features, min(self.idx, self.max_idx)), 0.0, True

        # Equation 4: use σ_{t-1} for current position, σ_{t-2} for previous position
        sig_t_1 = self.sigma[self.idx - 1]  # σ_{t-1}
        sig_t_2 = self.last_sig              # σ_{t-2}
        
        pos_current = action * (self.sigma_tgt / sig_t_1) if sig_t_1 > 0 else 0.0
        pos_prev = self.last_action * (self.sigma_tgt / sig_t_2) if sig_t_2 > 0 else 0.0
        
        gross = pos_current * self.returns[self.idx]
        tc = BP * self.prices[self.idx - 1] * abs(pos_current - pos_prev)
        reward = gross - tc

        self.last_action = action
        self.last_sig = sig_t_1
        done = self.idx >= self.max_idx - 1
        return get_feature_window(self.features, self.idx), reward, done


# ─── LSTM Network ─────────────────────────────────────────────────
class DQNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(FEATURE_DIM, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, ACTION_DIM)
        for name, p in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(p, gain=nn.init.calculate_gain('tanh'))
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)

    def forward(self, x):
        o, _ = self.lstm1(x);  o = F.leaky_relu(o, 0.01)
        o, _ = self.lstm2(o);  o = F.leaky_relu(o, 0.01)
        return self.fc(o[:, -1, :])


# ─── DQN Agent ────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self):
        self.q_net = DQNLSTM().to(DEVICE)
        self.target = DQNLSTM().to(DEVICE)
        self.target.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 50, 0.9)
        
        # Pre-allocate GPU tensors for batch training
        self.gpu_s = torch.zeros(BATCH_SIZE, SEQ_LEN, FEATURE_DIM, dtype=torch.float32, device=DEVICE)
        self.gpu_a = torch.zeros(BATCH_SIZE, dtype=torch.int64, device=DEVICE)
        self.gpu_r = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
        self.gpu_s2 = torch.zeros(BATCH_SIZE, SEQ_LEN, FEATURE_DIM, dtype=torch.float32, device=DEVICE)
        self.gpu_d = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
        
        # CPU replay buffer
        self.buffer_s = np.zeros((MEMORY_SIZE, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        self.buffer_a = np.zeros(MEMORY_SIZE, dtype=np.int64)
        self.buffer_r = np.zeros(MEMORY_SIZE, dtype=np.float32)
        self.buffer_s2 = np.zeros((MEMORY_SIZE, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        self.buffer_d = np.zeros(MEMORY_SIZE, dtype=np.float32)
        self.buf_pos = 0
        self.buf_len = 0
        self.train_steps = 0

    def push(self, s, a, r, s2, d):
        i = self.buf_pos % MEMORY_SIZE
        self.buffer_s[i] = s
        self.buffer_a[i] = a
        self.buffer_r[i] = r
        self.buffer_s2[i] = s2
        self.buffer_d[i] = d
        self.buf_pos += 1
        self.buf_len = min(self.buf_len + 1, MEMORY_SIZE)

    def sample(self):
        """Sample batch and copy to pre-allocated GPU tensors."""
        idx = np.random.randint(0, self.buf_len, BATCH_SIZE)
        # Copy directly to pre-allocated GPU tensors (avoids intermediate CPU tensor creation)
        self.gpu_s.copy_(torch.from_numpy(self.buffer_s[idx]).to(DEVICE, non_blocking=True))
        self.gpu_a.copy_(torch.from_numpy(self.buffer_a[idx]).to(DEVICE, non_blocking=True))
        self.gpu_r.copy_(torch.from_numpy(self.buffer_r[idx]).to(DEVICE, non_blocking=True))
        self.gpu_s2.copy_(torch.from_numpy(self.buffer_s2[idx]).to(DEVICE, non_blocking=True))
        self.gpu_d.copy_(torch.from_numpy(self.buffer_d[idx]).to(DEVICE, non_blocking=True))
        return self.gpu_s, self.gpu_a, self.gpu_r, self.gpu_s2, self.gpu_d

    def act(self, state, eps):
        if np.random.random() < eps:
            return np.random.randint(ACTION_DIM)
        with torch.no_grad():
            # Reuse a small buffer for single-state inference
            s = torch.from_numpy(state).unsqueeze(0).to(DEVICE, non_blocking=True)
            return self.q_net(s).argmax().item()

    def learn(self):
        if self.buf_len < BATCH_SIZE:
            return 0.0
        s, a, r, s2, d = self.sample()
        with torch.no_grad():
            next_a = self.q_net(s2).argmax(1)
            next_q = self.target(s2).gather(1, next_a.unsqueeze(1)).squeeze()
            tgt = r + (1 - d) * GAMMA * next_q
        cur = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        loss = F.mse_loss(cur, tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.opt.step()
        self.scheduler.step()
        self.train_steps += 1
        if self.train_steps % TAU == 0:
            self.target.load_state_dict(self.q_net.state_dict())
        return loss.item()

    def predict(self, state):
        """Predict position from state. Maps discrete action to continuous position.
        
        Paper-aligned: discrete action {-1, 0, +1} mapped to position.
        Action 0 → position -1.0 (short)
        Action 1 → position  0.0 (hold)
        Action 2 → position +1.0 (long)
        """
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).squeeze()
            action = q.argmax().item()  # Discrete action {0, 1, 2}
            # Map to position {-1, 0, +1}
            return float(action - 1)  # 0→-1, 1→0, 2→+1

    def save(self, path):
        torch.save({'q': self.q_net.state_dict(), 't': self.target.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        # Handle both old and new checkpoint formats
        if 'q' in ck:
            self.q_net.load_state_dict(ck['q'])
            self.target.load_state_dict(ck['t'])
        else:
            self.q_net.load_state_dict(ck.get('q_net', ck))
            self.target.load_state_dict(ck.get('target_net', ck))


# ─── Training ─────────────────────────────────────────────────────
DQN_DATA_DIR = os.path.join(ROOT, 'data', 'dqn_train')

def prepare_data(ticker, source='RAD', use_preprocessed=True):
    """Load contract data for training.
    
    If use_preprocessed=True, loads from data/dqn_train/{ticker}.npz
    (prepared by scripts/prepare_dqn_data.py using latest baseline config).
    """
    if use_preprocessed:
        path = os.path.join(DQN_DATA_DIR, f"{ticker}.npz")
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return {
                'prices': data['prices'],
                'returns': data['returns'],
                'sigma': data['sigma'],
                'features': data['features'],
                'dates': data['dates'],
                'source': str(data['source']),
            }
    
    # Fallback to live loading
    df = load_clc_full(ticker, source=source)
    if df is None or len(df) < 500:
        return None
    p = df['Close'].values.astype(float)
    r = np.zeros(len(p));  r[1:] = p[1:] - p[:-1]
    sig = pd.Series(r).ewm(span=EWMA_SPAN, adjust=False).std().values
    feats = build_all_features(p, r, sig)
    return {'prices': p, 'returns': r, 'sigma': sig, 'features': feats}


def train_contract(ticker, source='RAD', episodes=EPISODES, early_stop_patience=5, verbose=True):
    """Train DQN for a single contract with early stopping."""
    t0 = time.time()
    data = prepare_data(ticker, source)
    if data is None:
        print(f"  {ticker}: skip");  return False

    env = ContractEnv(data['prices'], data['returns'], data['sigma'], data['features'])
    agent = DQNAgent()
    rewards_log = []
    
    # Early stopping: check every episode, stop after `patience` consecutive non-improvements
    best_avg = -np.inf
    patience_counter = 0
    best_state = None
    best_ep = 0
    
    # Progress reporting every 10 episodes
    report_interval = max(1, episodes // 10)

    for ep in range(episodes):
        state = env.reset()
        total_r, done, steps = 0.0, False, 0
        while not done and steps < MAX_STEPS_PER_EP:
            eps = max(EPS_END, EPS_START - (ep * MAX_STEPS_PER_EP + steps) / EPS_DECAY_STEPS)
            a = agent.act(state, eps)
            ns, r, done = env.step(a)
            agent.push(state, a, r, ns, float(done))
            agent.learn()
            state = ns
            total_r += r
            steps += 1
        rewards_log.append(total_r)
        
        # Early stopping: check every episode
        episode_avg = total_r
        if episode_avg > best_avg:
            best_avg = episode_avg
            best_state = agent.q_net.state_dict().copy()
            best_ep = ep + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"  {ticker}: Early stop @ ep{ep+1} (best={best_avg:+.2f} @ ep{best_ep})")
                break
        
        if verbose and (ep + 1) % report_interval == 0:
            avg_r = np.mean(rewards_log[-report_interval:])
            elapsed = time.time() - t0
            eta = (episodes - ep - 1) * (elapsed / (ep + 1))
            print(f"  {ticker} ep {ep+1}/{episodes} avg_r={avg_r:+.4f} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    path = os.path.join(MODEL_DIR, f"{ticker}.pt")
    
    # Restore best model if early stopped
    if best_state is not None and patience_counter >= early_stop_patience:
        agent.q_net.load_state_dict(best_state)
        agent.target.load_state_dict(best_state)
        if verbose:
            print(f"  {ticker}: Restored best model (ep={best_ep})")
    
    agent.save(path)
    elapsed = time.time() - t0
    final_avg = np.mean(rewards_log[-report_interval:]) if len(rewards_log) >= report_interval else np.mean(rewards_log[-10:])
    if verbose:
        print(f"  {ticker}: ✅ {elapsed:.1f}s, final_avg_r={final_avg:+.4f}, stopped@ep{len(rewards_log)}")
    
    del agent, env, data
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return True


def train_asset_class(asset_name, episodes=EPISODES, early_stop_patience=5):
    """Train all contracts in one asset class with early stopping."""
    tickers = ASSET_CLASSES.get(asset_name, [])
    if not tickers:
        print(f"Unknown asset class: {asset_name}")
        return 0, 0
    
    print(f"\n{'='*60}")
    print(f"DQN Training: {asset_name} ({len(tickers)} contracts × {episodes} eps)")
    print(f"Early Stop: patience={early_stop_patience}")
    print(f"{'='*60}")
    
    t0 = time.time()
    ok, fail = 0, 0
    for i, tk in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] {tk}:")
        if train_contract(tk, episodes=episodes, early_stop_patience=early_stop_patience):
            ok += 1
        else:
            fail += 1
    
    elapsed = time.time() - t0
    per_contract = elapsed / len(tickers) if len(tickers) > 0 else 0
    print(f"\n{'='*60}")
    print(f"{asset_name}: {ok}/{len(tickers)} trained, {fail} failed")
    print(f"Time: {elapsed:.0f}s ({per_contract:.0f}s/contract)")
    print(f"{'='*60}")
    return ok, fail


def train_all(tickers=None, source='RAD', episodes=EPISODES):
    if tickers is None:
        tickers = []
        for tks in ASSET_CLASSES.values():
            tickers.extend(tks)

    print(f"DQN Training: {len(tickers)} contracts × {episodes} eps, device={DEVICE}")
    t0 = time.time()
    ok, fail = 0, 0
    for i, tk in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] {tk}:")
        if train_contract(tk, source=source, episodes=episodes):
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done: {ok}/{len(tickers)} trained, {fail} failed, {elapsed:.0f}s total")
    print(f"Models: {MODEL_DIR}/")
    print(f"{'='*60}")


# ─── Inference (for baseline_run integration) ─────────────────────
def strategy_dqn_positions(prices, ticker, source='RAD'):
    """Produce continuous position signal [-1, 1] for full price series."""
    path = os.path.join(MODEL_DIR, f"{ticker}.pt")
    if not os.path.exists(path):
        return np.zeros(len(prices))

    data = prepare_data(ticker, source)
    if data is None:
        return np.zeros(len(prices))

    agent = DQNAgent()
    agent.load(path)
    agent.q_net.eval()

    positions = np.zeros(len(prices))
    for t in range(WARMUP, len(prices)):
        state = get_feature_window(data['features'], t)
        positions[t] = agent.predict(state)

    del agent, data
    gc.collect()
    return positions


# ─── Status ───────────────────────────────────────────────────────
def status():
    models = set(f.replace('.pt', '') for f in os.listdir(MODEL_DIR) if f.endswith('.pt')) if os.path.exists(MODEL_DIR) else set()
    all_tk = [t for tks in ASSET_CLASSES.values() for t in tks]
    print(f"DQN: {len(models)}/{len(all_tk)} trained")
    for ac, tks in ASSET_CLASSES.items():
        done = sum(1 for t in tks if t in models)
        print(f"  {ac}: {done}/{len(tks)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'status'])
    p.add_argument('--all', action='store_true')
    p.add_argument('--asset', default=None)
    p.add_argument('--ticker', default=None)
    p.add_argument('--episodes', type=int, default=EPISODES)
    p.add_argument('--early-stop', type=int, default=5, help='Early stopping patience (default: 5)')
    p.add_argument('--source', default='RAD')
    a = p.parse_args()

    if a.cmd == 'status':
        status()
    elif a.cmd == 'train':
        if a.ticker:
            train_contract(a.ticker, source=a.source, episodes=a.episodes, early_stop_patience=a.early_stop)
        elif a.asset:
            train_asset_class(a.asset, episodes=a.episodes, early_stop_patience=a.early_stop)
        else:
            train_all(source=a.source, episodes=a.episodes)
