#!/usr/bin/env python3
"""
Pilot Test for Deep Reinforcement Learning Trading
CPU-based approaches that don't require GPU

Based on: "Deep Reinforcement Learning for Trading" (Zhang, Zohren, Roberts, 2019)

This pilot tests:
1. Data loading and preprocessing
2. Baseline strategies (Long, Sign, MACD)
3. Simple DQN with small network (CPU-friendly)
4. Performance metrics calculation
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = 'data/futures'
TRAIN_START = '2011-01-01'
TRAIN_END = '2017-06-30'
TEST_START = '2017-07-01'
TEST_END = '2019-12-31'

# Paper parameters
TRANSACTION_COST = 0.001  # 10 bps per trade
LOOKBACK = 50  # Days of history for state

# =============================================================================
# DATA LOADING
# =============================================================================

def load_futures_data():
    """Load all downloaded futures data."""
    futures_data = {}
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            ticker = filename.replace('.csv', '')
            filepath = os.path.join(DATA_DIR, filename)
            
            # yfinance saves with multi-level header, skip first 3 rows
            # Row 1: Price,Close,High,Low,Open,Volume
            # Row 2: Ticker,ES=F,ES=F,...
            # Row 3: Date,,,,
            # Row 4+: actual data
            df = pd.read_csv(filepath, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df = df.set_index('Date')
            df = df.dropna()  # Remove any NaN rows
            
            # Calculate daily returns
            df['Returns'] = df['Close'].pct_change()
            
            futures_data[ticker] = df
            
    print(f"✅ Loaded {len(futures_data)} futures contracts")
    return futures_data

def get_train_test_split(df):
    """Split data into train and test sets."""
    train = df[(df.index >= TRAIN_START) & (df.index < TRAIN_END)].copy()
    test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
    return train, test

# =============================================================================
# BASELINE STRATEGIES (No ML, CPU-friendly)
# =============================================================================

def strategy_long(returns):
    """Simple long position - buy and hold."""
    return np.ones(len(returns))

def strategy_sign(returns, lookback=252):
    """Sign of past returns (momentum)."""
    signals = np.zeros(len(returns))
    for i in range(lookback, len(returns)):
        past_return = np.sum(returns[i-lookback:i])
        signals[i] = np.sign(past_return)
    return signals

def strategy_macd(prices, fast=12, slow=26, signal=9):
    """MACD crossover strategy."""
    # Calculate MACD
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    
    # Generate signals
    signals = np.where(macd > signal_line, 1, -1)
    return signals

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_metrics(returns, positions):
    """Calculate performance metrics per paper's Table 2."""
    # Portfolio returns
    portfolio_returns = returns * positions
    
    # Shift for realistic execution (signal at t, trade at t+1)
    portfolio_returns = np.roll(portfolio_returns, -1)[:-1]
    
    # Annualized metrics (252 trading days)
    annual_return = np.mean(portfolio_returns) * 252
    annual_std = np.std(portfolio_returns) * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = annual_return / annual_std if annual_std > 0 else 0
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1
    sortino = annual_return / downside_std if downside_std > 0 else 0
    
    return {
        'E(R)': annual_return,
        'Std(R)': annual_std,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': max_drawdown,
        'Calmar': calmar,
        'Avg Position': np.mean(np.abs(positions)),
        'Turnover': np.mean(np.abs(np.diff(positions)))
    }

# =============================================================================
# SIMPLE DQN (CPU-friendly, small network)
# =============================================================================

class SimpleDQN:
    """
    Simplified DQN for CPU testing.
    Uses small network and discrete actions.
    """
    
    def __init__(self, state_size, n_actions=3, hidden_size=32):
        self.state_size = state_size
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        
        # Simple feedforward network (numpy only, no PyTorch/TensorFlow)
        np.random.seed(42)
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, n_actions) * 0.1
        self.b2 = np.zeros(n_actions)
        
        # Experience replay
        self.memory = []
        self.max_memory = 10000
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, state):
        """Forward pass through network."""
        h = self.relu(np.dot(state, self.W1) + self.b1)
        q_values = np.dot(h, self.W2) + self.b2
        return q_values
    
    def act(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
    
    def train(self, batch_size=32):
        """Train on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            
            # Calculate target
            current_q = self.forward(state)
            if done:
                target = reward
            else:
                next_q = self.forward(next_state)
                target = reward + self.gamma * np.max(next_q)
            
            # Simple gradient update
            error = target - current_q[action]
            h = self.relu(np.dot(state, self.W1) + self.b1)
            
            # Update weights
            self.W2[:, action] += self.learning_rate * error * h
            self.b2[action] += self.learning_rate * error
            
            # Backprop to hidden layer
            d_hidden = self.learning_rate * error * self.W2[:, action]
            d_hidden[h <= 0] = 0  # ReLU gradient
            self.W1 += np.outer(state, d_hidden)
            self.b1 += d_hidden
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

class FuturesEnv:
    """Simple trading environment for single asset."""
    
    def __init__(self, returns, lookback=50):
        self.returns = returns
        self.lookback = lookback
        self.reset()
    
    def reset(self):
        self.t = self.lookback
        self.position = 0  # -1, 0, 1
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """State: normalized returns over lookback period."""
        state = self.returns[self.t-self.lookback:self.t]
        # Normalize
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        return state
    
    def step(self, action):
        """
        Action: 0=short, 1=neutral, 2=long
        """
        # Map action to position
        new_position = action - 1  # -1, 0, 1
        
        # Calculate transaction cost
        trade_cost = abs(new_position - self.position) * TRANSACTION_COST
        
        # Get return
        ret = self.returns[self.t]
        
        # Calculate reward (return - transaction cost)
        reward = new_position * ret - trade_cost
        
        # Update
        self.position = new_position
        self.t += 1
        
        # Check if done
        if self.t >= len(self.returns) - 1:
            self.done = True
        
        return self._get_state(), reward, self.done

# =============================================================================
# MAIN PILOT TEST
# =============================================================================

def run_pilot_test():
    """Run pilot test on all available futures."""
    
    print("="*70)
    print("🧪 PILOT TEST: Deep Reinforcement Learning for Trading")
    print("="*70)
    print(f"Training: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing: {TEST_START} to {TEST_END}")
    print("="*70)
    
    # Load data
    futures_data = load_futures_data()
    
    # Select a few contracts for pilot testing
    pilot_contracts = ['ES=F', 'CL=F', 'GC=F', 'ZN=F', '6E=F']
    
    results = {}
    
    for ticker in pilot_contracts:
        if ticker not in futures_data:
            print(f"⚠️ {ticker} not found, skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"📊 Testing {ticker}")
        print(f"{'='*50}")
        
        df = futures_data[ticker]
        train, test = get_train_test_split(df)
        
        if len(train) < 100 or len(test) < 50:
            print(f"⚠️ Insufficient data for {ticker}")
            continue
        
        # Get returns
        train_returns = train['Returns'].dropna().values
        test_returns = test['Returns'].dropna().values
        test_prices = test['Close']
        
        # Initialize results dict
        results[ticker] = {'train': {}, 'test': {}}
        
        # --- BASELINE STRATEGIES ---
        
        # 1. Long only
        positions = strategy_long(test_returns)
        metrics = calculate_metrics(test_returns, positions)
        results[ticker]['test']['Long'] = metrics
        print(f"  Long: Sharpe={metrics['Sharpe']:.3f}, MDD={metrics['MDD']:.2%}")
        
        # 2. Sign strategy
        positions = strategy_sign(test_returns, lookback=50)
        metrics = calculate_metrics(test_returns, positions)
        results[ticker]['test']['Sign'] = metrics
        print(f"  Sign: Sharpe={metrics['Sharpe']:.3f}, MDD={metrics['MDD']:.2%}")
        
        # 3. MACD
        positions = strategy_macd(test_prices)
        metrics = calculate_metrics(test_returns, positions)
        results[ticker]['test']['MACD'] = metrics
        print(f"  MACD: Sharpe={metrics['Sharpe']:.3f}, MDD={metrics['MDD']:.2%}")
        
        # --- SIMPLE DQN ---
        print(f"\n  🤖 Training Simple DQN...")
        
        env = FuturesEnv(train_returns, lookback=LOOKBACK)
        agent = SimpleDQN(state_size=LOOKBACK, n_actions=3, hidden_size=32)
        
        # Training loop
        n_episodes = 10  # Small for CPU testing
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                action = agent.act(state, training=True)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train(batch_size=32)
                state = next_state
                total_reward += reward
            
            if (episode + 1) % 2 == 0:
                print(f"    Episode {episode+1}/{n_episodes}, Reward={total_reward:.4f}, Epsilon={agent.epsilon:.3f}")
        
        # Test DQN
        test_env = FuturesEnv(test_returns, lookback=LOOKBACK)
        state = test_env.reset()
        positions = []
        
        while not test_env.done:
            action = agent.act(state, training=False)
            positions.append(action - 1)  # Convert to -1, 0, 1
            state, _, done = test_env.step(action)
        
        positions = np.array(positions)
        metrics = calculate_metrics(test_returns[LOOKBACK:LOOKBACK+len(positions)], positions)
        results[ticker]['test']['DQN'] = metrics
        print(f"\n  DQN:  Sharpe={metrics['Sharpe']:.3f}, MDD={metrics['MDD']:.2%}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 PILOT TEST SUMMARY")
    print("="*70)
    
    for ticker, data in results.items():
        print(f"\n{ticker}:")
        for strategy, metrics in data['test'].items():
            print(f"  {strategy:8s}: Sharpe={metrics['Sharpe']:6.3f}, Sortino={metrics['Sortino']:6.3f}, MDD={metrics['MDD']:6.2%}")
    
    # Save results
    with open('pilot_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to pilot_test_results.json")
    
    return results

if __name__ == "__main__":
    run_pilot_test()