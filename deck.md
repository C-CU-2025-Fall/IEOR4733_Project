# Proposal Deck: Deep Reinforcement Learning for Trading

## Paper to Reproduce
**"Deep Reinforcement Learning for Trading"** by Zhang, Zohren, and Roberts (Oxford, 2019)  
📄 Paper Link: https://arxiv.org/pdf/1911.10107

---

## Slide 1: Topic & Brief Introduction

### What is the Topic?
Deep Reinforcement Learning (DRL) for optimal trading strategy development and portfolio management.

### Brief Introduction
This paper introduces a deep reinforcement learning framework that learns optimal trading strategies directly from market data. Unlike traditional approaches that rely on hand-crafted features and rules, the DRL agent learns to make trading decisions (buy, sell, hold) by maximizing a reward function based on portfolio returns while accounting for transaction costs and market impact.

**Key Innovation:** The paper demonstrates that DRL agents can outperform traditional benchmark strategies (e.g., buy-and-hold, moving average crossover) on real market data without explicit feature engineering.

---

## Slide 2: Overview of Literature and Methods

### Summary of Paper's Approach

The paper addresses portfolio optimization using three DRL algorithms:

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **DDPG** (Deep Deterministic Policy Gradient) | Actor-Critic | Continuous action spaces, deterministic policy |
| **PPO** (Proximal Policy Optimization) | Policy Gradient | Stable updates with clipped objective |
| **A2C** (Advantage Actor-Critic) | Actor-Critic | Parallel training, synchronous updates |

Deep Q-learning Networks (DQN) [34, 49], Policy Gradients (PG) [52] and Advantage Actor-Critic (A2C) [33]

原文文献综述
| **Fundamental analysis** | CAN-SLIM [43] compare current price with expectation to see if it is undervalued or overvalued | timing of enter and exit of trades is not specified|
| **Technical analysis** | Combination of indicators (e.g. Relative Strength Index (RSI), Bollinger Bands) | weak predictability due to lack of analysis of market|
| **Algorithum Trading** | Time series momentum strategies (baseline) | weak predictability due to market move sideways|
| **RL for Finance** | DQN (critic-approach,discrete,a state-action value function, Q, is constructed to represent how good a particular action is in a state); offline batch gradient ascent methods (actor-approach,continuous,e.g. max profits or Sharpe ratio)| suffers from large action spaces; NA|

### Paper's Methodology Framework

```
┌─────────────────────────────────────────────────────────┐
│  STATE SPACE (Observation)                              │
│  - Historical returns (past N days)                     │
│  - Portfolio weights (current allocation)               │
│  - Market indicators (volatility, volume)               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  DRL AGENT (DDPG / PPO / A2C)                           │
│  - Actor Network: State → Action (portfolio weights)    │
│  - Critic Network: (State, Action) → Q-value            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  ACTION SPACE                                           │
│  - Portfolio weights (continuous: 0 to 1)               │
│  - Rebalancing decisions at each time step              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  REWARD FUNCTION                                        │
│  - Differential Sharpe Ratio (risk-adjusted return)     │
│  - Transaction cost penalty                             │
└─────────────────────────────────────────────────────────┘
```

### How This Differs from Prior Work

| Prior Approach | Limitation | This Paper's Solution |
|----------------|------------|----------------------|
| Traditional strategies (e.g., mean-variance) | Static, require manual rebalancing | DRL learns dynamic rebalancing |
| Supervised learning (LSTM, CNN) | Predicts prices, not optimal actions | DRL directly optimizes portfolio weights |
| Classical RL (Q-learning) | Discrete actions, limited scalability | Continuous action space with DDPG/PPO |

---

## Slide 3: Strategy Overview (Data, Methodology, Evaluation Metrics)

### Data

| Data Type | Source | Period | Paper Reference |
|-----------|--------|--------|-----------------|
| S&P 500 constituent prices | Yahoo Finance | 2010-2023 | Section 3.1: "S&P 500 index constituents" |
| Risk-free rate (3M T-Bill) | FRED (DTB3) | 2010-2023 | Section 3.2: "3-month T-bill rate" |
| S&P 500 Index (benchmark) | Yahoo Finance (^GSPC) | 2010-2023 | Benchmark comparison |
| VIX (regime detection) | Yahoo Finance (^VIX) | 2010-2023 | Extension: market regime |

**Data Splits (per paper):**
- Training: ~60-70% of data
- Validation: ~15-20% of data
- Test: ~15-20% of data (out-of-sample)

### Methodology (Reproduction Plan)

1. **Data Pipeline**
   - Download and clean OHLCV data
   - Calculate returns, volatility, and technical indicators
   - Normalize features for neural network input

2. **DRL Agent Implementation**
   - Implement DDPG, PPO, A2C using stable-baselines3 or custom PyTorch
   - Design state representation (returns, portfolio weights)
   - Define action space (portfolio allocation)

3. **Backtest Engine**
   - Walk-forward validation (no lookahead bias)
   - Transaction cost modeling (10 bps per trade, per paper)
   - Realistic execution assumptions

4. **Deployed Application**
   - Streamlit dashboard for visualization
   - Performance metrics and risk analytics
   - Ability to run new simulations

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 (beat benchmark) |
| **Maximum Drawdown** | Largest peak-to-trough decline | < 20% |
| **Cumulative Return** | Total portfolio return | Beat S&P 500 |
| **Calmar Ratio** | Return / Max Drawdown | > 0.5 |
| **Turnover** | Portfolio rotation rate | Monitor for transaction cost impact |

---

## Slide 4: Identified Weaknesses & Tentative Timeline

### Identified Weaknesses in Paper's Methods

| Weakness | Description | Proposed Extension |
|----------|-------------|-------------------|
| **Regime Sensitivity** | Agent may perform differently in bull vs bear markets | Add VIX-based regime detection |
| **Transaction Costs** | Fixed 10 bps may not reflect real costs | Model realistic, time-varying costs |
| **Overfitting Risk** | Deep networks can overfit training period | Walk-forward validation, ensemble methods |
| **Interpretability** | Black-box decisions | Attention mechanism for feature importance |

### Tentative Timeline

| Phase | Period | Tasks | Deliverable |
|-------|--------|-------|-------------|
| **Pre-Mid Term** | Now - 3/10 | Paper analysis, deck, data collection | Proposal Deck ✅ |
| **Week 1-2** | 3/11 - 3/25 | Data pipeline, DRL agent implementation | Working prototype |
| **Week 3-4** | 3/26 - 4/8 | Backtest engine, transaction costs | Backtest results |
| **Week 5-6** | 4/9 - 4/22 | Robustness checks, regime analysis | Research findings |
| **Week 7-8** | 4/23 - 5/6 | Dashboard, final presentation | Deployed app |
| **Final** | 5/7+ | Complete system submission | All deliverables |

### Planned Extensions (Beyond Paper)

1. **Market Regime Detection** - Use HMM or VIX thresholds to identify bull/bear/sideways markets
2. **Adaptive Transaction Costs** - Model costs that vary with volatility and liquidity
3. **Walk-Forward Validation** - More rigorous out-of-sample testing than simple train/test split

---

## Evaluation Checklist (Per requirements.md - Option A)

### Research Presentation Requirements (Final)
- [ ] Summary of original paper
- [ ] Reproduction results
- [ ] Differences vs original
- [ ] Robustness checks
- [ ] Regime sensitivity
- [ ] Risk diagnostics

### Deployed Application Requirements (Final)
- [ ] Clean data pipeline
- [ ] Backtest engine
- [ ] Transaction cost modeling
- [ ] Performance dashboard
- [ ] Risk metrics
- [ ] Ability to run new simulations

### System Validation Requirements
- [ ] No lookahead bias
- [ ] Realistic execution assumptions
- [ ] Reproducibility
- [ ] Clear separation of training vs testing
- [ ] Sensitivity analysis

### Acceptable Delivery Formats
- [ ] Web app (Streamlit recommended)
- [ ] Interactive dashboard
- [ ] API-based system
- [ ] Modular Python framework
