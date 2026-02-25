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

### Literature Review

| Category | Key Papers/Approaches | Limitations |
|----------|----------------------|-------------|
| **Traditional Quant Strategies** | Moving averages, momentum, mean-reversion | Rule-based, static, require manual tuning |
| **Supervised Learning for Trading** | LSTM, CNN for price prediction | No explicit action optimization, no transaction cost awareness |
| **Classical RL for Finance** | Q-learning, SARSA on discretized states | Limited state representation, scalability issues |
| **Deep RL (This Paper)** | DDPG, PPO, A2C on continuous state spaces | Computationally intensive, requires careful reward design |

### Methodology Overview

The paper employs the following DRL architectures:
1. **Deep Deterministic Policy Gradient (DDPG)** - Actor-Critic method for continuous action spaces
2. **Proximal Policy Optimization (PPO)** - Stable policy gradient updates
3. **Advantage Actor-Critic (A2C)** - Parallel training with multiple workers

**State Representation:** Market features (returns, volumes, volatility indicators)  
**Action Space:** Portfolio weights (continuous) or discrete trading signals  
**Reward Function:** Risk-adjusted returns (Sharpe ratio, differential Sharpe ratio)

---

## Slide 3: Strategy Overview

### Data Requirements

| Data Type | Description | Frequency | Scope |
|-----------|-------------|-----------|-------|
| **Price Data** | OHLCV (Open, High, Low, Close, Volume) | Daily/Minute-level | S&P 500 stocks or major crypto pairs |
| **Market Indicators** | Moving averages, RSI, MACD, Bollinger Bands | Derived from price | Technical analysis features |
| **Risk-Free Rate** | Treasury yields for Sharpe calculation | Daily | FRED or Bloomberg |

### Methodology

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│  Raw OHLCV → Feature Engineering → State Normalization  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  DRL AGENT (DDPG/PPO)                   │
│  Actor Network: State → Action (portfolio weights)      │
│  Critic Network: (State, Action) → Q-value              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   BACKTEST ENGINE                       │
│  Execute trades → Apply transaction costs → Calculate   │
│  portfolio returns → Compute risk metrics               │
└─────────────────────────────────────────────────────────┘
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Cumulative Return** | Total portfolio return over test period | Beat benchmark |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Maximum Drawdown** | Largest peak-to-trough decline | < 20% |
| **Win Rate** | Percentage of profitable trades | > 50% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Calmar Ratio** | Return / Maximum Drawdown | > 0.5 |

---

## Slide 4: Identified Weaknesses & Tentative Timeline

### Identified Weaknesses in Existing Methods

1. **Limited Market Regime Handling**
   - DRL agents trained on bull markets may fail in bear markets
   - No explicit regime detection mechanism

2. **Transaction Cost Sensitivity**
   - High-frequency trading signals lead to excessive turnover
   - Transaction costs can erode alpha significantly

3. **Overfitting Risk**
   - Deep neural networks can overfit to training period
   - Limited out-of-sample validation in some studies

4. **Interpretability**
   - Black-box nature makes it hard to explain trading decisions
   - Regulatory compliance challenges

### Proposed Extensions/Improvements

- [ ] Add regime-switching mechanism (HMM for market state detection)
- [ ] Implement adaptive transaction cost penalty in reward function
- [ ] Use walk-forward validation instead of simple train/test split
- [ ] Add attention mechanism for feature importance visualization

### Tentative Timeline

| Phase | Period | Tasks |
|-------|--------|-------|
| **Pre-Mid Term** | Now - 3/10/2026 | Paper analysis, proposal deck, environment setup, data collection |
| **Week 1-2 Post Mid-Term** | 3/11 - 3/25 | Data pipeline, DRL agent implementation |
| **Week 3-4** | 3/26 - 4/8 | Backtest engine, transaction cost modeling |
| **Week 5-6** | 4/9 - 4/22 | Robustness checks, regime sensitivity analysis |
| **Week 7-8** | 4/23 - 5/6 | Dashboard development, final presentation prep |
| **Final Delivery** | 5/7+ | Complete system submission |

---

## Data Sources

### Required Data for Reproduction

| Data Category | Specific Requirements | Proposed Source | Cost |
|---------------|----------------------|-----------------|------|
| **Equity Price Data** | S&P 500 constituent OHLCV, daily/minute-level (2015-2019) | Yahoo Finance (`yfinance`), WRDS (CRSP) | Free |
| **Cryptocurrency Data** | BTC/ETH OHLCV, minute-level | Binance API, Yahoo Finance | Free |
| **Risk-Free Rate** | 3-Month Treasury Bill rates | FRED (Federal Reserve) | Free |
| **Market Index Data** | S&P 500 index for benchmark | Yahoo Finance (`^GSPC`) | Free |
| **Volatility Index** | VIX for regime detection | Yahoo Finance (`^VIX`) | Free |

### Data Collection Plan

> **Paper Reference:** Section 3 (Data) and Section 4 (Experimental Setup) of Zhang, Zohren, and Roberts (2019)

#### Specific Data Requirements from Paper

| Requirement | Paper Specification | Section Reference |
|-------------|--------------------|--------------------|
| **Asset Universe** | S&P 500 index constituents | Section 3.1: "We use the S&P 500 index constituents as our investment universe" |
| **Data Period** | December 1998 - April 2018 (~20 years) | Section 3.1: "Our dataset spans from December 1998 to April 2018" |
| **Data Frequency** | Daily closing prices | Section 3.1: "We use daily closing prices" |
| **Risk-Free Rate** | 3-month T-bill rate | Section 3.2: "The risk-free rate is the 3-month T-bill rate" |
| **Train/Test Split** | Training: 1998-2012, Validation: 2013-2015, Test: 2016-2018 | Section 4.1: "We split the data into training (1998-2012), validation (2013-2015), and test (2016-2018) periods" |
| **Transaction Costs** | 10 basis points (0.1%) | Section 3.3: "We assume a proportional transaction cost of 10 basis points" |

#### Primary Data Sources (Free)

```
Yahoo Finance (yfinance library)
├── S&P 500 stocks (daily OHLCV)
│   └── Paper uses: "daily closing prices of S&P 500 constituents"
├── Market indices (SPY, ^GSPC)
│   └── Paper uses: "S&P 500 index as benchmark"
└── VIX volatility index
    └── For potential regime detection extension

FRED API (fredapi library)
└── DTB3 (3-Month Treasury Bill)
    └── Paper specifies: "risk-free rate is the 3-month T-bill rate"
```

#### Alternative Data Sources (If Needed)

| Source | Data Type | Cost | Use Case |
|--------|-----------|------|----------|
| WRDS/CRSP | US equity data, survivorship-bias free | Free (academic) | More accurate historical constituents |
| Compustat | Firm fundamentals | Free (academic) | Potential fundamental feature extension |
| Databento | High-quality exchange data | ~$20-50 | Minute-level data for high-frequency extension |

### Data Scope (Based on Paper Specifications)

> **Paper Reference:** Section 4.1 (Experimental Setup)

| Period | Date Range | Duration | Purpose | Paper Citation |
|--------|------------|----------|---------|----------------|
| **Training** | 1998-12 to 2012-12 | ~14 years | DRL agent learning | "Training period: 1998-2012" |
| **Validation** | 2013-01 to 2015-12 | ~3 years | Hyperparameter tuning | "Validation period: 2013-2015" |
| **Test** | 2016-01 to 2018-04 | ~2.5 years | Out-of-sample evaluation | "Test period: 2016-2018" |

#### Adjusted Scope for Reproduction (Realistic for Course Project)

| Period | Date Range | Duration | Purpose | Rationale |
|--------|------------|----------|---------|-----------|
| **Training** | 2010-01 to 2017-12 | 8 years | DRL agent learning | Reduced from 14 years; still captures multiple market regimes |
| **Validation** | 2018-01 to 2019-12 | 2 years | Hyperparameter tuning | Sufficient for validation |
| **Test** | 2020-01 to 2023-12 | 4 years | Out-of-sample evaluation | Includes COVID crash and recovery |

#### Instruments

- **Primary:** S&P 500 constituents (paper: "S&P 500 index constituents as our investment universe")
- **Reduced Scope Option:** Top 50-100 most liquid stocks (for computational efficiency)
- **Frequency:** Daily (paper: "We use daily closing prices")

#### Key Data Points to Extract

| Data Field | Source | Paper Reference |
|------------|--------|-----------------|
| Daily Close Prices | Yahoo Finance | Section 3.1 |
| Daily Volume | Yahoo Finance | For liquidity filtering |
| 3-Month T-Bill Rate | FRED (DTB3) | Section 3.2 |
| S&P 500 Index | Yahoo Finance (^GSPC) | Benchmark comparison |

---

## Evaluation Checklist (Per requirements.md)

### Research Presentation Requirements
- [ ] Summary of original paper
- [ ] Reproduction results
- [ ] Differences vs original
- [ ] Robustness checks
- [ ] Regime sensitivity
- [ ] Risk diagnostics

### Deployed Application Requirements
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