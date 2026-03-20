# Cloud Deployment Guide - DRL Trading

## IEOR4733 Project: Deep Reinforcement Learning for Trading

**Paper:** Zhang, Zohren, and Roberts (Oxford, 2019)  
**Link:** https://arxiv.org/pdf/1911.10107

---

## Quick Start on Cloud

### Option 1: Google Colab (Recommended - Free GPU)

1. **Upload this folder to Google Drive** under `/content/drive/MyDrive/IEOR4733/`

2. **Open the notebook:**
   - Go to https://colab.research.google.com/
   - File → Open → Upload → `drl_trading_cloud.ipynb`

3. **Enable GPU:**
   - Runtime → Change runtime type → T4 GPU

4. **Run all cells:**
   - Runtime → Run all

### Option 2: Kaggle (Free GPU)

1. Go to https://www.kaggle.com/code
2. New Notebook → Upload `drl_trading_cloud.ipynb`
3. Settings → Accelerator → GPU T4 x2
4. Run all cells

### Option 3: Local with GPU

```bash
# Create environment
conda create -n drl_trading python=3.10 -y
conda activate drl_trading

# Install dependencies
pip install yfinance stable-baselines3 gymnasium pandas numpy matplotlib tqdm torch

# Run notebook
jupyter notebook drl_trading_cloud.ipynb
```

---

## Files to Upload to Cloud

```
📁 IEOR4733_Project/
├── 📄 drl_trading_cloud.ipynb    ← Main notebook (REQUIRED)
├── 📄 cloud_readme.md            ← This file
├── 📁 data/
│   └── 📁 futures/               ← Pre-downloaded data (37 CSV files)
│       ├── ES=F.csv
│       ├── CL=F.csv
│       ├── GC=F.csv
│       └── ... (34 more)
└── 📁 results/                   ← Will be created during training
```

**Note:** If you don't upload `data/futures/`, the notebook will download data automatically (takes ~5 min).

---

## Expected Output

After running the notebook, you'll get:

### 1. Performance Summary Table

| Ticker | Strategy | Sharpe | Sortino | MDD | Calmar |
|--------|----------|--------|---------|-----|--------|
| ES=F | Long | 0.876 | 0.988 | -20.43% | 0.466 |
| ES=F | Sign | 0.278 | 0.323 | -17.14% | 0.115 |
| ES=F | MACD | 2.161 | 3.137 | -6.86% | 1.085 |
| ES=F | DQN | ? | ? | ? | ? |
| ES=F | PPO | ? | ? | ? | ? |
| ES=F | A2C | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |

### 2. Files Generated

```
📁 results/
├── performance_summary.csv    ← Main results table
├── all_results.json           ← Detailed metrics
└── sharpe_comparison.png      ← Bar chart

📁 models/
├── ES_DQN.zip                 ← Trained DQN model
├── ES_PPO.zip                 ← Trained PPO model
├── ES_A2C.zip                 ← Trained A2C model
└── ... (for each contract)
```

### 3. Export Archives

- `drl_trading_results.zip` - All results
- `drl_trading_models.zip` - All trained models

---

## Configuration

Key parameters (can be modified in the notebook):

```python
TRAIN_START = '2011-01-01'      # Training start
TRAIN_END = '2017-06-30'        # Training end (~6 years)
TEST_START = '2017-07-01'       # Test start
TEST_END = '2019-12-31'         # Test end (~2.5 years)

TRANSACTION_COST = 0.001        # 10 bps per trade (per paper)
LOOKBACK = 50                   # Days of history for state

# Training timesteps
PILOT_TIMESTEPS = 50000         # Quick test
FULL_TIMESTEPS = 100000         # Full training
```

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in agent definitions
DQN('MlpPolicy', env, batch_size=32, ...)  # Default is 64
```

### Data Download Fails
```python
# Data is pre-downloaded in data/futures/
# If missing, the notebook will download automatically
```

### Slow Training
```python
# Reduce timesteps for quick test
results = train_drl_agents(ticker, futures_data, total_timesteps=10000)
```

---

## Results Pickup

After training completes:

1. **Download from Colab:**
   - Click folder icon in left sidebar
   - Right-click `drl_trading_results.zip` → Download
   - Right-click `drl_trading_models.zip` → Download

2. **Or sync with Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp drl_trading_results.zip /content/drive/MyDrive/
   !cp drl_trading_models.zip /content/drive/MyDrive/
   ```

---

## Paper Comparison

The paper reports these approximate results for the test period:

| Asset Class | DQN Sharpe | PPO Sharpe | A2C Sharpe |
|-------------|------------|------------|------------|
| Commodities | ~0.5-0.8 | ~0.3-0.5 | ~0.4-0.6 |
| Equity Indexes | ~0.5-1.0 | ~0.3-0.6 | ~0.4-0.7 |
| Fixed Income | ~0.8-1.2 | ~0.6-0.8 | ~0.6-0.9 |
| FX | ~0.3-0.6 | ~0.2-0.4 | ~0.3-0.5 |

Your results may vary due to:
- Different data source (Yahoo vs Pinnacle CLC)
- Different random seed
- Hyperparameter settings

---

## Contact

For questions about this project, refer to:
- `README.md` - Project overview
- `data_sources_log.md` - Data documentation
- `deck.md` - Proposal slides

---

*Generated: February 2026*