# IEOR4733_Project
Final/Mid Term Project

## Deep Reinforcement Learning for Trading

### Paper to Reproduce

**"Deep Reinforcement Learning for Trading"** by Zhang, Zohren, and Roberts (Oxford, 2019)

This paper represents a massive paradigm shift from the previous two papers.

📄 **Paper Link:** [https://arxiv.org/pdf/1911.10107](https://arxiv.org/pdf/1911.10107)

#### Key Contributions
- Introduces a deep reinforcement learning framework for optimal execution and portfolio management
- Demonstrates the application of DRL agents in financial markets
- Provides empirical validation on real market data

---

## Project Timeline (Option A: Reproduce / Extend Existing Paper)

**Mid-Term Deadline:** March 10, 2026

### Phase 1: Pre-Mid Term (Before 3/10/2026)

- [ ] **Paper Analysis**
  - [ ] Read and understand the full paper
  - [ ] Identify key methodologies and algorithms
  - [ ] Document mathematical formulations
  - [ ] Note data requirements and sources

- [ ] **Proposal Deck (~4 slides)**
  - [ ] Slide 1: Topic & Brief Introduction
  - [ ] Slide 2: Overview of Literature and Methods
  - [ ] Slide 3: Strategy Overview (Data, Methodology, Evaluation Metrics)
  - [ ] Slide 4: Identified Weaknesses & Tentative Timeline

- [x] **Environment Setup**
  - [x] Set up development environment (Python, required libraries)
  - [ ] Configure GPU access (Colab Pro or cloud GPU) - *Optional for full DRL*
  - [x] Set up version control (Git repository)

- [x] **Data Sourcing** (see [data_sources_log.md](data_sources_log.md) for details)
  > ⚠️ **CORRECTION**: Paper uses **futures contracts**, not equities!
  - [x] Install packages: `yfinance`, `fredapi`, `pandas_datareader`
  - [x] Check Yahoo Finance futures coverage → **40/50 contracts available (80%)**
  - [x] Verify data covers: commodities ✅, equity indices ⚠️, fixed income ⚠️, FX ✅
  - [x] Download 37 unique futures contracts (2011-2019) → `data/futures/`
  - ❌ ~~Download S&P 500 constituent prices~~ (WRONG DATA TYPE)
  - ✅ Download risk-free rate (DTB3) → `data/risk_free_rate.csv` (still needed)
  - ✅ Download VIX data → `data/index_data.csv` (still useful)
  - [x] Run pilot test with baseline strategies + simple DQN → `pilot_test.py`
  
  **Yahoo Finance Coverage (Paper's Exact 50 Contracts from Appendix A):**
  | Asset Class | Available | Coverage |
  |-------------|-----------|----------|
  | Commodities | 22/25 | 88% ✅ |
  | Forex | 9/9 | 100% ✅ |
  | Fixed Income | 3/5 | 60% ⚠️ |
  | Equity Indexes | 6/11 | 55% ⚠️ |
  | **Total** | **40/50** | **80%** |

### Phase 2: Post-Mid Term (After 3/10/2026)

- [ ] **Implementation**
  - [ ] Build clean data pipeline
  - [ ] Implement DRL agent architecture
  - [ ] Develop backtest engine
  - [ ] Add transaction cost modeling

- [ ] **Research Presentation**
  - [ ] Summary of original paper
  - [ ] Document reproduction results
  - [ ] Compare differences vs original
  - [ ] Perform robustness checks
  - [ ] Analyze regime sensitivity
  - [ ] Compile risk diagnostics

- [ ] **Deployed Application**
  - [ ] Build performance dashboard
  - [ ] Implement risk metrics visualization
  - [ ] Enable ability to run new simulations
  - [ ] Ensure no lookahead bias
  - [ ] Validate realistic execution assumptions
  - [ ] Clear separation of training vs testing

- [ ] **Final Deliverables**
  - [ ] Complete research presentation
  - [ ] Functional deployed application (Streamlit/React/API)
  - [ ] Code documentation and reproducibility check
  - [ ] Sensitivity analysis
