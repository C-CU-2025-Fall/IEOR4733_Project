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

- [ ] **Environment Setup**
  - [ ] Set up development environment (Python, required libraries)
  - [ ] Configure GPU access (Colab Pro or cloud GPU)
  - [ ] Set up version control (Git repository)

- [x] **Data Sourcing** (see [data_sources_log.md](data_sources_log.md) for details)
  - [x] Install packages: `yfinance`, `fredapi`, `pandas_datareader`
  - [x] Download S&P 500 constituent prices → `data/sp500_prices.csv` (174,625 rows, 50 tickers)
  - [x] Download risk-free rate (DTB3) → `data/risk_free_rate.csv` (3,651 rows)
  - [x] Download S&P 500 index and VIX data → `data/index_data.csv` (7,044 rows)
  - [x] Verify data completeness and quality

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
