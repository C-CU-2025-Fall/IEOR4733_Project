# A2C Model Weights - Placeholder

> ⏳ **This directory is awaiting A2C model weights upload**

## Expected Structure

```
rl_models/a2c/
├── README.md                                      # This file
│
├── period1_checkpoints/                          # Period 1 (2011-2015) models
│   ├── a2c_Commodity_period1.pt
│   ├── a2c_Equity_Index_period1.pt
│   ├── a2c_Fixed_Income_period1.pt
│   ├── a2c_Forex_period1.pt
│   └── a2c_All_period1.pt
│
├── period2_checkpoints/                          # Period 2 (2016-2019) models
│   ├── a2c_Commodity_period2.pt
│   ├── a2c_Equity_Index_period2.pt
│   ├── a2c_Fixed_Income_period2.pt
│   ├── a2c_Forex_period2.pt
│   └── a2c_All_period2.pt
│
└── results/                                        # A2C evaluation results
    ├── a2c_results_wide.csv                       # Cumulative returns organized by asset and date
    └── metrics_report.csv                         # Metrics comparison report
```

## Model File Description

| File | Description |
|------|-------------|
| `a2c_*_period1.pt` | A2C model weights trained during Period 1 (2011-2015) |
| `a2c_*_period2.pt` | A2C model weights trained during Period 2 (2016-2019) |
| `a2c_results_wide.csv` | Cumulative return data per asset class (date-indexed) |

## CSV File Format

`a2c_results_wide.csv` should contain the following columns:

```
date (index), Commodity, Equity Index, Fixed Income, Forex, All
2011-01-01,    1.0,      1.0,          1.0,         1.0,    1.0
2011-01-02,    1.002,    1.005,        0.998,       1.001,  1.001
...
2019-12-31,    1.234,    1.567,        0.987,       1.123,  1.228
```

- **Index**: Date (YYYY-MM-DD format)
- **Columns**: Cumulative wealth paths per asset class (W_t = 1 + cumsum(daily_returns))

## Model Architecture Requirements

The A2C model should include:

1. **Actor Network** - generates position weights (0 ~ 1)
2. **Critic Network** - evaluates state value

3. **Input State** (9-dim):
   - Recent returns (5-day)
   - Volatility (realized vol)
   - Relative Strength Index (RSI)
   - etc.

4. **Output**:
   - Actor: position weight distribution
   - Critic: state value estimate

## Upload Instructions

Submit the following to this directory:

1. ✅ Model weights (`.pt` files)
2. ✅ Evaluation results CSV
3. ✅ Model configuration document (config.json)
4. ✅ Training logs (optional)

## Related Code Locations

- **Model implementation**: `src/core/models/a2c/`
- **Training script**: `src/scripts/run_a2c_model.py`
- **Test script**: `tests/test_a2c.py`

## Usage Example

```python
from src.core.models.a2c import A2CEvaluator
import torch

# Load model
evaluator = A2CEvaluator(
    model_path="rl_models/a2c/period1_checkpoints/",
    data_path="data/",
    asset_class="Commodity"
)

# Evaluate strategy
returns = evaluator.evaluate(
    start_date="2011-01-01",
    end_date="2015-12-31"
)
```

---

**Upload deadline**: TBD
**Owner**: [To be assigned]
