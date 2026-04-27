# Forex Feature Correlation Matrix

Averaged across 9 Forex contracts (AN,BN,CN,DX,FN,JN,MP,NK,SN), r1, after WARMUP=252.

## Feature Names
- 0: price_norm (normalized close / 60d rolling std)
- 1: ret_21d (21-day return normalized)
- 2: ret_42d (42-day return normalized)
- 3: ret_63d (63-day return normalized)
- 4: ret_252d (252-day return normalized)
- 5: MACD (averaged multi-scale MACD)
- 6: RSI (30-day RSI normalized)
- 7: MACD_aux1
- 8: MACD_aux2

## Correlation Matrix

| | price_norm | ret_21d | ret_42d | ret_63d | ret_252d | MACD | RSI | MACD_aux1 | MACD_aux2 |
|---|---|---|---|---|---|---|---|---|---|
| price_norm | 1.000 | +0.070 | +0.091 | +0.130 | +0.152 | +0.085 | +0.151 | +0.201 | +0.120 |
| ret_21d | +0.070 | 1.000 | **+0.725** | **+0.617** | +0.314 | **+0.876** | **+0.755** | +0.392 | **+0.868** |
| ret_42d | +0.091 | **+0.725** | 1.000 | **+0.822** | +0.407 | **+0.716** | **+0.833** | +0.580 | **+0.900** |
| ret_63d | +0.130 | **+0.617** | **+0.822** | 1.000 | +0.472 | **+0.606** | **+0.801** | **+0.695** | **+0.858** |
| ret_252d | +0.152 | +0.314 | +0.407 | +0.472 | 1.000 | +0.331 | +0.516 | **+0.712** | +0.471 |
| MACD | +0.085 | **+0.876** | **+0.716** | **+0.606** | +0.331 | 1.000 | **+0.845** | +0.444 | **+0.856** |
| RSI | +0.151 | **+0.755** | **+0.833** | **+0.801** | +0.516 | **+0.845** | 1.000 | **+0.771** | **+0.874** |
| MACD_aux1 | +0.201 | +0.392 | +0.580 | **+0.695** | **+0.712** | +0.444 | **+0.771** | 1.000 | +0.620 |
| MACD_aux2 | +0.120 | **+0.868** | **+0.900** | **+0.858** | +0.471 | **+0.856** | **+0.874** | +0.620 | 1.000 |

## Key Observations

**High correlations (>0.7, bold above):**
- ret_42d ↔ MACD_aux2: +0.900 (最强)
- ret_42d ↔ ret_63d: +0.822
- ret_21d ↔ MACD: +0.876
- ret_21d ↔ MACD_aux2: +0.868
- RSI ↔ MACD_aux2: +0.874
- RSI ↔ MACD: +0.845
- RSI ↔ ret_42d: +0.833
- ret_63d ↔ MACD_aux2: +0.858

**Low correlations (<0.3):**
- price_norm 与所有其他 feature < 0.2 ✅ (设计目标：低相关性)
- ret_21d ↔ ret_252d: +0.314 (短期 vs 长期回报，低相关)
- MACD ↔ ret_252d: +0.331

**问题：**
- 6/9 features 之间高度相关 (>0.7)，实际有效维度可能只有 ~4
- MACD_aux2 与 ret_42d/ret_63d/ret_21d 高度冗余
- RSI 与几乎所有 features 都 >0.5

Source: drl_shared/prepare_features.py, structural_38 preset, FEATURE_DIM=9
Date: 2026-04-27
