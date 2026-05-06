# Paper Figures & Tables — Complete Reproduction Plan

## 论文原始 Exhibits 对照

| Paper | 内容 | 可做? | 数据 |
|-------|------|-------|------|
| **Table 1** | 超参数 | ✅ 已有 spec.py | — |
| **Table 2** | Portfolio vol-target 结果 | ✅ 需补 engine.py 1行 | audit .npz |
| **Table 3** | Raw signal (Appendix B) | ✅ Top-5 Q-ensemble 已有 | g06_audit |
| **Figure 1** | Cumulative trade returns (5 panels) | ✅ | portfolio_returns |
| **Figure 2** | TC sensitivity | ❌ 手续费相关，排除 | — |
| **Figure 3** | Per-contract boxplots | ❌ 缺 per-contract returns | 需重新回测 |

## 数据就绪

| 类别 | 位置 | 内容 |
|------|------|------|
| Portfolio returns | results/g06_audit/{Asset}/r1_s*.npz | portfolio_returns + dates, 10 seeds × 4 assets |
| Portfolio returns | results/gamma06_audit/Forex/ | r1+r2, 20 audit files |
| Q-ensemble 结果 | 已在内存 | Top-3/5/7 vs Long Only, full 2011-2019 |
| Per-contract | ❌ 缺失 | audit 只存了 portfolio-level |
| Plotting infra | ❌ 无 | 零 matplotlib/seaborn，需从零建 |

## 实施计划

### Wave 1: Paper Figure 1 — Cumulative Returns (5 panels)
**Paper layout**: 上行 Commodity/EqIdx/FI, 下行 Forex/Portfolio
**Our data**: Q-ensemble + Long Only portfolio_returns + dates
**Output**: PNG, 5 subplots, DQN vs Long Only overlay, 2011-2019, r1/r2 demarcation

### Wave 2: Paper Table 2 — Portfolio Vol Targeting
**需要**: engine.py `portfolio_metrics()` 加 `port_vol_target` 参数传递
**输出**: 9 metrics × 4 assets × Top-5 ensemble vs Paper Table 2

### Wave 3: Supplementary Figures (论文没有，但值得有)
- Rolling 252-day Sharpe (已有脚本框架)
- Drawdown curves
- Monthly returns heatmap (4×9 grid)
- Year-by-year performance bar chart

### Wave 4: Action Distribution (需重新回测存 positions)
**Wait** — audit npz 需要改存 per-timestep action_id
