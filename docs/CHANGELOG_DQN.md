# IEOR4733 DQN 项目记忆

## 2026-05-04 Pipeline 对齐 & Reward Fix

### 做了什么
1. **V1 回滚完成**: 9D features + V1 additive reward, stash@{0} 备份 V2/V3
2. **Reward timing bug 修复**: ContractEnv.step() 改用 self.last_position (论文 A_{t-1})
3. 清理重复 checkpoint (074328, 12D 模型)
4. 创建 `docs/DQN_PIPELINE.md` 架构文档
5. `scripts/alpha_decay_analysis.py` 重写, 用 baseline_run reward

### 数据管线结论
- baseline_run 和 training 用**同一个** load_clc_full, 价格完全相同
- 长度不同因为 start_date 不同 (2009 vs 2004), 不是数据源差异
- **曾经误报** "return correlation 0.03" — 纯粹是索引对齐错误, 不是真实差异
- On-the-fly features 和 npz features 在 test period 数学上相同

### 当前模型状态
- R1 (074157): Sharpe +0.10 (baseline_run reward), 2014 单年拉动
- R2 (074703): Sharpe -1.26, 全面亏损
- **两个模型都是在错误 reward (1-day lookahead) 下训练的**
- 需要用修正后的 reward 重训

### 教训
1. **先读文档再诊断** — 不要基于假设开始调试
2. **验证索引对齐** — 比较两个数组时先确认它们对应同一组日期
3. **不要连续叠加未验证的结论** — 一个错误结论会污染后续所有推理
4. **假警报比 bug 更危险** — 浪费时间、破坏信任

### 待做
- [ ] 用修正 reward 重训 Forex R1/R2
- [ ] 验证修正后训练/回测 reward 完全一致
- [ ] 分析模型为什么 Long=0% (Hold reward=0 bias)
- [ ] 其他 asset class features 重生成 (还是 12D 或旧日期范围)
