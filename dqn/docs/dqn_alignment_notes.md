# DQN Alignment Notes

## 两个关键概念区分

### 1️⃣ 对齐论文方法论 (Implementation Alignment)
**目标**：100% 复现论文的方法，即使论文本身有缺陷

**当前状态**：
- ✅ 超参数 (Table 1): LR=0.0001, γ=0.3, batch=64, LSTM[64,32]
- ✅ 训练期：2005-2010 (6 年，论文用 10+ 年但数据限制)
- ✅ 测试期：2011-2019 (9 年，与论文一致)
- ✅ 双 Q 网络 (Fixed Q-targets)
- ✅ Double DQN
- ✅ Gradient Clipping (0.5)
- ✅ LSTM + Leaky-ReLU
- ✅ 状态空间 8 维 (已修复多尺度 MACD)
- ⚠️ 动作空间：离散 {-1,0,+1} → 需确认论文用离散还是连续
- ⚠️ Reward 函数：需确认是否有 risk-adjustment

**待确认项**：
1. 论文 DQN 用离散动作还是连续动作？
2. 论文 reward 函数是否有 risk-adjustment 项？
3. 论文是否用 walk-forward validation？

### 2️⃣ 论文本身问题 (Paper Limitations)
**目标**：识别论文方法的潜在缺陷，作为未来改进方向

**可能的问题**（不影响对齐）：
- 训练数据只有 10 年，可能过拟合
- 50 合约独立训练，忽略合约间关系
- 离散动作空间可能限制策略表达力
- 无 risk-adjusted reward，可能过度冒险
- 无 walk-forward validation，时变性能未知

---

## 对齐方案 (专注 1️⃣)

### ✅ Step 1: 论文细节确认 (COMPLETED)
- ✅ **Action Space**: 离散 {-1, 0, +1} (output_dim=3, argmax 选择)
- ✅ **Reward 函数**: `reward = (action × vol_scale × return) - cost` (无 risk-adjustment)
- ✅ **训练期**: 论文用 2011-2015 (5 年)，我们用 2005-2010 (6 年，数据限制)
- ✅ **测试期**: 2011-2019 (9 年，与论文 Table 3 一致)
- ✅ **网络**: LSTM[64,32] + Leaky-ReLU + output_dim=3
- ✅ **Double DQN**: 主网络选动作，目标网络算 Q 值
- ✅ **Fixed Q-targets**: Target Network 每 1000 步更新

### ⚠️ Step 2: 当前实现差异
- ✅ **动作空间**: 离散 {-1,0,+1} → **已对齐**
- ✅ **Reward**: 简单 reward → **已对齐**
- ✅ **State Space**: 8 维 (含多尺度 MACD) → **已对齐**
- ⚠️ **训练数据**: 2005-2010 (6 年) vs 论文 2011-2015 (5 年) → **数据窗口不同但长度相似**
- ⚠️ **推理输出**: 当前用 softmax 转连续 position → **应改为离散动作映射**

### Step 3: 待修复项
- [ ] **推理时动作映射**: 离散动作 → position (baseline_run 中处理)
- [ ] **验证对齐**: 单合约训练，对比原始代码结果
- [ ] **记录差异**: 如与论文 Table 3 DQN 结果有差异，记录并分析

### Step 4: 训练计划
- [ ] 重新训练 50 合约 (100 episodes, 2005-2010 数据)
- [ ] 测试 2011-2019 对齐度
- [ ] 对比 Long Only 基线

---

**Last Updated**: 2026-04-22
