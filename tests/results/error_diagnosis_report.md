# RAD_v2 生成错误诊断报告

**生成日期**: 2026-04-12  
**测试期**: 2011-01-01 至 2019-12-31  
**总合约数**: 96

---

## 文件夹结构

```
IEOR4733_Project/
├── data/CLC/
│   ├── *_RAD.CSV          # CLC 官方原始数据
│   ├── *_NON.CSV          # CLC 官方连续合约
│   ├── *_RAD_v2*.CSV      # 我们生成的 RAD_v2 文件
│   └── *_rollovers.csv    # 从 ASC 提取的换月数据
├── tests/
│   ├── *.py               # 生成/验证脚本
│   └── results/           # 测试输出、对比报告、验证结果
└── config/
    ├── TEMP/              # ASC 原始文件
    └── roll_rules_*.json  # 换月规则配置
```

---

## 错误类型分析

### 1. MISSING_NON (1 个合约)

**合约**: AP

**原因**: 缺少 `AP_NON.CSV` 文件

**解决**: 需要检查 CLC 数据源是否提供 AP 合约的 NON 文件

---

### 2. NO_ROLLS_IN_TEST / NO_ROLLS_AFTER_2011 (43 个合约)

**合约列表**:
SP, MW, ZG, TU, ZT, NG, ZM, SM, LC, ZU, TA, ZH, LH, ZA, BO, CB, CC, CL, CR, CT, DA, DT, DX, EC, ED, FB, FC, FN, FX, GC, GI, HO, JN, JO, KC, KW, LB, MP, NK, NR, O__, PL, SB, SI

**原因**: 
- 这些合约的换月数据在 2011 年之前就结束了
- 例如 SP 合约：最后一次换月是 1999-12-08，但测试期从 2011-01-01 开始
- **这不是代码 bug，是数据本身的限制**

**影响**:
- 无法在测试期内应用换月 ratio 调整
- 生成的 RAD_v2 与 NON 完全相同（ratio=1.0）
- 与 CLC RAD 对比时相关性可能很低（因为 CLC RAD 有历史累积的 ratio）

**建议**:
- 方案 A: 将测试期提前到合约换月数据开始的年份
- 方案 B: 对这些合约使用"最后一次已知 ratio"外推
- 方案 C: 在论文中说明这些合约被排除在回测之外

---

### 3. RAD_ZERO_FIRST_HALF (27 个合约)

**合约列表**:
FA, ED, PA, HG, NG, SM, LC, ZU, TA, ZH, BG, LH, FC, BC, BO, CB, CC, CL, CR, CT, DA, DT, DX, EC, FB, FN, FX

**原因**:
- **CLC 官方 RAD 文件本身的数据质量问题**
- 这些合约的 RAD 文件在 2011 年后的前半段价格全是 0
- 例如 FA_RAD.CSV：2011-01-03 至约 2015 年，Close 列全为 0

**影响**:
- `scale = first_rad / first_v2 = 0 / X = 0`
- 相关性计算：`corr([0,0,0,...], [X,Y,Z,...]) = NaN`
- 无法进行有效的 RAD vs RAD_v2 对比验证

**验证**（以 FA 为例）:
```
FA RAD 2011 年数据: 2223 行
Close 列统计:
  mean: 3.46  (大部分是 0，少量非零值拉高平均)
  50%:  0.00  (中位数是 0)
  75%:  0.00  (75% 的值都是 0)
```

**解决**:
- 这是 CLC 数据源的问题，不是我们的 bug
- 建议：联系 CLC 数据提供方或使用其他数据源验证
- 或者：在论文中说明这些数据质量问题并排除这些合约

---

### 4. corr = NaN

**原因**: 上述问题的综合结果

| 根本原因 | 导致 NaN 的机制 |
|---------|---------------|
| RAD_ZERO_FIRST_HALF | 一方数据全 0，方差为 0，相关性未定义 |
| NO_ROLLS_IN_TEST | 测试期内没有换月，ratio 始终为 1.0，与 CLC RAD 的累积 ratio 不匹配 |
| scale = 0 | first_rad = 0，对齐因子为 0 |

---

## 正常工作的合约

**成功标准**:
- ✅ 有 NON 文件
- ✅ 有 rollover 文件
- ✅ 2011 年后有换月记录
- ✅ CLC RAD 文件数据有效（非 0）
- ✅ 相关性 ≥ 0.99

**示例**（正常合约）:
```
ES: 67 rolls, corr=1.000000, scale=1.000000 ✓
DJ: 17 rolls, corr=0.999997, scale=0.916231 ✓
ND: 11 rolls, corr=0.999999, scale=0.970038 ✓
YM: 29 rolls, corr=0.999988, scale=0.971803 ✓
```

---

## 建议下一步

1. **数据质量审查**: 标记所有 RAD 文件有 0 值问题的合约，考虑替换数据源

2. **测试期调整**: 
   - 当前：2011-2019（9 年）
   - 问题：43 个合约换月数据在 2011 年前结束
   - 建议：改为 2000-2019 或按合约可用数据动态调整

3. **回测验证**: 
   - 对"正常"合约运行完整回测
   - 对比 Sharpe/Sortino 指标
   - 验证 RAD_v2 是否能复现论文结果

4. **文档说明**: 
   - 在论文中说明数据质量问题
   - 列出被排除的合约及原因
   - 说明测试期选择的原因

---

## 已修复的问题

### ✅ ASC 文件读取乱码

**问题**: 运行 `extract_rollover_from_asc.py` 时出现 `\x00\` 乱码

**原因**: ASC 文件是二进制格式，包含空字节

**修复**: 使用二进制模式读取
```python
# 修复前
with open(asc_file, 'r') as f:  # 文本模式

# 修复后
with open(asc_file, 'rb') as f:  # 二进制模式
    line = raw_line.decode('ascii', errors='ignore').strip()
```

### ✅ 硬编码路径

**问题**: 脚本中使用绝对路径，无法从其他目录运行

**修复**: 使用 `PROJECT_ROOT = Path(__file__).resolve().parent.parent`

### ✅ 文件夹结构混乱

**问题**: 验证报告、中间结果与原始数据混在 `data/CLC/`

**修复**: 
- `data/CLC/`: 只保留数据文件（RAD, NON, RAD_v2, rollovers）
- `tests/results/`: 存放测试输出、对比报告、验证结果
