# 📊 Trading Strategy Simulator - 使用指南

## 功能概述

这个 Streamlit 应用提供完整的交易策略回测、分析和风险管理能力，满足教师关于部署应用的要求。

### ✅ 核心功能

#### 1. **清洁数据管道** (TAB 5 - 数据管道)
- 从 `data/CLC/` 加载 CLCData RAD 格式
- 自动验证价格、移除缺失值
- 向前调整价格处理
- 支持多资产类别 (商品、股指、固定收益、外汇、组合)

#### 2. **回测引擎** (TAB 1 - 策略对比)
- 支持 3 种核心策略: Long Only, Sign(R), MACD
- 多资产类别并行计算
- 头寸自动调整 (根据目标波动率 σ)
- 可视化累积收益对比图表

#### 3. **交易成本建模** (集成到 `baseline_run.py`)
- 自动计算合约滚动成本
- 头寸调整相关交易成本
- 已纳入回测收益计算

#### 4. **性能仪表板** (TAB 2 - 性能指标)
- 关键指标计算和展示:
  - **总收益率**: 回测期间总体收益
  - **年化收益**: 几何平均年化收益
  - **年化波动**: 日度收益标准差年化
  - **Sharpe 比**: 风险调整后收益
  - **最大回撤**: 最坏的峰谷下跌

#### 5. **风险指标** (TAB 3 - 风险分析)
- 最大回撤 (MDD) 对比
- 波动率 (Volatility) 对比
- 可视化图表 (柱状图)

#### 6. **运行新模拟** (TAB 4 - 敏感性分析)
- 参数敏感性分析界面
- 调整目标波动率 σ ∈ [0.03, 0.15]
- 实时计算性能影响
- 三维分析图表 (收益 vs σ, Sharpe vs σ, 波动率 vs σ)

---

## 启动应用

### 方法 1: 使用启动脚本 (推荐)

```bash
cd /Users/ladymie/Documents/GitHub/IEOR4733_Project
chmod +x run_app.sh
./run_app.sh
```

访问: **http://localhost:8501**

### 方法 2: 手动启动

```bash
cd /Users/ladymie/Documents/GitHub/IEOR4733_Project
pip install -r streamlit_requirements.txt
streamlit run src/app/main.py
```

---

## 使用界面说明

### 左侧边栏 - 配置面板

| 参数 | 说明 | 默认值 |
|------|------|--------|
| **开始日期** | 回测起始日期 | 2011-01-01 |
| **结束日期** | 回测结束日期 | 2019-12-31 |
| **资产类别** | 多选要分析的资产 | 全选 (5 个) |
| **交易策略** | 多选要对比的策略 | Long Only, Sign(R) |
| **目标波动率 σ** | 头寸调整目标 | 0.063 |

### TAB 页说明

**1️⃣ 策略对比 - 📈 Cumulative Returns**
- 按资产类别分页展示
- 每个图表显示所选策略的累积收益曲线
- 绿色背景区域 = 策略盈利

**2️⃣ 性能指标 - 💹 Performance Dashboard**
- 表格格式展示所有指标
- 支持 CSV 下载
- 可在 Excel 中进一步处理

**3️⃣ 风险分析 - 📊 Risk Metrics**
- 左侧: 最大回撤柱状图 (风险指标)
- 右侧: 年化波动柱状图 (风险度量)
- 红色 = 较高风险, 蓝色 = 波动情况

**4️⃣ 敏感性分析 - 🔧 Parameter Sensitivity**
- 选择单一资产 + 单一策略
- 自动扫描 σ ∈ [0.03, 0.15]
- 三个图表:
  - 左: 年化收益 vs σ 趋势
  - 中: Sharpe 比 vs σ 曲线
  - 右: 年化波动 vs σ 曲线

**5️⃣ 数据管道 - 📥 Data Pipeline**
- 展示数据清洁流程
- 列出数据来源和格式
- 质量指标总结

---

## 数据要求

### 输入数据格式
```
data/CLC/filename.csv
Columns: Date, Open, High, Low, Close, Volume, OI
```

### 支持的资产类别
```python
ASSET_CLASSES = {
    'Commodity': {...},      # 5 个大宗商品合约
    'Equity Index': {...},   # 5 个股票指数合约
    'Fixed Income': {...},   # 3 个债券合约
    'Forex': {...},          # 3 个外汇对
    'All': {...}             # 10 个流动性最好的合约
}
```

---

## 配置和自定义

### 修改默认参数

编辑 `config.py`:
```python
BP = 20                          # 基点 (对于头寸调整)
TRADING_DAYS = 252              # 年化因子
SIGMA_TARGET = 0.063            # 默认目标波动率
```

### 修改策略参数

编辑 `strategies.py`:
```python
SIGN_LOOKBACK = 20              # Sign(R) 回溯期
MACD_FAST = 12                  # MACD 快线参数
MACD_SLOW = 26                  # MACD 慢线参数
MACD_SIGNAL = 9                 # MACD 信号线参数
```

---

## 示例工作流

### 场景 1: 对比策略性能
1. 左侧选择 **开始/结束日期**: 2011-01-01 ~ 2019-12-31
2. **资产类别**: 选择 "All"
3. **策略**: 选择 "Long Only" + "Sign(R)"
4. TAB 1 查看累积收益对比
5. TAB 2 查看性能指标表格
6. TAB 3 查看风险对比图表

### 场景 2: 分析风险特征
1. TAB 3 查看最大回撤
2. TAB 4 进行敏感性分析
3. 调整 σ 观察风险变化
4. 识别最优参数

### 场景 3: 生成报告
1. TAB 2 性能指标表
2. 点击 "📥 下载指标CSV"
3. 在 PowerPoint/Word 中粘贴

---

## 故障排查

### 问题 1: "加载数据失败"
**解决**: 确保 `data/CLC/` 目录存在且包含 CSV 文件
```bash
ls -la data/CLC/ | head
```

### 问题 2: 图表不显示
**解决**: 确保已安装所有依赖
```bash
pip install -r streamlit_requirements.txt
```

### 问题 3: 应用响应缓慢
**解决**: 减少选中的资产类别或使用更小的时间范围

---

## 性能说明

| 操作 | 耗时 | 缓存 |
|------|------|------|
| 加载单个策略 | ~2-3 秒 | ✅ |
| 绘制图表 | ~1 秒 | 否 |
| 敏感性分析 (14 点) | ~20-30 秒 | ❌ |
| 性能指标计算 | 实时 | ✅ |

---

## 部署到云服务

### Streamlit Cloud 部署
1. 将项目上传到 GitHub
2. 访问 [streamlit.io/cloud](https://streamlit.io/cloud)
3. 选择 "New app"
4. 选择仓库 → 分支 → `src/app/main.py`
5. 点击 "Deploy"

### Docker 部署
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r streamlit_requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "src/app/main.py"]
```

---

## 技术栈

- **前端**: Streamlit (Python 原生 Web 框架)
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **计算**: NumPy (矢量化运算)
- **缓存**: Streamlit @st.cache_data

---

## 相关文件

```
📁 项目根目录/
├── src/app/main.py              ← Streamlit 应用主文件
├── src/core/                    ← 核心模块
│   ├── data/loader.py           ← 数据加载
│   ├── strategies/base.py       ← 策略实现
│   ├── metrics/calculator.py    ← 性能计算
│   └── utils/indicators.py      ← 技术指标
├── data/CLC/                    ← 输入数据 (CLCData 格式)
├── baseline_run.py              ← 回测执行逻辑
├── config.py                    ← 全局配置
└── streamlit_requirements.txt    ← Python 依赖
```

---

**最后更新**: 2024-05
**作者**: IEOR 4733 Project Team
