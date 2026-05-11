"""
Trading Strategy Simulation & Analysis Platform
Main Streamlit application for interactive backtesting and risk analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules
from config import ASSET_CLASSES, EXCLUDED_CONTRACTS, SOURCE_OVERRIDES
from data_loader import load_clc_full
from metrics import compute_metrics, max_drawdown_from_path, cagr_from_path
from baseline_run import load_contracts, compute_contract_returns
from strategies import compute_long_only, compute_signr, compute_macd

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.title("🎯 模拟配置")

# Test period selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("开始日期", datetime(2011, 1, 1))
with col2:
    end_date = st.date_input("结束日期", datetime(2019, 12, 31))

# Asset class selection
selected_assets = st.sidebar.multiselect(
    "选择资产类别",
    options=list(ASSET_CLASSES.keys()),
    default=list(ASSET_CLASSES.keys())
)

# Strategy selection
strategies_available = ["Long Only", "Sign(R)", "MACD"]
selected_strategies = st.sidebar.multiselect(
    "选择交易策略",
    options=strategies_available,
    default=["Long Only", "Sign(R)"]
)

# Risk target (volatility)
sigma_target = st.sidebar.slider(
    "目标波动率 (σ)",
    min_value=0.03,
    max_value=0.15,
    value=0.063,
    step=0.01,
    help="日度目标波动率，用于头寸调整"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_strategy_data(asset_class, strategy, start_date, end_date, sigma_tgt):
    """Load strategy returns for given parameters."""
    try:
        raw = load_contracts(
            asset_class, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            EXCLUDED_CONTRACTS,
            SOURCE_OVERRIDES
        )
        if not raw:
            return None
        
        series_list = []
        for rd in raw:
            Rt = compute_contract_returns(rd, strategy, sigma_tgt)
            start, t1, dates = rd['start'], rd['t1'], rd['dates']
            slc = Rt[start:t1 + 1]
            aligned_dates = pd.to_datetime(dates[:len(slc)])
            series_list.append(pd.Series(slc[:len(aligned_dates)], index=aligned_dates))
        
        if not series_list:
            return None
        
        port_series = pd.DataFrame(series_list).T.sort_index().mean(axis=1)
        return port_series
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        return None

def compute_cumulative_wealth(returns_series):
    """Convert returns to cumulative wealth."""
    if returns_series is None or len(returns_series) == 0:
        return None
    wealth = 1.0 + np.cumsum(returns_series.values)
    return pd.Series(wealth, index=returns_series.index)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 交易策略模拟与分析平台")

# Tab structure
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 策略对比",
    "💹 性能指标",
    "📊 风险分析",
    "🔧 敏感性分析",
    "📥 数据管道"
])

# ============================================================================
# TAB 1: Strategy Comparison
# ============================================================================
with tab1:
    st.header("策略对比分析")
    st.markdown(f"**回测期间**: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    if selected_assets and selected_strategies:
        # Create tabs for each asset class
        asset_tabs = st.tabs([f"{asset}" for asset in selected_assets])
        
        for asset_idx, asset in enumerate(selected_assets):
            with asset_tabs[asset_idx]:
                fig, ax = plt.subplots(figsize=(14, 6))
                colors = {"Long Only": "#2E86AB", "Sign(R)": "#A23B72", "MACD": "#F18F01"}
                
                data_loaded = False
                for strategy in selected_strategies:
                    returns = load_strategy_data(asset, strategy, start_date, end_date, sigma_target)
                    if returns is not None and len(returns) > 0:
                        wealth = compute_cumulative_wealth(returns)
                        if wealth is not None:
                            ax.plot(wealth.index, wealth.values - 1.0, 
                                   label=strategy, linewidth=2, color=colors.get(strategy, "#888"))
                            data_loaded = True
                
                if data_loaded:
                    ax.set_xlabel("时间", fontsize=11, fontweight='bold')
                    ax.set_ylabel("累积收益率", fontsize=11, fontweight='bold')
                    ax.set_title(f"{asset} - 策略对比", fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=10)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning(f"⚠️ {asset} 无可用数据")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 2: Performance Metrics
# ============================================================================
with tab2:
    st.header("性能指标仪表板")
    
    if selected_assets and selected_strategies:
        metrics_data = []
        
        for asset in selected_assets:
            for strategy in selected_strategies:
                returns = load_strategy_data(asset, strategy, start_date, end_date, sigma_target)
                if returns is not None and len(returns) > 0:
                    wealth = compute_cumulative_wealth(returns)
                    if wealth is not None:
                        # Calculate metrics
                        total_return = (wealth.iloc[-1] - 1.0) * 100
                        annual_return = ((wealth.iloc[-1]) ** (252 / len(wealth)) - 1) * 100
                        annual_vol = returns.std() * np.sqrt(252) * 100
                        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                        max_dd = max_drawdown_from_path(wealth.values) * 100
                        
                        metrics_data.append({
                            "资产类别": asset,
                            "策略": strategy,
                            "总收益 (%)": f"{total_return:.2f}",
                            "年化收益 (%)": f"{annual_return:.2f}",
                            "年化波动 (%)": f"{annual_vol:.2f}",
                            "Sharpe比": f"{sharpe:.2f}",
                            "最大回撤 (%)": f"{max_dd:.2f}",
                        })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            
            # Download button
            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label="📥 下载指标CSV",
                data=csv,
                file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ 无性能数据")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 3: Risk Analysis
# ============================================================================
with tab3:
    st.header("风险指标分析")
    
    if selected_assets and selected_strategies:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("最大回撤对比")
            mdd_data = []
            
            for asset in selected_assets:
                for strategy in selected_strategies:
                    returns = load_strategy_data(asset, strategy, start_date, end_date, sigma_target)
                    if returns is not None and len(returns) > 0:
                        wealth = compute_cumulative_wealth(returns)
                        if wealth is not None:
                            mdd = max_drawdown_from_path(wealth.values) * 100
                            mdd_data.append({
                                "策略": f"{strategy}/{asset}",
                                "最大回撤 (%)": mdd
                            })
            
            if mdd_data:
                df_mdd = pd.DataFrame(mdd_data).sort_values("最大回撤 (%)")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(df_mdd["策略"], df_mdd["最大回撤 (%)"], color="#E74C3C")
                ax.set_xlabel("最大回撤 (%)", fontweight='bold')
                ax.set_title("策略最大回撤对比", fontweight='bold')
                st.pyplot(fig)
        
        with col2:
            st.subheader("波动率对比")
            vol_data = []
            
            for asset in selected_assets:
                for strategy in selected_strategies:
                    returns = load_strategy_data(asset, strategy, start_date, end_date, sigma_target)
                    if returns is not None and len(returns) > 0:
                        annual_vol = returns.std() * np.sqrt(252) * 100
                        vol_data.append({
                            "策略": f"{strategy}/{asset}",
                            "年化波动 (%)": annual_vol
                        })
            
            if vol_data:
                df_vol = pd.DataFrame(vol_data).sort_values("年化波动 (%)")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(df_vol["策略"], df_vol["年化波动 (%)"], color="#3498DB")
                ax.set_xlabel("年化波动 (%)", fontweight='bold')
                ax.set_title("策略波动率对比", fontweight='bold')
                st.pyplot(fig)
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 4: Sensitivity Analysis
# ============================================================================
with tab4:
    st.header("敏感性分析")
    st.markdown("分析目标波动率对策略性能的影响")
    
    if selected_assets and selected_strategies:
        # Select one asset and strategy for sensitivity analysis
        col1, col2 = st.columns(2)
        
        with col1:
            sensitivity_asset = st.selectbox("选择资产", selected_assets)
        
        with col2:
            sensitivity_strategy = st.selectbox("选择策略", selected_strategies)
        
        # Range of sigma targets
        sigma_range = np.arange(0.03, 0.16, 0.01)
        sensitivity_results = []
        
        progress_bar = st.progress(0)
        for idx, sigma_val in enumerate(sigma_range):
            returns = load_strategy_data(
                sensitivity_asset, 
                sensitivity_strategy, 
                start_date, 
                end_date, 
                sigma_val
            )
            
            if returns is not None and len(returns) > 0:
                wealth = compute_cumulative_wealth(returns)
                if wealth is not None:
                    annual_return = ((wealth.iloc[-1]) ** (252 / len(wealth)) - 1) * 100
                    annual_vol = returns.std() * np.sqrt(252) * 100
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    sensitivity_results.append({
                        "σ_target": f"{sigma_val:.3f}",
                        "年化收益 (%)": annual_return,
                        "Sharpe比": sharpe,
                        "年化波动 (%)": annual_vol
                    })
            
            progress_bar.progress((idx + 1) / len(sigma_range))
        
        if sensitivity_results:
            df_sens = pd.DataFrame(sensitivity_results)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Plot 1: Return vs Sigma
            axes[0].plot(df_sens.index, df_sens["年化收益 (%)"], marker='o', color="#2E86AB", linewidth=2)
            axes[0].set_xlabel("σ_target 指数", fontweight='bold')
            axes[0].set_ylabel("年化收益 (%)", fontweight='bold')
            axes[0].set_title("收益敏感性", fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Sharpe vs Sigma
            axes[1].plot(df_sens.index, df_sens["Sharpe比"], marker='s', color="#A23B72", linewidth=2)
            axes[1].set_xlabel("σ_target 指数", fontweight='bold')
            axes[1].set_ylabel("Sharpe比", fontweight='bold')
            axes[1].set_title("Sharpe比敏感性", fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Volatility vs Sigma
            axes[2].plot(df_sens.index, df_sens["年化波动 (%)"], marker='^', color="#F18F01", linewidth=2)
            axes[2].set_xlabel("σ_target 指数", fontweight='bold')
            axes[2].set_ylabel("年化波动 (%)", fontweight='bold')
            axes[2].set_title("波动率敏感性", fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(df_sens, use_container_width=True)
        else:
            st.warning("⚠️ 敏感性分析无数据")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 5: Data Pipeline
# ============================================================================
with tab5:
    st.header("📥 数据管道")
    st.markdown("""
    ### 数据清理流程
    
    1. **原始数据读取**: CLCData RAD格式
    2. **数据验证**: 
       - 检查价格有效性（Close > 0）
       - 移除缺失数据
       - 按日期排序
    3. **特征工程**:
       - 向前调整（Forward Adjusted）价格
       - 计算收益率
       - 头寸调整（根据目标波动率）
    4. **输出**: 清洁数据供策略使用
    
    ### 支持的数据格式
    - **输入**: CLCData RAD CSV (Date, Open, High, Low, Close, Volume, OI)
    - **输出**: 时间序列数据，已清洗和对齐
    
    ### 质量指标
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("数据周期", "2011-2019 (9 年)")
    with col2:
        st.metric("资产类别", "5 (大宗商品、股指、固定收益、外汇、组合)")
    with col3:
        st.metric("交易日数", "~2,300 天/资产")
    
    st.markdown("""
    ### 数据来源
    - 🗄️ **主数据源**: `data/CLC/` (CLCData原始格式)
    - 📊 **预处理**: `data_loader.py` 中的 `load_clc_full()` 函数
    - ✅ **验证**: 自动检查缺失值、异常值、数据对齐
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>🎓 IEOR 4733 期末项目 - 交易策略模拟平台</p>
    <p>数据驱动 | 可复现 | 低向前偏差</p>
</div>
""", unsafe_allow_html=True)
