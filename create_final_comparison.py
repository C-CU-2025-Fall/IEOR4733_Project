#!/usr/bin/env python3
"""
创建三栏对比图: 论文 vs MLP vs LSTM
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 数据 ==========

# 论文 Table 2 (All Portfolio)
paper_strategies = ['Long', 'Sign(R)', 'MACD', 'DQN', 'PG', 'A2C']
paper_sharpes = [0.058, 0.441, 0.091, 1.288, 0.754, 1.050]

# MLP结果 (平均)
mlp_strategies = ['Long', 'MACD', 'DQN', 'PPO', 'A2C']
mlp_sharpes = [0.666, 2.367, 7.432, 10.154, -1.037]

# LSTM结果 (平均)
lstm_strategies = ['Long', 'MACD', 'DQN', 'PPO', 'A2C']
lstm_sharpes = [0.533, 0.690, -1.355, -0.175, -0.227]

# 颜色
colors_paper = {'Long': '#1f77b4', 'Sign(R)': '#ff7f0e', 'MACD': '#2ca02c', 
                'DQN': '#d62728', 'PG': '#9467bd', 'A2C': '#8c564b'}
colors_ours = {'Long': '#1f77b4', 'MACD': '#2ca02c', 
               'DQN': '#d62728', 'PPO': '#9467bd', 'A2C': '#8c564b'}

# ========== 创建图表 ==========
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Paper vs MLP vs LSTM: Sharpe Ratio Comparison', fontsize=14, fontweight='bold')

# 左图: 论文
ax1 = axes[0]
bars1 = ax1.bar(paper_strategies, paper_sharpes, 
                color=[colors_paper[s] for s in paper_strategies],
                alpha=0.8, edgecolor='black')
ax1.set_title('Paper (Zhang et al. 2019)\nAll Portfolio', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sharpe Ratio')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax1.set_ylim(-1.5, 2)
for bar, val in zip(bars1, paper_sharpes):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', fontsize=9)

# 中图: MLP
ax2 = axes[1]
bars2 = ax2.bar(mlp_strategies, mlp_sharpes,
                 color=[colors_ours[s] for s in mlp_strategies],
                 alpha=0.8, edgecolor='black')
ax2.set_title('Our Implementation (MLP)\n[256, 256, 256]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Sharpe Ratio')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax2.set_ylim(-1.5, 12)
for bar, val in zip(bars2, mlp_sharpes):
    ax2.text(bar.get_x() + bar.get_width()/2., 
             bar.get_height() + (0.2 if val > 0 else -0.3),
             f'{val:.2f}', ha='center', fontsize=9, 
             fontweight='bold' if val > 0 else 'normal')

# 右图: LSTM
ax3 = axes[2]
bars3 = ax3.bar(lstm_strategies, lstm_sharpes,
                 color=[colors_ours[s] for s in lstm_strategies],
                 alpha=0.8, edgecolor='black')
ax3.set_title('Our Implementation (LSTM)\n[64, 32] (Paper Config)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Sharpe Ratio')
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax3.set_ylim(-1.5, 2)
for bar, val in zip(bars3, lstm_sharpes):
    ax3.text(bar.get_x() + bar.get_width()/2., 
             bar.get_height() + (0.05 if val > 0 else -0.1),
             f'{val:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('comparison_all.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: comparison_all.png")

# ========== 打印汇总 ==========
print("\n" + "="*80)
print("📊 完整对比汇总 (Sharpe Ratio)")
print("="*80)

print(f"\n{'Strategy':<10} {'Paper':>10} {'MLP':>10} {'LSTM':>10}")
print("-" * 45)

all_strategies = ['Long', 'MACD', 'DQN', 'A2C']
for strat in all_strategies:
    paper_val = paper_sharpes[paper_strategies.index(strat)] if strat in paper_strategies else '-'
    mlp_val = mlp_sharpes[mlp_strategies.index(strat)] if strat in mlp_strategies else '-'
    lstm_val = lstm_sharpes[lstm_strategies.index(strat)] if strat in lstm_strategies else '-'
    
    paper_str = f'{paper_val:.3f}' if isinstance(paper_val, float) else str(paper_val)
    mlp_str = f'{mlp_val:.3f}' if isinstance(mlp_val, float) else str(mlp_val)
    lstm_str = f'{lstm_val:.3f}' if isinstance(lstm_val, float) else str(lstm_val)
    
    print(f"{strat:<10} {paper_str:>10} {mlp_str:>10} {lstm_str:>10}")

# PPO
print(f"{'PPO':<10} {'-':>10} {mlp_sharpes[mlp_strategies.index('PPO')]:>10.3f} {lstm_sharpes[lstm_strategies.index('PPO')]:>10.3f}")

print("\n" + "="*80)
print("🔍 关键发现")
print("="*80)
print("1. MLP (256x3) 表现最佳: DQN 7.43, PPO 10.15")
print("2. LSTM (64,32) 表现不如 MLP: 可能需要更多训练数据/调参")
print("3. 论文 DQN: 1.29, 我们 MLP DQN: 7.43 (提升 477%)")
print("4. A2C 在所有版本中表现不稳定")
