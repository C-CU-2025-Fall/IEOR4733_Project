#!/usr/bin/env python3
"""
创建论文 vs 我们的结果对比图
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ========== 论文 Table 2 数据 (Volatility Scaled) ==========
paper_data = {
    'Commodity': {
        'Long':   {'E(R)': -0.710, 'Std(R)': 0.979, 'Sharpe': -0.726, 'Sortino': -1.177, 'MDD': 0.350, 'Calmar': -0.140},
        'Sign(R)': {'E(R)': 0.347, 'Std(R)': 0.980, 'Sharpe': 0.354, 'Sortino': 0.606, 'MDD': 0.116, 'Calmar': 0.119},
        'MACD':   {'E(R)': -0.171, 'Std(R)': 0.978, 'Sharpe': -0.175, 'Sortino': -0.293, 'MDD': 0.190, 'Calmar': -0.060},
        'DQN':    {'E(R)': 0.703, 'Std(R)': 0.973, 'Sharpe': 0.723, 'Sortino': 1.275, 'MDD': 0.066, 'Calmar': 0.501},
        'PG':     {'E(R)': 0.062, 'Std(R)': 0.982, 'Sharpe': 0.063, 'Sortino': 0.106, 'MDD': 0.039, 'Calmar': 0.023},
        'A2C':    {'E(R)': 0.223, 'Std(R)': 0.955, 'Sharpe': 0.234, 'Sortino': 0.399, 'MDD': 0.141, 'Calmar': 0.091},
    },
    'Equity Index': {
        'Long':   {'E(R)': 0.668, 'Std(R)': 0.970, 'Sharpe': 0.688, 'Sortino': 1.102, 'MDD': 0.132, 'Calmar': 0.509},
        'Sign(R)': {'E(R)': 0.228, 'Std(R)': 0.966, 'Sharpe': 0.236, 'Sortino': 0.374, 'MDD': 0.344, 'Calmar': 0.077},
        'MACD':   {'E(R)': 0.016, 'Std(R)': 0.962, 'Sharpe': 0.017, 'Sortino': 0.027, 'MDD': 0.311, 'Calmar': 0.006},
        'DQN':    {'E(R)': 0.629, 'Std(R)': 0.970, 'Sharpe': 0.648, 'Sortino': 1.038, 'MDD': 0.161, 'Calmar': 0.381},
        'PG':     {'E(R)': 0.432, 'Std(R)': 0.967, 'Sharpe': 0.447, 'Sortino': 0.714, 'MDD': 0.242, 'Calmar': 0.185},
        'A2C':    {'E(R)': 0.473, 'Std(R)': 0.929, 'Sharpe': 0.510, 'Sortino': 0.798, 'MDD': 0.124, 'Calmar': 0.328},
    },
    'Fixed Income': {
        'Long':   {'E(R)': 0.680, 'Std(R)': 0.975, 'Sharpe': 0.698, 'Sortino': 1.180, 'MDD': 0.061, 'Calmar': 0.444},
        'Sign(R)': {'E(R)': 0.214, 'Std(R)': 0.972, 'Sharpe': 0.221, 'Sortino': 0.363, 'MDD': 0.080, 'Calmar': 0.083},
        'MACD':   {'E(R)': 0.219, 'Std(R)': 0.967, 'Sharpe': 0.228, 'Sortino': 0.380, 'MDD': 0.065, 'Calmar': 0.123},
        'DQN':    {'E(R)': 0.908, 'Std(R)': 0.972, 'Sharpe': 0.935, 'Sortino': 1.617, 'MDD': 0.062, 'Calmar': 0.543},
        'PG':     {'E(R)': 0.705, 'Std(R)': 0.974, 'Sharpe': 0.724, 'Sortino': 1.225, 'MDD': 0.061, 'Calmar': 0.436},
        'A2C':    {'E(R)': 0.699, 'Std(R)': 0.979, 'Sharpe': 0.714, 'Sortino': 1.203, 'MDD': 0.067, 'Calmar': 0.408},
    },
    'FX': {
        'Long':   {'E(R)': -0.344, 'Std(R)': 0.973, 'Sharpe': -0.353, 'Sortino': -0.590, 'MDD': 0.423, 'Calmar': -0.097},
        'Sign(R)': {'E(R)': -0.297, 'Std(R)': 0.973, 'Sharpe': -0.306, 'Sortino': -0.502, 'MDD': 0.434, 'Calmar': -0.111},
        'MACD':   {'E(R)': 0.006, 'Std(R)': 0.970, 'Sharpe': 0.007, 'Sortino': 0.011, 'MDD': 0.329, 'Calmar': 0.002},
        'DQN':    {'E(R)': 0.528, 'Std(R)': 0.967, 'Sharpe': 0.546, 'Sortino': 0.955, 'MDD': 0.183, 'Calmar': 0.313},
        'PG':     {'E(R)': 0.248, 'Std(R)': 0.967, 'Sharpe': 0.257, 'Sortino': 0.438, 'MDD': 0.240, 'Calmar': 0.124},
        'A2C':    {'E(R)': 0.316, 'Std(R)': 0.963, 'Sharpe': 0.328, 'Sortino': 0.561, 'MDD': 0.165, 'Calmar': 0.201},
    },
    'All (Portfolio)': {
        'Long':   {'E(R)': 0.055, 'Std(R)': 0.975, 'Sharpe': 0.058, 'Sortino': 0.092, 'MDD': 0.071, 'Calmar': 0.013},
        'Sign(R)': {'E(R)': 0.429, 'Std(R)': 0.972, 'Sharpe': 0.441, 'Sortino': 0.737, 'MDD': 0.038, 'Calmar': 0.201},
        'MACD':   {'E(R)': 0.089, 'Std(R)': 0.978, 'Sharpe': 0.091, 'Sortino': 0.153, 'MDD': 0.008, 'Calmar': 0.035},
        'DQN':    {'E(R)': 1.258, 'Std(R)': 0.976, 'Sharpe': 1.288, 'Sortino': 2.220, 'MDD': 0.002, 'Calmar': 1.025},
        'PG':     {'E(R)': 0.740, 'Std(R)': 0.980, 'Sharpe': 0.754, 'Sortino': 1.247, 'MDD': 0.012, 'Calmar': 0.480},
        'A2C':    {'E(R)': 1.024, 'Std(R)': 0.975, 'Sharpe': 1.050, 'Sortino': 1.785, 'MDD': 0.007, 'Calmar': 0.685},
    },
}

# ========== 我们的结果 ==========
our_data = {
    'ES=F': {
        'Long':   {'E(R)': 0.138, 'Std(R)': 0.130, 'Sharpe': 1.063, 'Sortino': 1.204, 'MDD': -0.204, 'Calmar': 0.677},
        'MACD':   {'E(R)': 0.271, 'Std(R)': 0.129, 'Sharpe': 2.098, 'Sortino': 3.004, 'MDD': -0.069, 'Calmar': 3.953},
        'DQN':    {'E(R)': 0.692, 'Std(R)': 0.116, 'Sharpe': 5.939, 'Sortino': 9.592, 'MDD': -0.062, 'Calmar': 11.095},
        'PPO':    {'E(R)': 0.623, 'Std(R)': 0.114, 'Sharpe': 5.443, 'Sortino': 14.293, 'MDD': -0.040, 'Calmar': 15.599},
        'A2C':    {'E(R)': -0.143, 'Std(R)': 0.130, 'Sharpe': -1.100, 'Sortino': -1.494, 'MDD': -0.387, 'Calmar': -0.369},
    },
    'CL=F': {
        'Long':   {'E(R)': 0.112, 'Std(R)': 0.309, 'Sharpe': 0.362, 'Sortino': 0.497, 'MDD': -0.443, 'Calmar': 0.252},
        'MACD':   {'E(R)': 0.767, 'Std(R)': 0.305, 'Sharpe': 2.513, 'Sortino': 4.144, 'MDD': -0.132, 'Calmar': 5.825},
        'DQN':    {'E(R)': 2.489, 'Std(R)': 0.256, 'Sharpe': 9.735, 'Sortino': 16.734, 'MDD': -0.113, 'Calmar': 21.959},
        'PPO':    {'E(R)': 3.254, 'Std(R)': 0.227, 'Sharpe': 14.349, 'Sortino': 141.780, 'MDD': -0.013, 'Calmar': 249.018},
        'A2C':    {'E(R)': -0.317, 'Std(R)': 0.308, 'Sharpe': -1.029, 'Sortino': -1.406, 'MDD': -0.719, 'Calmar': -0.441},
    },
    'GC=F': {
        'Long':   {'E(R)': 0.064, 'Std(R)': 0.112, 'Sharpe': 0.572, 'Sortino': 0.822, 'MDD': -0.137, 'Calmar': 0.466},
        'MACD':   {'E(R)': 0.275, 'Std(R)': 0.110, 'Sharpe': 2.490, 'Sortino': 4.240, 'MDD': -0.052, 'Calmar': 5.263},
        'DQN':    {'E(R)': 0.601, 'Std(R)': 0.091, 'Sharpe': 6.622, 'Sortino': 13.978, 'MDD': -0.020, 'Calmar': 30.378},
        'PPO':    {'E(R)': 0.901, 'Std(R)': 0.084, 'Sharpe': 10.669, 'Sortino': 56.771, 'MDD': -0.008, 'Calmar': 113.106},
        'A2C':    {'E(R)': -0.109, 'Std(R)': 0.111, 'Sharpe': -0.983, 'Sortino': -1.390, 'MDD': -0.369, 'Calmar': -0.296},
    },
}

# 颜色定义
paper_colors = {'Long': '#1f77b4', 'Sign(R)': '#ff7f0e', 'MACD': '#2ca02c', 
                'DQN': '#d62728', 'PG': '#9467bd', 'A2C': '#8c564b'}
our_colors = {'Long': '#1f77b4', 'MACD': '#2ca02c', 
              'DQN': '#d62728', 'PPO': '#9467bd', 'A2C': '#8c564b'}

# ========== 图1: 论文 Table 2 ==========
fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Paper Results (Table 2: Volatility Scaled)\n"Deep Reinforcement Learning for Trading" (Zhang et al., 2019)', 
              fontsize=14, fontweight='bold')

asset_classes = list(paper_data.keys())
paper_strategies = ['Long', 'Sign(R)', 'MACD', 'DQN', 'PG', 'A2C']

for idx, asset in enumerate(asset_classes):
    ax = axes1[idx // 3, idx % 3]
    data = paper_data[asset]
    
    x = np.arange(len(paper_strategies))
    sharpe_vals = [data[s]['Sharpe'] for s in paper_strategies]
    colors = [paper_colors[s] for s in paper_strategies]
    
    bars = ax.bar(x, sharpe_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title(asset, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(paper_strategies, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Sharpe Ratio')
    
    # 添加数值标签
    for bar, val in zip(bars, sharpe_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + (0.05 if val > 0 else -0.15),
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', 
                fontsize=8, fontweight='bold')

# 删除多余的子图
axes1[1, 2].axis('off')
plt.tight_layout()
plt.savefig('paper_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: paper_results.png")

# ========== 图2: 我们的结果 ==========
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle('Our Results (Paper-Aligned Implementation)', fontsize=14, fontweight='bold')

contracts = list(our_data.keys())
our_strategies = ['Long', 'MACD', 'DQN', 'PPO', 'A2C']

for idx, contract in enumerate(contracts):
    ax = axes2[idx]
    data = our_data[contract]
    
    x = np.arange(len(our_strategies))
    sharpe_vals = [data[s]['Sharpe'] for s in our_strategies]
    colors = [our_colors[s] for s in our_strategies]
    
    bars = ax.bar(x, sharpe_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title(contract, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(our_strategies, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Sharpe Ratio')
    
    # 添加数值标签
    for bar, val in zip(bars, sharpe_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height + (0.2 if val > 0 else -0.5),
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', 
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('our_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: our_results.png")

# ========== 图3: 对比汇总 ==========
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Paper vs Our Results: Sharpe Ratio Comparison', fontsize=14, fontweight='bold')

# 左图: 论文 "All (Portfolio)" 结果
ax1 = axes3[0]
paper_all = paper_data['All (Portfolio)']
paper_strats = ['Long', 'Sign(R)', 'MACD', 'DQN', 'PG', 'A2C']
paper_sharpes = [paper_all[s]['Sharpe'] for s in paper_strats]
bars1 = ax1.bar(paper_strats, paper_sharpes, color=[paper_colors[s] for s in paper_strats], 
                alpha=0.8, edgecolor='black')
ax1.set_title('Paper: All Contracts Portfolio', fontsize=12, fontweight='bold')
ax1.set_ylabel('Sharpe Ratio')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax1.set_ylim(-0.5, 1.5)
for bar, val in zip(bars1, paper_sharpes):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', fontsize=10)

# 右图: 我们的平均结果
ax2 = axes3[1]
our_avg = {
    'Long': np.mean([our_data[c]['Long']['Sharpe'] for c in contracts]),
    'MACD': np.mean([our_data[c]['MACD']['Sharpe'] for c in contracts]),
    'DQN': np.mean([our_data[c]['DQN']['Sharpe'] for c in contracts]),
    'PPO': np.mean([our_data[c]['PPO']['Sharpe'] for c in contracts]),
    'A2C': np.mean([our_data[c]['A2C']['Sharpe'] for c in contracts]),
}
our_strats = list(our_avg.keys())
our_sharpes = list(our_avg.values())
bars2 = ax2.bar(our_strats, our_sharpes, color=[our_colors[s] for s in our_strats], 
                alpha=0.8, edgecolor='black')
ax2.set_title('Our Results: Average (ES=F, CL=F, GC=F)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sharpe Ratio')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
for bar, val in zip(bars2, our_sharpes):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
             f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved: comparison.png")

# ========== 打印详细对比表 ==========
print("\n" + "="*80)
print("📊 详细对比表 (Sharpe Ratio)")
print("="*80)

print("\n【论文结果 - All Contracts Portfolio】")
print(f"{'Strategy':<10} {'Sharpe':>10} {'Sortino':>10} {'MDD':>10} {'Calmar':>10}")
print("-" * 50)
for strat in paper_strats:
    d = paper_all[strat]
    print(f"{strat:<10} {d['Sharpe']:>10.3f} {d['Sortino']:>10.3f} {d['MDD']:>10.3f} {d['Calmar']:>10.3f}")

print("\n【我们的结果 - 平均值】")
print(f"{'Strategy':<10} {'Sharpe':>10} {'Sortino':>10} {'MDD':>10} {'Calmar':>10}")
print("-" * 50)
for strat in our_strats:
    avg_sharpe = our_avg[strat]
    avg_sortino = np.mean([our_data[c][strat]['Sortino'] for c in contracts])
    avg_mdd = np.mean([our_data[c][strat]['MDD'] for c in contracts])
    avg_calmar = np.mean([our_data[c][strat]['Calmar'] for c in contracts])
    print(f"{strat:<10} {avg_sharpe:>10.3f} {avg_sortino:>10.3f} {avg_mdd:>10.3f} {avg_calmar:>10.3f}")

print("\n" + "="*80)
print("📈 关键发现")
print("="*80)
print(f"论文最佳 DQN Sharpe:  1.288")
print(f"我们最佳 PPO Sharpe: {our_avg['PPO']:.3f} (提升 {((our_avg['PPO']/1.288)-1)*100:.1f}%)")
print(f"我们最佳 DQN Sharpe: {our_avg['DQN']:.3f} (提升 {((our_avg['DQN']/1.288)-1)*100:.1f}%)")

print("\n✅ 所有图片已生成！")
print("  - paper_results.png  (论文 Table 2)")
print("  - our_results.png    (我们的结果)")
print("  - comparison.png     (对比图)")
