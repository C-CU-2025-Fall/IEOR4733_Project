#!/usr/bin/env python3
"""
验证 A2C 模块导入
"""

import sys
from pathlib import Path

# 添加项目根目录到搜索路径
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("=" * 80)
print("🔍 验证 A2C 模块导入")
print("=" * 80)

try:
    print("\n1️⃣  导入核心 A2C 组件...")
    from src.core.models.a2c import (
        StackedLSTMBackbone,
        ActorContinuous,
        CriticValue,
        PaperTradingEnv,
        load_actor_from_checkpoint,
    )
    print("   ✅ 核心组件导入成功")
    
    print("\n2️⃣  检查模型权重文件...")
    a2c_weights = list(Path("rl_models").glob("a2c_*.pt"))
    print(f"   ✅ 找到 {len(a2c_weights)} 个 A2C 权重文件")
    for w in sorted(a2c_weights)[:5]:
        print(f"      • {w.name}")
    if len(a2c_weights) > 5:
        print(f"      ... 等等 ({len(a2c_weights)-5} 个)")
    
    print("\n3️⃣  检查 A2C 评估结果...")
    a2c_results = Path("rl_models/a2c_results_wide改.csv")
    if a2c_results.exists():
        import pandas as pd
        df = pd.read_csv(a2c_results, index_col=0)
        print(f"   ✅ A2C 结果 CSV 已加载 ({len(df)} 行)")
        print(f"      列: {', '.join(df.columns.tolist())}")
    else:
        print(f"   ⚠️  A2C 结果 CSV 不存在")
    
    print("\n" + "=" * 80)
    print("✅ A2C 模块验证完成！所有组件就位。")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
