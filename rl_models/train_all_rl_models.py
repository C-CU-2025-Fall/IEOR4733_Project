#!/usr/bin/env python3
"""
统一 RL 模型训练脚本

支持 DQN、PG、A2C 三种模型
- DQN: Fixed Q-targets + Double DQN
- PG: Policy Gradient (Monte Carlo)
- A2C: Advantage Actor-Critic (实时更新)

使用方法（从项目根目录运行）：
  python rl_models/train_all_rl_models.py              # 训练所有模型
  python rl_models/train_all_rl_models.py dqn         # 仅训练 DQN
  python rl_models/train_all_rl_models.py pg          # 仅训练 PG
  python rl_models/train_all_rl_models.py a2c         # 仅训练 A2C
"""

import sys
import time
import subprocess
from datetime import datetime
import os

# 模型脚本（相对路径）
MODELS = {
    'dqn': 'train_dqn_paper_aligned.py',
    'pg': 'train_pg_paper_aligned.py',
    'a2c': 'train_a2c_paper_aligned.py'
}

def main():
    print("="*80)
    print("🔥 深度强化学习模型统一训练")
    print("="*80)
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 确定要训练的模型
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
        if model_name not in MODELS:
            print(f"❌ 未知模型：{model_name}")
            print(f"   可选：{', '.join(MODELS.keys())}")
            sys.exit(1)
        models_to_train = [model_name]
    else:
        models_to_train = list(MODELS.keys())
    
    print(f"📊 待训练模型：{', '.join(models_to_train)}")
    print("="*80)
    
    start_time = time.time()
    results = {}
    
    # 获取当前脚本所在目录（rl_models/）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for model_name in models_to_train:
        script = MODELS[model_name]
        script_path = os.path.join(script_dir, script)
        
        print(f"\n🚀 开始训练 {model_name.upper()}...")
        print(f"   脚本：{script_path}")
        print("-"*80)
        
        try:
            result = subprocess.run(['python', script_path], check=True)
            results[model_name] = 'SUCCESS'
            print(f"✅ {model_name.upper()} 训练完成")
        except subprocess.CalledProcessError as e:
            results[model_name] = 'FAILED'
            print(f"❌ {model_name.upper()} 训练失败（错误代码：{e.returncode}）")
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "="*80)
    print("📊 训练总结")
    print("="*80)
    for model_name, status in results.items():
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"  {status_icon} {model_name.upper()}: {status}")
    print("="*80)
    print(f"总耗时：{elapsed:.1f} 分钟")
    print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 检查是否全部成功
    if all(status == "SUCCESS" for status in results.values()):
        print("\n🎉 所有模型训练完成！")
        return 0
    else:
        print("\n⚠️  部分模型训练失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
