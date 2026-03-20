#!/usr/bin/env python3
"""
系统资源测试 - 测试 CPU/内存/GPU 并行能力
"""

import os
import time
import numpy as np
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import torch

print("=" * 80)
print("🔍 系统资源测试")
print("=" * 80)

# 基本系统信息
print(f"\n【系统信息】")
print(f"CPU 核心数: {mp.cpu_count()}")
print(f"CPU 逻辑核心: {os.cpu_count()}")
print(f"总内存: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"可用内存: {psutil.virtual_memory().available / 1024**3:.1f} GB")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA 版本: {torch.version.cuda}")

print(f"\n{'='*80}")
print("【CPU 并行测试】")
print(f"{'='*80}")

def cpu_task(n):
    """CPU 密集型任务"""
    start = time.time()
    # 矩阵乘法
    a = np.random.randn(500, 500)
    b = np.random.randn(500, 500)
    for _ in range(100):
        c = np.dot(a, b)
    return time.time() - start

# 串行测试
print(f"\n串行测试 (4个任务)...")
start = time.time()
for i in range(4):
    t = cpu_task(i)
print(f"串行耗时: {time.time() - start:.2f}s")

# 多进程测试
print(f"\n多进程测试 (4个任务, {min(4, mp.cpu_count())} workers)...")
start = time.time()
with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
    results = list(executor.map(cpu_task, range(4)))
print(f"多进程耗时: {time.time() - start:.2f}s")
print(f"加速比: {(time.time() - start) / (sum(results)):.1f}x")

print(f"\n{'='*80}")
print("【内存带宽测试】")
print(f"{'='*80}")

# 大数组复制
print(f"\n测试 1GB 内存复制...")
arr = np.random.randn(128 * 1024 * 1024)  # ~1GB
start = time.time()
arr_copy = arr.copy()
elapsed = time.time() - start
bandwidth = 1.0 / elapsed  # GB/s
print(f"耗时: {elapsed:.3f}s")
print(f"带宽: {bandwidth:.2f} GB/s")

print(f"\n{'='*80}")
print("【GPU 测试】")
print(f"{'='*80}")

if torch.cuda.is_available():
    # GPU 计算测试
    print(f"\nGPU 矩阵乘法测试 (4096x4096)...")
    
    # CPU
    a_cpu = torch.randn(4096, 4096)
    b_cpu = torch.randn(4096, 4096)
    start = time.time()
    for _ in range(10):
        c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"CPU 耗时: {cpu_time:.3f}s")
    
    # GPU
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU 耗时: {gpu_time:.3f}s")
    print(f"GPU 加速比: {cpu_time/gpu_time:.1f}x")
    
    # GPU 显存使用
    print(f"\nGPU 显存使用:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
else:
    print("GPU 不可用")

print(f"\n{'='*80}")
print("【资源监控建议】")
print(f"{'='*80}")

cpu_count = mp.cpu_count()
mem_available = psutil.virtual_memory().available / 1024**3

print(f"\n推荐并行配置:")
print(f"  CPU workers: {min(cpu_count - 1, 4)} (留1核给系统)")
print(f"  每进程内存: {mem_available / cpu_count:.1f} GB")

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU batch size: {int(gpu_mem * 1024 / 4)} (每GB约256)")

print(f"\n{'='*80}")
print("✅ 资源测试完成!")
print(f"{'='*80}")
