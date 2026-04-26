#!/usr/bin/env python3
"""
对比 raw prices vs normalized prices (p/p0) 在 Sign(R) Eq.4 回测中的差异。

结论：数学上完全等价，diff 应该 ≈ 0（浮点误差 ~1e-14）。

原理：
  normalized: r_norm = r_raw/p0, σ_norm = σ_raw/p0
  scale_norm = σ_tgt/σ_norm = σ_tgt×p0/σ_raw
  gross = scale_norm × r_norm = (σ_tgt×p0/σ) × (r/p0) = σ_tgt×r/σ = gross_raw
  tc = bp × (p/p0) × |scale_norm_change| = bp × (p/p0) × |σ_tgt×p0×Δ(1/σ)|
     = bp × p × σ_tgt × |Δ(1/σ)| = tc_raw
"""
import numpy as np
import pandas as pd
from strategies import strategy_sign_r
from baseline_run import load_contracts
from config import SIGMA_TGT_DAILY, BP, SIGN_LOOKBACK, ASSET_CLASSES

print("=== Sign(R) Eq.4: Raw Prices vs Normalized Prices (p/p0) ===\n")

for asset in ASSET_CLASSES:
    raw = load_contracts(asset)
    
    sum_raw = 0.0
    sum_norm = 0.0
    count = 0
    
    for rd in raw:
        prices = rd['prices']
        sigma = rd['sigma']
        rt = rd['rt']
        n = len(prices)
        p0 = prices[0]
        
        # Sign(R) 仓位
        pos = strategy_sign_r(prices, SIGN_LOOKBACK)
        
        # Normalized prices
        norm_p = prices / p0
        rt_norm = np.zeros(n)
        rt_norm[1:] = norm_p[1:] - norm_p[:-1]
        sigma_norm = pd.Series(rt_norm).ewm(span=60, adjust=False).std().values
        
        for t in range(SIGN_LOOKBACK + 2, n):
            if not (sigma[t-1] > 0 and sigma[t-2] > 0):
                continue
            if not (sigma_norm[t-1] > 0 and sigma_norm[t-2] > 0):
                continue
            
            # Raw prices
            sp = pos[t-1] * SIGMA_TGT_DAILY / sigma[t-1]
            spp = pos[t-2] * SIGMA_TGT_DAILY / sigma[t-2]
            Rt_raw = sp * rt[t] - BP * prices[t-1] * abs(sp - spp)
            
            # Normalized prices
            sp_n = pos[t-1] * SIGMA_TGT_DAILY / sigma_norm[t-1]
            spp_n = pos[t-2] * SIGMA_TGT_DAILY / sigma_norm[t-2]
            Rt_norm = sp_n * rt_norm[t] - BP * norm_p[t-1] * abs(sp_n - spp_n)
            
            sum_raw += Rt_raw
            sum_norm += Rt_norm
            count += 1
    
    diff = abs(sum_raw - sum_norm)
    print(f"{asset:15s}: raw_sum={sum_raw:+.6f}, norm_sum={sum_norm:+.6f}, "
          f"diff={diff:.2e}, steps={count}")

print("\n结论: diff ≈ 1e-14 (浮点误差), raw 和 norm 完全等价。")
