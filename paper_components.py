#!/usr/bin/env python3
"""
论文核心组件模块
包含: 
1. Differential Sharpe Ratio 奖励函数
2. 多时间尺度状态空间
3. Volatility Scaling
"""

import numpy as np
from typing import List, Tuple, Optional

# =============================================================================
# 1. Differential Sharpe Ratio 奖励函数
# =============================================================================

class DifferentialSharpeRatio:
    """
    论文 Equation 7-8: Differential Sharpe Ratio
    
    论文原文:
    "We use the differential Sharpe ratio (Moody et al., 1998) as the reward function,
    which allows the agent to optimise the risk-adjusted returns directly."
    
    公式:
    ΔSharpe_t = (R_t * Sharpe_{t-1} - 0.5 * R_t^2) / (t * σ_t)
    
    其中:
    - R_t: 时刻 t 的组合收益
    - Sharpe_{t-1}: 上一时刻的 Sharpe ratio
    - σ_t: 到时刻 t 的收益标准差
    
    递推公式 (使用指数移动平均):
    A_t = A_{t-1} + η * (R_t - A_{t-1})
    B_t = B_{t-1} + η * (R_t^2 - B_{t-1})
    
    Sharpe_t = A_t / sqrt(B_t - A_t^2)
    
    ΔSharpe_t = Sharpe_t - Sharpe_{t-1}
    """
    
    def __init__(self, eta: float = 0.01):
        """
        Args:
            eta: 学习率，控制历史数据的衰减速度
                 论文默认值通常在 0.01-0.1 之间
        """
        self.eta = eta
        self.reset()
    
    def reset(self):
        """重置所有状态"""
        self.t = 0
        self.A_t = 0.0  # 收益均值 (一阶矩)
        self.B_t = 0.0  # 收益平方均值 (二阶矩)
        self.Sharpe_t = 0.0  # 当前 Sharpe
        self.returns_history = []
    
    def update(self, R_t: float) -> float:
        """
        更新并返回 DSR 奖励
        
        Args:
            R_t: 当前时刻的组合收益
            
        Returns:
            DSR: Differential Sharpe Ratio (作为奖励信号)
        """
        self.t += 1
        self.returns_history.append(R_t)
        
        # 指数移动平均更新
        delta_A = R_t - self.A_t
        delta_B = R_t**2 - self.B_t
        
        A_t_new = self.A_t + self.eta * delta_A
        B_t_new = self.B_t + self.eta * delta_B
        
        # 计算 Sharpe ratio
        variance = B_t_new - A_t_new**2
        if variance > 1e-10:  # 避免数值问题
            Sharpe_new = A_t_new / np.sqrt(variance)
        else:
            Sharpe_new = 0.0
        
        # 计算 DSR
        DSR = Sharpe_new - self.Sharpe_t
        
        # 更新状态
        self.A_t = A_t_new
        self.B_t = B_t_new
        self.Sharpe_t = Sharpe_new
        
        return DSR
    
    def get_sharpe(self) -> float:
        """获取当前 Sharpe ratio"""
        return self.Sharpe_t
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        if len(self.returns_history) == 0:
            return {}
        
        returns = np.array(self.returns_history)
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe': self.Sharpe_t,
            'A_t': self.A_t,
            'B_t': self.B_t,
            't': self.t
        }


# =============================================================================
# 2. 多时间尺度状态空间
# =============================================================================

class MultiTimeScaleState:
    """
    论文 Section 3.1: State Space
    
    论文原文:
    "We adopt time series momentum features along with technical indicators 
    to represent state space."
    
    状态空间包含:
    1. 多时间尺度动量特征: [r_5, r_10, r_25, r_50, r_100, r_200]
    2. 技术指标: MACD, RSI, Bollinger Bands, ATR
    3. 价格位置
    4. 波动率
    5. 历史统计
    """
    
    # 论文中的动量窗口
    MOMENTUM_WINDOWS = [5, 10, 25, 50, 100, 200]
    
    # 技术指标参数
    MACD_PARAMS = {'fast': 12, 'slow': 26, 'signal': 9}
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2
    ATR_PERIOD = 14
    
    def __init__(self, lookback: int = 200):
        """
        Args:
            lookback: 最大回看窗口
        """
        self.lookback = lookback
    
    def compute(self, prices: np.ndarray, returns: np.ndarray, 
                t: int) -> np.ndarray:
        """
        计算时刻 t 的状态向量
        
        Args:
            prices: 价格序列
            returns: 收益率序列
            t: 当前时刻索引
            
        Returns:
            state: 状态向量
        """
        features = []
        
        # 1. 多时间尺度动量 (论文核心特征)
        momentum_features = self._compute_momentum(returns, t)
        features.extend(momentum_features)
        
        # 2. MACD 信号 (论文 Table 2)
        macd_features = self._compute_macd(prices, t)
        features.extend(macd_features)
        
        # 3. RSI (技术指标)
        rsi = self._compute_rsi(prices, t)
        features.append(rsi)
        
        # 4. Bollinger Bands 位置
        bb_position = self._compute_bb_position(prices, t)
        features.append(bb_position)
        
        # 5. ATR (波动率指标)
        atr = self._compute_atr(prices, t)
        features.append(atr)
        
        # 6. 价格动量位置 (0-1)
        price_position = self._compute_price_position(prices, t)
        features.append(price_position)
        
        # 7. 波动率 (20日)
        volatility = self._compute_volatility(returns, t)
        features.append(volatility)
        
        # 8. 当前收益
        features.append(returns[t] if t < len(returns) else 0)
        
        # 9. 历史夏比 (20日滚动)
        hist_sharpe = self._compute_rolling_sharpe(returns, t)
        features.append(hist_sharpe)
        
        # 10. 收益偏度 (20日)
        skewness = self._compute_skewness(returns, t)
        features.append(skewness)
        
        # 转换为 numpy 数组并归一化
        state = np.array(features, dtype=np.float32)
        
        # 裁剪到合理范围
        state = np.clip(state, -10, 10)
        
        # 处理 NaN
        state = np.nan_to_num(state, nan=0.0)
        
        return state
    
    def _compute_momentum(self, returns: np.ndarray, t: int) -> List[float]:
        """计算多时间尺度动量"""
        momentum = []
        for window in self.MOMENTUM_WINDOWS:
            if t >= window:
                # 累积收益
                mom = np.sum(returns[t-window:t])
                # 归一化
                mom = mom / np.sqrt(window)  # 按时间尺度归一化
            else:
                mom = 0.0
            momentum.append(mom)
        return momentum
    
    def _compute_macd(self, prices: np.ndarray, t: int) -> List[float]:
        """计算 MACD 信号"""
        if t < self.MACD_PARAMS['slow']:
            return [0.0, 0.0]
        
        window = prices[t-self.MACD_PARAMS['slow']*2:t+1]
        
        # EMA
        ema_fast = self._ema(window, self.MACD_PARAMS['fast'])
        ema_slow = self._ema(window, self.MACD_PARAMS['slow'])
        
        macd = ema_fast - ema_slow
        
        # 信号线 (简化)
        signal = macd * 0.5  # 近似信号线
        
        # MACD 信号 (论文 Equation 3)
        # φ(MACD) = MACD * exp(-MACD^2/4) / 0.89
        macd_signal = macd * np.exp(-macd**2 / 4) / 0.89 if abs(macd) < 10 else 0
        
        return [macd_signal, signal]
    
    def _ema(self, data: np.ndarray, span: int) -> float:
        """计算指数移动平均"""
        if len(data) < span:
            return np.mean(data)
        alpha = 2 / (span + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _compute_rsi(self, prices: np.ndarray, t: int) -> float:
        """计算 RSI"""
        if t < self.RSI_PERIOD + 1:
            return 50.0  # 中性值
        
        deltas = np.diff(prices[t-self.RSI_PERIOD:t+1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi / 100 - 0.5  # 归一化到 [-0.5, 0.5]
    
    def _compute_bb_position(self, prices: np.ndarray, t: int) -> float:
        """计算 Bollinger Bands 位置"""
        if t < self.BB_PERIOD:
            return 0.5
        
        window = prices[t-self.BB_PERIOD:t+1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std < 1e-8:
            return 0.5
        
        # 当前价格在 bands 中的位置 (0-1)
        upper = mean + self.BB_STD * std
        lower = mean - self.BB_STD * std
        
        position = (prices[t] - lower) / (upper - lower)
        return np.clip(position, 0, 1)
    
    def _compute_atr(self, prices: np.ndarray, t: int) -> float:
        """计算 ATR (Average True Range)"""
        if t < self.ATR_PERIOD + 1:
            return 0.02
        
        # 简化: 使用价格波动
        window = prices[t-self.ATR_PERIOD:t+1]
        atr = np.std(window) / np.mean(window)  # 归一化 ATR
        
        return atr * 10  # 放大
    
    def _compute_price_position(self, prices: np.ndarray, t: int) -> float:
        """计算价格在近期范围中的位置"""
        window = 50
        if t < window:
            return 0.5
        
        window_prices = prices[t-window:t+1]
        min_price = np.min(window_prices)
        max_price = np.max(window_prices)
        
        if max_price - min_price < 1e-8:
            return 0.5
        
        position = (prices[t] - min_price) / (max_price - min_price)
        return position
    
    def _compute_volatility(self, returns: np.ndarray, t: int) -> float:
        """计算 20日年化波动率"""
        window = 20
        if t < window:
            return 0.2  # 默认 20%
        
        vol = np.std(returns[t-window:t]) * np.sqrt(252)
        return vol
    
    def _compute_rolling_sharpe(self, returns: np.ndarray, t: int) -> float:
        """计算滚动夏比"""
        window = 20
        if t < window:
            return 0.0
        
        window_returns = returns[t-window:t]
        mean = np.mean(window_returns)
        std = np.std(window_returns)
        
        if std < 1e-8:
            return 0.0
        
        sharpe = mean / std * np.sqrt(252)
        return sharpe
    
    def _compute_skewness(self, returns: np.ndarray, t: int) -> float:
        """计算收益偏度"""
        window = 20
        if t < window:
            return 0.0
        
        window_returns = returns[t-window:t]
        mean = np.mean(window_returns)
        std = np.std(window_returns)
        
        if std < 1e-8:
            return 0.0
        
        skew = np.mean(((window_returns - mean) / std)**3)
        return skew
    
    def get_state_dimension(self) -> int:
        """获取状态维度"""
        # 6 动量 + 2 MACD + 1 RSI + 1 BB + 1 ATR + 1 价格位置 + 1 波动率 + 1 收益 + 1 夏比 + 1 偏度
        return 16


# =============================================================================
# 3. Volatility Scaling
# =============================================================================

class VolatilityScaler:
    """
    论文 Section 4.3: Volatility Scaling
    
    论文原文:
    "We scale all signals to have a constant volatility of 10% per annum 
    to make results comparable across different assets and time periods."
    
    公式:
    Position_scaled = Position_raw * (σ_target / σ_current)
    
    其中:
    - σ_target: 目标年化波动率 (10%)
    - σ_current: 当前年化波动率 (20日滚动)
    """
    
    def __init__(self, target_vol: float = 0.10, 
                 lookback: int = 20,
                 max_leverage: float = 10.0,
                 min_leverage: float = 0.1):
        """
        Args:
            target_vol: 目标年化波动率 (默认 10%)
            lookback: 波动率计算窗口 (默认 20日)
            max_leverage: 最大杠杆限制
            min_leverage: 最小杠杆限制
        """
        self.target_vol = target_vol
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        
        # 历史波动率记录
        self.vol_history = []
    
    def scale(self, position: float, returns: np.ndarray, t: int) -> float:
        """
        应用波动率缩放
        
        Args:
            position: 原始仓位
            returns: 收益率序列
            t: 当前时刻
            
        Returns:
            scaled_position: 缩放后的仓位
        """
        # 计算当前波动率
        current_vol = self._compute_current_vol(returns, t)
        
        # 计算缩放因子
        if current_vol > 1e-8:
            scaling = self.target_vol / current_vol
        else:
            scaling = 1.0
        
        # 应用杠杆限制
        scaling = np.clip(scaling, self.min_leverage, self.max_leverage)
        
        # 缩放仓位
        scaled_position = position * scaling
        
        # 记录
        self.vol_history.append(current_vol)
        
        return scaled_position
    
    def _compute_current_vol(self, returns: np.ndarray, t: int) -> float:
        """计算当前年化波动率"""
        if t < self.lookback:
            return self.target_vol  # 默认目标波动率
        
        window_returns = returns[t-self.lookback:t]
        daily_vol = np.std(window_returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        if len(self.vol_history) == 0:
            return {}
        
        return {
            'avg_vol': np.mean(self.vol_history),
            'min_vol': np.min(self.vol_history),
            'max_vol': np.max(self.vol_history),
            'target_vol': self.target_vol
        }
    
    def reset(self):
        """重置历史"""
        self.vol_history = []


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 核心组件测试")
    print("=" * 80)
    
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    returns = np.diff(prices) / prices[:-1]
    
    # 1. 测试 DSR
    print("\n【1. Differential Sharpe Ratio】")
    dsr = DifferentialSharpeRatio(eta=0.01)
    rewards = []
    for i, r in enumerate(returns[:100]):
        reward = dsr.update(r)
        rewards.append(reward)
    
    stats = dsr.get_statistics()
    print(f"  最终 Sharpe: {stats['sharpe']:.3f}")
    print(f"  平均收益: {stats['mean_return']:.4f}")
    print(f"  收益标准差: {stats['std_return']:.4f}")
    
    # 2. 测试状态空间
    print("\n【2. 多时间尺度状态空间】")
    state_builder = MultiTimeScaleState(lookback=200)
    state = state_builder.compute(prices, returns, t=250)
    print(f"  状态维度: {len(state)}")
    print(f"  状态示例 (t=250): {state[:6]}...")  # 前6个是动量特征
    
    # 3. 测试波动率缩放
    print("\n【3. Volatility Scaling】")
    scaler = VolatilityScaler(target_vol=0.10)
    scaled_positions = []
    raw_positions = np.random.randn(len(returns)) * 0.5  # 随机仓位
    
    for i in range(250, len(returns)):
        scaled = scaler.scale(raw_positions[i], returns, i)
        scaled_positions.append(scaled)
    
    vol_stats = scaler.get_statistics()
    print(f"  平均波动率: {vol_stats['avg_vol']:.2%}")
    print(f"  波动率范围: [{vol_stats['min_vol']:.2%}, {vol_stats['max_vol']:.2%}]")
    print(f"  目标波动率: {vol_stats['target_vol']:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成!")
    print("=" * 80)
