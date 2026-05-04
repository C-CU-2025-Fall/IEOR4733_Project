"""
时间序列FFT Regime检测 (修复版本)

核心改进：
1. 修复时间范围显示问题（1970-01-01错误）
2. 支持输出特定时间段的 soft probability（用于A2C）
3. 正确对齐时间索引

A2C 时间段配置：
- Period 1: Train (2005-01-01 ~ 2010-12-31), Test (2011-01-01 ~ 2015-12-31)
- Period 2: Train (2010-01-01 ~ 2015-12-31), Test (2016-01-01 ~ 2019-12-31)

Author: Fixed implementation
Date: 2026-04-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StateMatrixBuilder:
    """根据论文定义构建 60×9 状态矩阵"""
    
    @staticmethod
    def compute_rsi(prices, window=30):
        """计算RSI指标"""
        if len(prices) < window:
            return 50.0
        
        deltas = np.diff(prices[-window:])
        gains = np.sum(deltas[deltas > 0])
        losses = np.sum(np.abs(deltas[deltas < 0]))
        
        if losses == 0:
            return 100.0 if gains > 0 else 50.0
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi, 0, 100)
    
    @staticmethod
    def compute_macd(prices, fast=8, slow=24):
        """计算MACD"""
        if len(prices) < slow:
            return 0.0
        
        ema_fast = pd.Series(prices[-slow:]).ewm(span=fast, adjust=False).mean().values[-1]
        ema_slow = pd.Series(prices[-slow:]).ewm(span=slow, adjust=False).mean().values[-1]
        return ema_fast - ema_slow
    
    @staticmethod
    def compute_volatility(prices, span=60):
        """计算日波动率"""
        if len(prices) < 2:
            return 0.01
        
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        if len(returns) < span:
            return np.std(returns)
        
        ewma_vol = pd.Series(returns[-span:]).ewm(span=span, adjust=False).std().values[-1]
        return max(ewma_vol, 0.0001)
    
    @classmethod
    def build_state_matrix_for_asset(cls, prices: np.ndarray, time_idx: int) -> Optional[np.ndarray]:
        """
        为某个时间点构建完整的60×9状态矩阵
        
        Args:
            prices: 历史价格序列
            time_idx: 时间索引
        
        Returns:
            state_matrix: (60, 9) 或 None
        """
        if time_idx < 60:
            return None
        
        prices = np.array(prices, dtype=float)
        prices = np.nan_to_num(prices, nan=0.0)
        
        # 取过去60天的数据
        price_window = prices[time_idx-60:time_idx+1]
        
        if len(price_window) < 61:
            return None
        
        state_matrix = np.zeros((60, 9))
        
        # 对每个时间点构建状态向量
        for row_idx in range(60):
            price_hist = price_window[:row_idx+1]
            p_current = price_hist[-1]
            
            state_matrix[row_idx, 0] = (p_current - price_hist[0]) / (price_hist[0] + 1e-8)
            
            daily_vol = cls.compute_volatility(price_hist)
            for j, period in enumerate([21, 42, 63, 252]):
                if len(price_hist) > period:
                    r = (p_current - price_hist[-period-1]) / (price_hist[-period-1] + 1e-8)
                    state_matrix[row_idx, 1+j] = r / (daily_vol * np.sqrt(252) + 1e-8)
            
            for j, (fast, slow) in enumerate([(8, 24), (16, 48), (32, 96)]):
                if len(price_hist) >= slow:
                    state_matrix[row_idx, 5+j] = cls.compute_macd(price_hist, fast=fast, slow=slow)
            
            state_matrix[row_idx, 8] = cls.compute_rsi(price_hist, window=30)
        
        return state_matrix


def extract_fft_features(state_matrix: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    对60×9状态矩阵提取FFT特征
    
    Args:
        state_matrix: (60, 9) 矩阵
        n_components: FFT保留分量数
    
    Returns:
        features: (180,) 特征向量
    """
    if state_matrix is None or state_matrix.shape != (60, 9):
        return None
    
    features = []
    
    for col_idx in range(9):
        feature_series = state_matrix[:, col_idx]
        feature_series = np.nan_to_num(feature_series, nan=0.0)
        
        # FFT
        fft_vals = np.fft.fft(feature_series)
        real_part = np.real(fft_vals[:n_components])
        imag_part = np.imag(fft_vals[:n_components])
        
        features.extend(real_part)
        features.extend(imag_part)
    
    return np.array(features)


def detect_regimes_for_asset_class_timeseries(
    clc_data: Dict,
    asset_class_tickers: List[str],
    asset_class_name: str,
    n_regimes: int = 3,
    n_fft_components: int = 10,
    min_data_length: int = 312,
    date_range: Optional[Tuple[str, str]] = None
) -> Dict:
    """
    对单个资产类进行时间序列的regime检测
    
    Args:
        clc_data: Dict[ticker, DataFrame] - RAD数据
        asset_class_tickers: 该资产类的ticker列表
        asset_class_name: 资产类名称
        n_regimes: regime数量
        n_fft_components: FFT保留分量数
        min_data_length: 最小数据长度要求
        date_range: 可选的日期范围 (start_date, end_date)，格式: "YYYY-MM-DD"
                   如果指定，则只使用该范围内的数据
    
    Returns:
        result: 包含regime标签、概率等的字典
    """
    logger.info(f"\n处理 {asset_class_name} ({len(asset_class_tickers)} 个资产)...")
    
    # 如果指定了日期范围，打印信息
    if date_range:
        date_start, date_end = date_range
        logger.info(f"  时间范围: {date_start} ~ {date_end}")
    
    # Step 1: 加载数据并检查长度
    asset_prices = {}
    asset_dates_list = {}
    valid_tickers = []
    
    # 转换日期范围
    date_start_dt = pd.to_datetime(date_range[0]) if date_range else None
    date_end_dt = pd.to_datetime(date_range[1]) if date_range else None
    
    for ticker in asset_class_tickers:
        try:
            if ticker not in clc_data:
                continue
            
            df = clc_data[ticker]
            if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
                continue
            
            # 【新增】按日期范围过滤
            df_filtered = df.copy()
            if date_range:
                mask = (df_filtered['Date'] >= date_start_dt) & (df_filtered['Date'] <= date_end_dt)
                df_filtered = df_filtered[mask]
            
            prices = df_filtered['Close'].values.astype(float)
            prices = prices[~np.isnan(prices)]
            
            if len(prices) < min_data_length:
                logger.debug(f"  ⚠ {ticker}: 数据不足 ({len(prices)} < {min_data_length})")
                continue
            
            asset_prices[ticker] = prices
            # 保存对应的日期
            close_notna = ~df_filtered['Close'].isna()
            asset_dates_list[ticker] = df_filtered.loc[close_notna, 'Date'].values
            
            valid_tickers.append(ticker)
            logger.info(f"  ✓ {ticker}: {len(prices)} 个交易日")
        
        except Exception as e:
            logger.warning(f"  ✗ {ticker}: {e}")
    
    if not asset_prices:
        logger.error(f"  {asset_class_name} 没有有效的数据!")
        return None
    
    logger.info(f"  ✓ 加载了 {len(asset_prices)} 个有效资产")
    
    # Step 2: 确定共同的时间范围和日期
    min_len = min(len(prices) for prices in asset_prices.values())
    
    # 对齐所有资产到相同长度（取最后 min_len 个数据点）
    aligned_dates = None
    for ticker in asset_prices:
        if len(asset_prices[ticker]) > min_len:
            asset_prices[ticker] = asset_prices[ticker][-min_len:]
        
        # 使用第一个ticker的日期作为参考（因为对齐后都相同）
        if aligned_dates is None:
            dates = asset_dates_list[ticker]
            # 取最后min_len个日期
            if len(dates) >= min_len:
                aligned_dates = dates[-min_len:]
            else:
                aligned_dates = dates
    
    if aligned_dates is None:
        logger.error("无法获取日期信息")
        return None
    
    # 转换为DatetimeIndex以便正确处理
    aligned_dates = pd.DatetimeIndex(aligned_dates)
    
    # 调试日志
    logger.debug(f"  aligned_dates 类型: {type(aligned_dates)}")
    logger.debug(f"  aligned_dates[0]: {aligned_dates[0]}")
    logger.debug(f"  aligned_dates[-1]: {aligned_dates[-1]}")
    
    date_start = aligned_dates[0].strftime('%Y-%m-%d') if len(aligned_dates) > 0 else 'N/A'
    date_end = aligned_dates[-1].strftime('%Y-%m-%d') if len(aligned_dates) > 0 else 'N/A'
    logger.info(f"  时间范围: {date_start} 至 {date_end} ({len(aligned_dates)} 个交易日)")
    
    # Step 3: 对每个时间点生成FFT特征
    builder = StateMatrixBuilder()
    fft_features_list = []
    valid_time_indices = []
    
    logger.info(f"  生成时间序列特征...")
    
    for time_idx in tqdm(range(60, min_len), desc=f"  {asset_class_name}"):
        # 对该时间点的所有资产
        state_matrices = []
        
        for ticker in valid_tickers:
            prices = asset_prices[ticker]
            state_matrix = builder.build_state_matrix_for_asset(prices, time_idx)
            
            if state_matrix is not None:
                state_matrices.append(state_matrix)
        
        if not state_matrices:
            continue
        
        # 对资产维求平均
        avg_state_matrix = np.mean(np.array(state_matrices), axis=0)
        
        # 提取FFT特征
        fft_features = extract_fft_features(avg_state_matrix, n_fft_components)
        
        if fft_features is not None:
            fft_features_list.append(fft_features)
            valid_time_indices.append(time_idx)
    
    if not fft_features_list:
        logger.error(f"  无法生成任何特征!")
        return None
    
    fft_features_array = np.array(fft_features_list)  # (N_samples, 180)
    # 直接使用对应的日期而不是索引操作
    # valid_time_indices 包含每个有效样本对应在 aligned_dates 中的位置
    sample_dates = [aligned_dates[int(idx)] for idx in valid_time_indices]
    valid_dates = pd.DatetimeIndex(sample_dates)
    
    logger.info(f"  ✓ 生成 {len(fft_features_array)} 个时间点的特征 ({fft_features_array.shape})")
    
    # Step 4: GMM聚类
    logger.info(f"  执行GMM聚类 (n_clusters={n_regimes})...")
    
    # 标准化特征
    scaler = StandardScaler()
    fft_scaled = scaler.fit_transform(fft_features_array)
    
    # 训练GMM
    gmm = GaussianMixture(
        n_components=n_regimes,
        random_state=42,
        n_init=10,
        max_iter=100
    )
    
    regime_labels = gmm.fit_predict(fft_scaled)
    soft_probs = gmm.predict_proba(fft_scaled)
    
    # 计算质量指标
    silhouette = silhouette_score(fft_scaled, regime_labels)
    
    logger.info(f"  ✓ 聚类完成")
    logger.info(f"    • 样本数: {len(regime_labels)}")
    logger.info(f"    • Silhouette Score: {silhouette:.4f}")
    
    # Step 5: 统计Regime分布
    regime_counts = [np.sum(regime_labels == i) for i in range(n_regimes)]
    logger.info(f"  📊 Regime分布:")
    for i, count in enumerate(regime_counts):
        pct = 100 * count / len(regime_labels)
        logger.info(f"    Regime {i}: {count:6d} ({pct:5.1f}%)")
    
    # Step 6: 构建结果DataFrame
    regime_df = pd.DataFrame({
        'date': valid_dates,
        'regime': regime_labels,
        'regime_prob_0': soft_probs[:, 0],
        'regime_prob_1': soft_probs[:, 1] if n_regimes > 1 else 0,
        'regime_prob_2': soft_probs[:, 2] if n_regimes > 2 else 0,
    })
    regime_df = regime_df.sort_values('date').reset_index(drop=True)
    
    result = {
        'asset_class': asset_class_name,
        'n_valid_assets': len(valid_tickers),
        'valid_tickers': valid_tickers,
        'n_time_points': len(regime_labels),
        'regime_labels': regime_labels,
        'soft_probs': soft_probs,
        'regime_counts': regime_counts,
        'silhouette_score': silhouette,
        'gmm_model': gmm,
        'fft_features': fft_features_array,
        'fft_scaler': scaler,
        'regime_df': regime_df,
        'dates': valid_dates,
    }
    
    logger.info(f"  ✅ {asset_class_name} 完成")
    
    return result


def detect_regimes_for_all_classes_timeseries(
    clc_data: Dict,
    asset_classes: Dict[str, List[str]],
    n_regimes: int = 3,
    date_range: Optional[Tuple[str, str]] = None
) -> Dict:
    """
    对所有资产类进行时间序列regime检测
    
    Args:
        clc_data: Dict[ticker, DataFrame]
        asset_classes: Dict[class_name, List[ticker]]
        n_regimes: regime数量
        date_range: 可选的日期范围 (start_date, end_date)
    
    Returns:
        results: Dict[asset_class, result]
    """
    logger.info("\n" + "="*80)
    logger.info("🎯 开始时间序列FFT Regime检测")
    logger.info("="*80)
    
    results = {}
    
    for asset_class, tickers in asset_classes.items():
        result = detect_regimes_for_asset_class_timeseries(
            clc_data=clc_data,
            asset_class_tickers=tickers,
            asset_class_name=asset_class,
            n_regimes=n_regimes,
            date_range=date_range
        )
        
        if result is not None:
            results[asset_class] = result
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ 所有资产类处理完成 ({len(results)}/{len(asset_classes)})")
    logger.info("="*80)
    
    return results


def extract_regime_for_period(
    regime_results: Dict,
    period_start: str,
    period_end: str,
    asset_classes: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    为特定时间段提取regime标签和soft probability（用于A2C）
    
    Args:
        regime_results: detect_regimes_for_all_classes_timeseries的输出结果
        period_start: 时间段开始 (str, format: "YYYY-MM-DD")
        period_end: 时间段结束 (str, format: "YYYY-MM-DD")
        asset_classes: 要提取的资产类列表，默认全部
    
    Returns:
        period_regime_data: Dict[asset_class, DataFrame]
        DataFrame 包含: date, regime, regime_prob_0, regime_prob_1, regime_prob_2
    """
    logger.info(f"\n提取 {period_start} 至 {period_end} 的regime数据...")
    
    period_start_dt = pd.to_datetime(period_start)
    period_end_dt = pd.to_datetime(period_end)
    
    if asset_classes is None:
        asset_classes = list(regime_results.keys())
    
    period_regime_data = {}
    
    for asset_class in asset_classes:
        if asset_class not in regime_results:
            logger.warning(f"  ⚠ {asset_class}: 不在结果中")
            continue
        
        regime_df = regime_results[asset_class]['regime_df'].copy()
        
        # 按时间段过滤
        mask = (regime_df['date'] >= period_start_dt) & (regime_df['date'] <= period_end_dt)
        filtered_df = regime_df[mask].reset_index(drop=True)
        
        if len(filtered_df) == 0:
            logger.warning(f"  ⚠ {asset_class}: 该时间段无数据")
            continue
        
        period_regime_data[asset_class] = filtered_df
        logger.info(f"  ✓ {asset_class}: {len(filtered_df)} 个交易日")
    
    return period_regime_data


def export_regime_for_a2c(
    regime_results: Dict,
    a2c_periods: List[Tuple[str, str, str]],
    output_dir: str = None
) -> Dict:
    """
    为A2C导出特定时间段的regime数据
    
    A2C periods 格式：
    [
        ("period_1_test", "2011-01-01", "2015-12-31"),
        ("period_2_test", "2016-01-01", "2019-12-31"),
    ]
    
    Args:
        regime_results: detect_regimes_for_all_classes_timeseries的输出结果
        a2c_periods: [(period_name, start_date, end_date), ...]
        output_dir: 输出目录
    
    Returns:
        a2c_data: Dict[period_name][asset_class] = DataFrame
    """
    from pathlib import Path
    
    if output_dir is None:
        output_dir = Path('./regime_detection/results')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("\n" + "="*80)
    logger.info("💾 导出A2C用的regime数据")
    logger.info("="*80)
    
    a2c_data = {}
    
    for period_name, start_date, end_date in a2c_periods:
        logger.info(f"\n{period_name}: {start_date} ~ {end_date}")
        
        period_data = extract_regime_for_period(
            regime_results=regime_results,
            period_start=start_date,
            period_end=end_date
        )
        
        a2c_data[period_name] = period_data
        
        # 导出为CSV
        for asset_class, df in period_data.items():
            filename = f"a2c_{period_name}_{asset_class.replace(' ', '_')}.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"  ✓ 导出: {filename}")
    
    logger.info(f"\n✅ 所有数据已导出到: {output_dir}")
    
    return a2c_data


def predict_regime_soft_probs(
    clc_data: Dict,
    asset_class_tickers: List[str],
    asset_class_name: str,
    trained_gmm,
    trained_scaler,
    n_fft_components: int = 10,
    min_data_length: int = 312,
    date_range: Optional[Tuple[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Apply a pre-trained GMM to generate regime soft probs for a new time period.

    This is the "predict-only" counterpart of detect_regimes_for_asset_class_timeseries.
    It reuses the same FFT feature extraction logic, then calls
    trained_scaler.transform() and trained_gmm.predict_proba() instead of
    fitting a new model.

    Args:
        clc_data:              Dict[ticker, DataFrame] – raw CLC price data
        asset_class_tickers:   List[str] tickers for this class
        asset_class_name:      Used only for logging
        trained_gmm:           Fitted GaussianMixture returned by detect_regimes_*
        trained_scaler:        Fitted StandardScaler returned by detect_regimes_*
        n_fft_components:      Must match what was used at fit time
        min_data_length:       Minimum # of prices required per ticker
        date_range:            Optional (start_date, end_date) strings to restrict data

    Returns:
        regime_df: DataFrame with columns
            date, regime, regime_prob_0, regime_prob_1, regime_prob_2
        or None if no valid data is found.
    """
    logger.info(f"\n[predict] {asset_class_name} soft-prob prediction...")
    if date_range:
        logger.info(f"  date_range: {date_range[0]} ~ {date_range[1]}")

    date_start_dt = pd.to_datetime(date_range[0]) if date_range else None
    date_end_dt   = pd.to_datetime(date_range[1]) if date_range else None

    # --- Step 1: load & filter data ---
    asset_prices = {}
    asset_dates_list = {}
    valid_tickers = []

    for ticker in asset_class_tickers:
        try:
            if ticker not in clc_data:
                continue
            df = clc_data[ticker]
            if not isinstance(df, pd.DataFrame):
                continue

            # detect column name style (uppercase Date/Close vs lowercase date/close)
            date_col  = 'Date'  if 'Date'  in df.columns else 'date'
            close_col = 'Close' if 'Close' in df.columns else 'close'

            if close_col not in df.columns:
                continue

            df_filtered = df.copy()
            if date_range and date_col in df_filtered.columns:
                mask = (
                    (df_filtered[date_col] >= date_start_dt) &
                    (df_filtered[date_col] <= date_end_dt)
                )
                df_filtered = df_filtered[mask]

            prices = df_filtered[close_col].values.astype(float)
            prices = prices[~np.isnan(prices)]

            if len(prices) < min_data_length:
                continue

            asset_prices[ticker] = prices

            close_notna = ~df_filtered[close_col].isna()
            asset_dates_list[ticker] = df_filtered.loc[close_notna, date_col].values
            valid_tickers.append(ticker)

        except Exception as e:
            logger.warning(f"  ✗ {ticker}: {e}")

    if not asset_prices:
        logger.error(f"  {asset_class_name}: no valid data for prediction")
        return None

    # --- Step 2: align lengths ---
    min_len = min(len(p) for p in asset_prices.values())

    aligned_dates = None
    for ticker in asset_prices:
        if len(asset_prices[ticker]) > min_len:
            asset_prices[ticker] = asset_prices[ticker][-min_len:]
        if aligned_dates is None:
            dates = asset_dates_list[ticker]
            aligned_dates = dates[-min_len:] if len(dates) >= min_len else dates

    aligned_dates = pd.DatetimeIndex(aligned_dates)
    logger.info(
        f"  time range: {aligned_dates[0].strftime('%Y-%m-%d')} ~ "
        f"{aligned_dates[-1].strftime('%Y-%m-%d')} ({min_len} days)"
    )

    # --- Step 3: build FFT features ---
    builder = StateMatrixBuilder()
    fft_features_list = []
    valid_time_indices = []

    for time_idx in tqdm(range(60, min_len), desc=f"  {asset_class_name} (predict)"):
        state_matrices = []
        for ticker in valid_tickers:
            sm = builder.build_state_matrix_for_asset(asset_prices[ticker], time_idx)
            if sm is not None:
                state_matrices.append(sm)

        if not state_matrices:
            continue

        avg_sm = np.mean(np.array(state_matrices), axis=0)
        feats  = extract_fft_features(avg_sm, n_fft_components)
        if feats is not None:
            fft_features_list.append(feats)
            valid_time_indices.append(time_idx)

    if not fft_features_list:
        logger.error(f"  {asset_class_name}: could not generate FFT features")
        return None

    fft_features_array = np.array(fft_features_list)

    # --- Step 4: apply trained scaler + GMM ---
    fft_scaled    = trained_scaler.transform(fft_features_array)
    regime_labels = trained_gmm.predict(fft_scaled)
    soft_probs    = trained_gmm.predict_proba(fft_scaled)

    n_components = trained_gmm.n_components
    sample_dates = [aligned_dates[int(idx)] for idx in valid_time_indices]
    valid_dates  = pd.DatetimeIndex(sample_dates)

    prob_cols = {f"regime_prob_{i}": soft_probs[:, i] for i in range(n_components)}
    # pad to 3 columns if n_components < 3
    for i in range(n_components, 3):
        prob_cols[f"regime_prob_{i}"] = 0.0

    regime_df = pd.DataFrame({"date": valid_dates, "regime": regime_labels, **prob_cols})
    regime_df = regime_df.sort_values("date").reset_index(drop=True)

    logger.info(f"  ✓ {asset_class_name}: {len(regime_df)} time points predicted")
    return regime_df


if __name__ == "__main__":
    """
    使用示例：
    
    from regime_detection.timeseries_fft_regime import (
        detect_regimes_for_all_classes_timeseries,
        export_regime_for_a2c
    )
    from data_loader import load_clc_full
    from config import ASSET_CLASSES
    
    # 加载数据
    clc_data = {}
    for asset_class, tickers in ASSET_CLASSES.items():
        for ticker in tickers:
            try:
                clc_data[ticker] = load_clc_full(ticker)
            except:
                pass
    
    # 执行时间序列regime检测
    results = detect_regimes_for_all_classes_timeseries(
        clc_data=clc_data,
        asset_classes=ASSET_CLASSES,
        n_regimes=3
    )
    
    # 为A2C导出特定时间段的数据
    a2c_periods = [
        ("period_1_test", "2011-01-01", "2015-12-31"),
        ("period_2_test", "2016-01-01", "2019-12-31"),
    ]
    
    a2c_regime_data = export_regime_for_a2c(
        regime_results=results,
        a2c_periods=a2c_periods,
        output_dir='/Users/ladymie/Documents/GitHub/IEOR4733_Project/regime_detection/results'
    )
    
    # 获取某个时间段的结果
    period_1_test = a2c_regime_data['period_1_test']['Commodity']
    print(period_1_test.head())
    """
    pass
