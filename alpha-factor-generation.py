import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Step 1: 下载数据
def download_data(ticker, start_date, interval='15m'):
    logging.info(f"Fetching {ticker} data from {start_date} with interval {interval}.")
    data = yf.download(ticker, start=start_date, interval=interval)
    return data

# Step 2: 计算 Alpha 因子
def calculate_alpha_factors(data, window=20, fast=12, slow=26, signal=9):
    """
    计算多种 Alpha 因子
    :param data: 包含交易数据的 DataFrame
    :param window: 滚动窗口大小（默认 20）
    :param fast: MACD 快速 EMA 周期（默认 12）
    :param slow: MACD 慢速 EMA 周期（默认 26）
    :param signal: MACD 信号线 EMA 周期（默认 9）
    :return: 包含 Alpha 因子的数据
    """
    # MACD 因子
    data['ema_fast'] = data['Close'].ewm(span=fast).mean()
    data['ema_slow'] = data['Close'].ewm(span=slow).mean()
    data['macd'] = data['ema_fast'] - data['ema_slow']
    data['macd_signal'] = data['macd'].ewm(span=signal).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']

    # 均线交叉因子 (SMA Crossover)
    data['sma_short'] = data['Close'].rolling(window=fast).mean()
    data['sma_long'] = data['Close'].rolling(window=slow).mean()
    data['sma_crossover'] = (data['sma_short'] - data['sma_long']) / data['sma_long']

    # 相对成交量因子 (Relative Volume)
    data['relative_volume'] = (data['Volume'] - data['Volume'].rolling(window).mean()) / data['Volume'].rolling(window).mean()

    # ATR 因子
    data['high_low'] = data['High'] - data['Low']
    data['high_close'] = np.abs(data['High'] - data['Close'].shift(1))
    data['low_close'] = np.abs(data['Low'] - data['Close'].shift(1))
    data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
    data['atr'] = data['true_range'].rolling(window).mean()

    # Williams %R 因子
    data['williams_r'] = (data['High'].rolling(window).max() - data['Close']) / (
            data['High'].rolling(window).max() - data['Low'].rolling(window).min()) * -100

    # 加权价格变动因子 (Weighted Price Momentum)
    for i in range(1, window + 1):
        data[f'weighted_return_{i}'] = (data['Close'] - data['Close'].shift(i)) / data['Close'].shift(i) / i
    data['weighted_price_momentum'] = data[[f'weighted_return_{i}' for i in range(1, window + 1)]].sum(axis=1)

    # 清理无效数据
    data = data.dropna()
    return data

# Step 3: 保存因子数据
def save_to_excel(data, filename="alpha_factors_extended.xlsx"):
    """
    将因子数据保存到 Excel 文件
    :param data: 包含 Alpha 因子的数据
    :param filename: 文件名（默认 "alpha_factors_extended.xlsx"）
    """
    logging.info(f"Saving alpha factors to {filename}.")
    data.to_excel(filename)

# Step 4: 可视化因子
def plot_alpha_factors(data, factors):
    """
    可视化多种 Alpha 因子
    :param data: 包含因子数据的 DataFrame
    :param factors: 因子列表
    """
    plt.figure(figsize=(14, len(factors) * 3))
    for i, factor in enumerate(factors, 1):
        plt.subplot(len(factors), 1, i)
        plt.plot(data.index, data[factor], label=factor)
        plt.title(f"{factor} Over Time")
        plt.xlabel("Time")
        plt.ylabel(factor)
        plt.legend()
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 配置参数
    start_date = "2024-10-17"
    ticker = "TSLA"
    factors_to_plot = [
        'macd', 'macd_signal', 'macd_diff', 'sma_crossover', 'relative_volume',
        'atr', 'williams_r', 'weighted_price_momentum'
    ]

    # 下载数据
    data = download_data(ticker, start_date)

    # 计算 Alpha 因子
    data = calculate_alpha_factors(data)

    # 保存到 Excel 文件
    save_to_excel(data)

    # 可视化因子
    plot_alpha_factors(data, factors_to_plot)