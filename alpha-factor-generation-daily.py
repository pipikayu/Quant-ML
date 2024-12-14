import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Step 1: 下载日线数据
def download_daily_data(ticker, start_date):
    """
    下载指定股票的日线数据
    """
    logging.info(f"Fetching daily data for {ticker} from {start_date}.")
    try:
        data = yf.download(ticker, start=start_date, interval="1d")
        if data.empty:
            logging.error(f"No data returned for {ticker}.")
            return pd.DataFrame()  # Return an empty DataFrame
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# Step 2: 动态调整窗口
def dynamic_window(data, base_window=20, k=5):
    """
    根据波动性动态调整窗口大小
    """
    volatility = data['Close'].pct_change().rolling(base_window).std()
    dynamic_windows = (base_window * (1 + k * volatility)).round().clip(lower=10, upper=100).fillna(base_window)
    return dynamic_windows

# Step 3: 计算 RSI 和相关因子
def calculate_rsi_factors(data, window=14):
    """
    计算 RSI 和相关因子
    """
    delta = data['Close'].diff(1)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_mean_dev'] = (data['rsi'] - data['rsi'].rolling(window).mean()) / data['rsi'].rolling(window).mean()
    data['rsi_momentum'] = data['rsi'] - data['rsi'].shift(window)
    return data

# Step 4: 计算 MACD、SMA、EMA 和金叉死叉类指标
def calculate_crossover_signals(data, sma_short=50, sma_long=200, ema_short=12, ema_long=26, signal=9):
    """
    计算 MACD、SMA、EMA 及金叉死叉指标
    """
    # SMA 金叉/死叉
    data['sma_short'] = data['Close'].rolling(window=sma_short).mean()
    data['sma_long'] = data['Close'].rolling(window=sma_long).mean()
    data['sma_crossover'] = np.where(data['sma_short'] > data['sma_long'], 1, -1)

    # EMA 金叉/死叉
    data['ema_short'] = data['Close'].ewm(span=ema_short).mean()
    data['ema_long'] = data['Close'].ewm(span=ema_long).mean()
    data['ema_crossover'] = np.where(data['ema_short'] > data['ema_long'], 1, -1)

    # MACD 金叉/死叉
    data['macd'] = data['ema_short'] - data['ema_long']
    data['macd_signal'] = data['macd'].ewm(span=signal).mean()
    data['macd_crossover'] = np.where(data['macd'] > data['macd_signal'], 1, -1)

    return data

# Step 5: 计算 KDJ 因子
def calculate_kdj(data, window=14):
    """
    计算 KDJ 指标
    """
    low_n = data['Low'].rolling(window).min()
    high_n = data['High'].rolling(window).max()
    data['%K'] = (data['Close'] - low_n) / (high_n - low_n) * 100
    data['%D'] = data['%K'].rolling(3).mean()
    data['%J'] = 3 * data['%K'] - 2 * data['%D']
    return data

# Step 6: 计算其他动量和价格变化类因子
def calculate_other_factors(data, dynamic_windows):
    """
    计算动量、价格变化等因子
    """
    base_window = int(dynamic_windows.median())
    data['momentum'] = data['Close'] - data['Close'].shift(base_window)
    data['price_roc'] = (data['Close'] - data['Close'].shift(base_window)) / data['Close'].shift(base_window)
    data['volume_pct_change'] = (data['Volume'] - data['Volume'].shift(base_window)) / data['Volume'].shift(base_window)
    data['bias'] = (data['Close'] - data['Close'].rolling(base_window).mean()) / data['Close'].rolling(base_window).mean()
    return data

# Step 7: 保存因子数据
def save_to_excel(data, filename="alpha_factors_combined.xlsx"):
    """
    保存因子数据到 Excel 文件
    """
    logging.info(f"Saving alpha factors to {filename}.")
    data.to_excel(filename)

# Step 8: 可视化因子
def plot_alpha_factors(data, factors):
    """
    可视化多种因子
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
    start_date = "2020-01-01"
    ticker = "TSLA"
    factors_to_plot = [
        'rsi', 'rsi_mean_dev', 'rsi_momentum', 'macd', 'macd_signal', 'macd_crossover',
        'sma_crossover', 'ema_crossover', '%K', '%D', '%J', 'momentum', 'price_roc', 'bias'
    ]

    # 下载 Tesla 日线数据
    data = download_daily_data(ticker, start_date)

    if data.empty:
        logging.error("No data available. Exiting program.")
        exit(1)  # Exit if no data is available

    # 动态调整窗口
    dynamic_windows = dynamic_window(data)

    # 计算 RSI 和相关因子
    data = calculate_rsi_factors(data)

    # 计算 MACD、SMA、EMA 和金叉死叉类指标
    data = calculate_crossover_signals(data)

    # 计算 KDJ 因子
    data = calculate_kdj(data)

    # 计算动量和价格变化类因子
    data = calculate_other_factors(data, dynamic_windows)

    # 保存到 Excel 文件
    save_to_excel(data)

    # 可视化因子
    plot_alpha_factors(data, factors_to_plot)