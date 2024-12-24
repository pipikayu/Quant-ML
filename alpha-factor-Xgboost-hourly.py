import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import yfinance as yf
import logging
import time
import datetime
from tensorflow.keras.initializers import HeNormal
import os
import xgboost as xgb
import sys
import matplotlib
matplotlib.use('Agg')

def download_hourly_data(ticker):
    logging.info(f"Fetching hourly data for {ticker}.")
    
    # Calculate the start date for 730 days ago from yesterday
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=720)
    
    # Initialize an empty DataFrame to store the data
    all_data = pd.DataFrame()
    
    # Set the initial start and end dates for the first 60-day window
    current_start_date = start_date
    current_end_date = current_start_date + datetime.timedelta(days=60)
    
    # Loop to fetch data in 60-day windows
    while current_start_date < end_date:
        retries = 5
        for attempt in range(retries):
            try:
                data = yf.download(ticker, start=current_start_date.strftime("%Y-%m-%d"), end=current_end_date.strftime("%Y-%m-%d"), interval="1h")
                if data.empty:
                    raise ValueError("Downloaded data is empty.")
                all_data = pd.concat([all_data, data])
                break  # Exit the retry loop if successful
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(10)
                else:
                    logging.error("All attempts to download data failed.")
                    return None
        
        # Update the start and end dates for the next window
        current_start_date = current_end_date
        current_end_date = current_start_date + datetime.timedelta(days=60)
        if current_end_date > end_date:
            current_end_date = end_date
    
    return all_data

def calculate_alpha_factors(data):
    logging.info("Calculating alpha factors...")
    window = 14
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['macd'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    # KDJ计算
    low_min = data['Low'].rolling(window=9).min()
    high_max = data['High'].rolling(window=9).max()
    rsv = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['K'] = rsv.ewm(com=2, adjust=False).mean()
    data['D'] = data['K'].ewm(com=2, adjust=False).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']

    # 金叉银叉
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['golden_cross'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)  # 金叉
    data['death_cross'] = np.where(data['SMA_50'] < data['SMA_200'], 1, 0)  # 银叉

    # 神奇九转（示例实现）
    data['magic_nine'] = (data['Close'] - data['Close'].shift(9)) / data['Close'].shift(9) * 100  # 计算百分比变化

    # 布林带
    data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['Bollinger_Lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)

    # 动量
    data['momentum'] = data['Close'].pct_change(periods=10)  # 过去10天的动量

    # 成交量变化率
    data['volume_change'] = data['Volume'].pct_change()

    # 平均真实范围（ATR）
    data['high_low'] = data['High'] - data['Low']
    data['high_close'] = abs(data['High'] - data['Close'].shift())
    data['low_close'] = abs(data['Low'] - data['Close'].shift())
    data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
    data['ATR'] = data['tr'].rolling(window=14).mean()

    data.fillna(0, inplace=True)
    logging.info("Alpha factors calculated.")

    # Convert timezone-aware datetime columns to timezone-unaware
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # 保存因子到Excel
    data.to_excel("alpha_factors.xlsx", index=True)
    logging.info("Alpha factors saved to alpha_factors.xlsx.")
    
    return data

def prepare_data(data, ticker):
    logging.info("Preparing data...")
    features = [
        'rsi', 'macd', 'macd_signal', 
        'K', 'D', 'J',  
        'SMA_50', 'SMA_200', 
        'golden_cross', 'death_cross', 
        'magic_nine', 
        'Bollinger_Upper', 'Bollinger_Lower', 
        'momentum', 'volume_change', 
        'ATR', 'high_low', 'high_close', 'low_close', 'tr'
    ]
    
    # Fill missing values with -1 for the features
    data[features] = data[features].fillna(-1)

    # Check for infinite values and replace them
    if np.isinf(data[features]).any().any():
        logging.warning("Infinite values found in the data. Replacing with NaN.")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Check for NaN values after replacing infinities
    if data[features].isnull().any().any():
        logging.warning("NaN values found in the data. Filling with -1.")
        data[features].fillna(-1, inplace=True)

    # Check if data is empty after filling missing values
    if data.empty:
        logging.error("Data is empty after filling missing values.")
        return None, features  # Return None to indicate failure

    # Check if any of the features are missing
    if not all(feature in data.columns for feature in features):
        logging.error("Not all features are present in the data.")
        return None, features  # Return None to indicate failure

    # 计算未来 10 个周期的最大和最小收盘价
    data = data.iloc[::-1]
    data['future_max_close'] = data['Close'].rolling(window=10).max()
    data['future_min_close'] = data['Close'].rolling(window=10).min()    
    # 反转数据以恢复原始顺序
    data = data.iloc[::-1]
    
    data['future_min_close'] = data['future_min_close'].fillna(data['Close'][ticker])  # 用当前收盘价填充
    data['future_max_close'] = data['future_max_close'].fillna(data['Close'][ticker])  # 用当前收盘价填充
    # 初始化 action 列为持有
    data['action'] = 2  # 默认设置为持有
    # 先设置买入信号
    data.loc[(data['future_max_close'] >= data['Close'][ticker] * 1.03) & (data['future_min_close'] >= data['Close'][ticker] * 0.99), 'action'] = 0  # 买入
    # 然后设置卖出信号，只有在当前 action 仍为持有时才会更新
    data.loc[data['future_min_close'] <= data['Close'][ticker] * 0.98, 'action'] = 1  # 卖出

    # 标准化特征，不包括 action 列
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    logging.info("Data preparation completed.")
    
    # 打印特征数
    logging.info(f"Number of features: {len(features)}")
    
    # 保存特征到本地文件，包括 action 列
    features_df = data[features + ['action'] + ['future_max_close'] + ['future_min_close'] ]
    features_df.to_csv("features.csv", index=False)  # 保存为 CSV 文件
    logging.info("Features saved to features.csv.")
    
    return data, features

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = y_true.get_label()
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # Sigmoid
        loss = -alpha * (1 - y_pred) ** gamma * y_true * np.log(y_pred) - \
               (1 - alpha) * y_pred ** gamma * (1 - y_true) * np.log(1 - y_pred)
        return 'focal_loss', np.mean(loss)
    return focal_loss_fixed

def train_agent(data, features, initial_balance=100000.0, transaction_cost=0.001, train_split=0.8, model_save_path="xgboost_model.json"):
    """
    训练 XGBoost 智能体，基于给定的市场数据和特征。
    """
    # 确保目标动作列存在
    if 'action' not in data.columns:
        logging.error("Action column is missing in training data.")
        return None, None, features, []

    # 划分训练集和测试集
    train_size = int(len(data) * train_split)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    if train_data.empty:
        logging.error("Training data is empty. Cannot train the agent.")
        return None, None, features, []

    # 准备训练数据
    X_train = train_data[features]
    y_train = train_data['action']  # 使用构造的 action 列

    # 训练 XGBoost 模型
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=100,
        reg_alpha=1,# 添加L1正则化参数
        class_weight='balanced'
    )

    # 记录训练过程中的损失
    evals = [(X_train, y_train)]  # 训练集
    
    model.fit(X_train, y_train, eval_set=evals, verbose=True)

    # 保存模型
    model.save_model(model_save_path)
    logging.info(f"Model saved to {model_save_path}.")  # 记录保存信息

    return model, test_data  # 返回模型和测试数据

def backtest(model, test_data, features, initial_balance=100000.0, transaction_cost=0.001, model_path="xgboost_model.json"):
    """
    使用训练好的智能体进行回测，评估策略表现，并输出专业回测指标。
    """
    cash, holdings = initial_balance, 0
    portfolio_values = []  # 保存每个时间步的资产价值
    daily_returns = []  # 保存每日收益率
    buy_signals, sell_signals = [], []
    strong_buy_signals, strong_sell_signals = [], []
    stop_loss_threshold = 0.07  # 7%止损阈值

    # 确保测试数据的特征列与训练时一致
    if not all(feature in test_data.columns for feature in features):
        logging.error("Test data does not contain all required features.")
        return None, [], []

    for t in range(len(test_data) - 1):
        state = test_data[features].iloc[t].values.reshape(1, -1)
        
        # 使用 predict_proba 方法进行预测
        probabilities = model.predict_proba(state)  # 获取每个类别的概率
        logging.info(f"Probabilities: {probabilities}")  # 打印概率到日志
        action = np.argmax(probabilities)  # 获取概率最高的类别索引
        current_price = test_data['Close'].iloc[t].iloc[0]
        
        # 动作逻辑
        if action == 0 and cash >= current_price * (1 + transaction_cost) and probabilities[0][0] >= 0.4:  # 买入
            max_shares = int(cash // (current_price * (1 + transaction_cost)))
            cash -= max_shares * current_price * (1 + transaction_cost)
            holdings += max_shares
            buy_signals.append(test_data.index[t])
            buy_price = current_price  # 记录买入价格

        elif action == 1 and holdings > 0:  # 卖出
            cash += holdings * current_price * (1 - transaction_cost)
            holdings = 0
            sell_signals.append(test_data.index[t])

        # 检查止损条件
        if holdings > 0 and current_price < buy_price * (1 - stop_loss_threshold):  # 如果当前价格低于买入价格的93%
            cash += holdings * current_price * (1 - transaction_cost)  # 卖出
            holdings = 0
            sell_signals.append(test_data.index[t])
            logging.info(f"Stop loss triggered at {current_price:.2f} on {test_data.index[t]}.")

        if probabilities[0][0] >= 0.5:
            strong_buy_signals.append(test_data.index[t])
        elif probabilities[0][1] >= 0.5:
            strong_sell_signals.append(test_data.index[t])

        # 计算当前组合的资产价值
        portfolio_value = cash + holdings * current_price
        portfolio_values.append(portfolio_value)

        # 计算日收益率（避免除零）
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)

        # 日志输出
        logging.info(f"Time: {test_data.index[t]}, Action: {action}, Cash: {cash:.2f}, Holdings: {holdings}, "
                     f"Portfolio Value: {portfolio_value:.2f}")

    # 计算回测指标
    portfolio_values = np.array(portfolio_values)
    cumulative_return = (portfolio_values[-1] - initial_balance) / initial_balance
    annualized_return = (1 + cumulative_return) ** (252 / len(test_data)) - 1
    max_drawdown = np.max((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values))
    volatility = np.std(daily_returns) * np.sqrt(252)  # 年化波动率
    sharpe_ratio = (np.mean(daily_returns) * 252) / (volatility + 1e-10)  # 避免除零
    win_rate = np.sum(np.array(daily_returns) > 0) / len(daily_returns)

    # 输出回测指标到日志
    logging.info("\n=== Backtest Summary ===")
    logging.info(f"Initial Balance: {initial_balance:.2f}")
    logging.info(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")
    logging.info(f"Cumulative Return: {cumulative_return * 100:.2f}%")
    logging.info(f"Annualized Return: {annualized_return * 100:.2f}%")
    logging.info(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    logging.info(f"Annualized Volatility: {volatility * 100:.2f}%")
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logging.info(f"Win Rate: {win_rate * 100:.2f}%")
    
    return portfolio_values, buy_signals, sell_signals, strong_buy_signals, strong_sell_signals


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # 计算当前日期并转换为字符串
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BABA"
    output_dir = f"./data/Xgboost-Hourly-{current_date}/{ticker}-Xgboost-Hourly-{current_date}"
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging to save to a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=os.path.join(output_dir, 'trading_log.txt'),
        filemode='w'
    )

    data = download_hourly_data(ticker)
    if data is not None:
        data = calculate_alpha_factors(data)
        data, features = prepare_data(data, ticker)
        
        if data is None:
            logging.error("Data preparation failed. Exiting.")
            exit()

        model, test_data = train_agent(data, features, model_save_path=os.path.join(output_dir, "xgboost_model.json"))
        
        if test_data is not None and not test_data.empty:
            portfolio_values, buy_signals, sell_signals, strong_buy_signals, strong_sell_signals = backtest(model, test_data, features, model_path=os.path.join(output_dir, "xgboost_model.json"))
        else:
            logging.error("No valid data available for backtesting.")
            exit()

        plt.figure(figsize=(14, 7))
        if test_data is not None:
            plt.plot(test_data['Close'], label='Close Price')
            buy_signal_indices = [test_data.index.get_loc(ts) for ts in buy_signals]
            plt.scatter(test_data.index[buy_signal_indices], test_data['Close'].iloc[buy_signal_indices], marker='^', color='g', label='Buy Signal', alpha=1)
            strong_buy_signal_indices = [test_data.index.get_loc(ts) for ts in strong_buy_signals]
            plt.scatter(test_data.index[strong_buy_signal_indices], test_data['Close'].iloc[strong_buy_signal_indices], marker='^', color='yellow', label='Strong Buy Signal', alpha=1)
            sell_signal_indices = [test_data.index.get_loc(ts) for ts in sell_signals]
            plt.scatter(test_data.index[sell_signal_indices], test_data['Close'].iloc[sell_signal_indices], marker='v', color='r', label='Sell Signal', alpha=1)
            strong_sell_signal_indices = [test_data.index.get_loc(ts) for ts in strong_sell_signals]
            plt.scatter(test_data.index[strong_sell_signal_indices], test_data['Close'].iloc[strong_sell_signal_indices], marker='v', color='orange', label='Strong Sell Signal', alpha=1)
            plt.title(f"Trading Strategy - Buy & Sell Signals for {ticker}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"trading_strategy_signals_{ticker}.png"))
        else:
            logging.error("No valid data available for plotting.")

        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_values, label="Portfolio Value")
        plt.title(f"Portfolio Value Over Time for {ticker}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"portfolio_value_over_time_{ticker}.png"))