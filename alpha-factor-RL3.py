import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import yfinance as yf
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def download_daily_data(ticker, start_date):
    logging.info(f"Fetching daily data for {ticker} from {start_date}.")
    data = yf.download(ticker, start=start_date, interval="1d")
    return data

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

    # 保存因子到Excel
    data.to_excel("alpha_factors.xlsx", index=True)
    logging.info("Alpha factors saved to alpha_factors.xlsx.")
    
    return data

def prepare_data(data):
    logging.info("Preparing data...")
    features = ['rsi', 'macd', 'macd_signal']
    
    # Check if data is empty
    if data.empty:
        logging.error("No data available to prepare.")
        return data, features  # Return empty data and features if no data is available

    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    logging.info("Data preparation completed.")
    return data, features

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = float(target)
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(data, features, initial_balance=100000, transaction_cost=0.001, train_split=0.75):
    train_size = int(len(data) * train_split)
    train_data = data.iloc[:train_size]
    
    # Check if train_data is empty
    if train_data.empty:
        logging.error("Training data is empty. Cannot train the agent.")
        return None, None, None  # Return None values to indicate failure

    test_data = data.iloc[train_size:]
    state_size = len(features)
    action_size = 3
    agent = DQLAgent(state_size, action_size)
    batch_size = 32
    logging.info("Starting training...")
    for episode in range(10):
        state = train_data[features].iloc[0].values.reshape(1, -1)
        cash, holdings = initial_balance, 0
        accumulated_reward = 0
        previous_buy_price = 0
        for t in range(len(train_data) - 1):
            current_price = train_data['Close'].iloc[t]
            next_price = train_data['Close'].iloc[t + 1]
            action = agent.act(state)

            if action == 0:  # 买入逻辑
                max_shares = int(cash // (current_price.iloc[0] * (1 + transaction_cost)))
                if max_shares > 0:
                    transaction_cost_value = max_shares * current_price.iloc[0] * transaction_cost
                    cash -= max_shares * current_price.iloc[0] * (1 + transaction_cost)
                    holdings += max_shares
                    accumulated_reward -= transaction_cost_value
                    previous_buy_price = current_price.iloc[0]
            elif action == 1:  # 卖出逻辑
                if holdings > 0:
                    transaction_profit = holdings * (current_price.iloc[0] - previous_buy_price)
                    transaction_cost_value = holdings * current_price.iloc[0] * transaction_cost
                    cash += holdings * current_price.iloc[0] * (1 - transaction_cost)
                    holdings = 0
                    accumulated_reward += transaction_profit - transaction_cost_value
            else:  # 持有逻辑
                pass

            next_state = train_data[features].iloc[t + 1].values.reshape(1, -1)
            agent.remember(state, action, accumulated_reward, next_state, t == len(train_data) - 2)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Log key data
            logging.info(f"Episode: {episode + 1}, Step: {t + 1}, Cash: {cash:.2f}, Holdings: {holdings}, Pre_But_Price:{previous_buy_price}, "
                         f"Accumulated Reward: {accumulated_reward:.2f}")

    return agent, test_data, features

def backtest(agent, test_data, features, initial_balance=100000, transaction_cost=0.001):
    cash, holdings = initial_balance, 0
    portfolio_values = []
    buy_signals, sell_signals = [], []
    for t in range(len(test_data) - 1):
        state = test_data[features].iloc[t].values.reshape(1, -1)
        action = agent.act(state)
        current_price = test_data['Close'].iloc[t]
        if action == 0 and cash >= current_price * (1 + transaction_cost):  # 买入
            max_shares = int(cash // (current_price * (1 + transaction_cost)).iloc[0])
            cash -= max_shares * current_price * (1 + transaction_cost)
            holdings += max_shares
            buy_signals.append(test_data.index[t])
        elif action == 1 and holdings > 0:  # 卖出
            cash += holdings * current_price * (1 - transaction_cost)
            holdings = 0
            sell_signals.append(test_data.index[t])
        portfolio_values.append(cash + holdings * current_price)
    return portfolio_values, buy_signals, sell_signals

if __name__ == "__main__":
    ticker, start_date = "TSLA", "2020-01-01"
    data = download_daily_data(ticker, start_date)
    data = calculate_alpha_factors(data)
    data, features = prepare_data(data)
    agent, test_data, features = train_agent(data, features)

    # Check if agent and test_data are valid before backtesting
    if agent is not None and test_data is not None:
        portfolio_values, buy_signals, sell_signals = backtest(agent, test_data, features)
    else:
        logging.error("Agent training failed or test data is empty. Backtesting cannot proceed.")
        exit()  # Prevents the code from trying to plot None values

    plt.figure(figsize=(14, 7))
    # Check if test_data is valid before plotting
    if test_data is not None:
        plt.plot(test_data['Close'], label='Close Price')
        plt.scatter(test_data.index[buy_signals], test_data['Close'].iloc[buy_signals], marker='^', color='g', label='Buy Signal', alpha=1)
        plt.scatter(test_data.index[sell_signals], test_data['Close'].iloc[sell_signals], marker='v', color='r', label='Sell Signal', alpha=1)
        plt.title("Trading Strategy - Buy & Sell Signals")
        plt.legend()
        plt.show()
    else:
        logging.error("No valid test data available for plotting.")

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.legend()
    plt.show()