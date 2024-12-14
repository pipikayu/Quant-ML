import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import yfinance as yf
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 数据下载函数
def download_daily_data(ticker, start_date):
    logging.info(f"Fetching daily data for {ticker} from {start_date}.")
    return yf.download(ticker, start=start_date, interval="1d")

# Alpha因子计算函数
def calculate_alpha_factors(data):
    logging.info("Calculating alpha factors...")
    window = 14

    # RSI因子
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_mean_dev'] = (data['rsi'] - data['rsi'].rolling(window).mean()) / data['rsi'].rolling(window).mean()
    data['rsi_momentum'] = data['rsi'] - data['rsi'].shift(window)

    # MACD因子
    ema_short = data['Close'].ewm(span=12).mean()
    ema_long = data['Close'].ewm(span=26).mean()
    data['macd'] = ema_short - ema_long
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_crossover'] = np.where(data['macd'] > data['macd_signal'], 1, -1)

    # KDJ因子
    low_n = data['Low'].rolling(window).min()
    high_n = data['High'].rolling(window).max()
    data['%K'] = (data['Close'] - low_n) / (high_n - low_n) * 100
    data['%D'] = data['%K'].rolling(3).mean()
    data['%J'] = 3 * data['%K'] - 2 * data['%D']

    # 动量因子
    data['momentum'] = data['Close'] - data['Close'].shift(window)
    data['price_roc'] = (data['Close'] - data['Close'].shift(window)) / data['Close'].shift(window)
    data['bias'] = (data['Close'] - data['Close'].rolling(window).mean()) / data['Close'].rolling(window).mean()

    data.fillna(0, inplace=True)
    logging.info("Alpha factors calculated.")
    return data

# 数据准备
def prepare_data(data):
    logging.info("Preparing data...")
    features = [
        'rsi', 'rsi_mean_dev', 'rsi_momentum', 'macd', 'macd_signal', 'macd_crossover',
        '%K', '%D', '%J', 'momentum', 'price_roc', 'bias'
    ]
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    logging.info("Data preparation completed.")
    return data, features

# 强化学习模型
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
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
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
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = float(target) if isinstance(target, (int, float)) else float(target.iloc[0])
            states.append(state[0])
            targets.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets), batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 训练与回测
def train_agent(data, features, initial_balance=100000, train_split=0.75):
    train_size = int(len(data) * train_split)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    state_size = len(features)
    action_size = 3
    agent = DQLAgent(state_size, action_size)
    batch_size = 32
    logging.info("Starting training...")
    for episode in range(10):
        state = train_data[features].iloc[0].values.reshape(1, -1)
        cash = initial_balance
        holdings = 0
        start_time = time.time()
        for t in range(len(train_data) - 1):
            action = agent.act(state)
            next_state = train_data[features].iloc[t + 1].values.reshape(1, -1)
            reward = train_data['Close'].iloc[t + 1] - train_data['Close'].iloc[t] if action == 0 else 0
            reward = float(reward.iloc[0]) if isinstance(reward, pd.Series) else float(reward)
            agent.remember(state, action, reward, next_state, t == len(train_data) - 2)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            logging.info(f"Episode {episode + 1}, Step {t}, Reward: {reward:.2f}")
        logging.info(f"Episode {episode + 1} completed in {time.time() - start_time:.2f}s")
    return agent, test_data, features

def backtest(agent, test_data, features):
    cash, holdings = 100000, 0
    portfolio_values = []
    buy_signals, sell_signals = [], []
    for t in range(len(test_data) - 1):
        action = agent.act(test_data[features].iloc[t].values.reshape(1, -1))
        price = test_data['Close'].iloc[t]
        if action == 0 and cash > price:
            holdings += 1
            cash -= price
            buy_signals.append(test_data.index[t])
        elif action == 1 and holdings > 0:
            holdings -= 1
            cash += price
            sell_signals.append(test_data.index[t])
        portfolio_values.append(cash + holdings * price)
    return portfolio_values, buy_signals, sell_signals

# 主程序
if __name__ == "__main__":
    ticker, start_date = "TSLA", "2020-01-01"
    data = download_daily_data(ticker, start_date)
    data = calculate_alpha_factors(data)
    data, features = prepare_data(data)
    agent, test_data, features = train_agent(data, features)
    portfolio_values, buy_signals, sell_signals = backtest(agent, test_data, features)

    # 可视化结果
    plt.figure(figsize=(14, 7))
    plt.plot(test_data['Close'], label='Close Price')
    plt.scatter(test_data.index[buy_signals], test_data['Close'].iloc[buy_signals], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(test_data.index[sell_signals], test_data['Close'].iloc[sell_signals], marker='v', color='r', label='Sell Signal', alpha=1)
    plt.title("Trading Strategy - Buy & Sell Signals")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.legend()
    plt.show()