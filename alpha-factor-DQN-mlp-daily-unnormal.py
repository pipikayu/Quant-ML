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
import datetime  # Add this import statement
from tensorflow.keras.initializers import HeNormal  # 导入 HeNormal 初始化器

# Configure logging to save to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename='trading_log.txt',  # Specify the log file name
    filemode='w'  # 'w' to overwrite the file each time, 'a' to append
)

def download_daily_data(ticker, start_date):
    logging.info(f"Fetching daily data for {ticker} from {start_date}.")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the latest date
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
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
    features = [
        'rsi', 'macd', 'macd_signal', 
        'K', 'D', 'J', 
        'SMA_50', 'SMA_200', 
        'golden_cross', 'death_cross', 
        'magic_nine', 
        'Bollinger_Upper', 'Bollinger_Lower', 
        'momentum', 'volume_change', 
        'ATR','high_low','high_close','low_close','tr'
    ]
    
    # Check if data is empty
    if data.empty:
        logging.error("No data available to prepare.")
        return data, features  # Return empty data and features if no data is available

    #scaler = StandardScaler()
    #data[features] = scaler.fit_transform(data[features])
    logging.info("Data preparation completed.")
    
    # 打印特征数
    logging.info(f"Number of features: {len(features)}")
    
    # 保存特征到本地文件
    features_df = data[features]
    features_df.to_csv("features.csv", index=False)  # 保存为 CSV 文件
    logging.info("Features saved to features.csv.")
    
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
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu', kernel_initializer=HeNormal()),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_initializer=HeNormal()),
            Dropout(0.3),
            Dense(self.action_size, activation='linear', kernel_initializer=HeNormal())
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss='mean_squared_error')
        
        # 打印模型结构
        model.summary()
        
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
            target = np.clip(target, -1e6, 1e6)  # 限制 target 的范围
            target_f[0][action] = np.float32(target.item())  # Extract scalar value
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, model_filename):
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
        #model_filename = f"dql_agent_model_{timestamp}.h5"  # 创建文件名
        self.model.save(model_filename)  # 保存模型
        logging.info(f"Model saved to {model_filename}.")  # 记录保存信息

    def load_agent_model(self, model_path):
        """Load a model from the specified path."""
        self.model = tf.keras.models.load_model(model_path, custom_objects={'mean_squared_error': tf.keras.losses.MeanSquaredError()})
        logging.info(f"Model loaded from {model_path}.")  # 记录加载信息

# 奖励函数的设计：如果卖出时股价涨幅较大，给予高奖励；如果持有股价下跌，给予惩罚。
def calculate_reward(current_value, previous_value, action, last_action, holding_period, transaction_cost=0.001):
    """
    根据收益率、交易成本和持有时间步数计算奖励。
    """
    if previous_value == 0:
        previous_value = 1e-10  # 避免除零

    # 计算收益率
    reward = (current_value - previous_value) / previous_value

    # 动作奖励逻辑
    if action == 0:  # 买入
        reward -= transaction_cost  # 买入时扣除交易成本
    elif action == 1:  # 卖出
        reward -= transaction_cost  # 卖出时扣除交易成本
        if current_value > previous_value:
            reward += 0.0005  # 小额正向奖励
        else:
            reward -= 0.0005  
    elif action == 2:  # 持有
        if holding_period == 5:  # 如果连续持有
            reward += 0.001  # 小额正向奖励

    # 计算回撤
    drawdown = (previous_value - current_value) / previous_value
    if drawdown >= 0.05:  # 如果回撤超过10%
        reward -= 0.02 # 添加惩罚

    return reward

def train_agent(data, features, initial_balance=100000.0, transaction_cost=0.001, train_split=0.75, model_save_path="dql_agent_model.h5"):
    """
    训练 DQN 智能体，基于给定的市场数据和特征。
    """
    train_size = int(len(data) * train_split)
    train_data = data.iloc[:train_size]
    
    if train_data.empty:
        logging.error("Training data is empty. Cannot train the agent.")
        return None, None, features, []

    test_data = data.iloc[train_size:]
    state_size = len(features)
    action_size = 3  # 动作空间：0=买入, 1=卖出, 2=持有
    agent = DQLAgent(state_size, action_size)
    batch_size = 32
    logging.info("Starting training...")
    epoch_num = 6
    epoch_rewards = []

    for episode in range(epoch_num):
        state = train_data[features].iloc[0].values.reshape(1, -1)
        cash, holdings = initial_balance, 0
        previous_value = initial_balance
        last_action = 2  # 初始为持有
        holding_period = 0  # 持有时间计数
        total_reward = 0
        cash = float(cash)
        for t in range(len(train_data) - 1):
            current_price = train_data['Close'].iloc[t]
            next_price = train_data['Close'].iloc[t + 1]
            action = agent.act(state)

            # 更新资金、持仓和持有期
            if action == 0:  # 买入逻辑
                max_shares = int(cash // (current_price * (1 + transaction_cost)).item())
                if max_shares > 0:
                    cash -= max_shares * current_price * (1 + transaction_cost)
                    holdings += max_shares
                    holding_period = 0  # 重置持有时间

            elif action == 1:  # 卖出逻辑
                if holdings > 0:
                    cash += holdings * current_price * (1 - transaction_cost)
                    holdings = 0
                    holding_period = 0  # 重置持有时间

            elif action == 2:  # 持有逻辑
                holding_period += 1

            # 确保 holdings 是标量
            if isinstance(holdings, (pd.Series, np.ndarray)):
                holdings_value = holdings.item()  # 转换为标量
            else:
                holdings_value = holdings

            # Ensure holdings_value is a scalar
            holdings_value = holdings_value.item() if isinstance(holdings_value, (pd.Series, np.ndarray)) else holdings_value
            
            # 计算当前资产价值
            current_value = cash + holdings_value * next_price  # 确保 current_value 是标量
            current_value = current_value.item()  # Convert to scalar if it's a Series

            # 计算奖励
            reward = 0
            reward = calculate_reward(current_value, previous_value, action, last_action, holding_period, transaction_cost)
            total_reward += reward

            # 状态更新
            next_state = train_data[features].iloc[t + 1].values.reshape(1, -1)
            agent.remember(state, action, reward, next_state, t == len(train_data) - 2)
            state = next_state
            last_action = action
            previous_value = current_value

            # 经验回放
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # 日志输出
            cash = float(cash)  # Ensure scalar value is used
            current_price = float(current_price.iloc[0])
            #logging.info(f"Episode: {episode + 1}, Step: {t + 1}, Action: {action}, Cash: {cash:.2f}, Holdings: {holdings_value:.2f}, "
            #             f"Reward: {reward:.4f}, Total Value: {current_value:.2f}")
            logging.info(f"Episode: {episode + 1}, Action:{action} Step: {t + 1}, Cash: {cash:.2f}, Holdings: {holdings}, "
                         f"Current Price: {current_price:.2f}, Reward: {reward:.4f}， Total_reward:{total_reward:.4f}")

        # 记录每个 epoch 的总奖励
        total_value = cash + holdings_value * next_price
        epoch_reward = total_value - initial_balance
        epoch_rewards.append(epoch_reward)
        total_value = float(total_value)
        epoch_reward = float(epoch_reward)
        logging.info(f"Epoch: {episode + 1}, Total Value: {total_value:.2f}, Total Reward: {epoch_reward:.2f}")

    # 保存模型
    agent.save_model(model_save_path)
    return agent, test_data, features, epoch_rewards

def backtest(agent, test_data, features, initial_balance=100000.0, transaction_cost=0.001, model_path="dql_agent_model.h5"):
    """
    使用训练好的智能体进行回测，评估策略表现，并输出专业回测指标。
    """
    if model_path is not None:
        agent.load_agent_model(model_path)  # 加载训练好的模型
    
    cash, holdings = initial_balance, 0
    portfolio_values = []  # 保存每个时间步的资产价值
    daily_returns = []  # 保存每日收益率
    buy_signals, sell_signals = [], []

    for t in range(len(test_data) - 1):
        state = test_data[features].iloc[t].values.reshape(1, -1)
        action = agent.act(state)  # Ensure action is a scalar
        current_price = float(test_data['Close'].iloc[t])
        # 动作逻辑
        if action == 0 and cash >= current_price * (1 + transaction_cost):  # 买入
            max_shares = int(cash // (current_price * (1 + transaction_cost)))
            cash -= max_shares * current_price * (1 + transaction_cost)
            holdings += max_shares
            buy_signals.append(test_data.index[t])

        elif action == 1 and holdings > 0:  # 卖出
            cash += holdings * current_price * (1 - transaction_cost)
            holdings = 0
            sell_signals.append(test_data.index[t])

        # 计算当前组合的资产价值
        portfolio_value = cash + holdings * current_price
        portfolio_values.append(portfolio_value)

        # 计算日收益率（避免除零）
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)

        # 日志输出
        portfolio_value = float(portfolio_value)
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
    
    return portfolio_values, buy_signals, sell_signals


if __name__ == "__main__":
    ticker, start_date = "TSLA", "2020-01-01"
    data = download_daily_data(ticker, start_date)
    data = calculate_alpha_factors(data)
    data, features = prepare_data(data)
    
   
    agent, test_data, features, epoch_rewards = train_agent(data, features, model_save_path="dql_agent_model_daily.h5")
    
    state_size = len(features)
    action_size = 3  # 动作空间：0=买入, 1=卖出, 2=持有
    agent = DQLAgent(state_size, action_size)  # Reinitialize the agent

    # Check if agent and test_data are valid before backtesting
    if agent is not None and test_data is not None:
        portfolio_values, buy_signals, sell_signals = backtest(agent, test_data, features, model_path="dql_agent_model_daily.h5")
    else:
        logging.error("Agent training failed or test data is empty. Backtesting cannot proceed.")
        exit()  # Prevents the code from trying to plot None values

    plt.figure(figsize=(14, 7))
    # Check if test_data is valid before plotting
    if test_data is not None:
        plt.plot(test_data['Close'], label='Close Price')
        # Convert buy_signals to integer indices
        buy_signal_indices = [test_data.index.get_loc(ts) for ts in buy_signals]
        plt.scatter(test_data.index[buy_signal_indices], test_data['Close'].iloc[buy_signal_indices], marker='^', color='g', label='Buy Signal', alpha=1)
        # Convert sell_signals to integer indices
        sell_signal_indices = [test_data.index.get_loc(ts) for ts in sell_signals]
        plt.scatter(test_data.index[sell_signal_indices], test_data['Close'].iloc[sell_signal_indices], marker='v', color='r', label='Sell Signal', alpha=1)
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