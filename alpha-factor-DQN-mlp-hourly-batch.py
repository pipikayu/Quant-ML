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

def download_hourly_data(ticker, start_date):
    logging.info(f"Fetching hourly data for {ticker} from {start_date}.")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the latest date
    all_data = pd.DataFrame()  # 用于存储所有数据

    # 设置滑动窗口
    current_start_date = pd.to_datetime(start_date)
    while current_start_date < pd.to_datetime(end_date):
        # 计算当前请求的结束日期，最多获取60天的数据
        current_end_date = (current_start_date + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        
        try:
            data = yf.download(ticker, start=current_start_date.strftime("%Y-%m-%d"), end=current_end_date, interval="1h")
        except Exception as e:
            logging.error(f"Error fetching data for {ticker} from {current_start_date} to {current_end_date}: {e}")
            current_start_date += pd.Timedelta(days=60)  # 更新开始日期，继续尝试下一个时间段
            continue  # 继续尝试下一个时间段

        # Log the number of rows fetched
        logging.info(f"Fetched {len(data)} rows of data from {current_start_date.strftime('%Y-%m-%d')} to {current_end_date}.")  # 新增日志记录
        
        if data.empty:
            logging.warning(f"No data fetched from {current_start_date.strftime('%Y-%m-%d')} to {current_end_date}.")  # Log warning for empty data
            current_start_date += pd.Timedelta(days=60)  # 更新开始日期，继续尝试下一个时间段
            continue  # 继续尝试下一个时间段
        
        all_data = pd.concat([all_data, data])  # 拼接数据
        current_start_date = pd.to_datetime(current_end_date) + pd.Timedelta(hours=1)  # 更新开始日期为当前结束日期的下一小时

        # 如果获取的数据超过730天，停止获取
        if (datetime.datetime.now() - pd.to_datetime(current_start_date)).days > 730:
            break

    # 保存获取到的原始数据到本地文件
    all_data.to_csv(f"{ticker}_hourly_data.csv")  # 保存为 CSV 文件
    logging.info(f"Hourly data saved to {ticker}_hourly_data.csv.")

    return all_data

def calculate_alpha_factors(data):
    logging.info("Calculating alpha factors...")
    
    # Convert timezone-aware datetimes to timezone-unaware
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)  # Remove timezone information

    # Check the columns in the DataFrame
    logging.info(f"Data columns: {data.columns.tolist()}")  # Log the columns present in the DataFrame

    if 'Close' not in data.columns:
        logging.error("The 'Close' column is missing from the data.")
        return data  # Return the data without processing if 'Close' is missing

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

    # Check for NaN or infinite values
    if data[features].isnull().any().any():
        logging.error("Data contains NaN values.")
        data[features].fillna(0, inplace=True)  # Optionally fill NaN values with 0 or handle as needed

    # Check for infinite values before replacement
    if np.isinf(data[features]).any().any():
        logging.warning("Data contains infinite values before replacement.")

    # Ensure there are no infinite values
    data[features] = data[features].replace([np.inf, -np.inf], 0)  # Ensure infinite values are replaced
    # Check for infinite values after replacement
    if np.isinf(data[features]).any().any():
        logging.error("Data still contains infinite values after replacement.")
        # Add logging to identify which features contain infinite values
        logging.error(f"Features with infinite values: {data[features][np.isinf(data[features])].dropna(how='all')}")
        raise ValueError("Data contains infinite values after replacement.")

    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
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
        # 按时间顺序从 memory 中选择 minibatch
        # 采样时要保证时间上的连续性
        minibatch = []
        for i in range(batch_size):
            idx = random.randint(0, len(self.memory) - 2)  # 防止越界，确保 next_state 存在
            state, action, reward, next_state, done = self.memory[idx]
            minibatch.append((state, action, reward, next_state, done))

        # 提取所有状态和下一个状态
        states = np.array([x[0] for x in minibatch]).reshape(batch_size, -1)
        next_states = np.array([x[3] for x in minibatch]).reshape(batch_size, -1)
        

        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # 计算 next_states 对应的最大 Q 值
        next_q_values = self.model.predict(next_states, verbose=0)  # 批量预测
        next_q_values = np.amax(next_q_values, axis=1)

        # 计算 target
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        targets = np.clip(targets, -1e6, 1e6)  # 确保目标值不爆炸

        # 获取当前状态对应的 Q 值
        target_f = self.model.predict(states, verbose=0)  # 批量预测

        # 更新每个样本的 target_f，目标是根据动作索引来更新
        for i in range(batch_size):
            target_f[i][actions[i]] = targets[i]

        # 执行一次批量训练
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # 更新 epsilon，逐步降低探索概率
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
def calculate_reward(current_value, previous_value, action, last_action, holding_period, buy_price, next_price, transaction_cost=0.001):
    """
    根据收益率、交易成本、持有时间和回撤计算奖励。
    """
    if previous_value == 0:
        previous_value = 1e-10  # 避免除零

    reward = 0  # 初始化奖励

    # 动作奖励逻辑
    if action == 0:  # 买入
        reward = (current_value - previous_value) / previous_value  # 买入后的资产增值
        reward -= transaction_cost  # 扣除交易成本

    elif action == 1:  # 卖出
        reward = (previous_value - current_value) / previous_value
        reward -= transaction_cost  # 扣除交易成本

    elif action == 2:  # 持有
        price_change = (current_value - previous_value) / previous_value  # 计算价格变化
        reward += price_change  # 根据市场表现奖励或惩罚持有
        if holding_period >= 5:  # 连续持有的奖励
            reward += 0.001

    # 回撤惩罚
    drawdown = (previous_value - current_value) / previous_value
    if drawdown >= 0.05:  # 如果回撤超过5%
        reward -= 0.05 * drawdown  # 动态惩罚，回撤越大惩罚越大

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
    epoch_num = 50
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
                cash += holdings * current_price * (1 - transaction_cost)  # 计算卖出后现金
                holdings = 0  # 清空持仓

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
            if t + 1 < len(train_data):  # 确保不越界
                next_price = train_data['Close'].iloc[t + 1]  # 获取下一个时间步的价格
            else:
                next_price = 0.0
            reward = calculate_reward(current_value, previous_value, action, last_action, holding_period, next_price, transaction_cost)
            total_reward += reward
            
            # 状态更新
            next_state = train_data[features].iloc[t + 1].values.reshape(1, -1)
            agent.remember(state, action, reward, next_state, t == len(train_data) - 2)
            state = next_state
            last_action = action
            #previous_value = current_value

            # 经验回放
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # 日志输出
            cash = float(cash)  # Ensure scalar value is used
            current_price = float(current_price.iloc[0])
            #logging.info(f"Episode: {episode + 1}, Step: {t + 1}, Action: {action}, Cash: {cash:.2f}, Holdings: {holdings_value:.2f}, "
            #             f"Reward: {reward:.4f}, Total Value: {current_value:.2f}")
            logging.info(f"Episode: {episode + 1},Action:{action},Step: {t + 1},Cash: {cash:.2f},Holdings: {holdings}, "
                         f"Current Price: {current_price:.2f},Current Value:{current_value:.2f},Previous Value:{previous_value:.2f},Reward:{reward:.4f},Total_reward:{total_reward:.4f}")
            previous_value = current_value
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
    ticker, start_date = "TSLA", (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")  # 设置为730天前的日期
    data = download_hourly_data(ticker, start_date)
    data = calculate_alpha_factors(data)
    data, features = prepare_data(data)
    
    agent, test_data, features, epoch_rewards = train_agent(data, features, model_save_path="dql_agent_mlp_model_hourly.h5")
    
    state_size = len(features)
    action_size = 3  # 动作空间：0=买入, 1=卖出, 2=持有
    agent = DQLAgent(state_size, action_size)  # Reinitialize the agent

    # Check if agent and test_data are valid before backtesting
    if agent is not None and test_data is not None:
        portfolio_values, buy_signals, sell_signals = backtest(agent, test_data, features, model_path="dql_agent_mlp_model_hourly.h5")
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