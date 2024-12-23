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
import os  # Add this import statement
from tensorflow.keras.regularizers import l1  # 添加L1正则化器的导入
import sys  # 添加导入语句
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def download_daily_data(ticker, start_date):
    logging.info(f"Fetching daily data for {ticker} from {start_date}.")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the latest date
    
    retries = 5  # 最大重试次数
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if data.empty:
                raise ValueError("No data returned from yfinance.")
            return data
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:  # 如果不是最后一次尝试
                time.sleep(10)  # 等待 10 秒后重试
            else:
                logging.critical("All attempts to fetch data failed.")
                raise  # 重新抛出异常以便处理

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
    def __init__(self, state_size, action_size, transaction_cost=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.transaction_cost = transaction_cost
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu', kernel_initializer=HeNormal()),  # 添加L1正则化
            Dropout(0.3),
            Dense(64, activation='relu', kernel_initializer=HeNormal()),  # 添加L1正则化
            Dropout(0.3),
            Dense(self.action_size, activation='linear', kernel_initializer=HeNormal())  # 最后一层不需要正则化
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss='mean_squared_error')
        
        # 打印模型结构
        model.summary()
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        predicted_action = np.argmax(q_values[0])
        if predicted_action == 0 and np.max(q_values) < self.transaction_cost * 100:  # 买入动作预测收益过低
            return 2, q_values  # 持有
        elif predicted_action == 1 and np.max(q_values) < self.transaction_cost * 100:
            return 2, q_values  # 持有
        return predicted_action, q_values

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
        targets = np.clip(targets, -10, 10)  # 确保目标值不爆炸

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
    
    def replay_n(self, batch_size, n_step=3):
        # Ensure there are enough samples in memory
        if len(self.memory) < batch_size + n_step:
            return  # Not enough samples to replay

        # From memory, sample continuous time sequences
        valid_indices = [i for i in range(len(self.memory) - n_step)]
        minibatch_start_indices = np.random.choice(valid_indices, batch_size, replace=False)

        memory_list = list(self.memory)
        minibatch = [memory_list[i:i + n_step] for i in minibatch_start_indices]

        states = []
        actions = []
        targets = []
        # 遍历每个时间序列样本
        for sequence in minibatch:
            assert len(sequence) == n_step, "Sequence length is not consistent with n_step."

            G = 0  # 初始化累积奖励
            gamma_acc = 1  # 折扣因子的累积值
            # 从后往前计算 n 步奖励
            for step in reversed(range(len(sequence))):  # 从后往前遍历时间序列
                state, action, reward, next_state, done = sequence[step]
                G = reward + gamma_acc * G  # 累积奖励
                gamma_acc *= self.gamma  # 折扣因子递增
                if done:  # 如果遇到终止状态，停止累积
                    break

            # 加入未来 Q 值（仅适用于非终止状态）
            _, action, _, next_state, done = sequence[0]  # 当前序列的起始样本
            if not done:
                next_q_value = np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
                next_q_value = np.clip(next_q_value, -1e4, 1e4)  # 限制未来 Q 值范围
                G += gamma_acc * next_q_value

            # 保存状态、动作和目标值
            states.append(state)
            actions.append(action)
            targets.append(G)

        # 转换为 NumPy 数组
        states = np.array(states).reshape(batch_size, -1)  # Ensure correct shape
        actions = np.array(actions).reshape(batch_size, -1)
        targets = np.array(targets).reshape(batch_size, -1)

        # Ensure states has the correct shape for the model
        if len(states.shape) != 2 or states.shape[1] != self.state_size:
            raise ValueError(f"Invalid states shape: {states.shape}. Expected shape: (batch_size, {self.state_size}).")

        # Get current Q values
        target_f = self.model.predict(states, verbose=0)
        for i in range(batch_size):
            assert 0 <= actions[i] < self.action_size, f"Invalid action {actions[i]} detected."
            target_f[i][actions[i]] = targets[i]

        # 批量更新模型
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # 动态调整 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
   
    def replay_n2(self, batch_size, n_step=3):
        """
        优化的经验回放函数，支持 n 步奖励从后往前计算，提升训练效率。
        """
        if len(self.memory) < batch_size + n_step:
            return  # Not enough samples to replay

        # 从 memory 中随机采样连续时间序列
        valid_indices = [i for i in range(len(self.memory) - n_step)]
        minibatch_start_indices = np.random.choice(valid_indices, batch_size, replace=False)
        memory_list = list(self.memory)
        minibatch = [memory_list[idx:idx + n_step] for idx in minibatch_start_indices]
        # 批量构建数据
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for sequence in minibatch:
            rewards_seq = np.array([s[2] for s in reversed(sequence)])
            dones_seq = np.array([s[4] for s in reversed(sequence)])
            gamma_vector = np.cumprod([1] + [self.gamma] * (len(rewards_seq) - 1))[:len(rewards_seq)]  # 折扣因子向
            G = np.dot(rewards_seq, gamma_vector)  # 矢量化计算奖励
            if not dones_seq[0]:
                next_states.append(sequence[0][3])
                rewards.append(G)
                actions.append(sequence[0][1])
                states.append(sequence[0][0])
            #print(sequence[0][1],rewards_seq, G)

        next_states = np.array(next_states).reshape(batch_size, -1)
        if len(next_states) > 0:
            next_q_values = self.model.predict(np.array(next_states), verbose=0)
            max_next_q_values = np.amax(next_q_values, axis=1)
            rewards += self.gamma * max_next_q_values

        # 构造目标 Q 值
        states = np.array(states).reshape(batch_size, -1)
        actions = np.array(actions).reshape(batch_size, -1)
        rewards = np.array(rewards).reshape(batch_size, -1)

        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(len(actions)), actions] = rewards
        # 批量更新模型
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # 动态调整 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


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
def calculate_reward(current_value, previous_value, action, last_action, holding_period, next_price, transaction_cost=0.001):
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
        reward = price_change
        if price_change <= 0.005:
            reward -= 0.001
        #reward += price_change  # 根据市场表现奖励或惩罚持有
        if holding_period == 5:  # 连续持有的奖励
            reward += 0.001

    # 回撤惩罚
    drawdown = (previous_value - current_value) / previous_value
    if drawdown >= 0.05:  # 如果回撤超过5%
        reward -= 0.5 * drawdown  # 动态惩罚，回撤越大惩罚越大

    if abs(reward) < 0.001:
        reward -= 0.01  # 惩罚低收益交易
    return reward

def calculate_reward_2(current_value, previous_value, action, last_action, holding_period, current_price, next_price, holdings, transaction_cost=0.001):
    """
    改进后的奖励函数，增加对未来趋势、持有时间、回撤的动态调整。
    """
    if previous_value == 0:
        previous_value = 1e-10  # 避免除零

    reward = 0  # 初始化奖励
    price_change = 0.0
    
    # 动作奖励逻辑
    if action == 0:  # 买入
        reward = (current_value - previous_value) / previous_value  # 买入后的资产增值
        reward -= transaction_cost  # 扣除交易成本

    elif action == 1:  # 卖出
        #reward = (previous_value - current_value) / previous_value
        reward -= transaction_cost  # 扣除交易成本
        # 增加对未来价格下跌的奖励
        future_trend_penalty = (current_price - next_price) / current_price
        # Ensure future_trend_penalty is a scalar
        future_trend_penalty = future_trend_penalty.item()  # Convert to scalar if it's a Series
        if future_trend_penalty > 0:
            reward += future_trend_penalty
        else:
            reward -= abs(future_trend_penalty)

    elif action == 2:  # 持有
        price_change = (next_price - current_price) / current_price
        if holdings > 0:  # 持股期持有
            reward += price_change * 0.5  # 股价变化的奖励
            # 持有时间奖励
            if price_change >= 0.01:
                reward += holding_period * 0.0001
        else:  # 非持股期持有
            if price_change <= 0:
                reward += abs(price_change) * 0.5  # 规避风险的奖励
            elif price_change > 0:
                reward -= price_change             # 错失机会的惩罚
    
    # 动态回撤惩罚
    drawdown = (previous_value - current_value) / previous_value
    if drawdown >= 0.05:
        reward -= drawdown

    logging.info(f"Action: {action}, Current Price: {current_price}, Next Price: {next_price}, Reward: {reward}, Drawdown: {drawdown}, Pct_change:{price_change}")  # New logging statement

    # 扩大奖励值范围
    reward *= 100.0
    return reward

def train_agent(data, features, initial_balance=100000.0, transaction_cost=0.001, train_split=0.8, model_save_path="dql_agent_model.h5"):
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
    batch_size = 64
    logging.info("Starting training...")
    epoch_num = 10
    epoch_rewards = []  # 记录每个 epoch 的总奖励
    q_values = []
    for episode in range(epoch_num):
        state = train_data[features].iloc[0].values.reshape(1, -1)
        cash, holdings = initial_balance, 0
        previous_value = initial_balance
        last_action = 2  # 初始为持有
        holding_period = 0  # 持有时间计数
        total_reward = 0.0  # Initialize as float instead of just 0
        cash = float(cash)
        for t in range(len(train_data) - 1):
            current_price = train_data['Close'].iloc[t].iloc[0]
            next_price = train_data['Close'].iloc[t + 1].iloc[0]
            action, q_values = agent.act(state)

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
                holding_period = 0  # 重置持有时间

            elif action == 2:  # 持有逻辑
                if holdings >= 0:
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
            reward = 0.0
            reward = calculate_reward_2(current_value, previous_value, action, last_action, holding_period, current_price, next_price, holdings,transaction_cost)
            total_reward += reward.item() if isinstance(reward, pd.Series) else reward
            
            # 状态更新
            next_state = train_data[features].iloc[t + 1].values.reshape(1, -1)
            agent.remember(state, action, reward, next_state, t == len(train_data) - 1)
            state = next_state
            last_action = action 

            # 经验回放
            n_step = 3
            #if len(agent.memory) > batch_size + n_step:
            #    agent.replay_n2(batch_size, n_step)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            
            # 日志输出
            logging.info(f"Episode: {episode + 1},Action:{action},Step: {t + 1},Cash: {cash:.2f},Holdings: {holdings}, "
                         f"Current Price: {current_price:.2f}, Next_price:{next_price:.2f},Current Value:{current_value:.2f},Previous Value:{previous_value:.2f},Reward:{reward},Total_reward:{total_reward:.4f}")
            previous_value = current_value
        # 记录每个 epoch 的总奖励
        total_value = cash + holdings_value * next_price
        epoch_reward = total_value - initial_balance
        epoch_rewards.append(epoch_reward)  # 记录每个 epoch 的总奖励
        total_value = float(total_value)
        epoch_reward = float(epoch_reward)
        logging.info(f"Epoch: {episode + 1}, Total Value: {total_value:.2f}, Total Reward: {epoch_reward:.2f}")

    # 在训练结束后绘制总奖励图
    plt.figure(figsize=(14, 7))
    plt.plot(epoch_rewards, label="Total Reward per Epoch")
    plt.title("Total Reward Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.close()  # 关闭图像以释放资源

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
        action,q_values = agent.act(state)  # Ensure action is a scalar
        print(q_values)
        current_price = test_data['Close'].iloc[t].iloc[0]
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
    # 获取 ticker 参数，默认为 "BABA"
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BABA"  # 处理命令行参数

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
    output_dir = f"./data/DQN-RL-Daily/{ticker}-{timestamp}"  # 创建文件夹名，添加 ticker 前缀
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹

    # Configure logging to save to a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=os.path.join(output_dir, 'trading_log.txt'),  # Update the log file path
        filemode='w'  # 'w' to overwrite the file each time, 'a' to append
    )

    start_date = "2020-01-01"
    print(ticker)
    data = download_daily_data(ticker, start_date)
    data = calculate_alpha_factors(data)
    data, features = prepare_data(data)
    

    # 继续执行后续代码
    agent, test_data, features, epoch_rewards = train_agent(data, features, model_save_path=os.path.join(output_dir, "dql_agent_mlp_model_daily.h5"))
    
    state_size = len(features)
    action_size = 3  # 动作空间：0=买入, 1=卖出, 2=持有
    agent = DQLAgent(state_size, action_size)  # Reinitialize the agent

    # Check if agent and test_data are valid before backtesting
    if agent is not None and test_data is not None:
        portfolio_values, buy_signals, sell_signals = backtest(agent, test_data, features, model_path=os.path.join(output_dir, "dql_agent_mlp_model_daily.h5"))
    else:
        logging.error("Agent training failed or test data is empty. Backtesting cannot proceed.")
        exit()  # Prevents the code from trying to plot None values

    plt.figure(figsize=(14, 7))
    if test_data is not None:
        plt.plot(test_data['Close'], label='Close Price')
        buy_signal_indices = [test_data.index.get_loc(ts) for ts in buy_signals]
        plt.scatter(test_data.index[buy_signal_indices], test_data['Close'].iloc[buy_signal_indices], marker='^', color='g', label='Buy Signal', alpha=1)
        sell_signal_indices = [test_data.index.get_loc(ts) for ts in sell_signals]
        plt.scatter(test_data.index[sell_signal_indices], test_data['Close'].iloc[sell_signal_indices], marker='v', color='r', label='Sell Signal', alpha=1)
        plt.title("Trading Strategy - Buy & Sell Signals")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "trading_strategy_signals.png"))  # 保存图像
        #plt.show()  # 显示图像
        plt.close()  # 关闭图像以释放资源
    else:
        logging.error("No valid test data available for plotting.")

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "portfolio_value_over_time.png"))  # 保存图像
    #plt.show()  # 显示图像
    plt.close()  # 关闭图像以释放资源