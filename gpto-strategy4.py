import tushare as ts
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO,A2C, SAC  # Import SAC for more advanced algorithm
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna

# 设置 Tushare 的 Token
ts.set_token('f3470607326d522f19d7019357d93814209d66c9aa27be00b1ad3c50')  # 请替换为你自己的 Tushare API Token
pro = ts.pro_api()


class TradingEnv(gym.Env):
    """自定义交易环境，用于强化学习"""
    def __init__(self, df, initial_capital=100000):
        super(TradingEnv, self).__init__()
        self.df = df.select_dtypes(include=[np.number])  # Select only numeric columns
        self.initial_capital = initial_capital
        self.current_step = 0
        self.done = False
        self.position = 0
        self.cash = initial_capital
        self.portfolio_value = initial_capital

        # 定义动作和观察空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.df.columns),),  # Ensure this matches the actual number of columns
            dtype=np.float32
        )

    def reset(self):
        """重置环境状态"""
        self.current_step = 0
        self.done = False
        self.position = 0
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        return self._next_observation()

    def _next_observation(self):
        """获取下一个观察值"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)
        
        observation = self.df.iloc[self.current_step].values
        if np.any(np.isnan(observation)):
            print(f"NaN values found in observation at step {self.current_step}")
            observation = np.nan_to_num(observation)  # Replace NaN with zero or another strategy
        
        # Normalize observation
        std_dev = np.std(observation)
        if std_dev > 0:
            observation = (observation - np.mean(observation)) / (std_dev + 1e-8)
        else:
            observation = np.zeros_like(observation)  # or handle it in another way

        return observation

    def step(self, action):
        """执行给定动作并返回结果"""
        if self.current_step >= len(self.df):
            self.done = True
            return np.zeros(self.observation_space.shape), 0, self.done, {}

        current_price = self.df.iloc[self.current_step]['close']
        reward = 0

        # 将连续动作转换为离散的买/卖/持有决策
        if action < -0.5:
            action = 2  # 卖出
        elif action > 0.5:
            action = 1  # 买入
        else:
            action = 0  # 持有

        transaction_cost = 0.001  # 假设交易成本为0.1%
        
        # 执行买入操作
        if action == 1 and self.cash > 0:
            self.position = self.cash * (1-transaction_cost)/ current_price
            self.cash = 0
            self.portfolio_value -= self.cash * transaction_cost  # 扣除交易成本
        # 执行卖出操作
        elif action == 2 and self.position > 0:
            self.cash = self.position * current_price * ( 1-transaction_cost)
            self.position = 0
            #self.portfolio_value -= self.portfolio_value * transaction_cost  # 扣除交易成本
            self.portfolio_value = self.cash
        # 更新投资组合价值
        self.portfolio_value = self.cash + self.position * current_price
        reward = self.portfolio_value - self.initial_capital

        # 更新当前步骤
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.done = True
            return np.zeros(self.observation_space.shape), reward, self.done, {}

        # 确保观察值不包含NaN
        next_observation = self._next_observation()
        if np.any(np.isnan(next_observation)):
            print(f"NaN values found in observation at step {self.current_step}")
            next_observation = np.nan_to_num(next_observation)  # 用零或其他策略替换NaN

        return next_observation, reward, self.done, {}

def calculate_alpha_factors(df):
    """计算经典和扩展的alpha因子"""
    # 使用NumPy进行更高效的计算
    close = df['close'].values
    vol = df['vol'].values
    high = df['high'].values
    low = df['low'].values

    # Momentum Factor
    momentum = np.diff(close, n=10) / close[:-10]
    df['momentum'] = np.concatenate(([0] * 10, momentum))  # Pad with zeros to match the DataFrame length

    # Volatility Factor
    df['historical_volatility'] = pd.Series(close).rolling(window=14).std().values

    # Volume Factor
    obv = np.cumsum(np.sign(np.diff(close)) * vol[1:])
    df['obv'] = np.concatenate(([0], obv))  # Prepend zero to match DataFrame length

    # Trend Factor
    df['sma'] = pd.Series(close).rolling(window=20).mean().values

    # Mean Reversion Factor
    rolling_mean = pd.Series(close).rolling(window=20).mean().values
    rolling_std = pd.Series(close).rolling(window=20).std().values
    df['z_score'] = (close - rolling_mean) / rolling_std

    # Extended Alpha Factors
    # Relative Strength Index (RSI)
    df['rsi'] = calculate_rsi(df['close'])

    # Moving Average Convergence Divergence (MACD)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

    # Bollinger Bands
    df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df['close'])

    # Parabolic SAR
    df['psar'] = calculate_parabolic_sar(df)

    # Beta
    df['beta'] = calculate_beta(df)

    # New Alpha Factors
    # Amihud Illiquidity
    df['amihud'] = (df['close'].diff().abs() / df['vol']).rolling(window=20).mean()

    # Stochastic Oscillator
    df['stochastic_k'] = ((df['close'] - df['low'].rolling(window=14).min()) /
                          (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * 100

    df.fillna(0, inplace=True)
    return df

def calculate_rsi(data, window=14):
    """计算相对强弱指数（RSI）"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """计算移动平均收敛/发散（MACD）"""
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal, macd - signal

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """计算布林带"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_parabolic_sar(df, af=0.02, max_af=0.2):
    """Calculate Parabolic SAR"""
    # This is a simplified version of the Parabolic SAR calculation
    # You may need to adjust it based on your specific requirements
    psar = df['close'].copy()
    psar.fillna(0, inplace=True)
    return psar

def calculate_beta(df, market_return=None):
    """计算Beta值"""
    if market_return is None:
        market_return = df['close'].pct_change()
    stock_return = df['close'].pct_change()
    covariance = stock_return.rolling(window=252).cov(market_return)
    variance = market_return.rolling(window=252).var()
    beta = covariance / variance
    return beta

def train_rl_agent(df, algorithm='PPO', n_runs=5):
    """Train RL agent using a specified algorithm."""
    env = DummyVecEnv([lambda: TradingEnv(df) for _ in range(4)])
    
    # Select the algorithm based on the input parameter
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, buffer_size=100000, learning_starts=1000, batch_size=64)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Run multiple simulations
    results = []
    for _ in range(n_runs):
        model.learn(total_timesteps=10000)
        results.append(evaluate_model(env, model, df))

    average_result = np.mean(results)
    print(f"Average result over {n_runs} runs: {average_result}")
    return model

def evaluate_model(env, model, df, print_trades=False):
    """Evaluate the trained model."""
    # Use a single instance of the environment for evaluation
    eval_env = TradingEnv(df)
    df.to_csv('alibaba_alpha_factors_1.csv')
    obs = eval_env.reset()
    total_reward = 0
    done = False
    returns = []
    status = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = eval_env.step(action)
        total_reward = rewards
        returns.append(total_reward)
        if print_trades:
            # Print buy/sell points
            current_date = df.index[eval_env.current_step - 1]
            current_price = df.iloc[eval_env.current_step - 1]['close']
            #print(f"Date={current_date}, Cash={eval_env.cash}, Price={current_price}, Shares={eval_env.position}"), action

            if action > 0.5 and status == 0:
                status = 1  
                print(f"Buy: Date={current_date}, Cash={eval_env.cash}, Price={current_price}, Shares={eval_env.position}, Profit={total_reward}")
            elif action < -0.5 and status == 1 :
                status = 0 
                print(f"Sell: Date={current_date}, Cash={eval_env.cash}, Price={current_price}, Shares={eval_env.position}, Profit={total_reward}")
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    print(f"Total Reward: {total_reward}, Sharpe Ratio: {sharpe_ratio}")
    return total_reward

def backtest_rl_strategy(ts_code, start_date, end_date, algorithm='PPO'):
    """Backtest RL strategy using a specified algorithm."""
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    # Convert 'trade_date' to datetime and set as index
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    
    # Sort the DataFrame by date in ascending order (from far to near)
    df.sort_values(by='trade_date', inplace=True)
    
    df = calculate_alpha_factors(df)
    df.to_csv('alibaba_alpha_factors.csv')
    
    # 按日期切分数据
    split_index = int(len(df) * 3 / 4)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # 训练模型
    model = train_rl_agent_with_optimized_params(train_df, algorithm=algorithm)

    # 在测试集上评估模型
    return evaluate_model(TradingEnv(test_df), model, test_df, True)

def optimize_rl_hyperparameters(df, algorithm='PPO'):
    """Optimize hyperparameters for a reinforcement learning model using Optuna."""
    
    def objective(trial):
        # Define the hyperparameter search space
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        n_steps = trial.suggest_int('n_steps', 128, 2048)
        gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
        ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
        
        # Initialize the environment
        env = DummyVecEnv([lambda: TradingEnv(df) for _ in range(4)])
        
        # Select the model based on the algorithm
        if algorithm == 'PPO':
            model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, ent_coef=ent_coef, verbose=0)
        elif algorithm == 'A2C':
            model = A2C('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, ent_coef=ent_coef, verbose=0)
        elif algorithm == 'SAC':
            model = SAC('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, verbose=0)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train the model
        model.learn(total_timesteps=10000)
        
        # Evaluate the model
        return evaluate_model(env, model, df)
    
    # Create an Optuna study object
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # Output the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    return study.best_params

def train_rl_agent_with_optimized_params(df, algorithm='PPO', n_runs=5):
    """Train RL agent using optimized hyperparameters."""
    # Optimize hyperparameters
    best_params = optimize_rl_hyperparameters(df, algorithm)
    
    # Initialize the environment with optimized parameters
    env = DummyVecEnv([lambda: TradingEnv(df) for _ in range(4)])
    
    # Select the algorithm based on the input parameter
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, **best_params, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, **best_params, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')
    elif algorithm == 'SAC':
        # Remove 'n_steps' from best_params for SAC
        sac_params = {k: v for k, v in best_params.items() if k != 'n_steps'}
        model = SAC('MlpPolicy', env, **sac_params, buffer_size=100000, learning_starts=1000, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Run multiple simulations
    results = []
    for _ in range(n_runs):
        model.learn(total_timesteps=10000)
        results.append(evaluate_model(env, model, df))

    average_result = np.mean(results)
    print(f"Average result over {n_runs} runs: {average_result}")
    return model

def main():
    """Main function to execute and compare the backtest strategies."""
    ts_code = '300750.SZ'
    start_date = '20210101'
    end_date = '20241206'

    # Fetch and prepare data
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.sort_values(by='trade_date', inplace=True)
    df = calculate_alpha_factors(df)

    # Split data into training and testing sets
    split_index = int(len(df) * 3 / 4)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Train and evaluate model with PPO
    print("Training and evaluating with PPO:")
    model_ppo = train_rl_agent_with_optimized_params(train_df, algorithm='PPO')
    final_value_ppo = evaluate_model(TradingEnv(test_df), model_ppo, test_df, True)
    print("Final Portfolio Value with PPO:", final_value_ppo)

    # Train and evaluate model with A2C
    print("\nTraining and evaluating with A2C:")
    model_a2c = train_rl_agent_with_optimized_params(train_df, algorithm='A2C')
    final_value_a2c = evaluate_model(TradingEnv(test_df), model_a2c, test_df, True)
    print("Final Portfolio Value with A2C:", final_value_a2c)

    # Train and evaluate model with SAC
    print("\nTraining and evaluating with SAC:")
    model_sac = train_rl_agent_with_optimized_params(train_df, algorithm='SAC')
    final_value_sac = evaluate_model(TradingEnv(test_df), model_sac, test_df, True)
    print("Final Portfolio Value with SAC:", final_value_sac)


    print("Final Portfolio Value with A2C:", final_value_a2c)
    print("Final Portfolio Value with PPO:", final_value_ppo)
    print("Final Portfolio Value with SAC:", final_value_sac)

    # 比较结果
    if final_value_ppo > final_value_a2c and final_value_ppo > final_value_sac:
        print("PPO策略表现更好")
    elif final_value_a2c > final_value_ppo and final_value_a2c > final_value_sac:
        print("A2C策略表现更好")
    elif final_value_sac > final_value_ppo and final_value_sac > final_value_a2c:
        print("SAC策略表现更好")
    else:
        print("多种策略表现相同")
if __name__ == "__main__":
    main()