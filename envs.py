import os
from functools import reduce
import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box, Dict
from pandas.tseries.frequencies import to_offset

np.seterr(all='raise')

logdir = 'logs'
modelsdir = 'models/PPO'


class TradingEnv(Env):
    def __init__(self, data, capital, ep_len, fee, env_id, log_freq=128, log=True, train=True):
        super(TradingEnv, self).__init__()
        self.env_id = env_id

        self.train = train
        self.log_freq = log_freq
        self.ep_len = ep_len
        self.intervals = list(data.keys())
        self.interval2offset = {interval: to_offset(interval) for interval in self.intervals}
        self.coins = list(data[self.intervals[0]].keys())
        self.fee = fee
        self.fiat = 'usdt'
        self.trade = False

        self.timestep = min([to_offset(interval) for interval in self.intervals])
        self.max_interval = max([to_offset(interval) for interval in self.intervals])
        self.min_interval = [interval for interval in self.intervals
                             if self.interval2offset[interval] == self.timestep][0]
        self.start_date = max([data[interval][coin].date.min() for interval in self.intervals for coin in self.coins])
        self.data = self.preprocess_data(data)
        self.end_date = min([data[interval][coin].date.max() for interval in self.intervals for coin in self.coins])
        # self.end_date = min(self.start_date + self.ep_len * self.timestep, self.end_date)
        self.curr_date = self.start_date
        self.date_idx = {interval: {'datetime': self.start_date, 'idx': 0} for interval in self.intervals}

        self.log = log
        self.logs = {}
        self.capital = capital
        self.portfolio = {coin: 0 for coin in self.coins}
        self.portfolio.update({self.fiat: self.capital})
        self.portfolio_alloc = np.array([0 for _ in self.coins] + [1])
        self.portfolio_value = self.capital
        self.action2coin = {idx: c for idx, c in enumerate(self.coins)}
        self.steps = 0
        self.trade_threshold = 0

        self.observation = {'portfolio_alloc': self.portfolio_alloc}
        self.produce_observation(self.curr_date)
        self.observation_space = {interval: Box(low=-np.inf, high=np.inf, shape=(d.shape[1],), dtype=np.float64)
                                  for interval, d in self.data['features'].items()}
        self.observation_space.update({'portfolio_alloc': Box(low=0, high=1, shape=(len(self.coins) + 1,),
                                                              dtype=np.float64)})
        self.observation_space.update({'reward': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)})
        self.observation_space = Dict(self.observation_space)

        self.action_space = Box(low=0, high=1, shape=(len(self.portfolio) + 1,), dtype=np.float64)
        self.reward = 0
        self.trades = 0
        self.done = False

        self.episode = -1
        self.logs = {'episode_info': np.zeros((self.ep_len // log_freq, 3)),
                     'trades': np.zeros((self.ep_len + 1, len(self.portfolio) + 2))}

    def reset(self):
        # self.date_idx = {interval: {'datetime': self.start_date, 'idx': 0} for interval in self.intervals}
        # self.curr_date = self.start_date
        if self.train:
            self.random_curr_date()
        self.episode += 1
        self.trades = 0
        self.done = False
        self.steps = 0

        self.portfolio = {coin: 0 for coin in self.coins}
        self.portfolio.update({self.fiat: self.capital})
        self.portfolio_alloc = np.array([0 for _ in self.coins] + [1])
        self.portfolio_value = self.capital

        self.observation = {'portfolio_alloc': self.portfolio_alloc, 'reward': np.zeros((1,))}
        self.produce_observation(self.curr_date)

        self.logs = {'episode_info': np.zeros((self.ep_len // self.log_freq, 3)),
                     'trades': np.zeros((self.ep_len + 1, len(self.portfolio) + 2))}

        return self.observation

    def step(self, action: np.ndarray):
        self.reward = self.take_action(action)

        if self.log:
            self.logger()

        self.curr_date += self.timestep
        self.produce_observation(self.curr_date)
        self.steps += 1

        if self.steps >= self.ep_len:
            os.makedirs(f'model_logs/{self.env_id}', exist_ok=True)
            trades = pd.DataFrame(np.round(self.logs['trades'][:self.trades + 1], 4),
                                  columns=['step', 'portfolio_value', *self.portfolio.keys()])
            trades.to_csv(f'model_logs/{self.env_id}/logs_{self.episode}_trades.csv', index=False)
            episode_info = pd.DataFrame(np.round(self.logs['episode_info'], 4),
                                        columns=['reward', 'portfolio_value', 'trade'])
            episode_info.to_csv(f'model_logs/{self.env_id}/logs_{self.episode}_episode_info.csv', index=False)
            self.done = True

        return self.observation, self.reward, self.done, {}

    def produce_observation(self, curr_datetime):
        for interval in self.intervals:
            key = f'features_{interval}'
            if curr_datetime - self.date_idx[interval]['datetime'] >= self.interval2offset[interval]:
                self.date_idx[interval]['idx'] += 1
                self.date_idx[interval]['datetime'] = curr_datetime
                self.observation[key] = self.data['features'][key][self.date_idx[interval]['idx']]
            if self.observation.get(key) is None:
                self.observation[key] = self.data['features'][key][self.date_idx[interval]['idx']]

        if self.trade:
            self.observation['portfolio_alloc'] = self.portfolio_alloc

        self.observation['reward'] = self.reward

    def preprocess_data(self, data):
        data = {interval: {coin: data[interval][coin].set_index('date', inplace=False) for coin in self.coins}
                for interval in self.intervals}
        data = {interval: {coin: data[interval][coin].loc[self.start_date:].reset_index()
                           for coin in self.coins}
                for interval in self.intervals}

        dates = {f'dates_{interval}': reduce(lambda a, b: pd.merge(a, b, on='date', how='outer'),
                                             [d['date']
                                              for coin, d in coins.items()]).iloc[2:]
                 for interval, coins in data.items()}

        prices = reduce(lambda a, b: pd.merge(a, b, on='date', how='outer'),
                        [d[['date', 'open']].rename(columns={'open': f'{coin}'})
                         for coin, d in data[self.min_interval].items()]).drop(columns=['date']).values[2:]

        price_pct_chg = {f'price_pct_chg_{interval}': reduce(lambda a, b: pd.merge(a, b, on='date', how='outer'),
                                                             [d[['date', 'open']].rename(
                                                                 columns={'open': f'open_{coin}'}
                                                             )
                                                                 for coin, d in coins.items()]).drop(
            columns=['date']).pct_change().values[1:-1]
                         for interval, coins in data.items()}

        volume = {f'volume_{interval}': reduce(lambda a, b: pd.merge(a, b, on='date', how='outer'),
                                               [d[['date', 'volume']].rename(columns={'volume': f'volume_{coin}'})
                                                for coin, d in coins.items()]).drop(columns=['date']).pct_change().values[1:-1]
                  for interval, coins in data.items()}

        for i in self.intervals:
            for c in self.coins:
                data[i][c]['volatility'] = ((data[i][c]['high'] - data[i][c]['low']) / data[i][c]['low'])

        volatility = {f'volatility_{interval}': reduce(lambda a, b: pd.merge(a, b, on='date', how='outer'),
                                                       [d[['date', 'volatility']].rename(
                                                           columns={'volatility': f'volatility_{coin}'}
                                                       )
                                                           for coin, d in coins.items()]).drop(columns=['date']).values[1:-1]
                      for interval, coins in data.items()}

        self.start_date = max([dates[f'dates_{interval}']['date'].min() for interval in self.intervals])

        date_mask = {f'{interval}': dates[f'dates_{interval}']['date'] >= self.start_date for interval in
                     self.intervals}
        prices = prices[date_mask[self.min_interval]]
        price_pct_chg = {f'price_pct_chg_{interval}': price_pct_chg[f'price_pct_chg_{interval}'][date_mask[interval]]
                         for interval in self.intervals}
        volume = {f'volume_{interval}': volume[f'volume_{interval}'][date_mask[interval]] for interval in
                  self.intervals}
        volatility = {f'volatility_{interval}': volatility[f'volatility_{interval}'][date_mask[interval]] for interval
                      in self.intervals}
        dates = {f'dates_{interval}': dates[f'dates_{interval}'][date_mask[interval]].reset_index(drop=True) for
                 interval in self.intervals}

        features = {f'features_{interval}': np.concatenate((price_pct_chg[f'price_pct_chg_{interval}'],
                                                            volume[f'volume_{interval}'],
                                                            volatility[f'volatility_{interval}']), axis=1)
                    for interval in self.intervals}

        for key in features.keys():
            features[key][np.isnan(features[key])] = 0
            features[key][np.isinf(features[key])] = 0

        return {'dates': dates, 'prices': prices, 'features': features}

    def get_prices(self, n=2):
        idx = self.date_idx[f'{self.min_interval}']['idx']
        max_idx = len(self.data['prices'])
        data_slice = self.data['prices'][idx: min(max_idx, idx+n), :]
        return {coin: data_slice[:, index] for index, coin in enumerate(self.coins)}

    def sharpe_ratio(self, prices):

        returns = self.get_returns(prices)

        return returns.mean() / returns.std()

    def get_returns(self, prices):
        if self.portfolio_alloc[:-1].sum() == 0:
            return np.zeros((1,))

        portfolio_values = np.array([prices[coin][1:] * self.portfolio_alloc[i] for i, coin in enumerate(prices)])

        portfolio_returns = portfolio_values[:, 1:] / portfolio_values[:, :-1] - 1
        portfolio_returns = portfolio_returns.sum(axis=0)

        return portfolio_returns

    def take_action(self, action):
        self.trade_threshold = action[-1]
        prices = self.get_prices(n=4)

        self.trade = self.trade_threshold >= 0.5
        if self.trade:
            self.trades += 1
            portfolio_alloc = self.softmax(action[:-1])
            self.portfolio_alloc = portfolio_alloc
            portfolio_alloc = {coin: portfolio_alloc[idx] for idx, coin in self.action2coin.items()}

            for coin, val in self.portfolio.items():
                if coin != self.fiat:
                    self.portfolio[self.fiat] += (1 - self.fee) * self.portfolio[coin] * prices[coin][0]
                    self.portfolio[coin] = 0

            for coin, val in portfolio_alloc.items():
                amount = (1 - self.fee) * self.portfolio[self.fiat] * val
                self.portfolio[coin] = amount / prices[coin][0]
                self.portfolio[self.fiat] -= amount

        portfolio_value_next = sum(self.portfolio[coin] * prices[coin][1:].mean() for coin in self.coins) + self.portfolio[self.fiat]
        portfolio_pct_change = (portfolio_value_next - self.portfolio_value) / self.capital
        self.portfolio_value = portfolio_value_next

        # reward = self.get_returns(prices).mean()
        # print(portfolio_pct_change, reward)
        reward = portfolio_pct_change
        return reward

    def logger(self):
        if self.steps % self.log_freq == 0:
            self.logs['episode_info'][self.steps // self.log_freq] = [self.reward, self.portfolio_value,
                                                                      int(self.trade)]
        if self.trade or self.steps == 0:
            idx = self.trades
            self.logs['trades'][idx, :] = [self.steps, self.portfolio_value, *list(self.portfolio_alloc)]

    def softmax(self, arr):
        exp_arr = np.exp(arr)
        return exp_arr / exp_arr.sum()

    def random_curr_date(self):
        idx = np.random.randint(0, self.data['dates'][f'dates_{self.min_interval}'].shape[0] - 16 - self.ep_len)
        self.set_curr_date(idx)

    def set_curr_date(self, idx):
        date = self.data['dates'][f'dates_{self.min_interval}']['date'].iloc[idx]
        for interval in self.intervals:
            interval_idx = np.argmax(self.data['dates'][f'dates_{interval}']['date'] > date) - 1
            interval_date = self.data['dates'][f'dates_{interval}']['date'].iloc[interval_idx]
            self.date_idx.update({interval: {'datetime': interval_date, 'idx': interval_idx}})
        self.curr_date = date
