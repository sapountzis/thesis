import datetime
import math
import os
from datetime import timedelta
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box, Dict, Discrete, MultiBinary
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pandas.tseries.frequencies import to_offset
from functools import reduce

np.seterr(all='raise')

logdir = 'logs'
modelsdir = 'models/PPO'


class TrainingEnv(Env):
    def __init__(self, df: dict, capital, past_steps=None, fiat='usdt', fee=0.001, loss_tolerance=0.2,
                 loss_penalty_coef=1, holding_penalty_coef=0.01, holding_penalty_threshold=0.4, reward_alpha=0.1,
                 fixed_episode_length=True):

        super(TrainingEnv, self).__init__()

        self.done = False
        self.data = df.copy()
        self.loss_tolerance = loss_tolerance
        self.loss_penatly_coef = loss_penalty_coef
        self.fee = fee
        self.fiat = fiat
        self.capital = capital
        self.portfolio_value = self.capital
        self.current_coin = self.fiat
        self.intervals = [interval for interval in self.data.keys()]
        self.past_steps = past_steps
        self.holding_penalty_coef = holding_penalty_coef
        self.holding_penalty_threshold = holding_penalty_threshold
        self.reward_alpha = reward_alpha
        self.max_ep_length = math.ceil(np.log(self.loss_tolerance)/np.log(1-self.fee)) if fixed_episode_length else np.inf
        print(f'Maximum Episode Length: {self.max_ep_length}')

        self.interval2offset = {interval: to_offset(interval) for interval in self.intervals}

        self.coins = set([c.removesuffix(self.fiat.lower())
                          for interval in self.data.keys() for c in self.data[interval].keys()])
        # initialize portfolio allocations
        self.portfolio = {coin: 0 for coin in self.coins}
        self.portfolio.update({self.fiat: self.capital})

        # initialize start and end training time
        self.start_time = max(d['date'].min().replace(hour=0, minute=0, second=0, microsecond=0)
                              for coins in self.data.values() for d in coins.values()) + timedelta(days=1)
        self.end_time = min(d['date'].max().replace(hour=0, minute=0, second=0, microsecond=0)
                            for coins in self.data.values() for d in coins.values()) - timedelta(days=1)

        self.data = self.data_preprocess(self.data)

        # initialize the interval of each step
        self.time_step = min([offset for offset in self.interval2offset.values()])
        # find minimum interval (string)
        self.min_interval = None
        for interval in self.intervals:
            if self.interval2offset[interval] == self.time_step:
                self.min_interval = interval

        # initialize maximum interval to calculate minimum time offest required
        self.max_interval = max([offset for offset in self.interval2offset.values()])
        # initialize current time := start time + required offset
        self.current_time = self.start_time + self.max_interval * self.past_steps
        self.current_time = self.rand_date(start=int(self.current_time.to_datetime64()),
                                           end=int((self.end_time - timedelta(days=1)).to_datetime64()), n=1)
        self.data_cols = ['open', 'low', 'high', 'close', 'volume']
        self.processed_cols = ['open', 'volatility', 'volume']

        # Optimization | initialize indices in data
        self.curr_idx = {f'{interval}': {'datetime': self.current_time, 'idx': -1}
                         for interval in self.intervals}

        # initialize observation space
        dict_space = {
            f'{k}': Box(low=-np.inf, high=np.inf, shape=(1, self.past_steps, self.data['data'][k].shape[-1]),
                        dtype=np.float64)
            for k in self.data['data'].keys()}
        dict_space.update({'portfolio': MultiBinary(len(self.portfolio))})
        self.observation_space = Dict(dict_space)
        # load initial observation data
        self.observation_data = {}
        self.produce_step_data(self.current_time, self.portfolio)
        # initialize action space
        self.action_space = Discrete(len(self.portfolio))
        # create action to coin converter
        self.action2coin = {idx: c for idx, c in enumerate(self.coins)}
        self.action2coin.update({len(self.coins): fiat})

        # debug / info
        self.steps = 0
        self.trades = 0
        self.trade_rate = 0
        self.max_steps = (self.end_time - self.current_time) // self.time_step
        self.holdings = {**{c: 0 for c in self.coins}, self.fiat: 0}
        self.holdings_pc = {k: 0 for k, v in self.holdings.items()}
        self.trade_capital_change = 0
        self.total_capital_change = 0
        self.fiat_holding = 0
        self.fiat_holding_pc = 0
        self.same_coin_hold = 0

    def reset(self, **kwargs):
        self.done = False
        self.current_time = self.start_time + self.max_interval * self.past_steps
        self.current_time = self.rand_date(start=int(self.current_time.to_datetime64()),
                                           end=int((self.end_time - timedelta(days=1)).to_datetime64()), n=1)
        # debug / info
        self.steps = 0
        self.trades = 0
        self.trade_rate = 0
        self.max_steps = (self.end_time - self.current_time) // self.time_step
        self.holdings = {**{c: 0 for c in self.coins}, self.fiat: 0}
        self.holdings_pc = {k: 0 for k, v in self.holdings.items()}
        self.trade_capital_change = 0
        self.total_capital_change = 0
        self.fiat_holding = 0
        self.fiat_holding_pc = 0
        self.same_coin_hold = 0

        self.portfolio = {coin: 0 for coin in self.coins}
        self.portfolio.update({self.fiat: self.capital})
        self.portfolio_value = self.capital
        self.current_coin = self.fiat
        self.produce_step_data(self.current_time, len(self.coins))
        # Optimization | initialize indices in data
        self.curr_idx = {f'{interval}': {'datetime': self.current_time, 'idx': -1}
                         for interval in self.intervals}
        return self.observation_data

    def step(self, action):
        reward = 0
        self.steps += 1

        action_coin = self.action2coin[action]

        self.holdings[self.current_coin] += 1
        self.holdings_pc = {k: v / self.steps for k, v in self.holdings.items()}

        if self.current_coin != action_coin:
            # get current prices
            self.same_coin_hold = 0
            prices_curr = self.get_prices(self.current_time)
            if action_coin == self.fiat:
                self.portfolio[self.current_coin] *= (1 - self.fee)
                self.portfolio[action_coin] = prices_curr[self.current_coin] * self.portfolio[self.current_coin]
            elif self.current_coin == self.fiat:
                self.portfolio[self.current_coin] *= (1 - self.fee)
                self.portfolio[action_coin] = self.portfolio[self.current_coin] / prices_curr[action_coin]
            else:
                self.portfolio[self.current_coin] *= (1 - self.fee)
                fiat = prices_curr[self.current_coin] * self.portfolio[self.current_coin]
                fiat *= (1 - self.fee)
                self.portfolio[action_coin] = fiat / prices_curr[action_coin]
            self.portfolio[self.current_coin] = 0
            self.current_coin = action_coin
            self.trades += 1
        else:
            self.same_coin_hold += 1
            # if action_coin == self.fiat:
            # reward -= self.same_coin_hold * self.fee * self.holding_penalty_coef
            # reward -= self.fee if self.holdings_pc[action_coin] > self.holding_penalty_threshold else 0

        # increment time
        self.current_time += pd.Timedelta(self.time_step)

        if self.current_coin == self.fiat:
            portfolio_value = self.portfolio[self.current_coin]
        else:
            # get next prices
            prices_next = self.get_prices(self.current_time)
            portfolio_value = self.portfolio[self.current_coin] * prices_next[self.current_coin]

        value_pc_diff_step = (portfolio_value - self.portfolio_value) / self.portfolio_value
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = portfolio_value

        maximum_training_reached = self.current_time + self.time_step >= self.end_time
        max_ep_length_reached = self.steps >= self.max_ep_length
        loss_tolerance_threshold_reached = self.portfolio_value <= self.capital * self.loss_tolerance
        if maximum_training_reached or loss_tolerance_threshold_reached or max_ep_length_reached:
            self.done = True

        self.trade_rate = self.trades / self.steps
        self.trade_capital_change = value_pc_diff_step
        self.total_capital_change = (self.portfolio_value - self.capital) / self.capital
        self.fiat_holding_pc = self.fiat_holding / self.steps

        holding_threshold_reached = self.holdings_pc[action_coin] > self.holding_penalty_threshold
        reward += self.reward_alpha * self.total_capital_change + value_pc_diff_step
        if holding_threshold_reached and (reward > 0):
            reward *= self.holding_penalty_coef

        # produce next observation data
        self.produce_step_data(self.current_time, action)

        info = {'trade_rate': self.trade_rate, 'fiat_holding_pc': self.fiat_holding_pc,
                'total_capital_change': self.total_capital_change}

        if self.done:
            info = {'trade_rate': self.trade_rate, 'total_capital_change': self.total_capital_change,
                    'ep_len_pc': self.steps / self.max_steps, 'fiat_holding_pc': self.fiat_holding / self.steps,
                    'maximum_steps_reached': maximum_training_reached}

        return self.observation_data, reward, self.done, info

    def render(self, mode="human"):

        feature_per_coin = 3
        fig, axes = plt.subplots(feature_per_coin * len(self.intervals), len(self.coins))
        fig.set_size_inches(4 * len(self.coins), 6 * len(self.intervals))

        for i_idx, interval in enumerate(self.intervals):
            for j_idx, coin in enumerate(sorted(self.coins)):
                # print(self.observation_data[f'{coin}_{interval}'])
                sel = self.observation_data[f'{coin}_{interval}'].set_index(
                    pd.date_range(end=self.current_time, periods=self.past_steps - 1, freq=interval))
                # mpf.plot(sel, type='candle', volume=axes[2 * i_idx + 1, j_idx], ax=axes[2 * i_idx, j_idx])
                axes[feature_per_coin * i_idx, j_idx].plot(sel.index, sel['open'])
                axes[feature_per_coin * i_idx + 1, j_idx].plot(sel.index, sel['volatility'])
                axes[feature_per_coin * i_idx + 2, j_idx].plot(sel.index, sel['volume'])
                axes[feature_per_coin * i_idx, j_idx].set_title(f'{coin}_{interval}', fontweight='bold', fontsize=30)
                axes[feature_per_coin * i_idx, j_idx].set_xticklabels([])
                axes[feature_per_coin * i_idx + 1, j_idx].set_xticklabels([])
                axes[feature_per_coin * i_idx + 2, j_idx].set_xticklabels([])
                if j_idx == 0:
                    axes[feature_per_coin * i_idx, j_idx].set_ylabel('Price', fontweight='bold', fontsize=20)
                    axes[feature_per_coin * i_idx + 1, j_idx].set_ylabel('Volatility', fontweight='bold', fontsize=20)
                    axes[feature_per_coin * i_idx + 2, j_idx].set_ylabel('Volume', fontweight='bold', fontsize=20)
        fig.tight_layout()
        # convert it to an OpenCV image/numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # convert canvas to image
        graph_image = np.array(fig.canvas.get_renderer()._renderer)

        plt.close('all')

        # it still is rgb, convert to opencv's default bgr
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)

        cv2.imshow('plot', graph_image)
        cv2.namedWindow("plot", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions

        graph_image = cv2.resize(graph_image, (150 * len(self.coins), 200 * len(self.intervals)))  # Resize image
        cv2.imshow("plot", graph_image)  # Show image
        cv2.waitKey(10)

    def close(self):
        cv2.destroyAllWindows()
        self.reset()

    def preprocess_step(self, data, datetime_val, key):
        interval = key.split('_')[1]
        idx = self.curr_idx[interval]['idx']
        if idx != -1:
            if datetime_val - self.curr_idx[interval]['datetime'] == self.interval2offset[interval]:
                self.curr_idx[interval]['datetime'] = datetime_val
                idx += 1
                self.curr_idx[interval]['idx'] = idx
                return np.expand_dims(data[idx - self.past_steps: idx], axis=0)

            else:
                if key in self.observation_data.keys():
                    return self.observation_data[key]
                else:
                    return np.expand_dims(data[idx - self.past_steps: idx], axis=0)
        else:
            idx = np.argmax(self.data['dates'][f'dates_{interval}'] > datetime_val) - 1
            self.curr_idx[interval]['datetime'] = datetime_val
            self.curr_idx[interval]['idx'] = idx
            return np.expand_dims(data[idx - self.past_steps: idx], axis=0)

    def produce_step_data(self, datetime_val, action):
        dict_space_res = {f'{key}': self.preprocess_step(d, datetime_val, key)
                          for key, d in self.data['data'].items()}

        dict_space_res.update({'portfolio': np.array([1 if i == action else 0 for i in range(len(self.portfolio))])})

        if any([d.shape != (1, self.past_steps, len(self.coins)) for key, d in dict_space_res.items() if
                key != 'portfolio']):
            print(self.current_time)
            print(self.start_time)
            print(self.end_time)
            for key, space_res in dict_space_res.items():
                print(key)
                print(space_res.shape)
                print(space_res)
            raise ValueError('Wrong shape of data')

        self.observation_data = dict_space_res

    def _volatility(self, coin_data, pctg=True):
        coin_data['volatility'] = coin_data['high'] - coin_data['low']
        if pctg:
            coin_data['volatility'] = coin_data['volatility'] / coin_data['low']

        coin_data = coin_data.drop(columns=['high', 'low', 'open', 'close', 'volume'])

        return coin_data

    def data_preprocess(self, data):

        dates = {f'dates_{interval}': reduce(lambda left, right: pd.merge(left, right, on=['date'], how='inner'),
                                             [d[d['date'] >= self.start_time]['date']
                                              for coin, d in coins.items()])
                 for interval, coins in data.items()}

        prices = {f'prices_{interval}': reduce(lambda left, right: pd.merge(left, right, on=['date'], how='inner'),
                                               [d[['date', 'open']][d['date'] >= self.start_time].rename(
                                                   columns={'open': f'{coin}_open'})
                                                   for coin, d in coins.items()]).drop(columns=['date']).values
                  for interval, coins in data.items()}

        volatility = {
            f'volatility_{interval}': reduce(lambda left, right: pd.merge(left, right, on=['date'], how='inner'),
                                             [self._volatility(d)[d['date'] >= self.start_time].rename(
                                                 columns={'volatility': f'{coin}_volatility'})
                                                 for coin, d in coins.items()]).drop(columns=['date']).values
            for interval, coins in data.items()}

        volumes = {f'volumes_{interval}': reduce(lambda left, right: pd.merge(left, right, on=['date'], how='inner'),
                                                 [d[['date', 'volume']][d['date'] >= self.start_time].rename(
                                                     columns={'volume': f'{coin}_volume'})
                                                     for coin, d in coins.items()]).drop(columns=['date']).values
                   for interval, coins in data.items()}

        return {'dates': dates, 'data': {**prices, **volatility, **volumes}}

    def get_prices(self, current_time):
        idx = self.curr_idx[f'{self.min_interval}']['idx']
        if idx == -1:
            datetime = self.data['dates'][f'dates_{self.min_interval}']
            idx = np.argmax(datetime > current_time)
        data_slice = self.data['data'][f'prices_{self.min_interval}'][idx, :]
        return {coin: data_slice[i] for i, coin in enumerate(self.coins)}

    def rand_date(self, start, end, n):
        unit = 10 ** 9
        start_u = start // unit
        end_u = end // unit

        res = np.datetime_as_string((unit * np.random.randint(start_u, end_u, n, dtype=np.int64)).view('M8[ns]'),
                                    unit='D')[0]
        res = datetime.datetime.fromisoformat(res)
        return res
