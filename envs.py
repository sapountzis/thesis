import datetime
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

np.seterr(all='raise')

logdir = 'logs'
modelsdir = 'models/PPO'


class TrainingEnv(Env):
    def __init__(self, capital, intervals=None, past_steps=None, fiat='USDT', data_dir='data_aligned',
                 fee=0.001, loss_tolerance=0.2, loss_penalty_coef=1):
        super(TrainingEnv, self).__init__()

        self.done = False

        self.loss_tolerance = loss_tolerance
        self.loss_penatly_coef = loss_penalty_coef
        self.fee = fee
        self.fiat = fiat
        self.capital = capital
        self.portfolio_value = self.capital
        self.current_coin = self.fiat
        self.intervals = intervals
        self.past_steps = past_steps

        self.interval2offset = {interval: to_offset(interval) for interval in self.intervals}

        # load and preprocess training data
        self.data = {f'{file.split(fiat)[0]}_{interval}': self.data_resampling(pd.read_feather(f'{data_dir}/{file}'),
                                                                               interval)
                     for file in os.listdir(data_dir) for interval in self.intervals}
        # extract coins from data
        self.coins = sorted(set([c.split('_')[0] for c in self.data.keys()]))
        # initialize portfolio allocations
        self.portfolio = {coin: 0 for coin in self.coins}
        self.portfolio.update({self.fiat: self.capital})

        # initialize start and end training time
        self.start_time = max(d['prices'][:, 0].min().replace(hour=0, minute=0, second=0, microsecond=0)
                              for d in self.data.values()) + timedelta(days=1)
        self.end_time = min(d['prices'][:, 0].max().replace(hour=0, minute=0, second=0, microsecond=0)
                            for d in self.data.values()) - timedelta(days=1)
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
        self.current_time = self.start_time + pd.Timedelta(self.max_interval * self.past_steps)
        self.current_time = self.rand_date(start=int(self.current_time.to_datetime64()),
                                           end=int(self.end_time.to_datetime64()), n=1)
        self.data_cols = ['open', 'low', 'high', 'close', 'volume']
        self.processed_cols = ['open', 'volatility', 'volume']

        # Optimization | initialize indices in data
        self.curr_idx = {f'{coin}_{interval}': {'datetime': self.current_time, 'idx': -1}
                         for coin in self.coins for interval in self.intervals}

        # initialize observation space
        dict_space = {
            f'{k}': Box(low=-np.inf, high=np.inf, shape=(1, self.past_steps, len(self.processed_cols)), dtype=float)
            for k in self.data.keys()}
        dict_space.update({'portfolio': MultiBinary(len(self.portfolio))})
        self.observation_space = Dict(dict_space)
        # load initial observation data
        self.observation_data = None
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
        self.fiat_holding = 0
        self.trade_capital_change = 0
        self.total_capital_change = 0
        self.fiat_holding = 0
        self.fiat_holding_pc = 0
        self.same_coin_hold = 0

    def reset(self, **kwargs):
        self.done = False
        self.current_time = self.start_time + pd.Timedelta(self.max_interval * self.past_steps)
        self.current_time = self.rand_date(start=int(self.current_time.to_datetime64()),
                                           end=int(self.end_time.to_datetime64()), n=1)
        # debug / info
        self.steps = 0
        self.trades = 0
        self.trade_rate = 0
        self.max_steps = (self.end_time - self.current_time) // self.time_step
        self.fiat_holding = 0
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
        self.curr_idx = {f'{coin}_{interval}': {'datetime': self.current_time, 'idx': -1}
                         for coin in self.coins for interval in self.intervals}
        return self.observation_data

    def step(self, action):
        reward = 0
        self.steps += 1

        action_coin = self.action2coin[action]

        if self.current_coin == self.fiat:
            self.fiat_holding += 1

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
            if action_coin == self.fiat:
                reward -= self.same_coin_hold * self.fee / 10

        # increment time
        self.current_time += + pd.Timedelta(self.time_step)

        if self.current_coin == self.fiat:
            portfolio_value = self.portfolio[self.current_coin]
        else:
            # get next prices
            prices_next = self.get_prices(self.current_time)
            portfolio_value = self.portfolio[self.current_coin] * prices_next[self.current_coin]

        value_pc_diff_step = (portfolio_value - self.portfolio_value) / self.portfolio_value
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = portfolio_value

        reward += value_pc_diff_step if value_pc_diff_step > 0 else value_pc_diff_step * self.loss_penatly_coef

        maximum_training_reached = self.current_time + self.time_step >= self.end_time
        loss_tolerance_threshold_reached = self.portfolio_value <= self.capital * self.loss_tolerance
        if maximum_training_reached or loss_tolerance_threshold_reached:
            self.done = True

        # produce next observation data
        self.produce_step_data(self.current_time, action)

        self.trade_rate = self.trades / self.steps
        self.trade_capital_change = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.total_capital_change = (self.portfolio_value - self.capital) / self.capital
        self.fiat_holding_pc = self.fiat_holding / self.steps

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

    def preprocess_step(self, data, datetime_val, k):
        idx = self.curr_idx[k]['idx']
        if idx != -1:
            if datetime_val - self.curr_idx[k]['datetime'] == self.interval2offset[k.split('_')[1]]:
                self.curr_idx[k]['datetime'] = datetime_val
                idx += 1
                self.curr_idx[k]['idx'] = idx
                return np.expand_dims(data['data'][idx - self.past_steps: idx], axis=0)

            else:
                return self.observation_data[k]
        else:
            idx = np.argmax(data['datetime'] > datetime_val) - 1
            self.curr_idx[k]['datetime'] = datetime_val
            self.curr_idx[k]['idx'] = idx
            return np.expand_dims(data['data'][idx - self.past_steps: idx], axis=0)

    def produce_step_data(self, datetime_val, action):
        dict_space_res = {k: self.preprocess_step(d['preprocessed'], datetime_val, k) for k, d in
                          self.data.items()}

        dict_space_res.update({'portfolio': np.array([1 if i == action else 0 for i in range(len(self.portfolio))])})

        self.observation_data = dict_space_res

    def data_resampling(self, data, interval):
        resampled = data.resample(interval, label='right', closed='right', origin='start_day', on='datetime').agg(
            {'open': 'first', 'low': 'min', 'high': 'max', 'close': 'last', 'volume': 'sum'}).reset_index()
        resampled['volatility'] = resampled['high'] - resampled['low']
        resampled = resampled.fillna(method='bfill', axis=0).fillna(method='ffill', axis=0)

        datetime = resampled['datetime'].values[1:]
        preprocessed = resampled[['open', 'volatility', 'volume']]
        preprocessed = preprocessed.replace(0, 0.0000001)
        preprocessed = np.log(preprocessed / preprocessed.shift(1)).dropna()
        preprocessed['datetime'] = datetime
        preprocessed = preprocessed[['datetime', 'open', 'volatility', 'volume']].to_numpy()

        prices = resampled[['datetime', 'open', 'close']].to_numpy()

        return {'prices': prices,
                'preprocessed': {'datetime': preprocessed[:, 0], 'data': preprocessed[:, 1:].astype(float)}}

    def get_prices(self, current_time):
        prices_list = [('', 0) for c in self.coins]
        for c_idx, c in enumerate(self.coins):
            idx = self.curr_idx[f'{c}_{self.min_interval}']['idx']
            if idx == -1:
                datetime = self.data[f'{c}_{self.min_interval}']['preprocessed']['datetime']
                idx = np.argmax(datetime > current_time) - 1
            prices_list[c_idx] = (c, self.data[f'{c}_{self.min_interval}']['prices'][idx, 2])

        return {price[0]: price[1] for price in prices_list}

    def rand_date(self, start, end, n):
        unit = 10 ** 9
        start_u = start // unit
        end_u = end // unit

        res = np.datetime_as_string((unit * np.random.randint(start_u, end_u, n, dtype=np.int64)).view('M8[ns]'),
                                    unit='D')[0]
        res = datetime.datetime.fromisoformat(res)
        return res
