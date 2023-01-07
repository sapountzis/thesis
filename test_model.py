import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import make_config
from loading_utils import load_train_df
from sb3_contrib import RecurrentPPO
from envs import TradingEnv
import seaborn as sns


def test_model(model, config):
    ep_len = config['env_kwargs']['episode_length']

    test_size = int(ep_len / 4)
    # pre_test_size = min(test_size, ep_len - 1)
    pre_test_size = 0
    total_size = test_size + pre_test_size

    step = config['intervals'][0]
    test_start = pd.date_range(start=config['train_start'], periods=ep_len - pre_test_size,
                               freq=config['intervals'][0])[-1]
    test_end = pd.date_range(start=test_start, periods=total_size, freq=config['intervals'][0])[-1]
    print(test_start, test_end)
    print(test_size)

    df = load_train_df(config['data_dir'], intervals=config['intervals'], coins=config['coins'],
                       fiat=config['fiat'], index_col='date',
                       end_date=config['train_end'], start_date=test_start)

    coin_data = {coin: df[step][coin].iloc[pre_test_size: total_size]['open'] for coin in df[step].keys()}
    coin_dates = {coin: df[step][coin].iloc[pre_test_size: total_size]['date'] for coin in df[step].keys()}

    sns.set(font_scale=3.5, style='whitegrid')
    fig, ax = plt.subplots(figsize=(40, 10))
    ax.set_title('Prices')
    for coin in coin_data.keys():
        ax.plot(coin_dates[coin], coin_data[coin] / coin_data[coin].iloc[0], linewidth=3)
    ax.legend(coin_data.keys())
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/prices-test.png')

    env = TradingEnv(df, capital=1000, ep_len=total_size + 1, fee=0.00022, env_id=f'{last_model_dir}_{checkpoint}',
                     log=False)

    obs = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    r = 0
    cap = 0
    coins = env.coins
    for i in range(pre_test_size):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones

    cap = [env.portfolio_value]
    for i in range(test_size):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        cap.append(env.portfolio_value)

    hold = np.mean([coin_data[coin] / coin_data[coin].iloc[0] for coin in coin_data.keys()], axis=0)
    cap = np.array(cap) / cap[0]
    print(hold)
    print(cap)
    fig, ax = plt.subplots(figsize=(40, 10))
    ax.set_title('Return')
    ax.plot(cap, linewidth=4)
    ax.plot(hold, linewidth=4)
    ax.legend(['Model', 'Buy & Hold'])
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/model_vs_hold-test.png')
    # print(f'hold return: {hold}')
    # print(f'Portfolio return: {env.portfolio_value / cap}')


if __name__ == '__main__':
    dirs = sorted(d.split('_')[0] for d in os.listdir('models'))
    last_model_dir = sorted(d for d in os.listdir('models') if d.startswith(dirs[-1]))[0]
    # checkpoint = sorted([int(chkpnt.removesuffix('.zip')) for chkpnt in os.listdir(f'models/{last_model_dir}')])[-1]
    checkpoint = f'{46000000}'
    model = RecurrentPPO.load(f'models/{last_model_dir}/{checkpoint}.zip')

    config = make_config('config.json')

    test_model(model, config)

