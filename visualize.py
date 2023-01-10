import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import os
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from config import make_config
from loading_utils import load_train_df
from sb3_contrib import RecurrentPPO
from envs import TradingEnv


if __name__ == '__main__':
    model_logs = 'model_logs'
    # model = sorted(os.listdir(model_logs))[-1]
    model_name = 'RecurrentPPO-1673182213'

    print(f'Loading model {model_name} logs')

    logs_dir = f'{model_logs}/{model_name}'
    episode_infos = sorted([f for f in os.listdir(logs_dir) if f.endswith('episode_info.csv')], key=lambda x: int(x.split('_')[1]))
    trades = sorted([f for f in os.listdir(logs_dir) if f.endswith('trades.csv')], key=lambda x: int(x.split('_')[1]))

    config = make_config('config.json')
    episode_timesteps = config['env_kwargs']['episode_length']
    step = config['intervals'][0]
    df = load_train_df(config['data_dir'], intervals=config['intervals'], coins=config['coins'],
                       fiat=config['fiat'], index_col='date',
                       end_date=config['train_end'], start_date=config['train_start'])

    trades_chkpnt = 200

    coin_data = {coin: data
                 for interval, coin in df.items() for coin, data in coin.items() if interval == step}

    coins = list(coin_data.keys())

    trade = [t for t in trades if str(trades_chkpnt) in t]
    trade = pd.read_csv(f'{logs_dir}/{trade[-1]}')
    trade['step'] = trade['step'].astype(int)
    dates = coin_data[list(coin_data.keys())[0]].loc[:, ['date']]
    dates['step'] = dates.index
    # alloc = pd.merge(dates, trade, on='step', how='left').fillna(method='ffill')
    trade = pd.merge(dates, trade, on='step', how='right').fillna(method='ffill')

    sns.set(font_scale=4.5, style='whitegrid')
    fig, axes = plt.subplots(len(coin_data), 1, figsize=(50, len(coin_data) * 5))
    for i, (coin, data) in enumerate(coin_data.items()):
        ax_idx = i
        axes[ax_idx].plot(data['date'], data['open'], linewidth=4, alpha=0.8)
        axes[ax_idx].set_ylabel(coin)
        axes[ax_idx].yaxis.set_major_locator(plt.MaxNLocator(4))
        # axes[ax_idx].scatter(trade['date'], data[data['date'].isin(trade['date'])]['open'],
        #                      color='red', s=10, zorder=2)
        axes[ax_idx].yaxis.set_major_formatter(FormatStrFormatter(f'%.{4-max(0, int(np.log10(data["open"].max())))}f'))

        if i != len(coin_data) - 1:
            axes[ax_idx].set_xticklabels([])

        if i == 0:
            axes[ax_idx].set_title(f'Prices')

    # for i, (coin, data) in enumerate(coin_data.items()):
    #     ax_idx = i, 1
    #     axes[ax_idx].plot(alloc['date'], alloc[coin])
    #     axes[ax_idx].yaxis.set_major_locator(plt.MaxNLocator(4))
    #     axes[ax_idx].xaxis.set_major_locator(plt.MaxNLocator(2))
    #     axes[ax_idx].axhline(alloc[coin].mean(), color='red', zorder=2, linewidth=4)
    #     axes[ax_idx].yaxis.set_major_formatter(FormatStrFormatter(f'%.2f'))
    #
    #     if i != len(coin_data) - 1:
    #         axes[ax_idx].set_xticklabels([])
    #
    #     if i == 0:
    #         axes[ax_idx].set_title(f'Portfolio %')
    fig.tight_layout()
    plt.show()
    fig.savefig(f'visualizations/trade-viz.png')

    sns.set(font_scale=3.5, style='whitegrid')

    trades_corr = trade[coins + ['usdt']].corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title('Trades Correlation')
    sns.heatmap(trades_corr, annot=True, ax=ax, vmin=-1, vmax=1, center=0, cmap='coolwarm')
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/alloc-corr.png')

    coin_corr = pd.concat([coin_data[coin].rename(columns={'open': coin})[coin]
                           for coin in coins], axis=1).corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title('Price Correlation')
    sns.heatmap(coin_corr, annot=True, ax=ax, vmin=-1, vmax=1, center=0, cmap='coolwarm')
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/price-corr.png')

    trade_rate = [pd.read_csv(f'{logs_dir}/{t}').shape[0] / episode_timesteps for t in trades]
    fig, ax = plt.subplots(figsize=(40, 10))
    ax.set_title('Trade Rate')
    ax.plot(np.arange(len(trade_rate)), trade_rate, linewidth=4)
    ax.set_xlabel('Episode')
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/trade-rate.png')

    # coin_data_mean = pd.concat([coin_data[coin].rename(columns={'open': coin})[coin] for coin in coins], axis=1)
    # coin_data_mean /= coin_data_mean.iloc[0]
    # coin_data_mean = coin_data_mean.mean(axis=1)
    # # episode_info = [e for e in episode_info if str(ckpnt) in e]
    # episode_info = [e for e in episode_infos if str(trades_chkpnt) in e]
    # episode_info = pd.read_csv(f'{logs_dir}/{episode_info[-1]}')
    # # episode_info = pd.read_csv(f'{logs_dir}/{episode_infos[-1]}')
    # x = np.arange(0, len(episode_info)) * config['log_freq']
    # fig, ax = plt.subplots(figsize=(40, 10))
    # ax.set_title('Return')
    # ax.plot(dates.iloc[x]['date'], episode_info['portfolio_value'] / config['env_kwargs']['capital'], linewidth=4)
    # ax.plot(dates['date'], coin_data_mean, linewidth=4)
    # ax.legend(['Model', 'Buy & Hold'])
    # fig.tight_layout()
    # fig.show()
    # fig.savefig(f'visualizations/model_vs_hold.png')

    model_chkpnt = 200000
    model = RecurrentPPO.load(f'models/{model_name}/{model_chkpnt}')

    n_samples = 20

    control_performance = np.zeros((n_samples, episode_timesteps))
    agent_performance = np.zeros((n_samples, episode_timesteps))

    rng = default_rng()
    random_starts = rng.integers(0, len(dates) - episode_timesteps - 16,  size=n_samples)

    lstm_states = None
    env = TradingEnv(df, capital=1, ep_len=episode_timesteps, fee=0.00022,
                     env_id=f'{model_name}-{model_chkpnt}', log=True, train=False)

    for i, start in enumerate(random_starts):
        print(f'Running {i + 1} / {len(random_starts)}')

        env.set_curr_date(start)
        obs = env.reset()
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)
        for t in range(episode_timesteps):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts,
                                                deterministic=True)
            obs, rewards, dones, info = env.step(action)
            agent_performance[i, t] = env.portfolio_value
        lstm_states = None

        coin_data_mean = pd.concat([coin_data[coin].rename(columns={'open': coin})[coin] for coin in coins], axis=1).iloc[start:start + episode_timesteps]
        coin_data_mean /= coin_data_mean.iloc[0]
        coin_data_mean = coin_data_mean.mean(axis=1)

        control_performance[i, :] = coin_data_mean.values

    print(agent_performance)
    print(control_performance)

    # percentage change
    control_return = (control_performance[:, :-1] - control_performance[:, 1:]) / control_performance[:, :-1]
    agent_return = (agent_performance[:, :-1] - agent_performance[:, 1:]) / agent_performance[:, :-1]

    # mean return
    mean_control_return = control_return.mean(axis=1)
    mean_agent_return = agent_return.mean(axis=1)

    # std return
    std_control_return = control_return.std(axis=1)
    std_agent_return = agent_return.std(axis=1)

    # sharpe ratio
    sharpe_control = mean_control_return / (std_control_return + 1e-8)
    sharpe_agent = mean_agent_return / (std_agent_return + 1e-8)

    # sortino std
    try:
        std_sortino_control_return = np.array([np.std(control_return[i, control_return[i, :] < 0]) for i in range(n_samples)])
    except Exception:
        std_sortino_control_return = np.zeros(n_samples)
    try:
        std_sortino_agent_return = np.array([np.std(agent_return[i, agent_return[i, :] < 0]) for i in range(n_samples)])
    except Exception:
        std_sortino_agent_return = np.zeros(n_samples)

    # sortino ratio
    sortino_control = mean_control_return / (std_sortino_control_return + 1e-8)
    sortino_agent = mean_agent_return / (std_sortino_agent_return + 1e-8)

    # violinplot sharpe
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title('Sharpe Ratio')
    sns.violinplot(data=[sharpe_control, sharpe_agent], ax=ax)
    sns.pointplot(data=[sharpe_control, sharpe_agent], ax=ax, estimator=np.mean, color='red')
    ax.set_xticklabels(['Buy & Hold', 'Model'])
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/sharpe.png')

    # boxplot sortino
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_title('Sortino Ratio')
    sns.violinplot(data=[sortino_control, sortino_agent], ax=ax)
    sns.pointplot(data=[sortino_control, sortino_agent], ax=ax, estimator=np.mean, color='red')
    ax.set_xticklabels(['Buy & Hold', 'Model'])
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/sortino.png')

    mean_control_performance = np.mean(control_performance, axis=0)
    mean_agent_performance = np.mean(agent_performance, axis=0)
    std_control_performance = np.std(control_performance, axis=0)
    std_agent_performance = np.std(agent_performance, axis=0)

    fig, ax = plt.subplots(figsize=(40, 10))
    ax.plot(np.arange(episode_timesteps), mean_control_performance, linewidth=4)
    ax.plot(np.arange(episode_timesteps), mean_agent_performance, linewidth=4)
    # fill std
    ax.fill_between(np.arange(episode_timesteps), mean_control_performance - std_control_performance, mean_control_performance + std_control_performance, alpha=0.2)
    ax.fill_between(np.arange(episode_timesteps), mean_agent_performance - std_agent_performance, mean_agent_performance + std_agent_performance, alpha=0.2)
    ax.legend(['Buy & Hold', 'Model'])
    ax.set_title('Return')
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/model_vs_hold.png')


