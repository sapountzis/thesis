import matplotlib.pyplot as plt
import numpy as np
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
    model = sorted(os.listdir(model_logs))[-1]

    print(f'Loading model {model} logs')

    logs_dir = f'{model_logs}/{model}'
    episode_infos = sorted([f for f in os.listdir(logs_dir) if f.endswith('episode_info.csv')], key=lambda x: int(x.split('_')[1]))
    trades = sorted([f for f in os.listdir(logs_dir) if f.endswith('trades.csv')], key=lambda x: int(x.split('_')[1]))

    config = make_config('config.json')
    episode_timesteps = config['env_kwargs']['episode_length']
    step = config['intervals'][0]
    df = load_train_df(config['data_dir'], intervals=config['intervals'], coins=config['coins'],
                       fiat=config['fiat'], index_col='date',
                       end_date=config['train_end'], start_date=config['train_start'])

    ckpnt = 2807

    coin_data = {coin: data
                 for interval, coin in df.items() for coin, data in coin.items() if interval == step}

    coins = list(coin_data.keys())

    trade = [t for t in trades if str(ckpnt) in t]
    trade = pd.read_csv(f'{logs_dir}/{trade[-1]}')
    trade['step'] = trade['step'].astype(int)
    dates = coin_data[list(coin_data.keys())[0]].loc[:, ['date']]
    dates['step'] = dates.index
    alloc = pd.merge(dates, trade, on='step', how='left').fillna(method='ffill')
    trade = pd.merge(dates, trade, on='step', how='right').fillna(method='ffill')

    sns.set(font_scale=4.5, style='whitegrid')
    fig, axes = plt.subplots(len(coin_data), 2, figsize=(50, len(coin_data) * 5), gridspec_kw={'width_ratios': [4, 1]})
    for i, (coin, data) in enumerate(coin_data.items()):
        ax_idx = i, 0
        axes[ax_idx].plot(data['date'], data['open'], linewidth=4, alpha=0.8)
        axes[ax_idx].set_ylabel(coin)
        axes[ax_idx].yaxis.set_major_locator(plt.MaxNLocator(4))
        axes[ax_idx].scatter(trade['date'], data[data['date'].isin(trade['date'])]['open'],
                             color='red', s=10, zorder=2)
        axes[ax_idx].yaxis.set_major_formatter(FormatStrFormatter(f'%.{4-max(0, int(np.log10(data["open"].max())))}f'))

        if i != len(coin_data) - 1:
            axes[ax_idx].set_xticklabels([])

        if i == 0:
            axes[ax_idx].set_title(f'Prices')

    for i, (coin, data) in enumerate(coin_data.items()):
        ax_idx = i, 1
        axes[ax_idx].plot(alloc['date'], alloc[coin])
        axes[ax_idx].yaxis.set_major_locator(plt.MaxNLocator(4))
        axes[ax_idx].xaxis.set_major_locator(plt.MaxNLocator(2))
        axes[ax_idx].axhline(alloc[coin].mean(), color='red', zorder=2, linewidth=4)
        axes[ax_idx].yaxis.set_major_formatter(FormatStrFormatter(f'%.2f'))

        if i != len(coin_data) - 1:
            axes[ax_idx].set_xticklabels([])

        if i == 0:
            axes[ax_idx].set_title(f'Portfolio %')
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

    coin_data_mean = pd.concat([coin_data[coin].rename(columns={'open': coin})[coin] for coin in coins], axis=1)
    coin_data_mean /= coin_data_mean.iloc[0]
    coin_data_mean = coin_data_mean.mean(axis=1)
    # episode_info = [e for e in episode_info if str(ckpnt) in e]
    episode_info = [e for e in episode_infos if str(ckpnt) in e]
    episode_info = pd.read_csv(f'{logs_dir}/{episode_info[-1]}')
    # episode_info = pd.read_csv(f'{logs_dir}/{episode_infos[-1]}')
    x = np.arange(0, len(episode_info)) * config['log_freq']
    fig, ax = plt.subplots(figsize=(40, 10))
    ax.set_title('Return')
    ax.plot(dates.iloc[x]['date'], episode_info['portfolio_value'] / config['env_kwargs']['capital'], linewidth=4)
    ax.plot(dates['date'], coin_data_mean, linewidth=4)
    ax.legend(['Model', 'Buy & Hold'])
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/model_vs_hold.png')

    trade_rate = [pd.read_csv(f'{logs_dir}/{t}').shape[0] / episode_timesteps for t in trades]
    fig, ax = plt.subplots(figsize=(40, 10))
    ax.set_title('Trade Rate')
    ax.plot(np.arange(len(trade_rate)), trade_rate, linewidth=4)
    ax.set_xlabel('Episode')
    fig.tight_layout()
    fig.show()
    fig.savefig(f'visualizations/trade-rate.png')
