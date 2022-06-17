import os
import asyncio
from datetime import datetime
import requests
import pandas as pd
from finrl.finrl_meta.preprocessor.preprocessors import data_split
# load_dotenv()

# api_key = os.getenv('COIN_API7')
fiat = 'USDT'
coins = ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOT', 'DOGE']
symbol_type = 'SPOT'
exchange = 'BINANCE'
date_start = '2021-01-01T00:00:00.0000000Z'
date_end = '2022-02-13T15:10:00.0000000Z'
data_dir = 'data'
aligned_dir = 'data_aligned'


class CoinAPI:
    def __init__(self, api_key, fiat='USDT', symbol_type='SPOT'):
        self.api_key = api_key
        self.fiat = fiat
        self.symbol_type = symbol_type
        self.INTERVAL_1MINUTE = '1MIN'
        self.INTERVAL_5MINUTE = '5MIN'
        self.INTERVAL_10MINUTE = '10MIN'
        self.INTERVAL_30MINUTE = '30MIN'
        self.INTERVAL_1HOUR = '1HRS'
        self.INTERVAL_4HOUR = '4HRS'
        self.INTERVAL_1DAY = '1DAY'

    def _get(self, path='', base='https://rest.coinapi.io/v1'):
        url = f'{base}/{path}'
        print(url)
        headers = {'X-CoinAPI-Key': api_key, 'Accept': 'application/json', 'Accept-Encoding': 'deflate, gzip'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code)
            print(response.text)
            raise IOError

    def get_max_date(self, coin_info: dict):
        return max(coin_info[k]['data_start'] for k in coin_info.keys())

    def get_coin_info(self, coins, exchange=None, symbol_type=None, date_threshold=None):

        def check_entry(entry):
            has_start_date = 'data_quote_start' in entry
            in_coins = entry['asset_id_base'] in coins
            in_fiat = entry['symbol_id'].endswith(self.fiat)

            return has_start_date and in_coins and in_fiat

        if coins:
            symbol_filter = None
            if exchange or symbol_type:
                symbol_filter = f'?filter_symbol_id={exchange if exchange else ""}{f"_{symbol_type}" if symbol_type else ""}'
            path = f'symbols{symbol_filter if symbol_filter else ""}'
            data = self._get(path=path)

            data_sel = [entry for entry in data if check_entry(entry)]

            if exchange:
                data_sel = [d for d in data_sel if exchange in d['symbol_id']]

            pair_data = {}

            for coin in coins:
                data_coin = [d for d in data_sel if d['asset_id_base'] == coin]
                min_ex = data_coin[0]['symbol_id']
                min_date = data_coin[0]['data_quote_start']

                for i in range(1, len(data_coin)):
                    if data_coin[i]['data_start'] < min_date:
                        min_ex = data_coin[i]['symbol_id']
                        min_date = data_coin[i]['data_quote_start']

                if date_threshold:
                    if min_date <= date_threshold:
                        pair_data.update({coin: {'symbol': min_ex, 'data_start': min_date}})
                else:
                    pair_data.update({coin: {'symbol': min_ex, 'data_start': min_date}})

            return pair_data

    def get_coin_hist_data(self, symbol, interval, start, end=None, limit=100000):
        # start = datetime.fromisoformat(start).isoformat()
        # end = datetime.fromisoformat(end).isoformat() if end else None
        path = f'ohlcv/{symbol}/history?period_id={interval}' \
               f'{f"&time_start={start}" if start else ""}' \
               f'{f"&time_end={end}" if end else ""}' \
               f'{f"&limit={limit}" if limit else ""}'

        return self._get(path=path)


def create_dataset(client: CoinAPI):
    os.makedirs(data_dir, exist_ok=True)
    coin_info = client.get_coin_info(coins, exchange=exchange, symbol_type=symbol_type)
    # max_date = client.get_max_date(coin_info)

    intervals = {'5min': client.INTERVAL_5MINUTE, '30min': client.INTERVAL_30MINUTE,
                 '4hour': client.INTERVAL_4HOUR, '1day': client.INTERVAL_1DAY}

    for i_label, interval in intervals.items():
        for coin in coins:
            finename = f'{exchange}_{coin}_{fiat}_{i_label}.csv'
            if finename not in os.listdir(data_dir):
                res = client.get_coin_hist_data(coin_info[coin]['symbol'], interval, start=date_start, end=date_end)

                df = pd.DataFrame(res)
                print(f'{exchange}, {coin}, {fiat}, {i_label} | Saving {finename}')
                df.to_csv(f'{data_dir}/{finename}', index=False)


def align_data():
    os.makedirs(aligned_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.endswith('hdf5')]

    dfs = {f: pd.read_hdf(f'{data_dir}/{f}') for f in files}
    datetime_end_min = min([df.index.values[-1] for df in dfs.values()])
    datetime_start_max = max([df.index.values[0] for df in dfs.values()])

    for f, df in dfs.items():
        df = df[(df.index >= datetime_start_max) & (df.index <= datetime_end_min)]
        print(df)
        df.index.name = 'datetime'
        df = df.reset_index()
        df.to_feather(f'{aligned_dir}/{f.replace("hdf5", "feather")}')

        tmp = df.resample('5min', label='right', closed='right', origin='start_day', on='datetime').agg(
            {'open': 'first', 'low': 'min', 'high': 'max', 'close': 'last', 'volume': 'sum'})

        print(tmp)
        print(tmp.reset_index()['datetime'])


if __name__ == '__main__':
    fiat = 'USDT'
    data = pd.read_feather('data/OHLC_candles/Crypto/minute_binance/BTCUSDT.feather')
    from loading_utils import load_feather_dir

    pairs = [p.split('.')[0] for p in os.listdir('data/OHLC_candles/Crypto/minute_binance') if fiat in p]

    data = load_feather_dir(path='data/OHLC_candles/Crypto/minute_binance', pairs=pairs, n_workers=4, resample='5min')

    from preprocessing_utils import construct_features
    train_end = '2021'
    train_start = None
    feature_config = (
        dict(name='candle_values', func_name='candle_values'),
    )
    data_pre = construct_features(candles_dict=data, feature_config=feature_config, train_end=train_end, normalize=True)

    data_train = {k: data_split(v.reset_index(), start='2021', end=None) for k, v in data_pre.items()}

    print(data_train)





