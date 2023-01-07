import os

import pandas as pd
from binance_historical_data import BinanceDataDumper
from datetime import date


if __name__ == '__main__':

    # data_dumper = BinanceDataDumper(
    #     path_dir_where_to_dump="binance_data",
    #     data_type="klines",  # aggTrades, klines, trades
    #     data_frequency="15m",  # argument for data_type="klines"
    # )
    #
    # data_dumper.dump_data(
    #     tickers=["BTCUSDT", "ETHUSDT", "ADAUSDT", "LTCUSDT", "VETUSDT", "XRPUSDT", "DOTUSDT", "AVAXUSDT", "SOLUSDT"],
    #     date_start=date(2021, 1, 1),
    #     date_end=date(2023, 1, 1),
    # )
    #
    # data_dumper.delete_outdated_daily_results()

    pairs = os.listdir("binance_data/all")

    for pair in pairs:
        if pair.endswith('.feather'):
            df = pd.read_feather(f'binance_data/all/{pair}')
            print(df.head())

