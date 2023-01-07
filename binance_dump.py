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

    pairs = os.listdir("binance_data/spot/monthly/klines")
    for pair in pairs:
        files = os.listdir(f"binance_data/spot/monthly/klines/{pair}/15m")
        columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                                   "Quote asset volume",  "Number of trades", "Taker buy base asset volume",
                                   "Taker buy quote asset volume", "Ignore"]
        df = pd.DataFrame(columns=columns)
        for file in files:
            tmp = pd.read_csv(f"binance_data/spot/monthly/klines/{pair}/15m/{file}", header=None, names=columns)
            df = pd.concat([df, tmp], axis=0)

        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.rename(columns={"Open time": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close",
                           "Volume": "volume"}, inplace=True)
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.sort_values(by='date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_feather(f"binance_data/all/{pair}.feather")
        df.to_csv(f"binance_data/all/{pair}.csv", index=False)
