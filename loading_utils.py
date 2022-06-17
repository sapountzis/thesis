from functools import partial
import pandas as pd
from pathlib import Path
import warnings


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start if start else True) & (df[target_date_col] < end if end else True)]
    data = data.sort_values([target_date_col], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def resample_candles(candles: pd.DataFrame, window, label="right", closed="right", dropna=True) -> pd.DataFrame:
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("Candle dataframe index is not a Datetimeindex")
    aggregation_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in candles.columns:
        aggregation_dict["volume"] = "sum"
    candles = candles.resample(window, label=label, closed=closed).agg(aggregation_dict).dropna()
    if dropna:
        return candles.dropna()
    return candles


def load_feather(path: Path, resample=None, index_col='date'):
    df = pd.read_feather(str(path))
    cols = df.columns.tolist()
    for c in ['open', 'high', 'low', 'close', index_col]:
        assert c in cols, f"{c} is missing from columns from {str(path)}. All cols in this df is {cols}"
    if index_col:
        df.set_index(index_col, inplace=True)
    if resample:
        df = resample_candles(df, window=resample)
    return path.stem.lower(), df


def load_feather_dir(path: Path, resample=None, pairs=None, index_col='date', n_workers=None):
    path = Path(path).expanduser()
    feather_dict = dict()
    dirlist = list(path.iterdir())
    flist = [f for f in dirlist if f.suffix == '.feather']
    if len(flist) < len(dirlist):
        ignored_files = set(dirlist) - set(flist)
        ignored_files = [f.name for f in ignored_files]
        print(f"Ignoring non-feather files: {','.join(ignored_files)}")
    if pairs is not None:
        pairs = [p.lower() for p in pairs]
        flist = filter(lambda f: f.stem.lower() in pairs, flist)
    if not n_workers:
        for f in flist:
            if f.suffix == '.feather':
                feather_dict[f.stem.lower()] = load_feather(f, resample=resample, index_col=index_col)[1]
    else:
        from multiprocessing import Pool
        p = Pool(n_workers)
        load_feather_par = partial(load_feather, resample=resample, index_col=index_col)
        for res in p.imap_unordered(load_feather_par, flist):
            feather_dict[res[0]] = res[1]
    for k in list(feather_dict.keys()):
        if (feather_dict[k].close == 0).any():
            warnings.warn(f"Found close price zero in candles of {k} pair. Deleting pair...")
            del feather_dict[k]
    return feather_dict


def load_train_df(path: Path, intervals=None, coins=None, fiat=None, index_col='date', end_date=None):
    pairs = [f'{c}{fiat}' for c in coins]
    data = load_feather_dir(path, pairs=pairs, index_col=index_col, n_workers=4)
    data = {k: data_split(v.reset_index(), start=None, end=end_date).set_index(index_col, inplace=False)
            for k, v in data.items()}
    interval_data = {interval: {k: resample_candles(v, interval).reset_index() for k, v in data.items()} for interval in intervals}

    return interval_data



if __name__ == '__main__':
    pass
