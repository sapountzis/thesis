import pandas as pd
import numpy as np
from typing import Union, Dict
import warnings


def cast_column_to_time(col: pd.Series):
    if isinstance(col.iloc[0], (pd.DatetimeIndex, pd.DatetimeScalar, pd.Timestamp)):
        return col
    else:
        return pd.to_datetime(col)


def get_time_index(candles: pd.DataFrame):
    if 'date' in candles.columns:
        return cast_column_to_time(candles['date'])
    elif 'time' in candles.columns:
        return cast_column_to_time(candles['time'])
    else:
        if isinstance(candles.index, (pd.DatetimeIndex)):
            return candles.index
        else:
            raise ValueError("Candle Dataframe does not seem to have any time index")


def candle_values(candles):
    return candles


def standard_price_diffs(candles, use_pct=True):
    """
    Construct the difference or pct difference between the current high and low prices
    to the current close prices
    :param candles:
    :param use_pct:
    :return:
    """
    high_diff = candles.high - candles.close
    low_diff = candles.close - candles.low
    if use_pct:
        high_diff /= candles.close
        low_diff /= candles.close
    return pd.DataFrame(data=dict(high_diff=high_diff, low_diff=low_diff))


def intra_bar_diffs(candles, columns, use_pct=True, smoothing_window=None, shift=1):
    df = candles[columns]
    smooth_str = ''
    df_smoothed = df
    if smoothing_window is not None:
        df_smoothed = df.rolling(window=smoothing_window, min_periods=1).mean()
        smooth_str = f"_{smoothing_window}"
    if not use_pct:
        diffs = df_smoothed.shift(shift) - df
    else:
        diffs = ((df_smoothed.shift(shift) - df) / df_smoothed.shift(shift))
        diffs = diffs.replace([np.inf, -np.inf], np.nan).bfill()
    new_col_names = {col: f"{col}_diff_{shift}{smooth_str}" for col in df.columns.tolist()}
    return diffs.rename(new_col_names, axis=1).bfill()


def hl_to_pclose(candles):
    hl = candles[['high', 'low']]
    close = candles['close'].shift(1)
    hl_diff = pd.DataFrame(((hl.values / close.bfill().values.reshape(-1, 1)) - 1), index=hl.index)
    hl_diff.columns = ['high_to_pclose', 'low_to_pclose']
    return hl_diff


def hl_volatilities(candles, smoothing_window):
    hl = candles[['high', 'low']]
    vol = hl.high / hl.low
    vol = vol.rolling(window=smoothing_window, min_periods=1).mean().ffill().bfill()
    vol.name = f'hl_volatility_{smoothing_window}w'
    return vol


def return_volatilities(candles, smoothing_window):
    returns = candles['close'].pct_change().bfill().ffill()
    vol = returns.rolling(window=smoothing_window, min_periods=1).std().ffill().bfill()
    vol.name = f"return_volatility_{smoothing_window}w"
    return vol


def time_feature_day(candles):
    time_index = get_time_index(candles)
    minute = time_index.hour * 60 + time_index.minute
    feature1 = np.sin(minute * 2 * np.pi / (60 * 24))
    feature2 = np.cos(minute * 2 * np.pi / (60 * 24))
    return pd.DataFrame(data=dict(
        day_sin=feature1, day_cos=feature2
    ), index=time_index)


def time_feature_week(candles):
    time_index = get_time_index(candles)
    minute = (time_index.dayofweek * 24 + time_index.hour) * 60 + time_index.minute
    fraction = minute / (7 * 60 * 24)
    feature1 = np.sin(fraction * 2 * np.pi)
    feature2 = np.cos(fraction * 2 * np.pi)
    return pd.DataFrame(data=dict(
        week_sin=feature1, week_cos=feature2
    ), index=time_index)


def time_feature_month(candles):
    time_index = get_time_index(candles)
    minutes = (time_index.day * 24 + time_index.hour) * 60 + time_index.minute
    total_minutes = (time_index.daysinmonth + 1) * 24 * 60
    fraction = minutes / total_minutes
    feature1 = np.sin(fraction * 2 * np.pi)
    feature2 = np.cos(fraction * 2 * np.pi)
    return pd.DataFrame(data=dict(
        month_sin=feature1, month_cos=feature2
    ), index=time_index)


def time_feature_year(candles):
    time_index = get_time_index(candles)
    start_year = pd.to_datetime(time_index.year.astype('str'))
    stop_year = pd.to_datetime((time_index.year + 1).astype('str'))
    time_ratio = (time_index - start_year) / (stop_year - start_year)
    feature1 = np.sin(2 * np.pi * time_ratio)
    feature2 = np.cos(2 * np.pi * time_ratio)
    return pd.DataFrame(data=dict(
        year_sin=feature1, year_cos=feature2
    ), index=time_index)


def next_return(candles):
    """
    DO NOT USE THIS FEATURE. ONLY FOR TESTING
    This feature includes the returns of the next time step. It is used
    to ensure that a model can fully take advantage of into in the input
    :param candles:
    :return:
    """
    future_high_return = (candles.high.shift(-1).ffill() / candles.close) - 1.
    future_low_return = 1. - (candles.low.shift(-1).ffill() / candles.close)

    return pd.DataFrame(data=dict(
        future_return=candles.close.pct_change().shift(-1).ffill(),
        future_high_return=future_high_return,
        future_low_return=future_low_return,
        future_high_open=(candles['high'] / candles['open']).shift(-1).ffill() - 1.,
        future_low_open=1. - (candles['low'] / candles['open']).shift(-1).ffill(),
    ))


def extract_feature_groups(candles,
                           features_config=(
                                   dict(func_name='inter_bar_changes', columns=['close', 'high', 'low'],
                                        use_pct=True),
                                   dict(func_name='internal_bar_diff', use_pct=True),
                                   dict(func_name='hl_to_pclose'))):
    feature_list = []
    feature_names = []
    for i, f in enumerate(features_config):
        f = f.copy()
        fname = f.pop('func_name')
        feature_names.append(f.pop('name') if 'name' in f else fname)

        # This should become a mapping dict[str, func]
        if fname == 'inter_bar_changes':
            feature_list.append(intra_bar_diffs(candles, **f))
        elif fname == 'candle_values':
            feature_list.append(candle_values(candles))
        elif fname == 'internal_bar_diff':
            feature_list.append(standard_price_diffs(candles, **f))

        elif fname == 'hl_to_pclose':
            feature_list.append(hl_to_pclose(candles, **f))

        elif fname == 'hl_volatilities':
            feature_list.append(hl_volatilities(candles, **f))

        elif fname == 'return_volatilities':
            feature_list.append(return_volatilities(candles, **f))

        elif fname == 'time_feature_day':
            feature_list.append(time_feature_day(candles))

        elif fname == 'time_feature_year':
            feature_list.append(time_feature_year(candles))

        elif fname == 'time_feature_month':
            feature_list.append(time_feature_month(candles))

        elif fname == 'time_feature_week':
            feature_list.append(time_feature_week(candles))

        elif fname == 'next_return':
            feature_list.append(next_return(candles))

    unique_names = list(set(feature_names))
    feature_names = np.asarray(feature_names)
    grouped_features = []
    for f in unique_names:
        idxs = np.argwhere(feature_names == f)[:, 0]
        grouped_features.append(pd.concat([feature_list[i] for i in idxs], axis=1))

    features = pd.concat(grouped_features, keys=unique_names, axis=1)
    assert not pd.isna(features).any().any()
    return features


def extract_statistics(features, per_group=True):
    if per_group:
        means = features.mean().groupby(axis=0, level=0).mean()
        stds = features.std().groupby(axis=0, level=0).mean()
    else:
        if type(features.columns) is pd.MultiIndex:
            means = features.mean()
            stds = features.std()
        else:
            means = features.mean()
            stds = features.std()
    return means, stds


def normalize_features(features, mean, std, clip=6.,
                       mean_ignore_atol=5e-5, check_nans=True):
    for grp in std.index.levels[0]:
        cur_mean = mean.loc[grp]
        cur_std = std.loc[grp]
        cur_features = features.loc[:, grp]
        ignore_mean = mean_ignore_atol is not None and np.allclose(cur_mean, 0, atol=mean_ignore_atol)
        assert not (cur_std == 0).any(), f"{grp} feature has 0 std"
        if ignore_mean:
            features.loc[:, grp] = (cur_features / cur_std).values
        else:
            features.loc[:, grp] = ((cur_features - cur_mean) / cur_std).values
        if check_nans:
            assert not pd.isna(features).any().any()
    if clip is not None:
        features = features.clip(-clip, clip)
    return features.astype(np.float32)


def construct_features(candles_dict, train_end, feature_config, normalize=True):
    train_end = pd.to_datetime(train_end)
    feature_dict, feature_stats = dict(), dict()
    ret_stats = {}
    feature_df_list = []
    for key, candles in candles_dict.items():
        features = extract_feature_groups(
            candles,
            feature_config
        ).astype(np.float32)
        split_idx = features.index.searchsorted(pd.to_datetime(train_end))
        feature_df_list.append(features.iloc[:split_idx])
        assert not pd.isna(features).any().any()
        if features.iloc[:split_idx].empty:
            warnings.warn(f"Detected empty features in {key}. Ignoring")
            continue
        feature_dict[key] = features
        # feature_stats[key] = (means, stds)
    total_df = pd.concat(feature_df_list, axis=0, copy=False)
    reconciled_columns = pd.MultiIndex.from_tuples(total_df.columns.to_flat_index())
    assert all(reconciled_columns.values == total_df.columns.values)
    total_df.columns = reconciled_columns
    means, stds = extract_statistics(total_df, per_group=False)
    if normalize:
        for k, v in feature_dict.items():
            feature_dict[k] = normalize_features(v, means, stds, mean_ignore_atol=3e-4, clip=None)
            assert not pd.isna(feature_dict[k]).any().any()
            feature_dict[k].attrs['feature_stats'] = ret_stats
    return feature_dict
