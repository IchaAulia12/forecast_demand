# preprocessing.py
import numpy as np
import pandas as pd

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def create_date_features(df):
    df = df.copy()
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    # note: use isocalendar().week to avoid deprecated .weekofyear
    df['week_of_year'] = df.date.dt.isocalendar().week.astype(int)
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df['is_wknd'] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

def lag_features(df, lags=[91,98,105,112,119,126,182,364,546,728]):
    df = df.copy()
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store','item'])['sales'].transform(lambda x: x.shift(lag)) + random_noise(df)
    return df

def roll_mean_features(df, windows=[365,546,730]):
    df = df.copy()
    for window in windows:
        df[f'sales_roll_mean_{window}'] = df.groupby(['store','item'])['sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type='triang').mean()
        ) + random_noise(df)
    return df

def ewm_features(df, alphas=[0.99,0.95,0.9,0.8,0.7,0.5], lags=[91,98,105,112,180,270,365,546,728]):
    df = df.copy()
    for alpha in alphas:
        for lag in lags:
            col = f'sales_ewm_alpha_{str(alpha).replace(".","")}_lag_{lag}'
            df[col] = df.groupby(['store','item'])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return df

def generate_all_features(df):
    """
    Input:
      df with columns: ['date' (datetime), 'store', 'item', 'sales'] (sales in raw scale OR log1p)
    Output:
      df with feature columns (one-hot for month & day_of_week)
    """
    df = df.copy()
    df = create_date_features(df)
    df = lag_features(df)
    df = roll_mean_features(df)
    df = ewm_features(df)
    # one-hot encoding for day_of_week and month (same as training)
    df = pd.get_dummies(df, columns=['day_of_week','month'], drop_first=False)
    return df

def prepare_features_for_model(df_features, model_cols, global_means):
    """
    - df_features: dataframe after generate_all_features
    - model_cols: list of columns used during training
    - global_means: dict {col: mean_value}
    Returns X (dataframe) with same columns as model_cols and missing filled using global_means
    """
    df = df_features.copy()

    # ensure all dummies exist: add missing cols as NaN
    for c in model_cols:
        if c not in df.columns:
            df[c] = np.nan

    # keep only model cols in same order
    X = df[model_cols].copy()

    # fill using global means where available
    # convert global_means to Series to align indices
    gm = pd.Series(global_means)
    X = X.fillna(gm)

    # any remaining NaN -> fill with 0 as fallback
    X = X.fillna(0)

    return X
