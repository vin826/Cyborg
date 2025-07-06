import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0)
    neg_mf = mf.where(tp < tp.shift(), 0)
    mfi = 100 - (100 / (1 + pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum()))
    return mfi

def compute_wavetrend(df: pd.DataFrame, channel_length: int = 9, average_length: int = 12) -> pd.DataFrame:
    tp = (df['high'] + df['low'] + df['close']) / 3
    esa = tp.ewm(span=channel_length).mean()
    d = abs(tp - esa).ewm(span=channel_length).mean()
    ci = (tp - esa) / (0.015 * d)
    wt1 = ci.ewm(span=average_length).mean()
    wt2 = wt1.rolling(4).mean()
    return pd.DataFrame({'wt1': wt1, 'wt2': wt2})

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr