import pandas as pd
from trading import indicators
from typing import Dict

def generate_return(data: pd.DataFrame) -> None:
  return_col = data["Close"].pct_change() * 100
  data = _add_new_columns(data, {"% Return": return_col})
  return data

def lag_features(lag: int = 1):
  def lag_feature_generator(data: pd.DataFrame):
    columns = data.columns
    new_columns = {}
    for i in range(1, lag + 1):
      for column in columns:
        new_columns[f'lag_{i}_{column}'] = data[column].shift(i)
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return lag_feature_generator

def return_categories(data: pd.DataFrame) -> None:
  day5_max = data["High"].rolling(window=2).max().shift(-2)
  data["5 day high"] = day5_max
  return_col = (day5_max - data["Close"]) / data["Close"]
  return_col.fillna(0, inplace=True)
  data["5 day highest return"] = return_col
  categories_col = (return_col >= 0.015).astype(int)
  data["5 day Return Buckets"] = categories_col
  return data

def sma(period: int):
  def sma_generator(data: pd.DataFrame):
    data = _add_new_columns(data, indicators.sma(data, period))
    return data.dropna()
  return sma_generator

def sma_diff(short_period: int, long_period: int):
  if short_period > long_period:
    temp = short_period
    short_period = long_period
    long_period = temp
  def sma_diff_generator(data: pd.DataFrame):
    new_columns = {}

    if f"MA{short_period}" in data:
      short_sma = data[f"MA{short_period}"]
    else:
      short_sma = indicators.sma(data, short_period)
      new_columns[f"MA{short_period}"] = short_sma

    if f"MA{long_period}" in data:
      long_sma = data[f"MA{long_period}"]
    else:
      long_sma = indicators.sma(data, long_period)
      new_columns[f"MA{long_period}"] = long_sma
  
    new_columns[f"MA diff:{short_period}-{long_period}"] = short_sma - long_sma
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return sma_diff_generator

def ema(period: int):
  def ema_generator(data: pd.DataFrame):
    data = _add_new_columns(data, indicators.ema(data, period))
    return data.dropna()
  return ema_generator

def ema_diff(short_period: int, long_period: int):
  if short_period > long_period:
    temp = short_period
    short_period = long_period
    long_period = temp
  def ema_diff_generator(data: pd.DataFrame):
    new_columns = {}

    if f"EMA{short_period}" in data:
      short_ema = data[f"EMA{short_period}"]
    else:
      short_ema = indicators.ema(data, short_period)
      new_columns[f"EMA{short_period}"] = short_ema

    if f"EMA{long_period}" in data:
      long_ema = data[f"EMA{long_period}"]
    else:
      long_ema = indicators.ema(data, long_period)
      new_columns[f"EMA{long_period}"] = long_ema
  
    new_columns[f"EMA diff:{short_period}-{long_period}"] = short_ema - long_ema
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return ema_diff_generator

def macd(short_period: int = 12,
         long_period: int = 26,
         signal_period: int = 9):
  if short_period > long_period:
    temp = short_period
    short_period = long_period
    long_period = temp
  def macd_generator(data: pd.DataFrame):
    macd = indicators.macd(data,
                           short_period = short_period,
                           long_period = long_period,
                           signal_period = signal_period)
    data = _add_new_columns(data, macd)
    return data.dropna()
  return macd_generator

def _add_new_columns(data: pd.DataFrame, new_columns: pd.DataFrame|Dict[str, pd.DataFrame]):
  data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
  return data
