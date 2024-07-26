import pandas as pd
from trading import indicators
from typing import Dict, List

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

def return_category(period: int, return_target: float):
  def return_category_generator(data: pd.DataFrame):
    new_columns = {}
    high = data['High'].rolling(window=period).max().shift(-period)
    new_columns[f"high_{period}"] = high

    max_return = (high - data['Open'].shift(-1)) / data['Open'].shift(-1)
    new_columns[f"max_return_{period}"] = max_return

    category = (max_return >= return_target).astype(int)
    new_columns[f"return_category_{period}_{return_target}"] = category
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return return_category_generator

def loss_category(period: int, loss_target: float):
  def loss_category_generator(data: pd.DataFrame):
    new_columns = {}
    low = data['Low'].rolling(window=period).min().shift(-period)
    new_columns[f"low_{period}"] = low

    max_loss = (low - data['Open'].shift(-1)) / data['Open'].shift(-1)
    new_columns[f"max_loss_{period}"] = max_loss

    category = (max_loss <= loss_target).astype(int)
    new_columns[f"loss_category_{period}_{loss_target}"] = category
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return loss_category_generator

def return_loss_category(period: int, return_target: float, loss_target: float):
  def return_loss_category_generator(data: pd.DataFrame):
    new_columns = {}
    
    if f"return_category_{period}_{return_target}" not in data:
      data = return_category(period, return_target)(data)
    if f"loss_category_{period}_{loss_target}" not in data:
      data = loss_category(period, loss_target)(data)

    new_values = (data[f"return_category_{period}_{return_target}"] > data[f"loss_category_{period}_{loss_target}"]).astype(int)
    new_columns = {f"return_loss_category_{period}_{return_target}_{loss_target}": new_values}    
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return return_loss_category_generator

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
    new_columns = []

    if f"MA{short_period}" in data:
      short_sma = data[f"MA{short_period}"]
    else:
      short_sma = indicators.sma(data, short_period)
      new_columns.append(short_sma)
      short_sma = short_sma[f"MA{short_period}"]

    if f"MA{long_period}" in data:
      long_sma = data[f"MA{long_period}"]
    else:
      long_sma = indicators.sma(data, long_period)
      new_columns.append(long_sma)
      long_sma = long_sma[f"MA{long_period}"]

    diff = pd.DataFrame({f"MA diff:{short_period}-{long_period}": short_sma - long_sma})
    new_columns.append(diff)
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
    new_columns = []

    if f"EMA{short_period}" in data:
      short_ema = data[f"EMA{short_period}"]
    else:
      short_ema = indicators.ema(data, short_period)
      new_columns.append(short_ema)
      short_ema = short_ema[f"EMA{short_period}"]

    if f"EMA{long_period}" in data:
      long_ema = data[f"EMA{long_period}"]
    else:
      long_ema = indicators.ema(data, long_period)
      new_columns.append(long_ema)
      long_ema = long_ema[f"EMA{long_period}"]

    diff = pd.DataFrame({f"EMA diff:{short_period}-{long_period}": short_ema - long_ema})
    new_columns.append(diff)
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return ema_diff_generator

def wma(period: int):
  def wma_generator(data: pd.DataFrame):
    data = _add_new_columns(data, indicators.wma(data, period))
    return data.dropna()
  return wma_generator

def wma_diff(short_period: int, long_period: int):
  if short_period > long_period:
    temp = short_period
    short_period = long_period
    long_period = temp
  def wma_diff_generator(data: pd.DataFrame):
    new_columns = []

    if f"WMA{short_period}" in data:
      short_wma = data[f"WMA{short_period}"]
    else:
      short_wma = indicators.wma(data, short_period)
      new_columns.append(short_wma)
      short_wma = short_wma[f"WMA{short_period}"]

    if f"WMA{long_period}" in data:
      long_wma = data[f"WMA{long_period}"]
    else:
      long_wma = indicators.wma(data, long_period)
      new_columns.append(long_wma)
      long_wma = long_wma[f"WMA{long_period}"]

    diff = pd.DataFrame({f"WMA diff:{short_period}-{long_period}": short_wma - long_wma})
    new_columns.append(diff)
    data = _add_new_columns(data, new_columns)
    return data.dropna()
  return wma_diff_generator

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

def _add_new_columns(data: pd.DataFrame, new_columns: pd.DataFrame|Dict[str, pd.Series]|List[pd.DataFrame]):
  if isinstance(new_columns, pd.DataFrame):
    data = pd.concat([data, new_columns], axis=1)
  elif isinstance(new_columns, Dict):
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
  elif isinstance(new_columns, List):
    data = pd.concat([data] + new_columns, axis=1)

  return data
