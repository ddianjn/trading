import numpy as np
import pandas as pd

def sma(data: pd.DataFrame, period: int):
  ma = data['Close'].rolling(window=period).mean()
  return pd.DataFrame({f'MA{period}': ma})

def ema(data: pd.DataFrame, period: int):
  """Calculates the Exponential Moving Average (EMA) for a given data series.

  Args:
    data: A pandas Series containing the data for which to calculate EMA.
    period: The number of periods for the EMA calculation.

  Returns:
    A pandas Series containing the calculated EMA.
  """

  alpha = 2 / (period + 1)
  ema = data['Close'].ewm(alpha=alpha, adjust=False).mean()
  return pd.DataFrame({f'EMA{period}': ema})

def macd(data: pd.DataFrame,
         short_period: int = 12,
         long_period: int = 26,
         signal_period: int = 9):
  """Calculates the MACD, Signal, and Histogram for a given data series.

  Args:
    data: A pandas Series containing the data for which to calculate MACD.
    short_period: The number of periods for the short-term EMA.
    long_period: The number of periods for the long-term EMA.
    signal_period: The number of periods for the signal line.

  Returns:
    A pandas DataFrame containing the MACD, Signal, and Histogram.
  """

  close_data = data['Close']
  short_ema = close_data.ewm(span=short_period, adjust=False).mean()
  long_ema = close_data.ewm(span=long_period, adjust=False).mean()
  macd = short_ema - long_ema
  signal = macd.ewm(span=signal_period, adjust=False).mean()
  histogram = macd - signal

  return pd.DataFrame({'MACD': macd, 'Signal': signal, 'Histogram': histogram})

def wma(data: pd.DataFrame,
        period: int,
        source: str = "Close"):
  """Calculates the Weighted Moving Average (WMA) of a pandas Series.

  Args:
    data: The pandas Series containing the data.
    period: The length of the WMA window.
    source: The column name of the data in the pandas Series.

  Returns:
    A pandas Series containing the WMA values.
  """
  weights = np.array([(period - i) * period for i in range(period)])
  wma = data[source].rolling(window=period).apply(lambda x: (x * weights[::-1]).sum() / weights.sum(), raw=True)
  return pd.DataFrame({f'WMA{period}': wma})
