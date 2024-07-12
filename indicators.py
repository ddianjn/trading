import numpy as np
import pandas as pd

def sma(data: pd.DataFrame, period: int):
  return data['Close'].rolling(window=period).mean()

def ema(data: pd.DataFrame, period: int):
  """Calculates the Exponential Moving Average (EMA) for a given data series.

  Args:
    data: A pandas Series containing the data for which to calculate EMA.
    period: The number of periods for the EMA calculation.

  Returns:
    A pandas Series containing the calculated EMA.
  """

  alpha = 2 / (period + 1)
  ema = data.ewm(alpha=alpha, adjust=False).mean()
  return ema
