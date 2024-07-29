import numpy as np
import pandas as pd

def sma(data: pd.DataFrame,
        period: int,
        source: str = "Close"):
  ma = data[source].rolling(window=period).mean()
  return pd.DataFrame({f'MA{period}': ma})

def ema(data: pd.DataFrame,
        period: int,
        source: str = "Close"):
  """Calculates the Exponential Moving Average (EMA) for a given data series.

  Args:
    data: A pandas Series containing the data for which to calculate EMA.
    period: The number of periods for the EMA calculation.
    source: The column name of the data in the pandas Series.

  Returns:
    A pandas Series containing the calculated EMA.
  """

  alpha = 2 / (period + 1)
  ema = data[source].ewm(alpha=alpha, adjust=False).mean()
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

def atr(data: pd.DataFrame, period: int):
  """Calculates the Average True Range (ATR) for a given DataFrame.

  Args:
    data: A pandas DataFrame containing the True Range ('tr') column.
    period: The number of periods for the ATR calculation.

  Returns:
    A pandas Series containing the ATR values.
  """

  high_low = data['High'] - data['Low']
  high_prev_close = abs(data['High'] - data['Close'].shift(1))
  low_prev_close = abs(data['Low'] - data['Close'].shift(1))
  tr = high_low.where(high_low > high_prev_close, high_prev_close).where(lambda x: x > low_prev_close, low_prev_close)

  atr = tr.rolling(window=period).mean()
  return pd.DataFrame({'tr': tr, f'atr{period}': atr})

def ravifxf(data,
            source: str = "Close",
            short_period: int = 4,
            long_period: int = 49):
  """Calculates RAVI FX Fisher.

  RAVI FX Fisher is a special implementation of RAVI using WMA moving averages and ATR and then normalized
  like Fisher Transform. If the histogram falls between the white lines, the market is too choppy to trade.
  What is RAVI?
  The Range Action Verification Index (RAVI) indicator shows the percentage difference between current prices
  and past prices to identify market trends. It is calculated based on moving averages of different lengths.

  Args:
    data: A pandas DataFrame.
    short_period: The number of periods for the short-term WMA.
    long_period: The number of periods for the long-term WMA.

  Returns:
    A pandas Series containing the ATR values.
  """
  fast_wma = wma(data, short_period, source)[f"WMA{short_period}"]
  slow_wma = wma(data, long_period, source)[f"WMA{long_period}"]
  fast_atr = atr(data, short_period)[f"atr{short_period}"]
  slow_atr = atr(data, long_period)[f"atr{long_period}"]
  maval = (fast_wma - slow_wma) * fast_atr / slow_wma / slow_atr * 100
  fish = (np.exp(2 * maval) - 1) / (np.exp(2 * maval) + 1)
  return pd.DataFrame({f'maval_{source}_{short_period}_{long_period}': maval,
                       f'fish_{source}_{short_period}_{long_period}': fish})

def andeanOscillator(data,
                     period: int = 50,
                     signal_period: int = 9):
  """Calculates Andean Oscillator.

  The proposed indicator aims to measure the degree of variations of individual up-trends and down-trends in
  the price, thus allowing to highlight the direction and amplitude of a current trend.

  Args:
    data: A pandas DataFrame.
    period: Determines the significance of the trends degree of variations measured by the indicator.
    signal_period: Moving average period of the signal line.

  Returns:
    A pandas Series containing the ATR values.
  """
  alpha = 2/(period+1)

  close = data['Close']
  open = data['Open']
  up1 = np.maximum(close, open)
  up2 = np.maximum(close * close, open * open)
  dn1 = np.minimum(close, open)
  dn2 = np.minimum(close * close, open * open)
  for i in range(1, len(up1)):
    up1[i] = max(up1[i], up1[i - 1] - (up1[i - 1] - close[i]) * alpha)
    up2[i] = max(up2[i], up2[i - 1] - (up2[i - 1] - close[i] * close[i]) * alpha)
    dn1[i] = min(dn1[i], dn1[i - 1] + (close[i] - dn1[i - 1]) * alpha)
    dn2[i] = min(dn2[i], dn2[i - 1] + (close[i] * close[i] - dn2[i - 1]) * alpha)

  bull = np.sqrt(dn2 - dn1 * dn1)
  bear = np.sqrt(up2 - up1 * up1)

  # Use a fake "Close" column to calculate ema.
  signal = ema(pd.DataFrame({"Close": np.maximum(bull, bear)}), signal_period)
  return pd.DataFrame({"Bull": bull, "Bear": bear, "Signal": signal[f'EMA{signal_period}']})
