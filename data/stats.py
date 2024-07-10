import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

def is_stationary(data: pd.DataFrame, significance: int = 0.05):
  # Visual Inspection
  plt.plot(data)
  plt.show()
  
  # Dickey-Fuller Test: NON-stationarity as the null hypothesis.
  adf_result = adfuller(data)
  print(f"ADF Statistic: {adf_result[0]}")
  # Compare p-value to a significance level (e.g., 0.05)
  print(f"p-value: {adf_result[1]}")
  
  # KPSS Test: stationarity as the null hypothesis.
  kpss_result = kpss(data)
  print(f"KPSS Statistic: {kpss_result[0]}")
  # Compare p-value to a significance level (e.g., 0.05)
  print(f"p-value: {kpss_result[1]}")

  return adf_result[1] < significance and kpss_result[1] > significance
