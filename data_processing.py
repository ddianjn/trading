import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def split_data(data: np.ndarray,
               validate_size: int = 0,
               test_size: int = 30,
               plot_split_chart:bool = False):
  # Split data into training and testing sets (adjust based on desired split)
  train_data = data.iloc[: -test_size - validate_size]
  validate_data = data.iloc[-test_size - validate_size : -test_size]
  test_data = data.iloc[-test_size :]

  if plot_split_chart:
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data["Close"], label="Train")
    plt.plot(validate_data.index, validate_data["Close"], label="Validate")
    plt.plot(test_data.index, test_data["Close"], label="Test")
    plt.show()

  return train_data, validate_data, test_data
