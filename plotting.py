import numpy as np
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def plot_candlestick(historical_data: pd.DataFrame, period: int|None = None):
  if period is None:
    data = historical_data.copy()
  else:
    data = historical_data[-period:].copy()

  # Parse dates and convert them to matplotlib's date format
  data.reset_index(inplace=True)
  data['Date'] = pd.to_datetime(data['Date']).apply(mdates.date2num)

  # Prepare the data for the candlestick chart
  quotes = data[['Date', 'Open', 'High', 'Low', 'Close']].values

  # Create the candlestick chart
  fig, ax = plt.subplots(figsize=(12, 8))

  candlestick_ohlc(ax, quotes, width=0.6, colorup='green', colordown='red', alpha=0.8)

  # Format the x-axis for dates
  ax.xaxis_date()
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(50))
  plt.xticks(rotation=45)

  # Add grid, title and labels
  ax.grid(True)
  plt.title(f'Historical Price Data (Last {period} Days)')
  plt.xlabel('Date')
  plt.ylabel('Price (USD)')

  # Show the plot
  plt.tight_layout()
  plt.show()

def plot_sequence_prediction_comparison(actual: pd.DataFrame,
                                        predicted: np.ndarray,
                                        prediction_indices,
                                        secondary_predicted: np.ndarray|None = None):
  # Plot actual vs predicted closing prices
  fig = plt.figure(figsize=(12, 10))
  if predicted.ndim == 1:
    predicted = predicted.reshape(-1, 1)
  look_forward = predicted.shape[1]
  for i in range(look_forward):
    ax = fig.add_subplot(look_forward, 1, i+1)
    ax.plot(actual.index, actual, label="Actual")
    ax.plot(prediction_indices[i:-look_forward+i], predicted[:, i], label=f"Predicted day {i+1}", linestyle='dashed')
    if secondary_predicted is not None:
      ax.plot(prediction_indices[i:-look_forward+i], secondary_predicted[:, i], linestyle='dotted', label=f"Secondary Predicted day {i+1} (2)")
    # ax.plot(train_data.index, train_data["Close"], label="Train")
    ax.legend()
    ax.grid(True)
  plt.show()
 
def plot_lines(title: str, values: pd.DataFrame):
  fig = plt.figure(figsize=(12, 8))
  for column in values.columns:
    plt.plot(values.index, values[column], label=column)
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()
