import pandas as pd
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.ticker as ticker

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
