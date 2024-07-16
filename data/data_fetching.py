import yfinance as yf
import numpy as np
from datetime import date

def download_data(stock_ticker: str,
                  start: str = "2018-01-01",
                  end: str|None = None,
                  interval: str = "1d"):
  # if end is None:
  #   end = date.today().strftime("%Y-%m-%d")
  # Load the data from Yahoo Finance
  # historical_data = yf.download(stock_ticker, start=start, end=end, interval=interval)
  stock = yf.Ticker(stock_ticker)
  historical_data = stock.history(start=start, end=end, interval=interval)
  print()

  # historical_data.reset_index(inplace=True)
  # Parse dates and convert them to matplotlib's date format
  # historical_data['Date'] = pd.to_datetime(historical_data['Date'])
  # historical_data['Date'] = historical_data['Date'].apply(mdates.date2num)

  # Other yf functions
  # historical_data = stock.history(period=period)
  # stock.info
  # stock.cashflow
  return historical_data
