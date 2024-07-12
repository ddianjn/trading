import numpy as np
import pandas as pd

def sma(data, period: int):
  return data['Close'].rolling(window=period).mean()
