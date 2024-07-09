import pandas as pd

def generate_return(data: pd.DataFrame) -> None:
  return_col = data["Close"].pct_change() * 100
  return_col.fillna(0, inplace=True)
  data["% Return"] = return_col
  return data
