import pandas as pd

def generate_return(new_key: str, data: pd.DataFrame) -> None:
  return_col = data["Close"].pct_change() * 100
  return_col.fillna(0, inplace=True)
  data[new_key] = return_col
  return new_key, data
