import pandas as pd

def generate_return(data: pd.DataFrame) -> None:
  return_col = data["Close"].pct_change() * 100
  return_col.fillna(0, inplace=True)
  data["% Return"] = return_col
  return data

def lag_features(lag = 1):
  def lag_feature_gnerator(data: pd.DataFrame):
    columns = data.columns
    new_columns = {}
    for i in range(1, lag + 1):
      for column in columns:
        new_columns[f'lag_{i}_{column}'] = data[column].shift(i)
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
    return data.dropna()
  return lag_feature_gnerator
