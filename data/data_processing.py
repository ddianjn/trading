import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime
from typing import List, Dict

def split_data(data: pd.DataFrame,
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


def normalize_data(train_data: pd.DataFrame,
                   validate_data: pd.DataFrame,
                   test_data: pd.DataFrame,
                   scalers: Dict|None = None,
                   scaler_creater = StandardScaler):
  if scalers is None:
    scalers = {}

  # train_data.reset_index(inplace=True)
  # test_data.reset_index(inplace=True)
  scaled_train_data = [[]] * len(train_data)
  scaled_validate_data = [[]] * len(validate_data)
  scaled_test_data = [[]] * len(test_data)
  # print(f"scaled_train_data\n{scaled_train_data[:3]}")
  # scaled_train_data = np.array(train_data["Date"])
  for feature in train_data.columns:
    if feature == "Date":
      continue
    # print(f"====\nfeature {feature}")
    train_feature_data = train_data[[feature]].values
    validate_feature_data = validate_data[[feature]].values
    test_feature_data = test_data[[feature]].values

    # scalers[feature] = MinMaxScaler(feature_range=(0,1))
    if not feature in scalers:
      scalers[feature] = scaler_creater()
      scaled_train_feature_data = scalers[feature].fit_transform(train_feature_data)
    else:
      scaled_train_feature_data = scalers[feature].transform(train_feature_data)
    if len(validate_feature_data) > 0:
      scaled_validate_feature_data = scalers[feature].transform(validate_feature_data)
    else:
      scaled_validate_feature_data = validate_feature_data
    if len(test_feature_data) > 0:
      scaled_test_feature_data = scalers[feature].transform(test_feature_data)
    else:
      scaled_test_feature_data = test_feature_data
    # print(f"feature_data\n{feature_data[-3:]}")
    # print(f"scaled_feature_data\n{scaled_feature_data[-3:]}")
    scaled_train_data = np.concatenate((scaled_train_data, scaled_train_feature_data), axis=1)
    if len(scaled_validate_data) > 0:
      scaled_validate_data = np.concatenate((scaled_validate_data, scaled_validate_feature_data), axis=1)
    if len(scaled_test_data) > 0:
      scaled_test_data = np.concatenate((scaled_test_data, scaled_test_feature_data), axis=1)
    # print(f"scaled_train_data\n{scaled_train_data[:3]}")
    # print(f"scaled_train_data\n{scaled_train_data[-3:]}")
    # scaled_train_data[feature] = np.array(scaled_feature_data.flatten())
    # print(scaled_train_data)

  scaled_train_data = np.array(scaled_train_data)
  scaled_validate_data = np.array(scaled_validate_data)
  scaled_test_data = np.array(scaled_test_data)

  scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
  scaled_validate_data = pd.DataFrame(scaled_validate_data, columns=train_data.columns)
  scaled_test_data = pd.DataFrame(scaled_test_data, columns=train_data.columns)
  return scaled_train_data, scaled_validate_data, scaled_test_data, scalers

def create_time_series_data(data: pd.DataFrame,
                            scaled_data = None,
                            features: List[str] = ["Close", "High", "Low", "Volume"],
                            output_features: List[str] = ['Close'],
                            look_back: int = 5,
                            look_forward: int = 5,
                            fill_forward: bool = False,
                            verbose: bool = False):
  if scaled_data is None:
    scaled_data = data

  if fill_forward:
    # Append a few more days for the look forward period
    new_rows = {}
    new_scaled_rows = {}
    for column in data.columns:
      new_rows[column] = [data.iloc[-1][column]] * look_forward
      new_scaled_rows[column] = [scaled_data.iloc[-1][column]] * look_forward
    new_rows["Date"] = [data.index[-1] + datetime.timedelta(days=i) for i in range(1, look_forward+1)]
    new_rows = pd.DataFrame(new_rows)
    new_rows.set_index("Date", inplace=True)
    data = pd.concat([data, new_rows])
    if verbose:
      print(f"==data==\n{data[-look_forward-1:]}")
      print(f"==data size: {data.shape}==")
    new_scaled_rows = pd.DataFrame(new_scaled_rows)
    scaled_data = pd.concat([scaled_data, new_scaled_rows])
    data_indices = data.index.tolist()
    # Add one more day for prediction
    data_indices.append(data_indices[-1] + datetime.timedelta(days=1))

  else:
    data_indices = data.index.tolist()

  input_data = scaled_data[features].values
  output_data = data[output_features].values
  if verbose:
    print("=====")
    print(f"==input_data==\n{input_data[:3]}\n")
    print(f"==output_data==\n{output_data[:3]}\n")

  data_x, data_y = [], []
  for i in range(look_back, len(data) - look_forward + 1):
    data_x.append(input_data[i-look_back:i])
    data_y.append(output_data[i:i+look_forward])
  if verbose:
    print(f"==data_x==\n{data_x[:3]}\n")
    print(f"==data_x size: {len(data_x)}==")
    print(f"==data_y==\n{data_y[:3]}\n")
    print(f"==data_y size: {len(data_y)}==")
  return data_x, data_y, data_indices
