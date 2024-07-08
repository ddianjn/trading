import numpy as np
import pandas as pd
import datetime
from trading.data import data_processing, data_fetching
from trading import plotting
from typing import List

class TrainingData:
  def __init__(self,
               train_data: pd.DataFrame,
               validate_data: pd.DataFrame,
               test_data: pd.DataFrame,
               train_x: np.ndarray,
               train_y: np.ndarray,
               validate_x: np.ndarray,
               validate_y: np.ndarray,
               test_x: np.ndarray,
               test_y: np.ndarray,
               train_indices: List,
               validate_indices: List,
               test_indices: List):
    self.train_data = train_data
    self.validate_data = validate_data
    self.test_data = test_data
    self.train_x = train_x
    self.train_y = train_y
    self.validate_x = validate_x
    self.validate_y = validate_y
    self.test_x = test_x
    self.test_y = test_y
    self.train_indices = train_indices
    self.validate_indices = validate_indices
    self.test_indices = test_indices


def prepare_sequence_data(stock_ticker:str,
                 start:str = "2018-01-01",
                 end:str = None,
                 interval:str = "1d",
                 features: List[str] = ["Close", "High", "Low", "Volume"],
                 output_features: List[str] = ['Close'],
                 feature_generators = {},
                 validate_size: int = 30,
                 test_size: int = 30,
                 look_back: int = 5,
                 look_forward: int = 5,
                 plot_candle_chart: bool = False,
                 plot_split_chart: bool = False,
                 print_raw_data: bool = False,
                 print_scaled_data: bool = False,
                 print_time_series_data: bool = False):
  stock_data = data_fetching.download_data(stock_ticker, start, end, interval = interval)
  if plot_candle_chart:
    plotting.plot_candlestick(stock_data)

  for new_key, generator in feature_generators.items():
    generator(new_key, stock_data)

  train_data, validate_data, test_data = data_processing.split_data(stock_data,
                                                    validate_size = validate_size,
                                                    test_size = test_size,
                                                    plot_split_chart = plot_split_chart)

  if print_raw_data:
    print(f"\nFeatures\n{train_data.columns}")
    print(f"train_data\n{train_data.tail()}")
    print(f"validate_data\n{validate_data.tail()}")
    print(f"test_data\n{test_data.tail()}")

  # Normalize
  scaled_train_data, scaled_validate_data, scaled_test_data = data_processing.normalize_data(train_data, validate_data, test_data)
  if print_scaled_data:
    print(f"scaled_train_data\n{scaled_train_data.tail()}")
    print(f"scaled_test_data\n{scaled_test_data.tail()}")

  # Create time series data
  train_x, train_y, train_indices = data_processing.create_time_series_data(
      train_data,
      scaled_train_data,
      features = features,
      output_features = output_features,
      look_back = look_back,
      look_forward = look_forward)
  validate_x, validate_y, validate_indices = data_processing.create_time_series_data(
      validate_data,
      scaled_validate_data,
      features = features,
      output_features = output_features,
      fill_forward = True,
      look_back = look_back,
      look_forward = look_forward)
  test_x, test_y, test_indices = data_processing.create_time_series_data(
      test_data,
      scaled_test_data,
      features = features,
      output_features = output_features,
      look_back = look_back,
      look_forward = look_forward,
      fill_forward = True,
      verbose = print_time_series_data)

  train_x = np.array(train_x)
  train_y = np.array(train_y)
  validate_x = np.array(validate_x)
  validate_y = np.array(validate_y)
  test_x = np.array(test_x)
  test_y = np.array(test_y)

  if print_time_series_data:
    print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
    print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")

  return TrainingData(train_data=train_data,
                      validate_data=validate_data,
                      test_data=test_data,
                      train_x=train_x,
                      train_y=train_y,
                      validate_x=validate_x,
                      validate_y=validate_y,
                      test_x=test_x,
                      test_y=test_y,
                      train_indices=train_indices,
                      validate_indices=validate_indices,
                      test_indices=test_indices)

def prepare_data(stock_ticker:str,
                 start:str = "2018-01-01",
                 end:str = None,
                 interval:str = "1d",
                 features: List[str] = ["Close", "High", "Low", "Volume"],
                 output_features: List[str] = ['Close'],
                 feature_generators = {},
                 validate_size: int = 30,
                 test_size: int = 30,
                 plot_candle_chart: bool = False,
                 plot_split_chart: bool = False,
                 print_raw_data: bool = False,
                 print_scaled_data: bool = False,
                 verbose: bool = False):
  stock_data = data_fetching.download_data(stock_ticker, start, end, interval = interval)
  if plot_candle_chart:
    plotting.plot_candlestick(stock_data)

  for new_key, generator in feature_generators.items():
    generator(new_key, stock_data)

  train_data, validate_data, test_data = data_processing.split_data(stock_data,
                                                    validate_size = validate_size,
                                                    test_size = test_size,
                                                    plot_split_chart = plot_split_chart)

  if print_raw_data:
    print(f"\nFeatures\n{train_data.columns}")
    print(f"train_data\n{train_data.tail()}")
    print(f"validate_data\n{validate_data.tail()}")
    print(f"test_data\n{test_data.tail()}")

  # Normalize
  scaled_train_data, scaled_validate_data, scaled_test_data = data_processing.normalize_data(train_data, validate_data, test_data)
  if print_scaled_data:
    print(f"scaled_train_data\n{scaled_train_data.tail()}")
    print(f"scaled_test_data\n{scaled_test_data.tail()}")

  # Create time series data
  train_x, train_y, train_indices = _create_feature_label_data(
      train_data,
      scaled_train_data,
      features = features,
      output_features = output_features)
  validate_x, validate_y, validate_indices = _create_feature_label_data(
      validate_data,
      scaled_validate_data,
      features = features,
      output_features = output_features,
      fill_forward = True)
  test_x, test_y, test_indices = _create_feature_label_data(
      test_data,
      scaled_test_data,
      features = features,
      output_features = output_features,
      fill_forward = True,
      verbose = verbose)

  train_x = np.array(train_x)
  train_y = np.array(train_y)
  validate_x = np.array(validate_x)
  validate_y = np.array(validate_y)
  test_x = np.array(test_x)
  test_y = np.array(test_y)

  if verbose:
    print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
    print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")

  return TrainingData(train_data=train_data,
                      validate_data=validate_data,
                      test_data=test_data,
                      train_x=train_x,
                      train_y=train_y,
                      validate_x=validate_x,
                      validate_y=validate_y,
                      test_x=test_x,
                      test_y=test_y,
                      train_indices=train_indices,
                      validate_indices=validate_indices,
                      test_indices=test_indices)

def _create_feature_label_data(data: pd.DataFrame,
                               scaled_data = None,
                               features: List[str] = ["Close", "High", "Low", "Volume"],
                               output_features: List[str] = ['Close'],
                               fill_forward: bool = False,
                               verbose: bool = False):
  if scaled_data is None:
    scaled_data = data

  if fill_forward:
    # Append a few more days for the look forward period
    new_row = data[-1:].copy()
    new_row.reset_index(inplace=True)
    new_row["Date"] = pd.to_datetime(data.index[-1] + datetime.timedelta(days=1))
    new_row.set_index("Date", inplace=True)
    data = pd.concat([data, new_row])
    data_indices = data.index.tolist()
    if verbose:
      print(f"==data==\n{data[-3:]}")
      print(f"==data size: {data.shape}==")
    new_scaled_rows = scaled_data[-1:].copy()
    scaled_data = pd.concat([scaled_data, new_scaled_rows])
    # Add one more day for prediction
    data_indices.append(data_indices[-1] + datetime.timedelta(days=1))
  else:
    data_indices = data.index.tolist()

  data_x = scaled_data[features].values
  data_y = data[output_features].values
  if verbose:
    print(f"==data_x==\n{data_x[:3]}\n")
    print(f"==data_x size: {len(data_x)}==")
    print(f"==data_y==\n{data_y[:3]}\n")
    print(f"==data_y size: {len(data_y)}==")
  return data_x, data_y, data_indices
