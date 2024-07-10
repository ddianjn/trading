import numpy as np
import matplotlib.pyplot as plt
from trading.ml import ml_data
from trading.ml.trainer import ModelTrainer
from trading import plotting
from typing import List, Dict

def train_and_eval_sequence(model_trainer: ModelTrainer,
                            stock_ticker: str,
                            start: str = "2018-01-01",
                            end: str|None = None,
                            interval: str = "1d",
                            features: List[str] = ["Close", "High", "Low", "Volume"],
                            output_features: List[str] = ['Close'],
                            feature_generators = {},
                            validate_size: int = 30,
                            test_size: int = 30,
                            look_back: int = 5,
                            look_forward: int = 5,
                            epochs: int = 200,
                            early_stop_patience: int = 30,
                            print_train_steps: bool = True):
  if print_train_steps:
    print(model_trainer.model)
  data, scalers = ml_data.prepare_sequence_data(
    stock_ticker,
    start = start,
    end = end,
    features = features,
    output_features = output_features,
    feature_generators = feature_generators,
    look_back = look_back,
    look_forward = look_forward,
    validate_size = validate_size,
    test_size = test_size)
  model_trainer.scalers = scalers
  model_trainer.features = features
  model_trainer.output_features = output_features
  model_trainer.feature_generators = feature_generators
  model_trainer.look_back = look_back
  model_trainer.look_forward = look_forward

  train_loss, validate_loss = model_trainer.fit_eval(
      data.train_x,
      data.train_y,
      data.validate_x,
      data.validate_y,
      epochs=epochs,
      early_stop_patience=early_stop_patience,
      print_train_steps=print_train_steps)
  return data, train_loss, validate_loss

def eval_sequence(model_trainier: ModelTrainer,
                  stock_ticker: str,
                  start: str = "2018-01-01",
                  end: str|None = None,
                  interval: str = "1d",
                  test_size: int = 30,
                  print_result: bool = True,
                  plot_result: bool = True):
  data, _ = ml_data.prepare_sequence_data(stock_ticker,
                              start = start,
                              end = end,
                              interval = interval,
                              scalers = model_trainier.scalers,
                              features = model_trainier.features,
                              output_features = model_trainier.output_features,
                              feature_generators = model_trainier.feature_generators,
                              look_back = model_trainier.look_back,
                              look_forward = model_trainier.look_forward,
                              validate_size = 1,
                              test_size = test_size)

  print(model_trainier.model)
  # Evaluate on test set (calculate MSE)
  test_loss, predictions = model_trainier.eval(data.test_x, data.test_y)

  if print_result:
    print(f'Test Loss: {test_loss:.4f}')
  # print(f"test_indices: {data.test_indices}\n\n")
  if plot_result:
    plotting.plot_sequence_prediction_comparison(data.test_data[model_trainier.output_features],
                                                 predictions,
                                                 data.test_indices[model_trainier.look_back:])
  return data, test_loss, predictions

def train_and_eval(model_trainer: ModelTrainer,
                   stock_ticker: str,
                   start: str = "2018-01-01",
                   end: str|None = None,
                   interval: str = "1d",
                   features: List[str] = ["Close", "High", "Low", "Volume"],
                   output_features: List[str] = ['Close'],
                   feature_generators = {},
                   validate_size: int = 30,
                   test_size: int = 30,
                   epochs: int = 200,
                   early_stop_patience: int = 30,
                   print_train_steps: bool = True):
  if print_train_steps:
    print(model_trainer.model)
  data, scalers = ml_data.prepare_data(
    stock_ticker,
    start = start,
    end = end,
    features = features,
    output_features = output_features,
    feature_generators = feature_generators,
    validate_size = validate_size,
    test_size = test_size)
  model_trainer.scalers = scalers
  model_trainer.features = features
  model_trainer.output_features = output_features
  model_trainer.feature_generators = feature_generators

  train_loss, validate_loss = model_trainer.fit_eval(
      data.train_x,
      data.train_y,
      data.validate_x,
      data.validate_y,
      epochs=epochs,
      early_stop_patience=early_stop_patience,
      print_train_steps=print_train_steps)
  return data, train_loss, validate_loss

def eval(model_trainier: ModelTrainer,
         stock_ticker: str,
         start: str = "2018-01-01",
         end: str|None = None,
         interval: str = "1d",
         test_size: int = 30,
         print_result: bool = True,
         plot_result: bool = True):
  data, _ = ml_data.prepare_data(stock_ticker,
                              start = start,
                              end = end,
                              interval = interval,
                              scalers = model_trainier.scalers,
                              features = model_trainier.features,
                              output_features = model_trainier.output_features,
                              feature_generators = model_trainier.feature_generators,
                              validate_size = 1,
                              test_size = test_size)

  print(model_trainier.model)
  # Evaluate on test set (calculate MSE)
  test_loss, predictions = model_trainier.eval(data.test_x, data.test_y)

  if print_result:
    print(f'Test Loss: {test_loss:.4f}')
  # print(f"test_indices: {data.test_indices}\n\n")
  if plot_result:
    plotting.plot_sequence_prediction_comparison(data.test_data[model_trainier.output_features],
                                                 predictions,
                                                 data.test_indices)
  return data, test_loss, predictions
