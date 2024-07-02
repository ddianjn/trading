import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import matplotlib.pyplot as plt
from trading.ml import ml_data
from trading.ml.train_eval import trainer
from typing import List

def train_and_eval(model_trainer: trainer.ModelTrainer,
                   stock_ticker: str,
                   scheduler: LRScheduler|None = None,
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
  data = ml_data.prepare_data(
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

  train_loss, validate_loss = model_trainer.fit_eval(
      data.train_x,
      data.train_y,
      data.validate_x,
      data.validate_y,
      epochs=epochs,
      early_stop_patience=early_stop_patience,
      print_train_steps=print_train_steps)
  return data, train_loss, validate_loss
