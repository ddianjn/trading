import numpy as np
from sklearn.metrics import mean_squared_error
from trading.ml.trainer import ModelTrainer
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple

class ARIMATrainer(ModelTrainer):
  def __init__(self, order: Tuple[int, int, int]):
    super().__init__(32)
    self.order = order
    self.train_x = None
    self.train_y = None
    self.val_x = None
    self.val_y = None

  def fit_eval(self,
               train_x,
               train_y,
               val_x,
               val_y,
               epochs = 100,
               early_stop_patience = 30,
               verbose = False,
               print_train_steps = True) -> (float, float):
    self.train_x = train_x
    self.train_y = train_y
    self.val_x = val_x
    self.val_y = val_y
    return 0, 0

  def eval(self, x, y, shuffle: bool = False) -> (float, np.ndarray):
    predictions = []
    loss = 0
    train_y = np.concatenate((self.train_y, self.val_y), axis=0)
    for yy in y:
      model = ARIMA(train_y, order=self.order)
      model_fit = model.fit()
      yy_hat = model_fit.forecast(steps=1)
      predictions.append(yy_hat[0])
      train_y = np.append(train_y, yy)
    loss = mean_squared_error(predictions, y)
    return loss, np.array(predictions)
    
