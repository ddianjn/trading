import numpy as np
from sklearn.metrics import mean_squared_error
from trading.ml.trainer import ModelTrainer
from xgboost import XGBRegressor

# Obvious caveat: Can't predict well if the price raeches all time high.
class XGBoostTrainer(ModelTrainer):
  def __init__(self, n_estimators: int = 1000):
    super().__init__(32)
    self.model = XGBRegressor(objective='reg:squarederror',
                              n_estimators=n_estimators)

  def fit_eval(self,
               train_x,
               train_y,
               val_x,
               val_y,
               epochs = 100,
               early_stop_patience = 30,
               verbose = False,
               print_train_steps = True) -> (float, float):
    self.model.fit(train_x, train_y)
    predictions = self.model.predict(val_x)
    val_loss = mean_squared_error(val_y, predictions)
    return np.NaN, val_loss

  def eval(self, x, y, shuffle: bool = False) -> (float, np.ndarray):
    predictions = self.model.predict(x)
    loss = mean_squared_error(y, predictions)
    return loss, predictions
    
