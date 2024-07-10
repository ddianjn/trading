import numpy as np
from trading.ml.trainer import ModelTrainer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

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
    
class XGBClassifierTrainer(ModelTrainer):
  def __init__(self,
               objective: str = 'binary:logitraw', # 'binary:logistic' for accuracy
               scale_pos_weight: int = 1,
               n_estimators: int = 100):
    super().__init__(32)
    self.model = XGBClassifier(n_estimators=n_estimators,
                               objective=objective,
                               scale_pos_weight=scale_pos_weight)

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
    val_loss = precision_score(val_y, predictions)
    return np.NaN, val_loss

  def eval(self, x, y, shuffle: bool = False) -> (float, np.ndarray):
    predictions = self.model.predict(x)
    loss = precision_score(y, predictions)
    return loss, predictions
