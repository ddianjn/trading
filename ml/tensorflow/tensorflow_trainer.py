import numpy as np
from trading.ml.trainer import ModelTrainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping

class TensorflowTrainer(ModelTrainer):
  def __init__(self,
               model: Sequential,
               batch_size: int = 64):
    super().__init__(batch_size)
    self.model = model

  def fit_eval(self,
               train_x,
               train_y,
               val_x,
               val_y,
               epochs = 100,
               early_stop_patience = 30,
               verbose = False,
               print_train_steps = True) -> (float, float):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=early_stop_patience,
                                   mode='min',
                                   restore_best_weights=True)
    history = self.model.fit(train_x,
                             train_y,
                             epochs=epochs,
                             batch_size=self.batch_size,
                             verbose=print_train_steps,
                             validation_data=(val_x, val_y),
                             callbacks=[early_stopping])
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    return train_loss, val_loss

  def eval(self, x, y, shuffle: bool = False) -> (float, np.ndarray):
    loss = self.model.evaluate(x, y)
    predictions = self.model.predict(x)
    return loss, predictions
    
