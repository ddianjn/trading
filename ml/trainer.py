import numpy as np

class ModelTrainer:
  def __init__(self, batch_size: int = 64):
    self.batch_size = batch_size

  def fit_eval(self,
               train_x,
               train_y,
               test_x,
               test_y,
               epochs = 100,
               early_stop_patience = 30,
               verbose = False,
               print_train_steps = True):
    min_train_loss = float('inf')
    min_train_loss_epoch = 0
    min_test_loss = float('inf')
    min_test_loss_epoch = 0

    for epoch in range(epochs):
      # Train on training set
      train_loss = self._fit_one_epoch(train_x, train_y, verbose = verbose)
      # Evaluate on test set (calculate MSE)
      test_loss, predictions = self.eval(test_x, test_y)

      # print
      if print_train_steps and (epoch+1)%5==0:
          # print(predictions[:2])
          print(f'Epoch [{epoch+1}/{epochs}] - Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

      # Early Stop
      if train_loss < min_train_loss:
        min_train_loss = train_loss
        min_train_loss_epoch = epoch
      if test_loss < min_test_loss:
        min_test_loss = test_loss
        min_test_loss_epoch = epoch
      if epoch - min_train_loss_epoch > early_stop_patience or epoch - min_test_loss_epoch > early_stop_patience * 2:
        print(f'Epoch [{epoch+1}/{epochs}] - Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        break

    return train_loss, test_loss

  def _fit_one_epoch(self,
          x,
          y,
          epoch: int,
          shuffle: bool = True,
          verbose = False) -> float64:

  def eval(self, x, y, shuffle: bool = False) -> (float64, np.ndarray):
    
