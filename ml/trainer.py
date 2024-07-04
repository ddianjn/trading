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
               print_train_steps = True) -> (float64, float64):
    """Train and evaluate the model.

    Returns:
      Train loss and validation loss.
    """

  def eval(self, x, y, shuffle: bool = False) -> (float64, np.ndarray):
    """Evaluate the model.

    Returns:
      Loss and the prediction result.
    """
    
