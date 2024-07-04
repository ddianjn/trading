import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from trading.ml.trainer import ModelTrainer

class PytorchTrainer(ModelTrainer):
  def __init__(self,
               model: nn.Module,
               optimizer: Optimizer,
               loss_fn: nn.Module = nn.MSELoss,
               scheduler: LRScheduler = None,
               batch_size: int = 64):
    super().__init__(batch_size)
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.scheduler = scheduler

  def _fit_one_epoch(self,
          x,
          y,
          epoch: int,
          shuffle: bool = True,
          verbose = False) -> float64:
    train_loader = self._create_data_loader(x,
                                            y,
                                            batch_size = self.batch_size,
                                            shuffle = shuffle)

    total_train_loss = 0.0
    self.model.train()
    for batch_x, batch_y in train_loader:
      # print(batch_x.shape)
      # print(batch_y.shape)
      batch_prediction = self.model(batch_x)
      if verbose:
        print(f"predictions\n{np.array(batch_prediction.tolist())[:2]}\n")
        print(f"actual\n{np.array(batch_y)[:2]}\n")
      loss = self.loss_fn(batch_prediction, batch_y)
      total_train_loss += loss.item()

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    train_loss = total_train_loss / len(train_loader)
    # state_dict = model.state_dict()
    # print(f"Train LSTM weight: {state_dict['lstm.weight_ih_l0']}")

    # Update learning rate
    if self.scheduler is not None:
      if print_train_steps and (epoch+1)%5==0:
        print(f"last_lr: {self.scheduler.get_last_lr()}")
      self.scheduler.step()
    return train_loss

  def eval(self, x, y, shuffle: bool = False) -> (float64, np.ndarray):
    # state_dict = model.state_dict()
    # print(f"Test LSTM weight: {state_dict['lstm.weight_ih_l0']}")

    test_loader = self._create_data_loader(x,
                                           y,
                                           batch_size = self.batch_size,
                                           shuffle = shuffle)

    total_test_loss = 0.0
    predictions = []
    self.model.eval()
    with torch.no_grad():
      for batch_x, batch_y in test_loader:
        batch_prediction = self.model(batch_x)

        loss = self.loss_fn(batch_prediction, batch_y)
        total_test_loss += loss.item()

        batch_prediction = np.array(batch_prediction.tolist())
        predictions.extend(batch_prediction)  # Append predicted prices
      test_loss = total_test_loss / len(test_loader)
    return test_loss, np.array(predictions)
  
  def _create_data_loader(self,
                          x,
                          y,
                          batch_size: int = 64,
                          shuffle: bool = True):
    x_tensor = torch.tensor(x, dtype = torch.float32)
    y_tensor = torch.tensor(y, dtype = torch.float32)

    # Create PyTorch datasets and dataloaders
    dataset = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset,
                             batch_size = batch_size,
                             shuffle = shuffle)
    return data_loader
    
