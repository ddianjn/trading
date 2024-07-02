import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import TensorDataset, DataLoader

class ModelTrainer():
  def __init__(self,
               model: nn.Module,
               optimizer: Optimizer,
               loss_fn: nn.Module = nn.MSELoss,
               scheduler: LRScheduler = None,
               batch_size: int = 64):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.scheduler = scheduler
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

    if self.scheduler is not None:
        self.scheduler.step()
        # print(f"last_lr: {self.scheduler.get_last_lr()}")

    return train_loss, test_loss

  def _fit_one_epoch(self,
          x,
          y,
          shuffle: bool = True,
          verbose = False):
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
    return train_loss

  def eval(self, x, y, shuffle: bool = False):
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
        # batch_prediction = scaler.inverse_transform(batch_prediction)
        predictions.extend(batch_prediction)  # Append predicted prices
      test_loss = total_test_loss / len(test_loader)
    return test_loss, predictions
  
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
    
