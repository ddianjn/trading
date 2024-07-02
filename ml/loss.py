import torch

def mse_percent(y_pred, y_true):
  return torch.mean(((y_pred - y_true) / y_true) ** 2)
