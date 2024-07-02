from torch import nn

class LSTMModel(nn.Module):
  def __init__(self,
               input_dim: int,
               hidden_dim: int,
               output_dim: int,
               num_layers: int,
               output_seq_dim: int):
    super(LSTMModel, self).__init__()
    self.lstm = nn.LSTM(input_dim,
                        hidden_dim,
                        num_layers=num_layers,
                        batch_first=True)
    self.linear = nn.Linear(hidden_dim, output_dim)
    self.output_seq_dim = output_seq_dim

  def forward(self, x):
    # Pass through LSTM layer
    y, (h_n, c_n) = self.lstm(x)
    # Use last hidden state for prediction
    # out = self.linear(h_n[-1])
    out = self.linear(y[:, -self.output_seq_dim:, :])
    return out
