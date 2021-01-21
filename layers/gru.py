import torch
import torch.nn as nn

class TransitionGRUCell(nn.Module):
  def __init__(
    self, hidden_size, bias=True,
  ):
    """
    hidden_size (int): hidden state dimension
    bias (bool): default: True, if false not use bias on every single linear layer
    """
    super().__init__()

    self.reset_gate = nn.Sequential(
      nn.Linear(hidden_size, hidden_size, bias=bias),
      nn.ReLU(),
    )
    self.update_gate = nn.Sequential(
      nn.Linear(hidden_size, hidden_size, bias=bias),
      nn.ReLU(),
    )
    self.h_hat_linear = nn.Linear(hidden_size, hidden_size, bias=bias)

  def forward(self, h_minus_one):
    h_hat = torch.tanh(self.reset_gate(h_minus_one) * self.h_hat_linear(h_minus_one))
    h_t = (1 - self.update_gate(h_minus_one)) * h_minus_one + self.update_gate(h_minus_one) * h_hat

    return h_t
    

class TransitionGRU(nn.Module):
  def __init__(
    self, hidden_size, num_layer=1, bias=True,
  ):
    """
    hidden_size (int): hidden state dimension
    num_layer (int): number of TransitionGRUCell
    bias (bool): default: True, if false not use bias on every single linear layer
    """
    super().__init__()
    self.gru_cells = [TransitionGRUCell(hidden_size, bias) for _ in range(num_layer)]

  def forward(self, hidden_zero):
    hidden = hidden_zero

    for cell in self.gru_cells:
      hidden = cell(hidden)
    
    return hidden

class LinearTransformationEnhancedGRUCell(nn.Module):
  def __init__(self, input_size, hidden_size, bias=True):
    """
    input_size (int): input sequence dimension
    hidden_size (int): hidden state dimension
    bias (bool): default: True, if false not use bias on every single linear layer
    """
    super().__init__()

    self.reset_gate = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU()
    )
    self.update_gate = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
    )
    self.h_l_linear = nn.Linear(hidden_size, hidden_size)
    self.x_l_linear = nn.Linear(input_size, hidden_size)
    self.x_h_linear = nn.Linear(input_size, hidden_size)
    self.h_h_linear = nn.Linear(hidden_size, hidden_size)
    self.H = nn.Linear(input_size, hidden_size)

  def forward(self, x, h_minus_one):
    l_t = torch.relu(self.x_l_linear(x) + self.h_h_linear(h_minus_one))
    h_hat = torch.tanh(self.x_h_linear(x) + self.reset_gate(x) * self.h_h_linear(h_minus_one)) + l_t * self.H(x)
    h_t = (1 - self.update_gate(x)) * h_minus_one + self.update_gate(x) * h_hat

    return h_t

class LinearTransformationEnhancedGRU(nn.Module):
  def __init__(self, input_size, hidden_size, num_layer=1, bias=True):
    """
    input_size (int): input sequence dimension
    hidden_size (int): hidden state dimension
    num_layer (int): number of LinearTransformationEnhancedGRUCell
    bias (bool): default: True, if false not use bias on every single linear layer
    """
    super().__init__()
    self.gru_cells = [LinearTransformationEnhancedGRUCell(input_size, hidden_size, bias) for _ in range(num_layer)]

  def forward(self, x, hidden_zero):
    hidden = hidden_zero
    for cell in self.gru_cells:
      hidden = cell(x, hidden)

    return hidden

if __name__ == '__main__':
  t_gru = TransitionGRU(5, num_layer=5)
  l_gru = LinearTransformationEnhancedGRU(10, 5, num_layer=5)
  
  x = torch.randn(10)
  h = torch.randn(5)

  t_gru_hidden = t_gru(h)
  
  l_gru_hidden = l_gru(x, h)

  print(t_gru_hidden)
  print(l_gru_hidden)