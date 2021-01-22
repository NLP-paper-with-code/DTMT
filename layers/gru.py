import torch
import torch.nn as nn

class TransitionGRUCell(nn.Module):
  """
  hidden_size (int): hidden state dimension
  bias (bool): default: True, if false not use bias on every single linear layer
  """
  def __init__(
    self, hidden_size, bias=True,
  ):
    super().__init__()

    self.reset_gate = nn.Sequential(
      nn.Linear(hidden_size, hidden_size, bias=bias),
      nn.Sigmoid(),
    )
    self.update_gate = nn.Sequential(
      nn.Linear(hidden_size, hidden_size, bias=bias),
      nn.Sigmoid(),
    )
    self.h_hat_linear = nn.Linear(hidden_size, hidden_size, bias=bias)

  def forward(self, h_minus_one):
    h_hat = torch.tanh(self.reset_gate(h_minus_one) * self.h_hat_linear(h_minus_one))
    h_t = (1 - self.update_gate(h_minus_one)) * h_minus_one + self.update_gate(h_minus_one) * h_hat

    return h_t

class LinearTransformationEnhancedGRUCell(nn.Module):
  """
  input_size (int): input sequence dimension
  hidden_size (int): hidden state dimension
  bias (bool): default: True, if false not use bias on every single linear layer
  """
  def __init__(self, input_size, hidden_size, bias=True):
    super().__init__()

    self.reset_gate = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.Sigmoid()
    )
    self.update_gate = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.Sigmoid(),
    )
    self.h_l_linear = nn.Linear(hidden_size, hidden_size)
    self.x_l_linear = nn.Linear(input_size, hidden_size)
    self.x_h_linear = nn.Linear(input_size, hidden_size)
    self.h_h_linear = nn.Linear(hidden_size, hidden_size)
    self.H = nn.Linear(input_size, hidden_size)

  def forward(self, x, h_minus_one):
    l_t = torch.sigmoid(self.x_l_linear(x) + self.h_h_linear(h_minus_one))
    h_hat = torch.tanh(self.x_h_linear(x) + self.reset_gate(x) * self.h_h_linear(h_minus_one)) + l_t * self.H(x)
    h_t = (1 - self.update_gate(x)) * h_minus_one + self.update_gate(x) * h_hat

    return h_t

class DeepTransitionRNN(nn.Module):
  """
  input_size (int): input sequence dimension
  hidden_size (int): hidden state dimension
  deep (int): number of TransitionGRUCell
  bidirection (bool): is bidirectional DeepTransitionRNN
  bias (bool): default: True, if false not use bias on every single linear layer
  """
  def __init__(self, input_size, hidden_size, bidirection=False, deep=1, bias=True):
    super().__init__()
    
    self.l_gru = LinearTransformationEnhancedGRUCell(input_size, hidden_size, bias)
    self.t_grus = nn.Sequential(*[TransitionGRUCell(hidden_size, bias) for _ in range(deep)])
    
    self.bidirection = bidirection
    if self.bidirection:
      self.reversed_l_gru = LinearTransformationEnhancedGRUCell(input_size, hidden_size, bias)
      self.reversed_t_grus = nn.Sequential(*[TransitionGRUCell(hidden_size, bias) for _ in range(deep)])

  def forward(self, x, h_0):
    h_sequences = []
    
    h = h_0
    
    if self.bidirection:
      reversed_h = h_0
      reversed_x = torch.flip(x, [1])

    batch_size, seq_len, _ = x.size()
    
    for t in range(seq_len):
      x_t = x[:, t, :]
      
      if self.bidirection:
        reversed_x_t = reversed_x[:, t, :]

      h = self.l_gru(x_t, h)
      h = self.t_grus(h)

      if self.bidirection:
        reversed_h = self.reversed_l_gru(reversed_x_t, reversed_h)
        reversed_h = self.reversed_t_grus(reversed_h)
        h_sequences.append(torch.cat((h, reversed_h), dim=-1))
      else:
        h_sequences.append(h)

    h_sequences = torch.stack(h_sequences).transpose(1, 0)

    return h_sequences

if __name__ == '__main__':
  x = torch.randn(2, 10, 10)
  
  rnn = DeepTransitionRNN(10, 10)
  h_0 = torch.zeros(2, 10)
  h = rnn(x, h_0)

  print(h)