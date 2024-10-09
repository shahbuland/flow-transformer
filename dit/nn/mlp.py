import torch
from torch import nn

class MLP(nn.Module):
  """
  Multilayer perceptron

  In terms of information flow discussed in course,
  this model essentially processes each word individually,
  i.e. information flow from input to output is only within words, not between them
  """
  def __init__(self, dim, dim_out = None):
    super().__init__()
    if dim_out is None:
      dim_out = dim

    self.fc1 = nn.Linear(dim, 4 * dim) # hiddden size in transformer MLPs is normally 4x the input size
    self.act = nn.GELU()
    self.fc2 = nn.Linear(4 * dim, dim_out)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x