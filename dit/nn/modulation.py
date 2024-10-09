import torch
from torch import nn
from torchtyping import TensorType

class ModulationOutput:
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha[:,None] # [b,1,d]
    self.beta = beta[:,None] # [b,1,d]
    self.gamma = gamma[:,None] # [b,1,d]

  def first_step(self, x : TensorType["b", "n", "d"]):
    return x * (1 + self.alpha) + self.beta

  def second_step(self, x):
    return x * self.gamma

class SimpleModulation(nn.Module):
    """
    Simple modulation with jsut shift and scale
    """
    def __init__(self, dim):
        super().__init__()

        self.act = nn.GELU()
        self.mod_params = nn.Linear(dim, 2 * dim)
    
    def forward(self, x, t):
        t = self.act(t)
        scale, shift = self.mod_params(t).chunk(2,dim=-1)
        return x * (1 + scale[:,None]) + shift[:,None]

# Makes modulation parameters given t embedding
class DoubleModBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.act = nn.GELU() # Following SD3
    self.fc = nn.Linear(dim, 6 * dim)

  def forward(self, t):
    t = self.act(t)
    params = self.fc(t) # They say an MLP, but I think this isn't needed
    alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2 = params.chunk(6, dim = -1) # Break into 6 parts
    return [
        ModulationOutput(alpha_1, beta_1, gamma_1),
        ModulationOutput(alpha_2, beta_2, gamma_2)
    ]