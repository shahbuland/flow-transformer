import torch
from torch import nn
from torchtyping import TensorType

from .normalization import norm

class ModulationOutput:
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha[:,None] # [b,1,d]
    self.beta = beta[:,None] # [b,1,d]
    self.gamma = gamma[:,None] # [b,1,d]

  def first_step(self, x : TensorType["b", "n", "d"]):
    return x * (1 + self.alpha) + self.beta

  def second_step(self, x):
    return x * self.gamma

class HyperSphereModulation(ModulationOutput):
  def first_step(self, x):
    # SLERP x with beta using alpha as an interpolation factor
    x = norm(x)
    beta = norm(self.beta)
    scale = (1. + self.alpha)
    return norm(x + scale * (beta - x))
  
  def second_step(self, x):
    return norm(x * self.gamma)

class SimpleModulation(nn.Module):
    """
    Simple modulation with jsut shift and scale
    """
    def __init__(self, dim, normalized = False):
        super().__init__()

        self.act = nn.GELU()
        self.mod_params = nn.Linear(dim, 2 * dim)
        self.normalized = normalized
    
    def forward(self, x, t):
        t = self.act(t)
        scale, shift = self.mod_params(t).chunk(2,dim=-1)

        if self.normalized:
          return x * (1 + scale[:,None]) + shift[:,None]
        else:
          shift = norm(shift)[:,None]
          scale = (1. + scale)[:,None]
          return norm(x + scale * (shift - x))

# Makes modulation parameters given t embedding
class DoubleModBlock(nn.Module):
  def __init__(self, dim, normalized = False):
    super().__init__()

    self.act = nn.GELU() # Following SD3
    self.fc = nn.Linear(dim, 6 * dim)

    if normalized:
      self.modulation_cls = HyperSphereModulation
    else:
      self.modulation_cls = ModulationOutput

  def forward(self, t):
    t = self.act(t)
    params = self.fc(t) # They say an MLP, but I think this isn't needed
    alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2 = params.chunk(6, dim = -1) # Break into 6 parts
    return [
        self.modulation_cls(alpha_1, beta_1, gamma_1),
        self.modulation_cls(alpha_2, beta_2, gamma_2)
    ]