import torch
from torch import nn

from .normalization import ScalingLayer

class MLP(nn.Module):
  """
  Multilayer perceptron

  In terms of information flow discussed in course,
  this model essentially processes each word individually,
  i.e. information flow from input to output is only within words, not between them
  """
  def __init__(self, dim, dim_out = None, d_middle = None, use_scale = False):
    super().__init__()
    if dim_out is None:
      dim_out = dim
    if d_middle is None:
      d_middle = 4 * dim

    self.uv = nn.Linear(dim, 4 * dim) # hiddden size in transformer MLPs is normally 4x the input size
    self.act = nn.SiLU()
    self.out = nn.Linear(2 * dim, dim_out)

    self.use_scale = use_scale
    if self.use_scale:
      self.scale_u = ScalingLayer(2*dim, 1, 1)
      self.scale_v = ScalingLayer(2*dim, 1, 1)
      self.scale_v_extra = dim ** .5

  def forward(self, x):
    u,v = self.uv(x).chunk(2,dim=-1)
    if self.use_scale:
      u = self.scale_u(u)
      v = self.scale_v(v) * self.scale_v_extra
    
    x = self.out(u * self.act(v))
    return x