import torch
from torch import nn

from .normalization import ScalingLayer, ConvScaling

class SwiGLUMLP(nn.Module):
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

    self.uv = nn.Linear(dim, 8 * dim, bias = not use_scale) # hiddden size in transformer MLPs is normally 4x the input size
    self.act = nn.SiLU()
    self.out = nn.Linear(4 * dim, dim_out, bias = not use_scale)

    self.use_scale = use_scale
    if self.use_scale:
      self.scale_u = ScalingLayer(4*dim, 1, 1)
      self.scale_v = ScalingLayer(4*dim, 1, 1)
      self.scale_v_extra = dim ** .5

  def forward(self, x):
    u,v = self.uv(x).chunk(2,dim=-1)
    if self.use_scale:
      u = self.scale_u(u)
      v = self.scale_v(v) * self.scale_v_extra
    
    x = self.out(u * self.act(v))
    return x

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
    self.act = nn.GELU()
    self.out = nn.Linear(4 * dim, dim_out)

    self.use_scale = use_scale
    if self.use_scale:
      self.scale = ScalingLayer(4*dim,1,1, extra = dim ** .5)

  def forward(self, x):
    x = self.uv(x)
    if self.use_scale:
      x = self.scale(x)
    x = self.out(x)
    return x

class MixFFN(nn.Module):
    def __init__(self, config : 'ModelConfig'):
        super().__init__()

        dim_in = config.d_model
        dim_middle = 4 * dim_in

        self.reshape_in = lambda x: eo.rearrange(
            x,
            'b (n_y n_x) d -> b d n_y n_x',
            n_y = config.sample_size // config.patch_size
        )
        self.reshape_out = lambda x: eo.rearrange(x,
            x,
            'b d n_y n_x -> b (n_y n_x) d'
        )

        self.act = nn.ReLU()
        self.conv_1 = nn.Conv2d(dim,dim_middle,1)
        self.conv_2 = nn.Conv2d(dim_middle, dim_middle, 3, padding = 1, groups = dim_middle)
        self.conv_3 = nn.Conv2d(dim_middle//2,dim,1)
    
        self.use_scale = False
        if config.normalized:
          self.use_scale = True
          self.scaling_1 = ConvScaling(dim_middle,1,1,extra=dim**.5)
          self.scaling_2 = ConvScaling(dim_middle,1,1,extra=dim**.5)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        x = self.reshape_in(x)
        x = self.conv_1(x) # [b, d, n_y, n_x]
        if self.use_scale:
          x = self.scaling_1(x)
        
        x = self.conv_2(x) # [b, d*4, n_y, n_x]
        if self.use_scale:
          x = self.scaling_2(x)
        gate, x = x.chunk(2, dim = 1) # 2*[b, d*2, n_y, n_x] 

        gate = self.act(gate)
        x = x * gate
        x = self.conv_3(x)
        x = self.reshape_out(x)