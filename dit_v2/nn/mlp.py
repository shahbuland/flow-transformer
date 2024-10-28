import torch
from torch import nn

import einops as eo

class MLP(nn.Module):
    def __init__(self, dim_in, dim_middle = None, dim_out = None):
        super().__init__()

        if dim_middle is None:
            dim_middle = 4 * dim_in
        if dim_out is None:
            dim_out = dim_in

        self.act = nn.GELU()
        self.fc_uv = nn.Linear(dim_in, dim_middle)
        self.fc_out = nn.Linear(dim_middle, dim_out)

    def forward(self, x):
        x = self.fc_uv(x)
        return self.fc_out(self.act(x))

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
        self.reshape_out = lambda x: eo.rearrange(
            x,
            'b d n_y n_x -> b (n_y n_x) d'
        )

        self.act = nn.ReLU()
        self.conv_1 = nn.Conv2d(dim_in,dim_middle,1)
        self.conv_2 = nn.Conv2d(dim_middle, dim_middle, 3, padding = 1, groups = dim_middle)
        self.conv_3 = nn.Conv2d(dim_middle//2,dim_in,1)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        x = self.reshape_in(x)
        x = self.conv_1(x) # [b, d, n_y, n_x]
        x = self.conv_2(x) # [b, d*4, n_y, n_x]
        gate, x = x.chunk(2, dim = 1) # 2*[b, d*2, n_y, n_x] 

        gate = self.act(gate)
        x = x * gate
        x = self.conv_3(x)
        x = self.reshape_out(x)
        return x