from transformers import AutoModel, AutoProcessor
import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
import einops as eo

from .mlp import MLP
from ..configs import ModelConfig
from ..utils import freeze

def dino_proc(x: TensorType["b", "c", "h", "w"]):
    """
    DINO processor as a function
    """
    # Convert from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Resize
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

    # Rescale
    x = x * 0.00392156862745098  # This is the rescale_factor

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    x = (x - mean) / std

    return x

# This only works for a factor of 2 currently
def patch_pool(x : TensorType["b", "n", "d"], config : ModelConfig):
    #x = eo.rearrange(x, 'b (n_p_h n_p_w) d -> b n_p_h n_p_w d', n_p_h = config.sample_size//config.patch_size)
    #x = eo.rearrange(x, '')
    return x

class REPA(nn.Module):
    def __init__(self, config : ModelConfig, dino_path = "facebook/dinov2-small"):
        super().__init__()

        self.pool_factor = config.repa_pool_factor

        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino.to(device='cuda',dtype=torch.half)
        self.mlp = MLP(
            config.d_model * (self.pool_factor ** 2),
            dim_out=self.dino.config.hidden_size,
            use_scale = False,
            d_middle = config.d_model * 4
        )
        self.batch_size = config.repa_batch_size
        
        self.patch_pool = None
        if self.pool_factor > 1:
            self.patch_pool = lambda x: patch_pool(x, config)
    
        freeze(self.dino)

    def to(self, *args, **kwargs): # don't touch dino
        self.mlp.to(*args, **kwargs)

    @torch.no_grad()
    def dino_features(self, x):
        # x is [b,c,h,w] [-1,1]
        inputs = dino_proc(x.half().cuda())
        input_batches = inputs.split(self.batch_size)

        h_all = []
        for batch in input_batches:
            h = self.dino(pixel_values=batch, output_hidden_states = True).hidden_states[-2][:,1:] # Skip CLS
            h_all.append(h)
        
        h_all = torch.cat(h_all)

        return h_all.to(x.dtype)

    def forward(self, x, features):
        # x [b,c,h,w]
        # features [b,n,d]

        if self.pool_factor > 1:
            patch_p

        h = self.dino_features(x)
        h_rft = self.mlp(features)
        # now both [b,n,d] in the same space

        h = F.normalize(h, p = 2, dim = -1)
        h_rft = F.normalize(h_rft, p = 2, dim = -1)

        cos_sims = torch.einsum('bnd,bnd->bn', h, h_rft)
        return -cos_sims.mean() # maximize cos_sims -> minimize -cos_sims


