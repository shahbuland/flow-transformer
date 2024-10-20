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

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    x = (x - mean) / std

    return x

# This only works for a factor of 2 currently
def patch_pool(x : TensorType["b", "n", "d"], config : ModelConfig):
    n_patches = config.sample_size // config.patch_size
    x = eo.rearrange(x, 'b (n_p_h n_p_w) d -> b n_p_h n_p_w d', n_p_h = n_patches)

    top_rows = x[:,::2]
    bottom_rows = x[:,1::2]

    top_rows = eo.rearrange(top_rows, 'b n_p_h n_p_w d -> b (n_p_h n_p_w) d')
    top_left_patch = top_rows[:,::2]
    top_right_patch = top_rows[:,1::2]

    bottom_rows = eo.rearrange(bottom_rows, 'b n_p_h n_p_w d -> b (n_p_h n_p_w) d')
    bottom_left_patch = bottom_rows[:,::2]
    bottom_right_patch = bottom_rows[:,1::2]

    # Now should all be [b,k,d] where k is n / 4
    # want [b,k,d*4]
    pooled = torch.cat([top_left_patch, top_right_patch, bottom_left_patch, bottom_right_patch], dim = -1)
    return pooled

class REPA(nn.Module):
    def __init__(self, config : ModelConfig, dino_path = "facebook/dinov2-giant"):
        super().__init__()

        self.pool_factor = config.repa_pool_factor
        self.normalized = config.normalized

        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino.to(device='cuda',dtype=torch.half)
        self.mlp = MLP(
            config.d_model * (self.pool_factor ** 2),
            dim_out=self.dino.config.hidden_size,
            use_scale = self.normalized,
            d_middle = config.d_model * 4
        )
        self.batch_size = config.repa_batch_size
        
        self.patch_pool = None
        if self.pool_factor > 1:
            self.patch_pool = lambda x: patch_pool(x, config)

        freeze(self.dino)

    @torch.no_grad()
    def dino_features(self, x):
        # x is [b,c,h,w] [-1,1]
        inputs = dino_proc(x)
        input_batches = inputs.split(self.batch_size)

        h_all = []
        for batch in input_batches:
            h = self.dino(pixel_values=batch, output_hidden_states = True).hidden_states[-2][:,1:] # Skip CLS
            h_all.append(h)
        
        h_all = torch.cat(h_all)

        return h_all.to(x.dtype)

    def feature_cos_sim(self, x, y):
        x = F.normalize(x, dim = -1)
        y = F.normalize(y, dim = -1)
        cos_sims = torch.einsum('bnd,bnd->bn', x, y)
        return -cos_sims.mean()
    
    def feature_mse(self, x, y):
        x = F.normalize(x)
        y = F.normalize(y)

        # both [b,n,d]
        return F.mse_loss(x, y)

    def forward(self, x, features):
        # x [b,c,h,w]
        # features [b,n,d]

        if self.pool_factor > 1:
            features = self.patch_pool(features)

        h = self.dino_features(x)
        h_rft = self.mlp(features)
        # now both [b,n,d] in the same space

        #return F.mse_loss(h, h_rft)
        #if self.normalized:
        return self.feature_cos_sim(h, h_rft)
        #else:
            #return self.feature_mse(h, h_rft)


