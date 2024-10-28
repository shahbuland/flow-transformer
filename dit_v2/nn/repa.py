from transformers import AutoModel, AutoProcessor
import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
import einops as eo

from .mlp import MLP
from dit.configs import ModelConfig
from dit.utils import freeze

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

class REPA(nn.Module):
    def __init__(self, config : ModelConfig, dino_path = "facebook/dinov2-giant"):
        super().__init__()

        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino.to(device='cuda',dtype=torch.half)
        self.mlp = MLP(
            config.d_model,
            dim_out=self.dino.config.hidden_size
        )
        self.batch_size = config.repa_batch_size

        freeze(self.dino)

    @torch.no_grad()
    def dino_features(self, x):
        # x is [b,c,h,w] [-1,1]
        inputs = dino_proc(x)
        input_batches = inputs.split(self.batch_size)

        h_all = []
        for batch in input_batches:
            h = self.dino(pixel_values=batch, output_hidden_states = True).hidden_states[-1][:,1:] # Skip CLS
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

        h = self.dino_features(x)
        h_rft = self.mlp(features)
        # now both [b,n,d] in the same space

        return self.feature_cos_sim(h, h_rft)