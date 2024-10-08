import torch
from torch import nn
from diffusers import AutoencoderTiny

class VAE(nn.Module):
    def __init__(self, force_batch_size = 16):
        super().__init__()

        self.model = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype = torch.half)
        self.model.cuda()
        self.model.half()

        self.force_batch_size = force_batch_size
    
    @torch.no_grad()
    def encode(self, x):
        x_dtype = x.dtype
        if self.force_batch_size is not None:
            chunks = x.split(self.force_batch_size)
            latents = [self.model.encode(chunk.half()).latents for chunk in chunks]
            return torch.cat(latents, dim=0).to(x_dtype)
        return self.model.encode(x.half()).latents.to(x_dtype)
    
    @torch.no_grad()
    def decode(self, x):
        x_dtype = x.dtype
        if self.force_batch_size is not None:
            chunks = x.split(self.force_batch_size)
            decoded = [self.forward(chunk.half()) for chunk in chunks]
            return torch.cat(decoded, dim=0).to(x_dtype)
        return self.forward(x.half()).to(x_dtype)
    
    @torch.no_grad()
    def forward(self, latents):
        return self.model.decode(latents).sample