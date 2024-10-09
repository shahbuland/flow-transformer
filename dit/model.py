import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
import einops as eo
import math

from .vae import VAE
from .utils import freeze, truncated_normal_init, mimetic_init, normal_init

from rotary_embedding_torch import RotaryEmbedding

from .configs import ModelConfig
from .nn.embeddings import TimestepEmbedding, AbsEmbedding
from .nn.modulation import SimpleModulation
from .nn.transformers import DiTBlock
from .nn.text_embedder import TextEmbedder

class RectFlowTransformer(nn.Module):
  def __init__(self, config: ModelConfig = ModelConfig()):
    super().__init__()

    self.config = config

    n_layers = config.n_layers
    d_model = config.d_model
    n_heads = config.n_heads
    patch_size = config.patch_size
    sample_size = config.sample_size
    channels = config.channels

    self.t_embedder = TimestepEmbedding(d_model)

    n_patches = (sample_size // patch_size) ** 2
    # Seen in repos that init'ing with std = 0.02 is better
    # You can use abs + rope but often times it's not needed
    self.pos_enc = AbsEmbedding(n_patches, d_model)

    # Shorter way to make many layers
    self.layers = nn.ModuleList([DiTBlock(config) for _ in range(n_layers)])

    # Make some patchify and depatchify functions given the params
    self.patchify = lambda x: eo.rearrange(x, 'b c (n_p_h p_h) (n_p_w p_w) -> b (n_p_h n_p_w) (p_h p_w c)',
                                           p_h = patch_size, p_w = patch_size)
    self.depatchify = lambda x: eo.rearrange(x, 'b (n_p_h n_p_w) (p_h p_w c) -> b c (n_p_h p_h) (n_p_w p_w)',
                                             p_h = patch_size, p_w = patch_size, n_p_h = sample_size//patch_size)

    patch_content = channels * patch_size ** 2

    self.proj_in = nn.Linear(patch_content, d_model)
    self.proj_out = nn.Linear(d_model, patch_content)
    
    #self.final_norm = RMSNorm(d_model)
    self.final_norm = nn.LayerNorm(d_model, elementwise_affine = False, eps = 1.0e-6)
    self.final_mod = SimpleModulation(d_model)

    if self.config.take_label:
      self.text_embedder = TextEmbedder(d_model)
      freeze(self.text_embedder)

    self.vae = None
    if config.use_vae:
        self.vae = VAE()
        freeze(self.vae)

    # Inits
    for layer in self.layers:
        pass
        #truncated_normal_init(layer)
        #mimetic_init(layer.qkv, layer.out, config.n_heads)
        
    truncated_normal_init(self.pos_enc)
  
  def encode_text(self, *args, **kwargs):
    return self.text_embedder.encode_text(*args, **kwargs)

  def forward(self, x):
    if self.config.take_label:
      x, ctx = x # c is list str
      ctx = self.text_embedder.encode_text(ctx)
      ctx = ctx.to(x.dtype).to(x.device)
    else:
      ctx = None

    if self.vae is not None:
        with torch.no_grad():
            x = self.vae.encode(x)

    b, c, h, w = x.shape

    with torch.no_grad():
        z = torch.randn_like(x) # Noise we will lerp with
        t = torch.randn(b, device = x.device, dtype = x.dtype).sigmoid() # log norm timesteps

        # exp here means expanded
        t_exp = eo.repeat(t, 'b -> b c h w', c = c, h = h, w = w) # Makes it the same shape as x and z so we can multiply

        # Based on ODE setup of going t: 0 -> 1 noise -> images
        # t = 0 should be noise
        lerpd = x * (1 - t_exp) + z * t_exp
        target = z-x # Velocity to predict

    x = self.denoise(lerpd, t, ctx)

    loss = ((x - target) ** 2).mean()
    return loss

  def denoise(self, x, t, c = None):
    x = self.patchify(x)
    x = self.proj_in(x)

    t_emb = self.t_embedder(t)
    
    for layer in self.layers:
      x = layer(x, t_emb, c)

    x = self.final_norm(x)
    x = self.final_mod(x, t_emb)
    x = self.proj_out(x)
    x = self.depatchify(x)

    return x


if __name__ == "__main__":
    import torch

    model = RectFlowTransformer(ModelConfig())

    # Create a random input tensor of shape [1, 3, 64, 64]
    input_tensor = torch.randn(1, 3, 64, 64)

    # Forward pass
    output = model(input_tensor)
    output.backward()

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value (loss): {output.item():.6f}")

    # Test denoise method
    t = torch.rand(1)  # Random timestep between 0 and 1
    denoised = model.denoise(input_tensor, t)
    print(f"Denoised output shape: {denoised.shape}")

