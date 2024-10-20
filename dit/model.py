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
from .nn.embeddings import TimestepEmbedding, AbsEmbedding, SphericalAdditiveLayer
from .nn.modulation import SimpleModulation
from .nn.transformers import DiTBlock
from .nn.text_embedder import TextEmbedder
from .nn.normalization import Norm, RMSNorm, norm_layer, norm_dit_block, norm, LayerNorm
from .nn.repa import REPA

class RFTCore(nn.Module):
  def __init__(self, config: ModelConfig = ModelConfig()):
    super().__init__()

    self.config = config

    n_layers = config.n_layers
    d_model = config.d_model
    n_heads = config.n_heads
    patch_size = config.patch_size
    sample_size = config.sample_size
    channels = config.channels
    self.normalized = config.normalized

    self.t_embedder = TimestepEmbedding(d_model)

    n_patches = (sample_size // patch_size) ** 2
    self.pos_enc = AbsEmbedding(n_patches, d_model)
    #self.pos_enc = SphericalAdditiveLayer(n_patches, d_model)

    self.layers = nn.ModuleList([DiTBlock(config) for _ in range(n_layers)])

    self.patchify = lambda x: eo.rearrange(x, 'b c (n_p_h p_h) (n_p_w p_w) -> b (n_p_h n_p_w) (p_h p_w c)',
                                            p_h=patch_size, p_w=patch_size)
    self.depatchify = lambda x: eo.rearrange(x, 'b (n_p_h n_p_w) (p_h p_w c) -> b c (n_p_h p_h) (n_p_w p_w)',
                                              p_h=patch_size, p_w=patch_size, n_p_h=sample_size//patch_size)

    patch_content = channels * patch_size ** 2

    self.proj_in = nn.Linear(patch_content, d_model)
    if self.config.take_label: 
      self.text_proj = nn.Linear(self.config.text_d_model, d_model)
      #freeze(self.text_proj)
    
    if not self.normalized:
      self.final_norm = LayerNorm(d_model)

    self.proj_out = nn.Linear(d_model, patch_content)
    self.final_mod = SimpleModulation(d_model, normalized = True)

    truncated_normal_init(self.pos_enc)
      
  def normalize(self):
    norm_layer(self.text_proj)
    norm_layer(self.proj_in)
    #self.pos_enc.normalize()
    #norm_layer(self.proj_out)
    for layer in self.layers:
      norm_dit_block(layer)

  def forward(self, x, t, c=None, output_hidden_states=False):
    if c is not None:
      c = self.text_proj(c)
      if self.normalized:
        c = norm(c)

    x = self.patchify(x)
    x = self.proj_in(x)
    x = self.pos_enc(x)
    if self.normalized:
      x = norm(x)

    t = self.t_embedder(t)

    h = []
    for layer in self.layers:
      x = layer(x, t, c)
      if output_hidden_states:
        h.append(x)

    if not self.normalized:
      x = self.final_norm(x)

    x = self.final_mod(x, t)
    x = self.proj_out(x)
    x = self.depatchify(x)

    if output_hidden_states:
      return x, h
    return x

class RectFlowTransformer(nn.Module):
  def __init__(self, config: ModelConfig = ModelConfig()):
    super().__init__()

    self.config = config

    self.core = RFTCore(config)

    if self.config.take_label:
      self.text_embedder = TextEmbedder(config.d_model)
      freeze(self.text_embedder)

    self.vae = None
    if config.use_vae:
        self.vae = VAE()
        freeze(self.vae)

    self.repa = None
    if config.repa_weight > 0.0:
      self.repa = REPA(self.config)

  def parameters(self):
    return self.core.parameters() # Only return what we need
  
  def encode_text(self, *args, **kwargs):
    return self.text_embedder.encode_text(*args, **kwargs)
  
  def normalize(self):
    if not self.config.normalized:
      return
    if self.repa is not None:
      norm_layer(self.repa.mlp.fc1)
      norm_layer(self.repa.mlp.fc2)
    self.core.normalize()

  def forward(self, x):
    if self.config.take_label:
      x, ctx = x # c is list str
      if self.config.cfg_prob > 0:
        mask = torch.rand(len(ctx)) < self.config.cfg_prob
        ctx = [c if not m else "" for c, m in zip(ctx, mask)]

      ctx = self.text_embedder.encode_text(ctx)
      ctx = ctx.to(x.dtype).to(x.device)
      
    else:
      ctx = None

    x_orig = x.clone()
    if self.vae is not None:
      with torch.no_grad():
        x = self.vae.encode(x)

    b, c, h, w = x.shape

    # prepare target and input
    with torch.no_grad():
      z = torch.randn_like(x) # Noise we will lerp with
      t = torch.randn(b, device = x.device, dtype = x.dtype).sigmoid() # log norm timesteps

      # exp here means expanded
      t_exp = eo.repeat(t, 'b -> b c h w', c = c, h = h, w = w) # Makes it the same shape as x and z so we can multiply

      # Based on ODE setup of going t: 0 -> 1 noise -> images
      # t = 0 should be noise
      lerpd = x * (1 - t_exp) + z * t_exp
      target = z-x # Velocity to predict

    extra = {}

    pred, h = self.denoise(lerpd, t, ctx, output_hidden_states=True)
    extra['last_hidden'] = h[-1]

    total_loss = 0.

    diff_loss = F.mse_loss(target, pred)
    extra['diff_loss'] = diff_loss.item()
    total_loss += diff_loss

    if self.training:
      if self.repa is None:
        repa_loss = 0.
        extra['repa_loss'] = 0.
      else:
        repa_loss = self.repa(x_orig, h[self.config.repa_layer_ind])
        total_loss += repa_loss * self.config.repa_weight
        extra['repa_loss'] = repa_loss.item()

    return total_loss, extra

  def denoise(self, x, t, c = None, output_hidden_states = False):
    return self.core(x,t,c,output_hidden_states)

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

