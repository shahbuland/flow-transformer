import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
import einops as eo
import math

from .vae import VAE
from .utils import (
  freeze, truncated_normal_init, mimetic_init, normal_init,
  log2, sample_discrete_timesteps, sample_step_size
)

from rotary_embedding_torch import RotaryEmbedding

from .configs import ModelConfig
from .nn.embeddings import TimestepEmbedding, AbsEmbedding, SphericalAdditiveLayer, StepEmbedding
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
    self.d_embedder = StepEmbedding(d_model, max_steps = self.config.base_steps)

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
    self.final_mod = SimpleModulation(d_model, normalized = False)

    truncated_normal_init(self.pos_enc)
      
  def normalize(self):
    norm_layer(self.text_proj)
    norm_layer(self.proj_in)
    norm_layer(self.t_embedder.mlp.fc1)
    norm_layer(self.t_embedder.mlp.fc2)
    norm_layer(self.d_embedder.mlp.fc1)
    norm_layer(self.d_embedder.mlp.fc2)
    #self.pos_enc.normalize()
    #norm_layer(self.proj_out)
    for layer in self.layers:
      norm_dit_block(layer)

  def forward(self, x, t, c=None, d=None, output_hidden_states=False):
    """
    x [b,c,h,w] image
    t [b,] timesteps
    c [b,n_text,d_text] text embeddings
    d [b,] step multiplier (0) 
    """
    if c is not None:
      c = self.text_proj(c)
      if self.normalized:
        c = norm(c)

    x = self.patchify(x)
    x = self.proj_in(x)
    x = self.pos_enc(x)
    if self.normalized:
      x = norm(x)

    t = norm(self.t_embedder(t))
    d = norm(self.d_embedder(d))
    t = norm(t + d)

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

    self.empty_embed = self.encode_text([""]) # [1,n,d]

  def grouped_parameters(self):
    res = list(self.core.parameters())
    if self.repa is not None:
      res += list(self.repa.mlp.parameters())
    return res
  
  def encode_text(self, *args, **kwargs):
    return self.text_embedder.encode_text(*args, **kwargs)
  
  def normalize(self):
    if not self.config.normalized:
      return
    if self.repa is not None:
      norm_layer(self.repa.mlp.fc1)
      norm_layer(self.repa.mlp.fc2)
    self.core.normalize()

  @torch.no_grad()
  def generate_sc_targets(self, x):
    if self.config.take_label:
      x, ctx = x
      ctx = self.text_embedder.encode_text(ctx)
      ctx = ctx.to(x.dtype).to(x.device)
      neg_ctx = self.empty_embed.repeat(x.shape[0],1,1).to(x.dtype).to(x.device)
    else:
      ctx = None

    if self.vae is not None:
      x = self.vae.encode(x)

    # Mostly the same, but we sample steps first then sample time based on those
    b,c,h,w = x.shape
    z = torch.randn_like(x)
    
    d_slow = sample_step_size(b, self.config.base_steps).to(device=x.device,dtype=x.dtype)
    cfg_mask = (d_slow == 128).float()[:,None,None,None]
    d_fast = d_slow / 2 # half as may steps -> faster

    dt_slow = -1./d_slow
    dt_fast = -1./d_fast

    # Since this t will be input to model being trained to do fast, 
    # use timesteps that make sense for step faster
    t = sample_discrete_timesteps(d_fast)

    x_exp = lambda x: eo.repeat(x, 'b -> b c h w',c=c,h=h,w=w)
    t_exp = x_exp(t)
    dt_exp = x_exp(dt_slow)

    # Sample slow to create target for training fast
    noisy = x * (1. - t_exp) + z * t_exp
    pred_1 = self.denoise(noisy, t, ctx, d_slow)
    if cfg_mask.any():
      pred_1_neg = self.denoise(noisy, t, neg_ctx, d_slow)
      pred_1 = torch.where(
        cfg_mask,
        pred_1_neg + self.config.sc_cfg * (pred_1 - pred_1_neg),
        pred_1
      )
    
    less_noisy = noisy + dt_exp * pred_1
    pred_2 = self.denoise(less_noisy, t + dt_slow, ctx, d_slow)
    if cfg_mask.any():
      pred_2_neg = self.denoise(noisy,t+dt_slow,neg_ctx,d_slow)
      pred_2 = torch.where(
        pred_2_neg + self.config.sc_cfg * (pred_2 - pred_2_neg)
      )

    sc_target = 0.5 * (pred_1 + pred_2) # avg two slow predictions
    return noisy, sc_target, t, ctx, d_fast

  def forward(self, x, sc_targets = None):
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
      #t = torch.randn(b, device = x.device, dtype = x.dtype).sigmoid() # log norm timesteps
      t = torch.rand(b, device = x.device, dtype = x.dtype) # U(0,1)
      d = torch.full((b,), self.config.base_steps, device=x.device, dtype=x.dtype)

      # exp here means expanded
      t_exp = eo.repeat(t, 'b -> b c h w', c = c, h = h, w = w) # Makes it the same shape as x and z so we can multiply

      # Based on ODE setup of going t: 0 -> 1 noise -> images
      # t = 0 should be noise
      lerpd = x * (1 - t_exp) + z * t_exp
      target = z-x # Velocity to predict

    extra = {}

    pred, h = self.denoise(lerpd, t, ctx, d, output_hidden_states=True)
    extra['last_hidden'] = h[-1]

    total_loss = 0.

    diff_loss = F.mse_loss(target, pred)
    extra['diff_loss'] = diff_loss.item()
    total_loss += diff_loss

    if self.training:
      if self.config.sc_weight > 0 and sc_targets is not None:
        sc_inputs, sc_targets, sc_t, sc_ctx, sc_d = sc_targets
        sc_pred = self.denoise(sc_inputs, sc_t, sc_ctx, sc_d)
        sc_loss = F.mse_loss(sc_targets, sc_pred)
        extra['sc_loss'] = sc_loss.item()

        total_loss += self.config.sc_weight * sc_loss

      if self.repa is not None:
        repa_loss = self.repa(x_orig, h[self.config.repa_layer_ind])
        total_loss += repa_loss * self.config.repa_weight
        extra['repa_loss'] = repa_loss.item()

    return total_loss, extra

  def denoise(self, x, t, c = None, steps = None, output_hidden_states = False):
    return self.core(x,t,c,steps,output_hidden_states)

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

