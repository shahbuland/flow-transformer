import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
import einops as eo
import math

from .vae import VAE
from .utils import freeze, truncated_normal_init, mimetic_init, normal_init

from rotary_embedding_torch import RotaryEmbedding

class RMSNorm(nn.Module):
    def __init__(self, d, eps = 1.0e-6):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x : TensorType["b", "n", "d"]):
        gain = (1 + self.g)[None,None,:] # Add a batch and sequence dim

        rms = (x.float().pow(2).mean(-1, keepdim = True) + self.eps).rsqrt() # [b, n]

        x = (x * rms.to(x.dtype))
        x = x * gain

        return x

class RoPEEmbedding(nn.Module):
    """
    "Flat" Version of RoPE
    """
    def __init__(self, dim, flash = False):
        super().__init__()

        self.rope = RotaryEmbedding(dim // 2)
        self.flash = flash
    
    def forward(self, q, k):
        if self.flash: # [b,n,h,d] -> [b,h,n,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)
        q, k = self.rope.rotate_queries_or_keys(q), self.rope.rotate_queries_or_keys(k)
        if self.flash: # [b,h,n,d] -> [b,n,h,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)
        return q,k

class RoPE2D(nn.Module):
    def __init__(self, dim, n_patches_h, n_patches_w, flash = False):
        super().__init__()

        self.rope_h = RotaryEmbedding(dim // 4)
        self.rope_w = RotaryEmbedding(dim // 4)
        self.flash = flash

        self.n_h = n_patches_h 
        self.n_w = n_patches_w

    def forward(self, q, k):
        if self.flash: # [b,n,h,d] -> [b,h,n,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)

        q_h, q_w = q.chunk(2, dim = -1)
        k_h, k_w = k.chunk(2, dim = -1)
        
        q_h = eo.rearrange(q_h, 'b h (n_h n_w) d -> (b n_w) h n_h d', n_h = self.n_h)
        k_h = eo.rearrange(k_h, 'b h (n_h n_w) d -> (b n_w) h n_h d', n_h = self.n_h)

        q_w = eo.rearrange(q_w, 'b h (n_h n_w) d -> (b n_h) h n_w d', n_w = self.n_w)
        k_w = eo.rearrange(k_w, 'b h (n_h n_w) d -> (b n_h) h n_w d', n_w = self.n_w)

        q_h, k_h = self.rope_h.rotate_queries_or_keys(q_h), self.rope_h.rotate_queries_or_keys(k_h)
        q_w, k_w = self.rope_w.rotate_queries_or_keys(q_w), self.rope_w.rotate_queries_or_keys(k_w)

        q_h = eo.rearrange(q_h, '(b n_w) h n_h d -> b h (n_h n_w) d', n_w=self.n_w)
        k_h = eo.rearrange(k_h, '(b n_w) h n_h d -> b h (n_h n_w) d', n_w=self.n_w)

        q_w = eo.rearrange(q_w, '(b n_h) h n_w d -> b h (n_h n_w) d', n_h=self.n_h)
        k_w = eo.rearrange(k_w, '(b n_h) h n_w d -> b h (n_h n_w) d', n_h=self.n_h)

        q = torch.cat([q_h, q_w], dim=-1)
        k = torch.cat([k_h, k_w], dim=-1)

        if self.flash: # [b,h,n,d] -> [b,n,h,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)

        return q,k        


def contiguous_qkv_chunk(qkv):
    q, k, v = qkv.chunk(3, dim = -1)
    return q.contiguous(), k.contiguous(), v.contiguous()

def head_split(x, n_heads, flash = False):
    if flash:
        return eo.rearrange(x, 'b n (h d) -> b n h d', h = n_heads)
    else:
        return eo.rearrange(x, 'b n (h d) -> b h n d', h = n_heads)
    
def head_merge(x, flash = False):
    if flash:
        return eo.rearrange(x, 'b n h d -> b n (h d)')
    else:
        return eo.rearrange(x, 'b h n d -> b n (h d)')

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -.5

    def forward(self, q, k, v):
        # [b, h, n, d] for all 3
        scores = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale
        scores = torch.softmax(scores, dim = -1) # [b, h, n_in, n_out]
        out = torch.einsum('bhnm,bhmd->bhmd', scores, v)
        return out

class MLP(nn.Module):
  """
  Multilayer perceptron

  In terms of information flow discussed in course,
  this model essentially processes each word individually,
  i.e. information flow from input to output is only within words, not between them
  """
  def __init__(self, dim, dim_out = None):
    super().__init__()
    if dim_out is None:
      dim_out = dim

    self.fc1 = nn.Linear(dim, 4 * dim) # hiddden size in transformer MLPs is normally 4x the input size
    self.act = nn.GELU()
    self.fc2 = nn.Linear(4 * dim, dim_out)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x

class ModulationOutput:
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha[:,None] # [b,1,d]
    self.beta = beta[:,None] # [b,1,d]
    self.gamma = gamma[:,None] # [b,1,d]

  def first_step(self, x : TensorType["b", "n", "d"]):
    return x * (1 + self.alpha) + self.beta

  def second_step(self, x):
    return x * self.gamma

class SimpleModulation(nn.Module):
    """
    Simple modulation with jsut shift and scale
    """
    def __init__(self, dim):
        super().__init__()

        self.act = nn.GELU()
        self.mod_params = nn.Linear(dim, 2 * dim)
    
    def forward(self, x, t):
        t = self.act(t)
        scale, shift = self.mod_params(t).chunk(2,dim=-1)
        return x * (1 + scale[:,None]) + shift[:,None]

# Makes modulation parameters given t embedding
class DoubleModBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.act = nn.GELU() # Following SD3
    self.fc = nn.Linear(dim, 6 * dim)

  def forward(self, t):
    t = self.act(t)
    params = self.fc(t) # They say an MLP, but I think this isn't needed
    alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2 = params.chunk(6, dim = -1) # Break into 6 parts
    return [
        ModulationOutput(alpha_1, beta_1, gamma_1),
        ModulationOutput(alpha_2, beta_2, gamma_2)
    ]

class DiTBlock(nn.Module):
  def __init__(self, config):
    super().__init__()

    d_model = config.d_model
    n_heads = config.n_heads
    flash = config.flash

    self.qkv = nn.Linear(d_model, 3 * d_model, bias = False)
    self.out = nn.Linear(d_model, d_model)
    self.mod = DoubleModBlock(d_model)

    #self.rope = RoPEEmbedding(d_model // n_heads, flash = flash)
    self.rope = RoPE2D(d_model // n_heads, config.sample_size // config.patch_size, config.sample_size // config.patch_size, flash = flash)

    self.mlp = MLP(d_model)
    if flash:
        from flash_attn import flash_attn_func
        self.attn = flash_attn_func
    else:
        self.attn = MultiHeadAttention(d_model, n_heads)

    #self.norm1 = RMSNorm(d_model)
    #self.norm2 = RMSNorm(d_model)
    self.norm1 = nn.LayerNorm(d_model, elementwise_affine = False, eps = 1.0e-6)
    self.norm2 = nn.LayerNorm(d_model, elementwise_affine = False, eps = 1.0e-6)

    self.d_model = d_model
    self.n_heads = n_heads
    self.flash = flash

    self.q_norm = RMSNorm(d_model // n_heads)
    self.k_norm = RMSNorm(d_model // n_heads)

    self.head_split = lambda x: head_split(x, n_heads = self.n_heads, flash = flash)
    self.head_merge = lambda x: head_merge(x, flash = flash)

  def forward(self, x : TensorType["b", "n", "d"], t_emb : TensorType["b", "d"]):
    mod1, mod2 = self.mod(t_emb)

    resid_1 = x.clone() # Clone x to have a residual signal
    x = self.norm1(x)
    x = mod1.first_step(x)

    qkv = self.qkv(x)
    q,k,v = contiguous_qkv_chunk(qkv)

    q = self.head_split(q)
    k = self.head_split(k)
    v = self.head_split(v)

    q = self.q_norm(q)
    k = self.k_norm(k)

    q, k = self.rope(q, k)

    # Now they are all unique objects in memory
    if self.flash:
        orig_dtype = q.dtype
        attn_out = self.attn(q.half(), k.half(), v.half()).to(orig_dtype)
    else:
        attn_out = self.attn(q,k,v)

    attn_out = self.head_merge(attn_out)
    attn_out = self.out(attn_out)
    attn_out = mod1.second_step(attn_out)

    x = attn_out + resid_1
    resid_2 = x.clone()

    x = self.norm2(x)
    x = mod2.first_step(x)
    x = self.mlp(x)
    x = mod2.second_step(x)

    x = x + resid_2
    return x

class TimestepEmbedding(nn.Module):
    def __init__(self, d_out, d_in = 512):
        super().__init__()

        self.mlp = MLP(d_in, d_out)
        self.d = d_in # Assume this is even

    def forward(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        # t is [B] tensor of timesteps ()
        t = t * 1000

        max_period = 10000 # This seems to always be assumed in all repos
        half = self.d // 2

        inds = torch.arange(half, device = t.device, dtype = t.dtype)
        freqs = (
            -math.log(max_period) * inds / half
        ).exp()

        embs = t[:,None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim = -1)

        return self.mlp(embs)

class AbsEmbedding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()

        self.embedding = nn.Parameter(torch.randn(seq_len, dim))
    
    def forward(self, x):
        # x: [b,n,d]
        p = eo.repeat(self.embedding, 'n d -> b n d', b = x.shape[0])
        return x + p

from .configs import ModelConfig

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

    self.vae = None
    if config.use_vae:
        self.vae = VAE()
        freeze(self.vae)

    # Inits
    for layer in self.layers:
        pass
        #truncated_normal_init(layer)
        #mimetic_init(layer.qkv, layer.out, config.n_heads)
    
    #truncated_normal_init(self.t_embedder)
    truncated_normal_init(self.pos_enc)
    #truncated_normal_init(self.proj_in)
    #truncated_normal_init(self.proj_out)
    #truncated_normal_init(self.final_norm)

  def forward(self, x):
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

    x = self.denoise(lerpd, t)

    loss = ((x - target) ** 2).mean()
    return loss

  def denoise(self, x, t):
    x = self.patchify(x)
    x = self.proj_in(x)

    t_emb = self.t_embedder(t)
    
    for layer in self.layers:
      x = layer(x, t_emb)

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

