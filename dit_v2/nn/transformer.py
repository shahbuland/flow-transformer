from torchtyping import TensorType
import torch
from flash_attn import flash_attn_func
from torch import nn

from .normalization import LayerNorm, RMSNorm
from .embeddings import RoPEEmbedding
from .modulation import DoubleModBlock
from .mlp import MLP, MixFFN

import einops as eo

def qkv_chunk(qkv):
    return [x.contiguous() for x in qkv.chunk(3, dim = -1)]

def kv_chunk(kv):
    return [x.contiguous() for x in kv.chunk(2, dim = -1)]

def head_split(x, n_heads):
    return eo.rearrange(x, 'b n (h d) -> b n h d', h = n_heads)

def head_merge(x):
    return eo.rearrange(x, 'b n h d -> b n (h d)')

class Attn(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model
        dim_head = config.d_model // config.n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias = False)
        self.out = nn.Linear(dim, dim)

        self.split = lambda x: head_split(x, n_heads = config.n_heads)
        self.merge = lambda x: head_merge(x)

        self.cross_qkv = nn.Linear(dim, 3*dim, bias = False)


        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.cross_q_norm = RMSNorm(dim_head)
        self.cross_k_norm = RMSNorm(dim_head)

        self.rope = RoPEEmbedding(dim_head)    
        self.attn_func = flash_attn_func

    def forward(self, x, y):
        _,n,_ = x.shape
        qkv = self.qkv(x)
        q,k,v = [self.split(x) for x in qkv_chunk(qkv)]

        q = self.q_norm(q)
        k = self.k_norm(k)

        cross_qkv = self.cross_qkv(y)
        c_q,c_k,c_v = [self.split(x) for x in qkv_chunk(cross_qkv)]
        c_q = self.cross_q_norm(c_q)
        c_k = self.cross_k_norm(c_k)

        q = torch.cat([q, c_q], 1)
        k = torch.cat([k, c_k], 1)
        v = torch.cat([v, c_v], 1)

        #q, k = self.rope(q,k)

        attn_out = self.attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)).to(q.dtype)
        attn_out = self.merge(attn_out)

        attn_out = attn_out[:,:n]

        return self.out(attn_out)

class LinearAttn(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model
        dim_head = config.d_model // config.n_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

        self.cross_kv = nn.Linear(dim, 2*dim)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.cross_k_norm = RMSNorm(dim_head)

        self.split = lambda x: head_split(x, n_heads = config.n_heads)
        self.merge = lambda x: head_merge(x)

    def forward(self, x, y):
        _,n,_ = x.shape
        q,k,v = [self.split(x) for x in qkv_chunk(self.qkv(x))]
        c_k,c_v = [self.split(x) for x in kv_chunk(self.cross_kv(y))]

        # all [b,n,h,d] now
        q = self.q_norm(q)
        k = self.k_norm(k)
        c_k = self.cross_k_norm(c_k)

        k = torch.cat([k,c_k],1)
        v = torch.cat([v,c_v],1)

        q = self.relu(q)
        k = self.relu(k)

        # einsum bmhd,bmhd->bhdd
        k_sum = k.sum(1) # [b,h,d]
        k = k.permute(0, 2, 3, 1) # [b,h,d,n]
        v = v.permute(0, 2, 1, 3) # [b,h,n,d]
        kv = torch.matmul(k,v) # [b,h,d,d]

        # [b,n,h,d] -> [b,h,n,d]
        q = q.permute(0, 2, 1, 3)
        attn_out = torch.matmul(q,kv) # [b,h,n,d]
        scale = torch.einsum('bhnd,bhd->bhn',q,k_sum).unsqueeze(-1) # [b,h,n,1]

        attn_out = attn_out/(scale + 1.0e-6)
        attn_out = attn_out.permute(0,2,1,3) # -> [b,n,h,d]
        x = self.merge(attn_out)
        return self.out(x)

class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mod = DoubleModBlock(config.d_model)
        #self.attn = Attn(config)
        self.attn = LinearAttn(config)
        #self.mlp = MLP(config.d_model)
        self.mlp = MixFFN(config)

        self.norm_1 = LayerNorm(config.d_model)
        self.norm_2 = LayerNorm(config.d_model)

    def forward(self, x, y, cond):
        mod1, mod2 = self.mod(cond)
        resid_1 = x.clone()

        x = self.norm_1(x)
        x = mod1.first_step(x)
        attn_out = self.attn(x,y)
        attn_out = mod1.second_step(attn_out)

        x = self.norm_2(x + resid_1)
        resid_2 = x.clone()

        x = mod2.first_step(x)
        x = self.mlp(x)
        x = mod2.second_step(x)

        return x + resid_2

class StackedDiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.n_layers)
        ])

    def forward(self, x, y, cond, output_hidden_states = False):
        h = []
        for block in self.blocks:
            x = block(x, y, cond)
            h.append(x.clone())

        if output_hidden_states:
            return x,h
        else:
            return x