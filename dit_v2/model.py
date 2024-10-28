import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from dit_v2.nn.transformer import StackedDiT
from dit_v2.nn.embeddings import (
    TimestepEmbedding, AbsEmbedding, StepEmbedding
)
from dit_v2.nn.modulation import FinalMod
from dit_v2.nn.repa import REPA
from .nn.vae import VAE
from dit_v2.nn.mlp import MLP

from dit.nn.text_embedder import TextEmbedder
from dit.utils import freeze, sample_discrete_timesteps, sample_step_size

class RFTCore(nn.Module):
    def __init__(self, config : 'ModelConfig'):
        super().__init__()

        self.blocks = StackedDiT(config)

        self.t_embed = TimestepEmbedding(config.d_model)
        self.d_embedder = StepEmbedding(config.d_model, max_steps = config.base_steps)
        self.pool_embedder = MLP(config.text_d_model, dim_out=config.d_model)

        n_patches = config.sample_size // config.patch_size
        patch_content = config.patch_size * config.patch_size * config.channels
        self.pos_enc = AbsEmbedding(n_patches**2, config.d_model)

        self.proj_out = nn.Linear(config.d_model, patch_content)
        self.text_proj = nn.Linear(config.text_d_model, config.d_model)

        self.final_mod = FinalMod(config.d_model)

        self.patch_proj = nn.Conv2d(
            config.channels,
            config.d_model,
            config.patch_size,
            config.patch_size
        )

        self.depatchify = lambda x: eo.rearrange(
            x,
            'b (n_p_h n_p_w) (p_h p_w c) -> b c (n_p_h p_h) (n_p_w p_w)',
            n_p_h = n_patches,
            p_h = config.patch_size,
            c = config.channels
        )
    
    def forward(self, x, y, ts, d, output_hidden_states = False):
        # x [b,c,h,w]
        # y [b,m,d]
        # ts [b,d]

        y_pool = y.clone().mean(1)
        y = self.text_proj(y)
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1,2)
        x = self.pos_enc(x)

        cond = self.t_embed(ts) + self.pool_embedder(y_pool) + self.d_embedder(d)

        h = None

        out = self.blocks(x, y, cond, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            x, h = out
        else:
            x = out
        
        x = self.final_mod(x, cond)
        x = self.proj_out(x)
        x = self.depatchify(x)

        if output_hidden_states:
            return x,h
        return x

class RectFlowTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.core = RFTCore(config)

        self.text_embedder = TextEmbedder()
        freeze(self.text_embedder)

        self.vae = VAE()
        freeze(self.vae)

        self.repa = None
        if self.config.repa_weight > 0.0:
            self.repa = REPA(self.config)
        
        self.empty_embed = self.encode_text([""])
    
    def encode_text(self, *args, **kwargs):
        return self.text_embedder.encode_text(*args, **kwargs)

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
        cfg_mask = (d_slow == 128)[:,None,None,None]
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
        pred_1 = self.denoise(noisy, ctx, t, d_slow)
        if cfg_mask.any():
            pred_1_neg = self.denoise(noisy, neg_ctx, t, d_slow)
            pred_1 = torch.where(
                cfg_mask,
                pred_1_neg + self.config.sc_cfg * (pred_1 - pred_1_neg),
                pred_1
            )
        
        less_noisy = noisy + dt_exp * pred_1
        pred_2 = self.denoise(less_noisy, ctx, t + dt_slow, d_slow)
        if cfg_mask.any():
            pred_2_neg = self.denoise(noisy, neg_ctx, t+dt_slow, d_slow)
            pred_2 = torch.where(
                cfg_mask,
                pred_2_neg + self.config.sc_cfg * (pred_2 - pred_2_neg),
                pred_2
            )

        sc_target = 0.5 * (pred_1 + pred_2) # avg two slow predictions
        return noisy, sc_target, t, ctx, d_fast

    def denoise(self, *args, **kwargs):
        # x, y, ts, d, output_h
        return self.core(*args, **kwargs)
    
    def forward(self, x, sc_targets = None):
        if self.config.take_label:
            x, ctx = x
            if self.config.cfg_prob > 0:
                mask = torch.rand(len(ctx)) < self.config.cfg_prob
                ctx = [c if not m else "" for c,m in zip(ctx,mask)]
            ctx = self.text_embedder.encode_text(ctx)
            ctx = ctx.to(x.dtype).to(x.device)
        else:
            ctx = None
        
        x_orig = x.clone()
        if self.vae is not None:
            with torch.no_grad():
                x = self.vae.encode(x)
        
        b,c,h,w = x.shape

        # Prepare input + target
        with torch.no_grad():
            z = torch.randn_like(x)
            t = torch.rand(b, device = x.device, dtype = x.dtype)
            d = torch.full((b,), self.config.base_steps, device = x.device, dtype = x.dtype)

            t_exp = eo.repeat(t, 'b -> b c h w', c=c,h=h,w=w)
            lerpd = x * (1 - t_exp) + z * t_exp
            target = z - x
        
        extra = {}

        pred, h = self.denoise(lerpd, ctx, t, d, output_hidden_states=True)
        total_loss = 0.

        diff_loss = F.mse_loss(target, pred)
        extra['diff_loss'] = diff_loss.item()
        total_loss += diff_loss

        if self.training:
            if self.config.sc_weight > 0 and sc_targets is not None:
                sc_inputs, sc_targets, sc_ts, sc_ctx, sc_d = sc_targets
                sc_pred = self.denoise(sc_inputs, sc_ctx, sc_ts, sc_d)
                sc_loss = F.mse_loss(sc_targets, sc_pred)
                extra['sc_loss'] = sc_loss.item()
                total_loss += self.config.sc_weight * sc_loss

            if self.repa is not None:
                repa_loss = self.repa(x_orig, h[self.config.repa_layer_ind])
                extra['repa_loss'] = repa_loss.item()
                total_loss += self.config.repa_weight * repa_loss
        
        return total_loss, extra