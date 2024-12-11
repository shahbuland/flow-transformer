import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from dit_v2.nn.transformer import StackedDiT
from dit_v3.nn.embeddings import (
    TimestepEmbedding, AbsEmbedding, StepEmbedding, TypeEmbedding
)
from dit_v2.nn.modulation import FinalMod
from dit_v2.nn.repa import REPA
from dit_v2.nn.vae import VAE
from dit_v2.nn.mlp import MLP

from dit.nn.text_embedder import TextEmbedder
from dit.utils import freeze, sample_discrete_timesteps, sample_step_size

class RFTCore(nn.Module):
    def __init__(self, config : 'ModelConfig'):
        super().__init__()

        self.blocks = StackedDiT(config)

        self.t_embed = TimestepEmbedding(config.d_model)
        self.d_embedder = StepEmbedding(config.d_model, max_steps = config.base_steps)
        #self.pool_embedder = MLP(config.text_d_model, dim_out=config.d_model)

        n_patches = config.sample_size // config.patch_size
        patch_content = config.patch_size * config.patch_size * config.channels
        #self.pos_enc = AbsEmbedding(n_patches**2, config.d_model)

        self.proj_out = nn.Linear(config.d_model, patch_content)
        self.all_ctrl_embed = MLP(config.n_controls, 4 * config.d_model, config.d_model)
        self.final_ctrl_embed = MLP(config.n_controls, 4 * config.d_model, config.d_model)

        self.final_mod = FinalMod(config.d_model)

        self.patch_proj = nn.Conv2d(
            config.channels,
            config.d_model,
            config.patch_size,
            config.patch_size
        )

        self.video_patch_proj = nn.Conv3d(
            config.channels,
            config.d_model,
            kernel_size=(config.temporal_patch_size,config.patch_size,config.patch_size),
            stride=(config.temporal_patch_size,config.patch_size,config.patch_size)
        )

        self.depatchify = lambda x: eo.rearrange(
            x,
            'b (n_p_h n_p_w) (p_h p_w c) -> b c (n_p_h p_h) (n_p_w p_w)',
            n_p_h = n_patches,
            p_h = config.patch_size,
            c = config.channels
        )

        self.prev_frames_embed = TypeEmbedding(config.d_model)
        self.prev_frame_embed = TypeEmbedding(config.d_model)
        self.control_embed = TypeEmbedding(config.d_model)
    
    def forward(self, x, y, z, c, ts, d, output_hidden_states = False):
        # x [b,c,h,w]
        # y [b,c,h,w] (prev frame)
        # z [b,n,c,h,w] (prev frames)
        # c [b,n+2,c] (controls)
        # ts [b,d]
        # d [b,d]

        # Downsample previous frames for efficiency
        z = torch.nn.functional.interpolate(
            z.flatten(0,1),  # Combine batch and sequence dims
            scale_factor=0.25,
            mode='bilinear',
            align_corners=False
        ).unflatten(0, z.shape[:2])  # Restore batch and sequence dims

        x = self.patch_proj(x)
        y = self.patch_proj(y)
        z = self.video_patch_proj(z.transpose(1,2))

        y = self.prev_frame_embed(y)
        z = self.prev_frames_embed(z)

        all_c = self.control_embed(self.all_ctrl_embed(c))
        final_c = self.final_ctrl_embed(c[:,-1])

        cond = self.t_embed(ts) + self.d_embedder(d) + final_c

        ctx = torch.cat([all_c, z, y.unsqueeze(1)], dim = 1)

        out = self.blocks(x, ctx, cond, output_hidden_states=output_hidden_states)
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
        # X is tuple of (prev_frames, input conditioning)
        x, controls = x # controls is [b,n,d]
        prev_frames = x[:,:-2] # [b,n-2,c,h,w]
        prev_frame = x[:,-2] # [b,c,h,w]
        avg_frame = x[:,:-1].mean(1,keepdim=True) # [b,n-1,c,h,w]
        crnt_frame = x[:,-1] # [b,c,h,w]

        #with torch.no_grad():
        #    crnt_frame_orig = self.vae.decode(crnt_frame)
        b = crnt_frame.shape[0]
        if self.config.cfg_prob > 0:
            mask = torch.rand(len(b)) < self.config.cfg_prob
            if mask.any():
                prev_frames = torch.where(
                    mask[:,None,None,None,None],
                    avg_frame[:,None].expand_as(prev_frames),
                    prev_frames
                )
                prev_frame = torch.where(
                    mask[:,None,None,None],
                    avg_frame[:,0],
                    prev_frame
                )
                controls = torch.where(
                    mask[:,None,None],
                    torch.zeros_like(controls),
                    controls
                )
        
        b,c,h,w = crnt_frame.shape

        # Prepare input + target
        with torch.no_grad():
            z = torch.randn_like(crnt_frame)
            t = torch.rand(b, device = x.device, dtype = x.dtype)
            d = torch.full((b,), self.config.base_steps, device = x.device, dtype = x.dtype)

            t_exp = eo.repeat(t, 'b -> b c h w', c=c,h=h,w=w)
            lerpd = x * (1 - t_exp) + z * t_exp
            target = z - x
        
        extra = {}

        pred, h = self.denoise(lerpd, prev_frame, prev_frames, controls, t, d, output_hidden_states=True)
        total_loss = 0.

        diff_loss = F.mse_loss(target, pred)
        extra['diff_loss'] = diff_loss.item()
        total_loss += diff_loss

        if self.training:
            # TODO update self consistency
            if self.config.sc_weight > 0 and sc_targets is not None:
                sc_inputs, sc_targets, sc_ts, sc_ctx, sc_prev_frames, sc_controls, sc_d = sc_targets
                sc_pred = self.denoise(sc_inputs, sc_ctx, sc_prev_frames, sc_controls, sc_ts, sc_d)
                sc_loss = F.mse_loss(sc_targets, sc_pred)
                extra['sc_loss'] = sc_loss.item()
                total_loss += self.config.sc_weight * sc_loss

            if self.repa is not None:
                repa_loss = self.repa(crnt_frame_orig, h[self.config.repa_layer_ind])
                extra['repa_loss'] = repa_loss.item()
                total_loss += self.config.repa_weight * repa_loss
        
        return total_loss, extra


if __name__ == "__main__":
    from .configs import ModelConfig, TrainConfig
    from .data import FrameDataset

    # Initialize configs
    model_config = ModelConfig()
    train_config = TrainConfig()

    # Create model
    model = RectFlowTransformer(model_config)

    # Create dataset and get a batch
    dataset = FrameDataset("./train_data", 
                          image_size=model_config.image_size,
                          frame_count=model_config.temporal_patch_size+1)
    dataloader = dataset.create_loader(batch_size=2, num_workers=0)
    
    # Get first batch
    frames, controls = next(iter(dataloader))
    
    # Split into current frame, previous frame, and previous frames sequence
    crnt_frame = frames[:,-1]  # Last frame
    prev_frame = frames[:,-2]  # Second to last frame
    prev_frames = frames[:,:-2] # All earlier frames
    
    # Try a forward pass
    loss, extra = model(
        (crnt_frame,
        prev_frame, 
        prev_frames,
        controls),
        sc_targets=None
    )
    
    print(f"Loss: {loss.item()}")
    print("Extra metrics:", extra)

