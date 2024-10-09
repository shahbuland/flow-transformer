from diffusers import FlowMatchEulerDiscreteScheduler
import wandb
import einops as eo
from torchtyping import TensorType
import torch

class Sampler:
    def __init__(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3)

    @torch.no_grad()
    def sample(self, n_samples, model, prompts = None, n_steps = 40):
        c = None
        if prompts is not None:
            assert len(prompts) == n_samples
            c = model.encode_text(prompts)
            
        sample_shape = (model.config.channels, model.config.sample_size, model.config.sample_size)
        sample_shape = (n_samples,) + sample_shape
        self.scheduler.set_timesteps(n_steps)

        timesteps = self.scheduler.timesteps / 1000
        sigmas = self.scheduler.sigmas

        noisy = torch.randn(*sample_shape)

        # Get device and dtype from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Move noisy, timesteps, and sigmas to the same device and dtype as the model
        noisy = noisy.to(device=device, dtype=dtype)
        timesteps = timesteps.to(device=device, dtype=dtype)
        sigmas = sigmas.to(device=device, dtype=dtype)

        if c is not None:
            c = c.to(device=device, dtype=dtype)
            
        for i, t in enumerate(timesteps):
            dt = sigmas[i+1] - sigmas[i]
            pred = model.denoise(noisy, t, c)
            noisy += pred * dt
        
        if model.vae is None:
            return noisy
        else:
            return model.vae.decode(noisy)

def to_wandb_image(x : TensorType["c", "h", "w"], caption : str = ""):
    """
    Turn tensor into wandb image for sampling
    """
    x = eo.rearrange(x, 'c h w -> h w c')
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    x = x.detach().cpu().numpy()
    return wandb.Image(x, caption = caption)

def to_wandb_batch(x):
    return [to_wandb_image(x_i) for x_i in x]