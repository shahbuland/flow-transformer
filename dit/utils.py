import math
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch import nn
import einops as eo 

def count_parameters(model):
    """
    Count and print the number of learnable parameters in a model.
    
    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def pretty_print_parameters(model):
    """
    Same as above func but doesn't return anything, just prints in a pretty format
    """
    params = count_parameters(model)
    formatted_params = params
    if params < 1_000_000:
        formatted_params = f"{params // 1000}K"
    elif params < 1_000_000_000:
        formatted_params = f"{params // 1_000_000}M"
    elif params < 1_000_000_000_000:
        formatted_params = f"{params // 1_000_000_000}B"
    else:
        formatted_params = f"{params // 1_000_000_000_000}T"
    
    print(f"Model has {formatted_params} trainable parameters.")

def freeze(module: nn.Module):
    """
    Set all parameters in a module to not require gradients.
    
    Args:
        module (nn.Module): The PyTorch module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module: nn.Module):
    """
    Set all parameters in a module to require gradients.
    
    Args:
        module (nn.Module): The PyTorch module to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True

def get_scheduler_cls(scheduler_name: str):
    """
    Returns the scheduler class based on the given name.

    Args:
        scheduler_name (str): The name of the scheduler.

    Returns:
        _LRScheduler: The scheduler class.

    Raises:
        ValueError: If an invalid scheduler name is provided.
    """
    scheduler_map = {
        "CosineDecayAfterWarmup": CosineDecayAfterWarmup,
        "CosineDecay": CosineDecay
    }

    scheduler_cls = scheduler_map.get(scheduler_name)
    if scheduler_cls is None:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")
    
    return scheduler_cls

class CosineDecayAfterWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineDecayAfterWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        elif self.last_epoch < self.warmup_steps + self.T_max:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / self.T_max
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]
        else:
            # Constant minimum learning rate
            return [self.eta_min for _ in self.base_lrs]

class CosineDecay(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_max:
            # Cosine decay
            progress = self.last_epoch / self.T_max
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]
        else:
            # Constant minimum learning rate
            return [self.eta_min for _ in self.base_lrs]
            
import time

class Stopwatch:
    def __init__(self):
        self.start_time = None

    def reset(self):
        """Prime the stopwatch for measurement."""
        self.start_time = time.time()

    def hit(self, samples: int) -> float:
        """
        Measure the average time per 1000 samples since the last reset.

        Args:
            samples (int): The number of samples processed.

        Returns:
            float: The time in seconds per 1000 samples.
        """
        if self.start_time is None:
            raise ValueError("Stopwatch must be reset before calling hit.")

        elapsed_time = time.time() - self.start_time
        avg_time_per_sample = elapsed_time / samples
        return avg_time_per_sample * 1000  # Return time per 1000 samples

def mimetic_init(qkv: nn.Linear, proj: nn.Linear, n_heads : int):
    """
    Initialize parameters in qkv and proj linear layers using mimetic initialization.

    :param qkv: Linear layer for query, key, and value projections
    :param proj: Linear layer for output projection
    :param n_heads: Number of attention heads
    """
    alpha_1 = 0.7
    beta_1 = 0.7
    alpha_2 = 0.4
    beta_2 = 0.4

    d = proj.in_features
    k = d // n_heads
    z_1 = torch.randn(d, d) / d
    z_2 = torch.randn(d, d) / d
    I = torch.eye(d)

    mat_1 = alpha_1 * z_1 + beta_1 * I
    u_1, sigma_1, v_1 = torch.svd(mat_1)
    
    sigma_1_diag = torch.diag(sigma_1)
    w_v = u_1 @ sigma_1_diag
    w_proj = v_1 @ torch.sqrt(sigma_1_diag)

    mat_2 = alpha_2 * z_2 + beta_2 * I
    u_2, sigma_2, v_2 = torch.svd(mat_2)
    
    sigma_2_diag = torch.diag(sigma_2)
    w_q = u_2[:,:k] @ torch.sqrt(sigma_2_diag[:k,:k])
    w_k = v_2[:,:k] @ torch.sqrt(sigma_2_diag[:k,:k])

    # Move w_q, w_k, and w_v to the same device as the layers
    w_q = w_q.to(qkv.weight.device)
    w_k = w_k.to(qkv.weight.device)
    w_v = w_v.to(qkv.weight.device)
    w_proj = w_proj.to(proj.weight.device)

    w_q = eo.repeat(w_q, 'd_in d_head -> d_in (n_heads d_head)', n_heads = n_heads)
    w_k = eo.repeat(w_k, 'd_in d_head -> d_in (n_heads d_head)', n_heads = n_heads)
    w_q = w_q.T # [d_out, d_in]
    w_k = w_k.T # [d_out, d_in]

    # Set weights for qkv
    with torch.no_grad():
        qkv.weight.data[:d, :] = w_q
        qkv.weight.data[d:2*d, :] = w_k
        qkv.weight.data[2*d:, :] = w_v

    # Set weights and bias for proj
    with torch.no_grad():
        proj.weight.data = w_proj.T
        if proj.bias is not None:
            proj.bias.data.zero_()

def truncated_normal_init(module: nn.Module, std: float = 0.02):
    """
    Initialize the parameters of a module using a truncated normal distribution.
    
    Args:
        module (nn.Module): The PyTorch module to initialize.
        std (float): The standard deviation of the normal distribution. Default is 0.02.
    """
    for p in module.parameters():
        nn.init.trunc_normal_(p, mean=0.0, std=std, a=-std*2, b=std*2)

def normal_init(module: nn.Module, std: float = 0.02):
    """
    Initialize the parameters of a module using a normal distribution.
    
    Args:
        module (nn.Module): The PyTorch module to initialize.
        std (float): The standard deviation of the normal distribution. Default is 0.02.
    """
    for p in module.parameters():
        nn.init.normal_(p, mean=0.0, std=std)

# Optimizers stuff
from .soap import SOAP

def get_extra_optimizer(name):
    if name.lower() == "soap":
        return SOAP