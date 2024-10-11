from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ModelConfig:
    n_layers : int = 12
    n_heads : int = 6
    d_model : int = 384
    image_size : int = 64
    sample_size : int = 64
    channels : int = 3
    patch_size : int = 4
    use_vae = False
    flash : bool = False
    take_label : bool = True # Take the batch as (pixel_values, label_str) instead of pixel_values
    cfg_prob : float = 0.1

@dataclass
class TrainConfig:
    dataset : str = "mnist"
    target_batch_size : int = 64
    batch_size : int = 16
    epochs : int = 100
    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-3,
        "eps": 1e-7,
        "betas" : (0.9, 0.96),
        "weight_decay" : 0.0
    })

    # scheduler
    scheduler: Optional[str] = None#"CosineDecayAfterWarmup"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        "warmup_steps": 400,  # Linear warmup over 1000 steps
        "T_max" : 1000000,
        "eta_min" : 1.0e-6
    })

    log_interval : int = 1
    sample_interval : int = 50
    n_samples : int = 8 # Number of samples to log each time (too many gets crowded)
    sample_prompts = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight"
    ]
    grad_clip : float = -1 # Clip grad norms to this value
    normalize_every : int = 1
    

@dataclass
class LoggingConfig:
    run_name : str = "mnist 40M (+ngpt +1e-3 lr)"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "mnist_sanity"