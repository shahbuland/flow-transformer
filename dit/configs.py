from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ModelConfig:
    n_layers : int = 4
    n_heads : int = 6
    d_model : int = 384
    image_size : int = 64
    sample_size : int = 64
    channels : int = 3
    patch_size : int = 4
    use_vae = False
    flash : bool = False
    take_label : bool = True # Take the batch as (pixel_values, label_str) instead of pixel_values

@dataclass
class TrainConfig:
    dataset : str = "mnist"
    target_batch_size : int = 64
    batch_size : int = 16
    epochs : int = 100
    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-4,
        "eps": 1e-7,
        "betas" : (0.9, 0.96)
    })

    # scheduler
    scheduler: Optional[str] = "CosineDecayAfterWarmup"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        "warmup_steps": 400,  # Linear warmup over 1000 steps
        "T_max" : 1000000,
        "eta_min" : 1.0e-6
    })

    log_interval : int = 1
    sample_interval : int = 50
    n_samples : int = 4 # Number of samples to log each time (too many gets crowded)
    sample_prompts = [f"A drawing of the digit {i}" for i in ["one", "two", "three", "four"]]
    grad_clip : float = -1 # Clip grad norms to this value
    

@dataclass
class LoggingConfig:
    run_name : str = "mnist tiny (+cond)"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "mnist_sanity"