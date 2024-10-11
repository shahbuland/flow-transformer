from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ModelConfig:
    n_layers : int = 16
    n_heads : int = 16
    d_model : int = 1024
    image_size : int = 256
    sample_size : int = 32
    channels : int = 16
    patch_size : int = 2
    use_vae = True
    flash : bool = True
    take_label : bool = True # Take the batch as (pixel_values, label_str) instead of pixel_values
    cfg_prob = 0.1

@dataclass
class TrainConfig:
    dataset : str = "coco"
    target_batch_size : int = 64
    batch_size : int = 64
    epochs : int = 100
    # optimizer
    opt : str = "soap"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 2.0e-4,
        "eps": 1e-6,
        "betas" : (0.9, 0.96),
        "weight_decay" : 0.0,
        "precondition_frequency" : 50
    })

    # scheduler
    scheduler: Optional[str] = None#"CosineDecayAfterWarmup"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        "warmup_steps": 400,  # Linear warmup over 1000 steps
        "T_max" : 1000000,
        "eta_min" : 1.0e-6
    })

    checkpoint_root_dir : str = "checkpoints"
    resume : bool = False

    log_interval : int = 1
    sample_interval : int = 50
    save_interval: int = 10000
    n_samples : int = 8 # Number of samples to log each time (too many gets crowded)
    sample_prompts = [
        "dog in park",
        "colorful bird on a tree branch",
        "red bicycle leaning against a white fence",
        "sandy beach",
        "coffee on a wooden table",
        "cat sleeping on a windowsill",
        "person holding an umbrella in the rain",
        "Fresh fruits arranged in a wicker basket"
    ]
    grad_clip : float = -1 # Clip grad norms to this value
    

@dataclass
class LoggingConfig:
    run_name : str = "COCO_27M"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "mnist_sanity"