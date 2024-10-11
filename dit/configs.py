from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ModelConfig:
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 768
    image_size : int = 512
    sample_size : int = 64
    channels : int = 4
    patch_size : int = 4
    use_vae = True
    flash : bool = True
    take_label : bool = True # Take the batch as (pixel_values, label_str) instead of pixel_values
    cfg_prob : float = 0.1

@dataclass
class TrainConfig:
    dataset : str = "coco"
    target_batch_size : int = 256
    batch_size : int = 128
    epochs : int = 100
    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-3,
        "eps": 1.0e-15,
        "betas" : (0.9, 0.96),
        "weight_decay" : 0.00,
        #"precondition_frequency" : 50
    })

    # scheduler
    scheduler: Optional[str] = None#"CosineDecayAfterWarmup"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        "warmup_steps": 400,  # Linear warmup over 1000 steps
        "T_max" : 1000000,
        "eta_min" : 1.0e-6
    })

    checkpoint_root_dir = "checkpoints"

    log_interval : int = 1
    sample_interval : int = 50
    save_interval : int = 2500
    resume : bool = True

    n_samples : int = 4 # Number of samples to log each time (too many gets crowded)
    sample_prompts = ["a dog in a park", "the blue sky", "the ocean", "the beach"]
    grad_clip : float = -1 # Clip grad norms to this value
    normalize_every : int = 1
    
@dataclass
class LoggingConfig:
    run_name : str = "coco 150M (adam, +ngpt, lr=1e-3)"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "mnist_sanity"