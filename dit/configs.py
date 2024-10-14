from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ModelConfig:
    # Transformer
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 768
    flash : bool = True

    # input/latent
    image_size : int = 512
    sample_size : int = 64
    channels : int = 4
    patch_size : int = 4
    use_vae = True
    
    # Guidance
    take_label : bool = True # Take the batch as (pixel_values, label_str) instead of pixel_values
    text_d_model : int = 512 # Hidden size of text embedding model
    cfg_prob : float = 0.1

    # REPA
    repa_weight : float = 1.0
    repa_batch_size : int = 32
    repa_layer_ind : int = 4
    repa_pool_factor : int = 1 # If 256 patches, matching dinov2small, set to 1, if doing 1024, set to 2

@dataclass
class TrainConfig:
    dataset : str = "coco"
    target_batch_size : int = 256
    batch_size : int = 128
    epochs : int = 100
    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-2,
        "eps": 1.0e-15,
        "betas" : (0.9, 0.96),
        "weight_decay" : 0.00,
        #"precondition_frequency" : 50
    })

    # scheduler
    scheduler: Optional[str] = "CosineDecay"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        "T_max" : 1000000,
        "eta_min" : 5.0e-6
    })

    # Saving
    checkpoint_root_dir = "checkpoints"

    log_interval : int = 1
    sample_interval : int = 50
    save_interval : int = 2500
    val_interval : int = 1000
    resume : bool = False

    # Sampling
    n_samples : int = 4 # Number of samples to log each time (too many gets crowded)
    sample_prompts = ["a dog in a park", "the blue sky", "the ocean", "the beach"]
    
    # Validating
    val_batch_mult = 4

    grad_clip : float = -1 # Clip grad norms to this value
    normalize_every : int = 1
    
@dataclass
class LoggingConfig:
    run_name : str = "coco 150M (repa, lr=1e-2)"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "mnist_sanity"

@dataclass
class SamplerConfig:
    n_steps : int = 100
    cfg_scale : float = 1.5