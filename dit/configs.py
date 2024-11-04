from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ModelConfig:
    # Transformer
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 768
    flash : bool = True
    normalized : bool = False

    # input/latent
    image_size : int = 256
    sample_size : int = 32
    channels : int = 4
    patch_size : int = 2
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

    # Shortcut models
    sc_weight : float = 1.0
    sc_cfg : float = 3.5
    sc_batch_frac : float = 0.25 # What percentage of training batch?
    delay_sc : int = 0 # Delay sc to this many steps after training starts
    base_steps : int = 128

@dataclass
class TrainConfig:
    dataset : str = "coco"
    target_batch_size : int = 256
    batch_size : int = 256
    epochs : int = 100
    # optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-4,
        "eps": 1.0e-15,
        "betas" : (0.9, 0.95),
        "weight_decay" : 0.1,
        #"precondition_frequency" : 50
    })

    # scheduler
    scheduler: Optional[str] = None#"CosineDecay"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        "T_max" : 15000,
        "eta_min" : 1.0e-4
    })

    # Saving
    checkpoint_root_dir = "checkpoints"

    log_interval : int = 1
    sample_interval : int = 100
    save_interval : int = 20000
    val_interval : int = 1000
    resume : bool = False

    # Sampling
    n_samples : int = 16 # Number of samples to log each time (too many gets crowded)
    sample_prompts = [
        "golden retriever",
        "tabby cat",
        "school bus",
        "tennis ball",
        "acoustic guitar",
        "monarch butterfly",
        "great white shark",
        "bald eagle",
        "red panda",
        "pineapple",
        "fire truck",
        "grand piano",
        "mountain bike",
        "polar bear",
        "peacock",
        "zebra"
    ]
    
    # Validating
    val_batch_mult = 2

    grad_clip : float = -1 # Clip grad norms to this value
    
@dataclass
class LoggingConfig:
    run_name : str = "coco 150M (RoPE, -Abs_enc, +Linformer)"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "mnist_sanity"

@dataclass
class SamplerConfig:
    n_steps : int = 128
    cfg_scale : float = 3.5
    fast_steps : int = 1
