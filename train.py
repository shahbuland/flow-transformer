from PIL import Image

import torch
from dit.data import create_loader
from dit.nn import RectFlowTransformer
from dit.trainer import Trainer
from dit.configs import ModelConfig, TrainConfig, LoggingConfig
from dit.utils import pretty_print_parameters

if __name__ == "__main__":
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    log_cfg = LoggingConfig()
    
    seed = 42
    torch.manual_seed(seed)

    model = RectFlowTransformer(model_cfg)
    pretty_print_parameters(model)
    trainer = Trainer(train_cfg, log_cfg, model_cfg)
    
    # Create the data loader using the configuration
    train_loader = create_loader(
        dataset_name=train_cfg.dataset,
        batch_size=train_cfg.batch_size,
        image_size=model_cfg.image_size,
        deterministic=True  # This will use a fixed seed internally
    )

    trainer.train(model, train_loader)