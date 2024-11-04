from torch.utils.data import DataLoader
import torch
from . import (
    mnist,
    imagenet,
    coco
)

def create_loader(dataset_name, batch_size, image_size, deterministic=True, split='train'):
    if dataset_name.lower() == 'mnist':
        dataset = mnist.CustomMNISTDataset(image_size=image_size)
    elif dataset_name.lower() == 'imagenet':
        dataset = imagenet.CustomImageNetDataset(image_size=image_size, split = split)
    elif dataset_name.lower() == 'coco':
        dataset = coco.CustomCOCODataset(image_size=image_size, split = split)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    generator = None
    if deterministic:
        generator = torch.Generator().manual_seed(0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        generator=generator,
        pin_memory = True,
        num_workers = 8,
        drop_last = True
    )