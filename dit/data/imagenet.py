from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image

# Define the transforms
def get_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: img.convert('RGB')),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

class CustomImageNetDataset(Dataset):
    def __init__(self, image_size=224):
        self.dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True)
        self.transform = get_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label
