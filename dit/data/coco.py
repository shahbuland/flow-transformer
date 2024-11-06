from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image

# Define the transforms
def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: img.convert('RGB')),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

class CustomCOCODataset(Dataset):
    def __init__(self, image_size=512, split = 'train'):
        self.dataset = load_dataset("HuggingFaceM4/COCO", split = split, trust_remote_code = True)
        self.transform = get_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['sentences']['raw']
        if self.transform:
            image = self.transform(image)
        return image, str(label)