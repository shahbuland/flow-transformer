from datasets import load_dataset
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

class CustomImageNetDataset(Dataset):
    def __init__(self, image_size=256, split = 'train'):
        self.dataset = load_dataset("ILSVRC/imagenet-1k", split=split, trust_remote_code=True, cache_dir = "../.cache/huggingface/datasets/ILSVRC___imagenet-1k")
        self.transform = get_transform(image_size)
        self.label_names = self.dataset.features['label'].names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row['image']
        label_idx = row['label']
        label = self.label_names[label_idx]

        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    dataset = CustomImageNetDataset(image_size=256)
    sample, label = dataset[0]
    print(f"Sample type: {type(sample)}")
    print(f"Sample tensor shape: {sample.shape}")
    print(label)

