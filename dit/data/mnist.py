from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

from PIL import Image

def label_to_sentence(label: int) -> str:
    """
    Maps MNIST labels (0-9) to descriptive sentences.

    Args:
        label (int): The MNIST label (0-9)

    Returns:
        str: A descriptive sentence for the given label
    """
    digit_names = [
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine"
    ]
    
    if 0 <= label <= 9:
        return f"A drawing of the digit {digit_names[label]}"
    else:
        raise ValueError("Label must be an integer between 0 and 9")

# Define the transforms
def get_transform(image_size):
    return transforms.Compose([
        transforms.ToTensor(), # [0,1]
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.Resize((image_size, image_size)),  # Resize to specified size
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Renormalize to [-1,1]
    ])

class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_size=64):
        self.dataset = load_dataset("mnist", split = 'train')

        self.transform = get_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = int(self.dataset[idx]['label'])
        if self.transform:
            image = self.transform(image)
            image = image.repeat(3, 1, 1) # Force RGB channels

        return image, label_to_sentence(label)