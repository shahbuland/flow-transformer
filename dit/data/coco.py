from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image

# Load the COCO dataset
coco_dataset = load_dataset("HuggingFaceM4/COCO", split="train")

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
    def __init__(self, image_size=256):
        self.dataset = coco_dataset
        self.transform = get_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['sentences']['raw']
        if self.transform:
            image = self.transform(image)
        return image, str(label)


# Add this at the end of the file, outside the class definition
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Create an instance of the dataset
    dataset = CustomCOCODataset(image_size=256)

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Get a sample batch

    sample_image, sample_label = next(iter(loader))

    # Print the shape of the image and the label
    print(f"Image shape: {sample_image.shape}")
    print(sample_label)

