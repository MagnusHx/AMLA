import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np


class GreekCharacterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to the Greek directory containing character folders
            transform: Optional transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all image paths and their corresponding labels
        for char_idx, char_folder in enumerate(sorted(self.root_dir.glob('character*'))):
            for img_path in sorted(char_folder.glob('*.png')):
                self.images.append(img_path)
                self.labels.append(char_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(greek_dir, batch_size=32, train_split=0.8, img_size=64):
    """
    Create train and validation DataLoaders for Greek character dataset.
    
    Args:
        greek_dir: Path to the Greek directory
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training
        img_size: Size to resize images to
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = GreekCharacterDataset(greek_dir, transform=transform)
    
    # Split into train and validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
