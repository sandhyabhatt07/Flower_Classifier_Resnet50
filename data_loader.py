"""
Data loading utilities for the flower classification project
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class FlowerDataset(Dataset):
    """Custom dataset for flower images"""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations
            img_dir (string): Directory with all the images (train/ or val/)
            transform (callable, optional): Transform to be applied on a sample
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Get all class indices from the CSV
        self.classes = sorted(self.data_frame['category'].unique())
        self.class_to_idx = {cls: int(cls) for cls in self.classes}  # numeric mapping (0–13)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        filename = row['filename']  # e.g. 'sunflower/abc.jpg'
        label = int(row['category'])  # numeric label 0–13

        # Construct full image path
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self):
        return sorted(self.class_to_idx.keys())

def get_data_loaders():
    """Create and return dataloaders for train and validation sets"""

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    # Create datasets
    train_dataset = FlowerDataset(
        csv_file=config.TRAIN_CSV,
        img_dir=config.TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = FlowerDataset(
        csv_file=config.VAL_CSV,
        img_dir=config.VAL_DIR,
        transform=val_transform
    )

    # Update number of classes in config
    config.NUM_CLASSES = len(train_dataset.get_classes())

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, train_dataset.get_classes()
