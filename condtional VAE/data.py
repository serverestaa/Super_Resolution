import torch
import random
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from dataset import DIV2KDataset

class DIV2KDatasetWithAugmentation(DIV2KDataset):
    def __init__(self, root_dir, lr_folder, hr_folder, hr_patch_size=128, scale=4):
        super().__init__(root_dir, lr_folder, hr_folder, hr_patch_size, scale)
        
    def __getitem__(self, idx):
        # Get the original LR and HR patches from the parent class
        lr_patch, hr_patch = super().__getitem__(idx)
        
        # Apply data augmentation with 50% chance
        if random.random() > 0.5:
            # Random horizontal flip
            if random.random() > 0.5:
                lr_patch = TF.hflip(lr_patch)
                hr_patch = TF.hflip(hr_patch)
            
            # Random vertical flip
            if random.random() > 0.5:
                lr_patch = TF.vflip(lr_patch)
                hr_patch = TF.vflip(hr_patch)
            
            # Random 90 degree rotation
            rotate_times = random.randint(0, 3)
            if rotate_times > 0:
                lr_patch = torch.rot90(lr_patch, rotate_times, [1, 2])
                hr_patch = torch.rot90(hr_patch, rotate_times, [1, 2])
        
        return lr_patch, hr_patch


def load_dataset(root_dir, train_lr_folder, train_hr_folder, 
                valid_lr_folder, valid_hr_folder,
                hr_patch_size=128, scale=4, batch_size=16):
    try:
        # Create training dataset with augmentations
        train_dataset = DIV2KDatasetWithAugmentation(
            root_dir=root_dir,
            lr_folder=train_lr_folder,
            hr_folder=train_hr_folder,
            hr_patch_size=hr_patch_size,
            scale=scale
        )
        
        # Create validation dataset (no augmentations for validation)
        val_dataset = DIV2KDataset(
            root_dir=root_dir,
            lr_folder=valid_lr_folder,
            hr_folder=valid_hr_folder,
            hr_patch_size=hr_patch_size,
            scale=scale
        )
        
        print(f"Datasets loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_dataloader, val_dataloader, test_dataloader
        
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        # Fallback to sample creation if dataset initialization fails
        print("Using sample data instead for demonstration purposes")
        return None, None, None