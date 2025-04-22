import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self,
                 root_dir,
                 lr_folder='DIV2K_train_LR_bicubic',
                 hr_folder='DIV2K_train_HR',
                 hr_patch_size=128,
                 scale=4):
        super().__init__()
        self.hr_dir = os.path.join(root_dir, hr_folder)
        self.lr_dir = os.path.join(root_dir, lr_folder)
 
        hr_files = [f for f in os.listdir(self.hr_dir)
                    if f.lower().endswith(('.png','.jpg','.jpeg'))]
        lr_files = [f for f in os.listdir(self.lr_dir)
                    if f.lower().endswith(('.png','.jpg','.jpeg'))]

        
        lr_map = {
            os.path.splitext(f)[0].lower(): f
            for f in lr_files
        }
 
        self.pairs = []
        for hr_name in hr_files:
            base = os.path.splitext(hr_name)[0]           
            candidate = f"{base}x4"                     
            lr_name = lr_map.get(candidate.lower())
            if not lr_name:
                raise ValueError(f"No LR match for HR file {hr_name}")
            self.pairs.append((
                os.path.join(self.lr_dir, lr_name),
                os.path.join(self.hr_dir, hr_name)
            ))

        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.scale = scale
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
       
        return len(self.pairs)

    def __getitem__(self, idx):
        
        lr_path, hr_path = self.pairs[idx]
 
        hr = Image.open(hr_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
 
        hr_w, hr_h = hr.size
        if hr_w < self.hr_patch_size or hr_h < self.hr_patch_size:
            raise ValueError('HR image smaller than patch size')
        x = random.randint(0, hr_w - self.hr_patch_size)
        y = random.randint(0, hr_h - self.hr_patch_size)
        hr_crop = hr.crop((x, y, x + self.hr_patch_size, y + self.hr_patch_size))
 
        lr_x, lr_y = x // self.scale, y // self.scale
        lr_crop = lr.crop((lr_x, lr_y, lr_x + self.lr_patch_size, lr_y + self.lr_patch_size))
 
        hr_tensor = self.to_tensor(hr_crop)
        lr_tensor = self.to_tensor(lr_crop)

        return lr_tensor, hr_tensor