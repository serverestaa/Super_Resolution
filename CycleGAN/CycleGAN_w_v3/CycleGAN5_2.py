import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

HR_TRAIN_DIR = "DIV2K/DIV2K_train_HR"
LR_TRAIN_DIR = "DIV2K/DIV2K_train_LR_bicubic_X4\X4"
HR_VAL_DIR   = "DIV2K/DIV2K_valid_HR"
LR_VAL_DIR   = "DIV2K/DIV2K_valid_LR_bicubic_X4\X4"

BATCH_SIZE   = 1
NUM_WORKERS  = 2
NUM_EPOCHS   = 100
LR           = 2e-4
BETA1        = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambda_idt = 0.1
PATCH_SIZE_LR = 64
PATCH_SIZE_HR = PATCH_SIZE_LR * 4

# ========== Dataset ==========
class DIV2KPairedDataset(Dataset):
    def __init__(self, root_dir,
                 lr_folder='DIV2K_train_LR_bicubic',
                 hr_folder='DIV2K_train_HR',
                 hr_patch_size=PATCH_SIZE_HR,
                 scale=4):
        super().__init__()
        self.hr_dir = os.path.join(root_dir, hr_folder)
        self.lr_dir = os.path.join(root_dir, lr_folder)

        hr_files = sorted([f for f in os.listdir(self.hr_dir)
                           if f.lower().endswith(('.png','.jpg','.jpeg'))])
        lr_files = sorted([f for f in os.listdir(self.lr_dir)
                           if f.lower().endswith(('.png','.jpg','.jpeg'))])

        lr_map = {os.path.splitext(f)[0].lower(): f for f in lr_files}
        self.pairs = []
        for hr_name in hr_files:
            base = os.path.splitext(hr_name)[0]
            candidate = f"{base}x4"
            lr_name = lr_map.get(candidate.lower())
            if lr_name is None:
                continue 
            self.pairs.append((
                os.path.join(self.lr_dir, lr_name),
                os.path.join(self.hr_dir, hr_name)
            ))

        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.scale = scale

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        hr = Image.open(hr_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
  
        hr_w, hr_h = hr.size
        x = np.random.randint(0, hr_w - self.hr_patch_size + 1)
        y = np.random.randint(0, hr_h - self.hr_patch_size + 1)
        hr_crop = hr.crop((x, y, x + self.hr_patch_size, y + self.hr_patch_size))
        lr_crop = lr.crop((x//self.scale, y//self.scale,
                           x//self.scale + self.lr_patch_size,
                           y//self.scale + self.lr_patch_size))

        hr_tensor = self.to_tensor(hr_crop)
        lr_tensor = self.to_tensor(lr_crop)
        return lr_tensor, hr_tensor


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.InstanceNorm2d) and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class GeneratorLR2HR(nn.Module):
    def __init__(self, n_residuals=9):
        super().__init__()
        ngf = 64
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7, 1, 0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_residuals):
            layers.append(ResidualBlock(ngf))
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf, ngf*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(inplace=True)
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, 7, 1, 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class GeneratorHR2LR(nn.Module):
    def __init__(self, n_residuals=9):
        super().__init__()
        ndf = 64
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ndf, 7, 1, 0),
            nn.InstanceNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, ndf*2, 3, 2, 1),
            nn.InstanceNorm2d(ndf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 3, 2, 1),
            nn.InstanceNorm2d(ndf*4),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_residuals):
            layers.append(ResidualBlock(ndf*4))
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ndf*4, 3, 7, 1, 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 64
        layers = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for mult in [2,4,8]:
            layers += [
                nn.Conv2d(ndf*mult//2, ndf*mult, 4, 2, 1),
                nn.InstanceNorm2d(ndf*mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers.append(nn.Conv2d(ndf*8, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# ========== SSIM  VGG‑perceptual loss ==========
vgg = models.vgg19(pretrained=True).features.to(DEVICE).eval()
for p in vgg.parameters():
    p.requires_grad = False

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_idx=21):
        super().__init__()
        self.vgg = vgg
        self.layer_idx = layer_idx
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        sr_norm = (sr + 1) / 2
        hr_norm = (hr + 1) / 2
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)
        std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)
        sr_in = (sr_norm - mean) / std
        hr_in = (hr_norm - mean) / std
        feat_sr = self.vgg[:self.layer_idx](sr_in)
        feat_hr = self.vgg[:self.layer_idx](hr_in)
        return self.criterion(feat_sr, feat_hr)


def evaluate_generator(G, loader, device):
    G.eval()
    psnr, ssim_vals = [], []
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = G(lr)
            sr_np = (sr.cpu()[0].permute(1,2,0).numpy()*0.5+0.5)
            hr_np = (hr.cpu()[0].permute(1,2,0).numpy()*0.5+0.5)
            psnr.append(peak_signal_noise_ratio(hr_np, sr_np, data_range=1))
            ssim_vals.append(structural_similarity(hr_np, sr_np, multichannel=True, data_range=1))
    return np.mean(psnr), np.mean(ssim_vals)

def show_results(G, dataset, device, samples=3):
    for i in range(samples):
        lr, hr = dataset[i]
        with torch.no_grad():
            sr = G(lr.unsqueeze(0).to(device))
        lr_img = lr.permute(1,2,0).numpy()*0.5+0.5
        sr_img = sr.cpu()[0].permute(1,2,0).numpy()*0.5+0.5
        hr_img = hr.permute(1,2,0).numpy()*0.5+0.5
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(lr_img); axs[0].set_title('LR'); axs[0].axis('off')
        axs[1].imshow(sr_img); axs[1].set_title('SR'); axs[1].axis('off')
        axs[2].imshow(hr_img); axs[2].set_title('HR'); axs[2].axis('off')
        plt.show()


def main():
    print("Using device:", DEVICE)
    train_ds = DIV2KPairedDataset(root_dir="DIV2K")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = None
    if os.path.isdir(HR_VAL_DIR) and os.path.isdir(LR_VAL_DIR):
        val_ds = DIV2KPairedDataset(root_dir="DIV2K",
                                    lr_folder="DIV2K_valid_LR_bicubic",
                                    hr_folder="DIV2K_valid_HR")
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)


    G = GeneratorLR2HR().to(DEVICE)
    F = GeneratorHR2LR().to(DEVICE)
    D_X = Discriminator().to(DEVICE)
    D_Y = Discriminator().to(DEVICE)
    G.apply(weights_init)
    F.apply(weights_init)
    D_X.apply(weights_init)
    D_Y.apply(weights_init)

    
    optimizer_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=LR, betas=(BETA1,0.999))
    optimizer_D = optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=LR, betas=(BETA1,0.999))
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=NUM_EPOCHS, eta_min=0)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=NUM_EPOCHS, eta_min=0)

    criterion_gan   = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    ssim_loss       = lambda x,y: 1 - ssim(x, y, data_range=1.0, size_average=True)
    perceptual_loss = VGGPerceptualLoss(layer_idx=21)

    lambda_cycle = 2.0
    lambda_ssim  = 0.1
    lambda_perc  = 0.01
    for epoch in range(1, NUM_EPOCHS+1):
        G.train(); F.train()
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        train_iter = tqdm(train_loader,
                      desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
                      leave=False)
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)

          
            fake_hr = G(lr_imgs)
            rec_lr  = F(fake_hr)
            fake_lr = F(hr_imgs)
            rec_hr  = G(fake_lr)

          
            loss_G_Y = criterion_gan(D_Y(fake_hr), torch.ones_like(D_Y(fake_hr)))
            loss_G_X = criterion_gan(D_X(fake_lr), torch.ones_like(D_X(fake_lr)))
            loss_cycle = (criterion_cycle(rec_lr, lr_imgs)
                          + criterion_cycle(rec_hr, hr_imgs))
            loss_ssim = ssim_loss(fake_hr, hr_imgs)
            loss_perc = perceptual_loss(fake_hr, hr_imgs)

            loss_G = (loss_G_Y + loss_G_X
                      + lambda_cycle * loss_cycle
                      + lambda_ssim  * loss_ssim
                      + lambda_perc  * loss_perc)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            
            loss_DX = 0.5 * (criterion_gan(D_X(hr_imgs), torch.ones_like(D_X(hr_imgs)))
                             + criterion_gan(D_X(fake_lr.detach()), torch.zeros_like(D_X(fake_lr))))
            loss_DY = 0.5 * (criterion_gan(D_Y(hr_imgs), torch.ones_like(D_Y(hr_imgs)))
                             + criterion_gan(D_Y(fake_hr.detach()), torch.zeros_like(D_Y(fake_hr))))
            loss_D = loss_DX + loss_DY

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            g_loss_epoch += loss_G.item()
            d_loss_epoch += loss_D.item()
            train_iter.set_postfix(G=loss_G.item(), D=loss_D.item())
        scheduler_G.step()
        scheduler_D.step()

        print(f"Epoch {epoch}/{NUM_EPOCHS}  "
              f"Loss_G: {g_loss_epoch/len(train_loader):.4f}  "
              f"Loss_D: {d_loss_epoch/len(train_loader):.4f}")

        if val_loader:
            psnr_val, ssim_val = evaluate_generator(G, val_loader, DEVICE)
            print(f"  Val PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

  
    torch.save(G.state_dict(), "generator_G_LR2HR_x4.pth")
    torch.save(F.state_dict(), "generator_F_HR2LR_x4.pth")

   
    if val_loader:
        show_results(G, val_ds, DEVICE)
    else:
        show_results(G, train_ds, DEVICE)

if __name__ == "__main__":
    main()