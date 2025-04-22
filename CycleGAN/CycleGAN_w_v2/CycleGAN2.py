import os
import random
import math
import urllib.request
import zipfile
from PIL import Image

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

# ==================== 1) Download DIV2K ====================
def download_div2k(root="div2k"):
    """
    Скачивает и распаковывает официальный DIV2K_train_HR.zip и DIV2K_train_LR_bicubic_X4.zip
    Возвращает пути к папке HR и к папке LR (X4).
    """
    os.makedirs(root, exist_ok=True)

    
    hr_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    lr_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"

    
    hr_zip = os.path.join(root, "DIV2K_train_HR.zip")
    hr_folder = os.path.join(root, "DIV2K_train_HR")
    if not os.path.exists(hr_folder):
        if not os.path.exists(hr_zip):
            print("Скачиваю HR zip...")
            urllib.request.urlretrieve(hr_url, hr_zip)
        print("Распаковываю HR zip...")
        with zipfile.ZipFile(hr_zip, 'r') as z:
            z.extractall(root)
    else:
        print("HR уже распакован.")

    # LR zip
    lr_zip = os.path.join(root, "DIV2K_train_LR_bicubic_X4.zip")
    lr_folder_parent = os.path.join(root, "DIV2K_train_LR_bicubic")
    lr_folder = os.path.join(lr_folder_parent, "X4") 
    if not os.path.exists(lr_folder):
        if not os.path.exists(lr_zip):
            print("Скачиваю LR zip...")
            urllib.request.urlretrieve(lr_url, lr_zip)
        print("Распаковываю LR zip...")
        with zipfile.ZipFile(lr_zip, 'r') as z:
            z.extractall(root)
    else:
        print("LR уже распакован.")

    return hr_folder, lr_folder


# ==================== 2) Dataset: unpaired LR vs HR ====================
class UnpairedImageDataset(Dataset):
    """
    Непарный датасет, где мы имеем 2 папки:
    - root_LR: domain A (Low-Resolution)
    - root_HR: domain B (High-Resolution)
    При каждом __getitem__() берётся случайный файл из LR-папки и случайный файл из HR-папки.
    """
    def __init__(self, root_LR, root_HR, transform_A=None, transform_B=None):
        super().__init__()
        self.root_LR = root_LR
        self.root_HR = root_HR
        self.files_LR = sorted([
            os.path.join(root_LR, f) for f in os.listdir(root_LR)
            if f.lower().endswith(('png','jpg','jpeg','bmp','tif'))
        ])
        self.files_HR = sorted([
            os.path.join(root_HR, f) for f in os.listdir(root_HR)
            if f.lower().endswith(('png','jpg','jpeg','bmp','tif'))
        ])
        self.transform_A = transform_A
        self.transform_B = transform_B

        if len(self.files_LR) == 0 or len(self.files_HR) == 0:
            raise RuntimeError("В одной из папок нет изображений (LR или HR).")

    def __len__(self):
        return max(len(self.files_LR), len(self.files_HR))

    def __getitem__(self, idx):
        lr_path = random.choice(self.files_LR)
        hr_path = random.choice(self.files_HR)

        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')

        if self.transform_A:
            lr_img = self.transform_A(lr_img)
        if self.transform_B:
            hr_img = self.transform_B(hr_img)

        return lr_img, hr_img


# ==================== 3) CycleGAN Model Definitions ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


#  G: LR -> HR (4x)
class GeneratorSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual=9, base_channels=64):
        super(GeneratorSR, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True)
        )
        res_blocks = []
        for _ in range(num_residual):
            res_blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()  

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.res_blocks(out1)
        out3 = self.up1(out2)
        out4 = self.up2(out3)
        out_final = self.conv_out(out4)
        return self.tanh(out_final)


#  F: HR -> LR
class GeneratorDown(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super(GeneratorDown, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            # Down x2
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4) 
        )

        self.out_conv = nn.Conv2d(base_channels*4, out_channels, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.down(x)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return self.tanh(x)


#  (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*4, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

def init_weights(net, scale=0.02):
    """Простейшая инициализация. Можно заменить на другую при желании."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

from tqdm import tqdm
def train_cyclegan_sr(lr_folder, hr_folder,
                      batch_size=4, n_epochs=20,
                      lambda_cycle=10.0,
                      lambda_identity=0.0, 
                      device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # "unpaired": domain A = LR, domain B = HR
    transform_A = transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    transform_B = transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    dataset = UnpairedImageDataset(lr_folder, hr_folder,
                                   transform_A=transform_A,
                                   transform_B=transform_B)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

   
    G = GeneratorSR().to(device)      # LR->HR
    F = GeneratorDown().to(device)    # HR->LR
    D_HR = Discriminator().to(device) 
    D_LR = Discriminator().to(device)
    for module in D_HR.modules():
        if isinstance(module, nn.Conv2d):
            nn.utils.spectral_norm(module)

    for module in D_LR.modules():
        if isinstance(module, nn.Conv2d):
            nn.utils.spectral_norm(module)
    init_weights(G, 0.02)
    init_weights(F, 0.02)
    init_weights(D_HR, 0.02)
    init_weights(D_LR, 0.02)

   
    mse_criterion = nn.MSELoss()
    l1_criterion  = nn.L1Loss()

    
    optimizer_G = torch.optim.Adam(
        list(G.parameters()) + list(F.parameters()),
        lr=5e-5, betas=(0.5, 0.999)
    )
    optimizer_D_HR = torch.optim.Adam(D_HR.parameters(), lr=5e-5, betas=(0.5, 0.999)) #увеличить learning rate(maybe)
    optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=5e-5, betas=(0.5, 0.999))
    def lambda_rule(epoch):
       
        decay_start = n_epochs // 2
        if epoch <= decay_start:
            return 1.0
        else:
        
            extra = epoch - decay_start
         
            decay_len = n_epochs - decay_start
            return 1.0 - float(extra) / float(decay_len)

    scheduler_G    = LambdaLR(optimizer_G,    lr_lambda=lambda_rule)
    scheduler_D_HR = LambdaLR(optimizer_D_HR, lr_lambda=lambda_rule)
    scheduler_D_LR = LambdaLR(optimizer_D_LR, lr_lambda=lambda_rule)

  
    for epoch in range(1, n_epochs+1):
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{n_epochs}]", leave=True)
        for i, (real_A, real_B) in enumerate(pbar, start=1):
            real_A = real_A.to(device)  # LR
            real_B = real_B.to(device)  # HR

            # ---------------------
            # 1) Train Generators
            # ---------------------
            optimizer_G.zero_grad()

            # G: A->B, F: B->A
            fake_B = G(real_A)
            recov_A = F(fake_B)

            fake_A = F(real_B)
            recov_B = G(fake_A)

            # GAN loss: G 
            pred_fake_B = D_HR(fake_B)
            loss_g_gan = mse_criterion(pred_fake_B, torch.ones_like(pred_fake_B))

            # GAN loss: F
            pred_fake_A = D_LR(fake_A)
            loss_f_gan = mse_criterion(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle-consistency
            loss_cycle_A = l1_criterion(recov_A, real_A)
            loss_cycle_B = l1_criterion(recov_B, real_B)

            # Identity 
            loss_id_A = 0.0
            loss_id_B = 0.0
            if lambda_identity > 0:
                # G(real_B) ~ real_B
                id_B = G(real_B)
                loss_id_B = l1_criterion(id_B, real_B)
                # F(real_A) ~ real_A
                id_A = F(real_A)
                loss_id_A = l1_criterion(id_A, real_A)

            loss_G_total = (loss_g_gan + loss_f_gan +
                            lambda_cycle*(loss_cycle_A + loss_cycle_B) +
                            lambda_identity*(loss_id_A + loss_id_B))

            loss_G_total.backward()
            optimizer_G.step()

            # -------------------------
            # 2) Train Discriminator D_HR
            # -------------------------
            optimizer_D_HR.zero_grad()

            # Real HR
            pred_real_B = D_HR(real_B)
            loss_D_real = mse_criterion(pred_real_B, torch.ones_like(pred_real_B))

            # Fake HR
            pred_fake_B = D_HR(fake_B.detach())
            loss_D_fake = mse_criterion(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_D_HR = 0.5*(loss_D_real + loss_D_fake)
            loss_D_HR.backward()
            optimizer_D_HR.step()

            # -------------------------
            # 3) Train Discriminator D_LR
            # -------------------------
            optimizer_D_LR.zero_grad()

            # Real LR
            pred_real_A = D_LR(real_A)
            loss_D_real = mse_criterion(pred_real_A, torch.ones_like(pred_real_A))

            # Fake LR
            pred_fake_A = D_LR(fake_A.detach())
            loss_D_fake = mse_criterion(pred_fake_A, torch.zeros_like(pred_fake_A))

            loss_D_LR = 0.5*(loss_D_real + loss_D_fake)
            loss_D_LR.backward()
            optimizer_D_LR.step()
            if i % 200 == 0:
                with torch.no_grad():
                    example_lr = real_A[:1]  
                    example_hr = real_B[:1] 
                    fake_hr = G(example_lr)  # LR->FakeHR
                    fake_lr = F(example_hr)  # HR->FakeLR

               
                fake_hr_01 = 0.5*(fake_hr + 1.0)
                fake_lr_01 = 0.5*(fake_lr + 1.0)

                save_image(fake_hr_01, f"D:\samples/fakeHR_epoch{epoch}_batch{i}.png")
                save_image(fake_lr_01, f"D:\samples/fakeLR_epoch{epoch}_batch{i}.png")
            pbar.set_postfix({
                "G_loss": f"{loss_G_total.item():.4f}",
                "D_HR": f"{loss_D_HR.item():.4f}",
                "D_LR": f"{loss_D_LR.item():.4f}",
                "cycleA": f"{loss_cycle_A.item():.4f}",
                "cycleB": f"{loss_cycle_B.item():.4f}"
            })
        scheduler_G.step()
        scheduler_D_HR.step()
        scheduler_D_LR.step()
        torch.cuda.empty_cache()
   
    torch.save(G.state_dict(), "G_LR2HR.pth")
    torch.save(F.state_dict(), "F_HR2LR.pth")
    torch.save(D_HR.state_dict(), "D_HR.pth")
    torch.save(D_LR.state_dict(), "D_LR.pth")

    print("Тренировка закончена!")



if __name__ == "__main__":
    hr_folder, lr_folder = download_div2k("div2k")

    train_cyclegan_sr(
        lr_folder=lr_folder,
        hr_folder=hr_folder,
        batch_size=2,
        n_epochs=40,  
        lambda_cycle=10.0,
        lambda_identity=0.0,  
        device="cuda"
    )

 