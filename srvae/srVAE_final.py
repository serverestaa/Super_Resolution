import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm


# train_lr_dir = "ML Train and Valid data/Train/DIV2K_train_LR_bicubic/X4"
# train_hr_dir = "ML Train and Valid data/Train/DIV2K_train_HR"
# val_lr_dir = "ML Train and Valid data/Val/DIV2K_valid_LR_bicubic/X4"
# val_hr_dir = "ML Train and Valid data/Val/DIV2K_valid_HR"

# --- 1. Dataset Definition ---
class SRDataset(Dataset):
    """
    PyTorch Dataset for super-resolution that loads low-resolution and high-resolution image pairs.
    It optionally performs random cropping and augmentation for training.
    """
    def __init__(self, lr_dir, hr_dir, patch_size=128, training=True):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.training = training
        # Get list of files
        # Assume HR images have names like "0001.png" and LR like "0001x4.png" or similar
        # We will match by file prefix (before extension or 'x4').
        hr_files = sorted([f for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))])
        lr_files = sorted([f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))])
        # Filter or match files: ensure lr and hr lists correspond by index
        # (This assumes the file ordering corresponds, which should be true if naming is consistent.)
        self.hr_files = hr_files
        self.lr_files = lr_files

        # Image transformation to tensor
        self.to_tensor = transforms.ToTensor()  # converts PIL image [0,255] to float tensor [0,1]

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load images

        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        lr_w, lr_h = lr_img.size  # note: PIL uses (width, height)
        hr_w, hr_h = hr_img.size

        if self.training:
            # Random crop:
            # Ensure patch_size is divisible by 4 for 4x downscaling.
            ps = self.patch_size
            lr_ps = ps // 4  # corresponding LR patch size
            # Random top-left for HR patch
            if hr_w < ps or hr_h < ps:
                # If image is smaller than patch (unlikely in DIV2K), resize or adjust
                hr_img = hr_img.resize((max(hr_w, ps), max(hr_h, ps)), Image.BICUBIC)
                lr_img = lr_img.resize((hr_img.width//4, hr_img.height//4), Image.BICUBIC)
                hr_w, hr_h = hr_img.size
            x = np.random.randint(0, hr_w - ps + 1)
            y = np.random.randint(0, hr_h - ps + 1)
            # Crop HR and corresponding LR region
            hr_crop = hr_img.crop((x, y, x+ps, y+ps))
            # Corresponding LR crop (note: LR is exactly 1/4 size of HR in DIV2K bicubic setting)
            lr_crop = lr_img.crop((x//4, y//4, x//4 + lr_ps, y//4 + lr_ps))
            # Random flips and rotations for augmentation
            if np.random.rand() < 0.5:
                hr_crop = hr_crop.transpose(Image.FLIP_LEFT_RIGHT)
                lr_crop = lr_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.5:
                hr_crop = hr_crop.transpose(Image.FLIP_TOP_BOTTOM)
                lr_crop = lr_crop.transpose(Image.FLIP_TOP_BOTTOM)
            # (We could also do 90-degree rotations with another random chance)
            lr_img, hr_img = lr_crop, hr_crop

        # Convert to tensor (0-1 range)
        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)
        return lr_tensor, hr_tensor

# --- 2. VGG Perceptual Feature Extractor ---
class VGGFeatureExtractor(nn.Module):
    """
    Uses a pretrained VGG19 network to extract intermediate features for perceptual loss.
    By default, outputs features from layers ReLU4_4 and ReLU5_4 (indices 26 and 35 in VGG19).
    """
    def __init__(self, layers=(26, 35)):
        super(VGGFeatureExtractor, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # Freeze VGG weights:
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        # Register mean and std buffers for normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        # Normalize input to VGG expected range
        x = (x - self.mean) / self.std
        features = []
        # Run through VGG19 layers and capture the specified outputs
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
            if i >= max(self.layers):
                break
        return features

# --- 3. srVAE Model Definition (Encoder, Prior, Decoder) ---
class ConvEncoder(nn.Module):
    """
    Encoder network: extracts features from input image(s) and produces latent distribution (mu, logvar).
    Supports variable input size by using adaptive pooling before the fully-connected layers.
    """
    def __init__(self, in_channels, base_channels=64, latent_dim=512):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        # Convolutional feature extractor
        layers = []
        # Use a sequence of conv blocks with increasing channels and occasional downsampling
        # (No BatchNorm as per SR best-practices, use ReLU activations)
        channels = base_channels
        # Block 1: conv (stride 1)
        layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Block 2: conv (stride 2) downsample
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Block 3: increase channels, stride 1
        layers.append(nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        channels *= 2
        # Block 4: downsample
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Block 5: increase channels
        layers.append(nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        channels *= 2
        # Block 6: downsample
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Block 7: increase channels
        layers.append(nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        channels *= 2
        # Block 8: downsample
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Now channels = base_channels * 8 (after 4 downs)
        self.conv_layers = nn.Sequential(*layers)
        # Adaptive pooling to a fixed small size (e.g., 4x4) to handle arbitrary input resolutions
        self.adap_pool = nn.AdaptiveAvgPool2d((4,4))
        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Linear(channels * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(channels * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        # Pool to fixed spatial size
        x = self.adap_pool(x)
        x = torch.flatten(x, start_dim=1)  # flatten to (B, features)
        # Linear layers to produce mu and log-var
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """
    Decoder network: generates a high-res residual image from latent code and low-res image.
    Implements progressive 2x upsampling using sub-pixel convolution (PixelShuffle).
    """
    def __init__(self, base_channels=64, latent_dim=512):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        # Fully-connected layer to project latent z to a feature map (start of decoder)
        self.fc = nn.Linear(latent_dim, base_channels*8 * 4 * 4)
        # Convolutional layers for upsampling (using PixelShuffle for 2x upscaling per block)
        # After fc projection, feature map has base_channels*8 channels at 4x4 spatial.
        self.conv1 = nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1)
        # Upsample blocks: conv outputting r^2*C channels then pixel shuffle (r=2)
        self.conv2 = nn.Conv2d(base_channels*8, base_channels*4 * 4, kernel_size=3, padding=1)   # prepares 2x up (C->4*C_out)
        self.conv3 = nn.Conv2d(base_channels*4, base_channels*2 * 4, kernel_size=3, padding=1)   # another 2x up
        self.conv4 = nn.Conv2d(base_channels*2, base_channels*1 * 4, kernel_size=3, padding=1)   # final 2x up (to base*1 channels)
        self.conv_out = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)  # output residual RGB image
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_lr, z):
        # x_lr: Low-res input image (tensor Bx3xhxw)
        # z: sampled latent vector (B x latent_dim)
        batch_size = z.size(0)
        # 1. Project latent to feature map
        feat = self.fc(z)                          # (B, base*8 * 4 * 4)
        feat = feat.view(batch_size, -1, 4, 4)     # reshape to (B, base*8, 4, 4)
        feat = self.relu(feat)
        # 2. Convolution + upsampling stages
        feat = self.relu(self.conv1(feat))         # (B, base*8, 4, 4)
        # Upsample by 2x (4->8) via PixelShuffle
        feat = self.relu(F.pixel_shuffle(self.conv2(feat), upscale_factor=2))  # (B, base*4, 8, 8)
        feat = self.relu(F.pixel_shuffle(self.conv3(feat), upscale_factor=2))  # (B, base*2, 16, 16)
        feat = self.relu(F.pixel_shuffle(self.conv4(feat), upscale_factor=2))  # (B, base*1, 32, 32) – note: if input LR was 32x32, now 32*2^3 = 256?
        # (Actually, the output spatial depends on input; we'll handle exact output size by adding the skip connection from bicubic.)
        residual = self.conv_out(feat)            # (B, 3, H, W) residual output
        # 3. Add bicubic upsampled LR to get final output
        x_up = F.interpolate(x_lr, scale_factor=4, mode='bicubic', align_corners=False)  # upsample input to HR size
        if residual.shape[-2:] != x_up.shape[-2:]:
            # Upsample the residual to the HR size (128×128 in your training setup)
            residual = F.interpolate(residual,
                                     size=x_up.shape[-2:],  # (H, W) of x_up
                                     mode="bilinear",  # or 'nearest'
                                     align_corners=False)
        output = x_up + residual
        return output

class SRVAE(nn.Module):
    """
    The complete Super-Resolution VAE model, including:
    - Encoder (posterior) that takes [LR_up, HR] and outputs q(z|X,Y)
    - Prior encoder that takes LR_up and outputs p(z|X)
    - Decoder that generates HR (residual) from z and LR.
    """
    def __init__(self, base_channels=64, latent_dim=512):
        super(SRVAE, self).__init__()
        # Posterior encoder sees 6-channel input (concatenated LR and HR)
        self.encoder = ConvEncoder(in_channels=6, base_channels=base_channels, latent_dim=latent_dim)
        # Prior encoder sees only 3-channel input (LR image upsampled)
        self.prior_enc = ConvEncoder(in_channels=3, base_channels=base_channels, latent_dim=latent_dim)
        # Decoder
        self.decoder = Decoder(base_channels=base_channels, latent_dim=latent_dim)

    def forward(self, x_lr, x_hr=None):
        """
        Forward pass.
        If x_hr (ground truth) is provided (training mode), returns:
           output SR image, mu_q, logvar_q, mu_p, logvar_p.
        If x_hr is None (inference mode), returns:
           output SR image (sampled from prior).
        """
        # Bicubic upsample the LR input for conditioning
        x_up = F.interpolate(x_lr, scale_factor=4, mode='bicubic', align_corners=False)
        if self.training and x_hr is not None:
            # In training, compute posterior and prior
            # Concatenate upsampled LR and HR as input to encoder
            enc_in = torch.cat([x_up, x_hr], dim=1)  # 6-channel input
            mu_q, logvar_q = self.encoder(enc_in)    # q_phi(z|x,y)
            mu_p, logvar_p = self.prior_enc(x_up)    # p_psi(z|x)
            # Reparameterization trick: sample z ~ N(mu_q, sigma_q)
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z = mu_q + eps * std_q
            # Decode
            output = self.decoder(x_lr, z)
            return output, mu_q, logvar_q, mu_p, logvar_p
        else:
            # Inference: use prior to sample z
            mu_p, logvar_p = self.prior_enc(x_up)
            std_p = torch.exp(0.5 * logvar_p)
            eps = torch.randn_like(std_p)
            z = mu_p + eps * std_p
            output = self.decoder(x_lr, z)
            return output

# --- 4. Discriminator (for adversarial loss, optional) ---
class Discriminator(nn.Module):
    """
    Simple PatchGAN-like discriminator for 4x SR images.
    Outputs a single probability (real/fake).
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        def conv_block(in_c, out_c, stride=1):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)]
            # Use BatchNorm for discriminator (except maybe first layer) for stability
            # We will add BN for layers after first.
            if in_c != 3:  # skip BN for first conv
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        layers = []
        # progressively downsample
        layers += conv_block(3, 64, stride=2)    # 3->64, 2x down
        layers += conv_block(64, 128, stride=2)  # 64->128, 2x down
        layers += conv_block(128, 256, stride=2) # 128->256, 2x down
        layers += conv_block(256, 512, stride=2) # 256->512, 2x down
        layers += conv_block(512, 512, stride=2) # 512->512, 2x down (if input was ~128x128 patch, now ~4x4)
        self.conv_layers = nn.Sequential(*layers)
        # final dense layer
        self.fc = nn.Linear(512 * 4 * 4, 1)  # assuming input patch ~128x128 => after 5 downs: ~4x4 feature map
        # (If training on variable sizes, could use GlobalAvgPool and then fc)
    def forward(self, x):
        feat = self.conv_layers(x)
        feat = torch.flatten(feat, start_dim=1)
        out = self.fc(feat)
        # Note: we'll apply sigmoid or BCEWithLogitsLoss externally, so output is raw logit.
        return out

def main():
    # ---------- paths ----------
    train_lr_dir = "ML Train and Valid data/Train/DIV2K_train_LR_bicubic/X4"
    train_hr_dir = "ML Train and Valid data/Train/DIV2K_train_HR"
    val_lr_dir   = "ML Train and Valid data/Val/DIV2K_valid_LR_bicubic/X4"
    val_hr_dir   = "ML Train and Valid data/Val/DIV2K_valid_HR"

    # ---------- device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- models ----------
    latent_dim = 512
    sr_vae = SRVAE(base_channels=64, latent_dim=latent_dim).to(device)
    disc   = Discriminator().to(device)

    # ---------- optimisers ----------
    optimizer_G = torch.optim.Adam(sr_vae.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(disc.parameters(),   lr=1e-4, betas=(0.9, 0.999))

    # ---------- losses ----------
    criterion_pixel = nn.L1Loss()
    criterion_bce   = nn.BCEWithLogitsLoss()
    vgg_extractor   = VGGFeatureExtractor(layers=(26, 35)).to(device)

    # ---------- data ----------
    train_dataset = SRDataset(train_lr_dir, train_hr_dir, patch_size=128, training=True)
    train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True,
                               num_workers=4, persistent_workers=True)
    val_dataset   = SRDataset(val_lr_dir, val_hr_dir, patch_size=128, training=False)
    val_loader    = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # ---------- loop ----------
    num_epochs        = 100
    λ_perc, λ_adv, β  = 1.0, 1e-3, 1.0

    for epoch in range(num_epochs):
        sr_vae.train()
        running = 0.0
        for lr_imgs, hr_imgs in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{num_epochs}",
                                 leave=False):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            sr_out, μq, logσ2_q, μp, logσ2_p = sr_vae(lr_imgs, hr_imgs)

            loss_pix  = criterion_pixel(sr_out, hr_imgs)
            loss_perc = sum(criterion_pixel(a,b) for a,b in
                            zip(vgg_extractor(sr_out), vgg_extractor(hr_imgs)))

            # KL(q||p)
            var_q, var_p = torch.exp(logσ2_q), torch.exp(logσ2_p)
            kl = 0.5*(logσ2_p - logσ2_q + (var_q + (μq-μp)**2)/(var_p+1e-8) - 1)
            loss_kl = kl.sum(1).mean()

            loss_adv_g = 0.0
            if λ_adv:
                pred_fake = disc(sr_out)
                loss_adv_g = criterion_bce(pred_fake, torch.ones_like(pred_fake))

            loss_G = loss_pix + λ_perc*loss_perc + β*loss_kl + λ_adv*loss_adv_g
            optimizer_G.zero_grad();  loss_G.backward();  optimizer_G.step()

            if λ_adv:
                optimizer_D.zero_grad()
                loss_D = 0.5*(criterion_bce(disc(hr_imgs), torch.ones_like(pred_fake)) +
                              criterion_bce(disc(sr_out.detach()), torch.zeros_like(pred_fake)))
                loss_D.backward();  optimizer_D.step()

            running += loss_G.item()

        print(f"Epoch {epoch+1}/{num_epochs}  |  loss={running/len(train_loader):.4f}")

    # save
    torch.save(sr_vae.state_dict(), "srVAE_4x.pth")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()