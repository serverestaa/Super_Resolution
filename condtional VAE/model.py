import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:18],  # relu3_4
            vgg[18:27], # relu4_4
        ])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        # Normalize input images to VGG expected range
        return (x - self.mean) / self.std
    
    def forward(self, x, y):
        # Ensure inputs are in range [0, 1]
        x = (x + 1) / 2  # If inputs are [-1, 1], normalize to [0, 1]
        y = (y + 1) / 2
        
        # Normalize
        x = self.normalize(x)
        y = self.normalize(y)
        
        # Compute loss
        loss = 0.0
        x_features = x
        y_features = y
        
        for block in self.blocks:
            x_features = block(x_features)
            y_features = block(y_features)
            loss += F.l1_loss(x_features, y_features)
            
        return loss


class CVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: LR image (3x32x32) -> latent space
        # We'll capture intermediate features for skip connections
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 64x32x32
            nn.LeakyReLU(0.2)
        )
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Flatten layer for latent space
        self.flatten = nn.Flatten()  # 512*4*4 = 8192
        
        # Latent space projections
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Decoder input - now incorporate the condition (LR image)
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Conditioning network to process LR image
        self.condition_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Upsampling network with skip connections
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Note the +256 in input channels to account for skip connection
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # +128 for skip connection
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # Additional upsampling layers
        # Use Pixel Shuffle (sub-pixel convolution) for more efficient upsampling
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64 + 64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # 64x128x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # Final layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Residual upsampling for LR image
        self.lr_upsampler = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )

    def encode(self, x):
        # Pass through encoder layers and save intermediate outputs
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)
        x4 = self.enc_conv4(x3)
        
        # Flatten and project to latent space
        encoded = self.flatten(x4)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        # Return latent parameters and skip features
        return mu, logvar, (x1, x2, x3, x4)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, condition, skip_features):
        # Unpack skip connection features
        skip1, skip2, skip3, skip4 = skip_features
        
        # Initial processing of condition (LR image)
        condition_features = self.condition_processor(condition)
        
        # Process latent vector
        x = self.decoder_input(z)
        x = x.view(-1, 512, 4, 4)
        
        # Upsampling with skip connections
        x = self.dec_conv1(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip3], dim=1)
        
        x = self.dec_conv2(x)
        x = torch.cat([x, skip2], dim=1)
        
        x = self.dec_conv3(x)
        x = torch.cat([x, skip1], dim=1)
        
        # Further upsampling
        x = self.upsample1(x)
        x = self.upsample2(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Residual learning: add upsampled LR image
        upsampled_lr = self.lr_upsampler(condition)
        x = x + upsampled_lr
        
        return x
        
    def forward(self, x):
        # x is the LR image
        mu, logvar, skip_features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, x, skip_features)
        return reconstructed, mu, logvar
        
    def sample(self, num_samples, condition):
        # Sample from latent space with a given LR image as condition
        z = torch.randn(num_samples, self.latent_dim).to(condition.device)
        
        # Generate dummy skip features (zeros with correct shapes)
        batch_size = condition.size(0)
        dummy_skip1 = torch.zeros(batch_size, 64, 32, 32).to(condition.device)
        dummy_skip2 = torch.zeros(batch_size, 128, 16, 16).to(condition.device)
        dummy_skip3 = torch.zeros(batch_size, 256, 8, 8).to(condition.device)
        dummy_skip4 = torch.zeros(batch_size, 512, 4, 4).to(condition.device)
        
        dummy_skips = (dummy_skip1, dummy_skip2, dummy_skip3, dummy_skip4)
        
        return self.decode(z, condition, dummy_skips)