# Super-Resolution Deep Learning Project

This repository contains multiple implementations of state-of-the-art deep learning models for image super-resolution. The project includes several different approaches to tackle the problem of upscaling low-resolution images to high-resolution versions.

## Project Overview

Image super-resolution (SR) is the process of recovering high-resolution (HR) images from low-resolution (LR) inputs. This project implements several cutting-edge deep learning approaches to SR:

1. **CycleGAN-based Super-Resolution**: Uses unpaired image-to-image translation with cycle consistency
2. **ESRGAN (Enhanced Super-Resolution GAN)**: State-of-the-art perceptual-driven approach
3. **SRVAE (Super-Resolution Variational Autoencoder)**: Probabilistic approach using VAEs
4. **CVAE (Conditional Variational Autoencoder)**: Conditional generative approach to super-resolution

## Models

### 1. CycleGAN for Super-Resolution

The CycleGAN implementation uses unpaired learning to translate between low-resolution and high-resolution image domains.

**Key Features:**
- Unpaired image-to-image translation
- Cycle consistency loss to preserve content
- Two-way mapping (LR→HR and HR→LR)
- Residual blocks for better feature extraction

**Architecture:**
- Generator G: LR → HR (4x upscaling)
- Generator F: HR → LR (4x downscaling)
- Discriminator D_HR: Distinguishes real/fake HR images
- Discriminator D_LR: Distinguishes real/fake LR images

### 2. ESRGAN (Enhanced Super-Resolution GAN)

ESRGAN improves upon the original SRGAN with architectural enhancements and a more sophisticated perceptual loss.

**Key Features:**
- Residual-in-Residual Dense Block (RRDB)
- Relativistic average GAN for more stable training
- VGG-based perceptual loss
- Produces more photorealistic results

### 3. SRVAE (Super-Resolution Variational Autoencoder)

SRVAE uses a probabilistic approach to super-resolution by modeling the distribution of high-resolution images.

**Key Features:**
- Encoder-decoder architecture with latent space
- KL divergence regularization
- Probabilistic generation of HR images
- Can generate multiple plausible HR versions of a single LR input

### 4. CVAE (Conditional Variational Autoencoder)

CVAE extends the VAE approach by conditioning the generation process on the input LR image.

**Key Features:**
- Conditional generation based on LR input
- Latent space modeling of HR image distribution
- Combines reconstruction loss with KL divergence
- Can generate diverse HR outputs for a single LR input

## Dataset

The project uses the DIV2K dataset (DIVerse 2K resolution high-quality images):
- 800 high-resolution training images
- LR versions created using bicubic downsampling with a scale factor of 4x

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- PIL (Pillow)
- numpy
- matplotlib (for visualization)
- Standard Python libraries: os, random, math, urllib.request, zipfile

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/super-resolution-project.git
cd super-resolution-project

# Install dependencies
pip install torch torchvision pillow numpy matplotlib
```

## Usage

### Data Preparation

The project includes code to automatically download and extract the DIV2K dataset:

```python
hr_folder, lr_folder = download_div2k("div2k")
```

### Training

#### CycleGAN

```python
python CycleGAN/CycleGAN_w_v1/CycleGAN.py
```

#### ESRGAN

```python
python ESRGAN/train_esrgan.py
```

#### SRVAE

```python
python SRVAE/train_srvae.py
```

#### CVAE

```python
python CVAE/train_cvae.py
```

### Configuration

Each model has its own set of hyperparameters that can be adjusted. For example, for CycleGAN:

```python
train_cyclegan_sr(
    lr_folder=lr_folder,
    hr_folder=hr_folder,
    batch_size=4,         # Adjust based on GPU memory
    n_epochs=10,          # Increase for better results
    lambda_cycle=10.0,    # Weight for cycle consistency loss
    lambda_identity=0.0,  # Weight for identity mapping loss
    device="cuda"         # Use "cpu" if no GPU is available
)
```

## Model Comparison

| Model | Strengths | Weaknesses | Best Use Case |
|-------|-----------|------------|---------------|
| CycleGAN | Works with unpaired data, preserves content structure | Less sharp details than perceptual methods | When paired training data is unavailable |
| ESRGAN | Highest perceptual quality, photorealistic textures | Can hallucinate details, computationally intensive | When visual quality is the priority |
| SRVAE | Probabilistic approach, multiple possible outputs | Less sharp than adversarial methods | When diversity of solutions is desired |
| CVAE | Controlled generation, good balance of fidelity and diversity | Complex training dynamics | When controllable generation is needed |

## Results

The models produce different types of results:

- **CycleGAN**: Focuses on structure preservation with cycle consistency
- **ESRGAN**: Produces the most visually pleasing results with fine texture details
- **SRVAE/CVAE**: Generate diverse possible HR outputs for a single LR input

## Project Structure

```
super-resolution/
├── CycleGAN/
│   ├── CycleGAN_w_v1/
│   │   └── CycleGAN.py
│   ├── CycleGAN_w_v2/
│   │   └── CycleGAN2.py
│   └── CycleGAN_w_v3/
│       └── CycleGAN5_2.py
├── ESRGAN/
│   └── [ESRGAN implementation files]
├── SRVAE/
│   └── [SRVAE implementation files]
├── CVAE/
│   └── [CVAE implementation files]
├── div2k/
│   ├── DIV2K_train_HR/
│   └── DIV2K_train_LR_bicubic/X4/
└── README.md
```

## Extending the Project

Possible extensions:
- Add quantitative evaluation metrics (PSNR, SSIM, LPIPS)
- Implement real-world image testing pipeline
- Add more recent SR architectures (SwinIR, EDSR, etc.)
- Create a unified inference interface for all models
- Add web demo with Gradio or Streamlit

## License

[Specify your license here]

## Acknowledgements

- DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- CycleGAN paper: https://arxiv.org/abs/1703.10593
- ESRGAN paper: https://arxiv.org/abs/1809.00219
- VAE paper: https://arxiv.org/abs/1312.6114Project

This repository contains multiple implementations of state-of-the-art deep learning models for image super-resolution, including CycleGAN, ESRGAN, SRVAE, and CVAE approaches.

## Project Overview

Image super-resolution (SR) is the process of recovering high-resolution (HR) images from low-resolution (LR) inputs. This project implements several cutting-edge deep learning approaches to SR with a focus on 4x upscaling.

## Models

### 1. CycleGAN for Super-Resolution

The CycleGAN implementation uses unpaired learning to translate between low-resolution and high-resolution image domains.

**Implementation Details:**
- **Architecture:**
  - Generator G (LR→HR): Uses 6 residual blocks and 2 pixel shuffle upsampling layers (2x each) for 4x total upscaling
  - Generator F (HR→LR): Uses downsampling convolutions and 2 residual blocks for 4x downscaling
  - Discriminators: PatchGAN-style with 4 convolutional layers
- **Loss Functions:**
  - Adversarial Loss: MSE between discriminator predictions and target labels
  - Cycle Consistency Loss: L1 distance between original and reconstructed images (weighted by λ_cycle=10.0)
  - Optional Identity Loss: L1 distance between input and output when input is from target domain
- **Training Details:**
  - Batch size: 4
  - Learning rate: 2e-4 with Adam optimizer (β1=0.5, β2=0.999)
  - Random horizontal and vertical flips for data augmentation
  - Images resized to 256×256 during training
  - Weights initialized with normal distribution (scale=0.02)

**Code Structure:**
```
CycleGAN/
├── CycleGAN_w_v1/
│   └── CycleGAN.py  # Main implementation with dataset, models, and training
├── CycleGAN_w_v2/
│   └── CycleGAN2.py # Enhanced version
└── CycleGAN_w_v3/
    └── CycleGAN5_2.py # Further improvements
```

### 2. ESRGAN (Enhanced Super-Resolution GAN)

ESRGAN improves upon the original SRGAN with architectural enhancements and a more sophisticated perceptual loss.

**Key Features:**
- Residual-in-Residual Dense Block (RRDB)
- Relativistic average GAN for more stable training
- VGG-based perceptual loss
- Produces more photorealistic results

### 3. SRVAE (Super-Resolution Variational Autoencoder)

SRVAE uses a probabilistic approach to super-resolution by modeling the distribution of high-resolution images.

**Key Features:**
- Encoder-decoder architecture with latent space
- KL divergence regularization
- Probabilistic generation of HR images
- Can generate multiple plausible HR versions of a single LR input

### 4. CVAE (Conditional Variational Autoencoder)

CVAE extends the VAE approach by conditioning the generation process on the input LR image.

**Key Features:**
- Conditional generation based on LR input
- Latent space modeling of HR image distribution
- Combines reconstruction loss with KL divergence
- Can generate diverse HR outputs for a single LR input

## Dataset

The project uses the DIV2K dataset (DIVerse 2K resolution high-quality images):
- 800 high-resolution training images
- LR versions created using bicubic downsampling with a scale factor of 4x

The dataset is automatically downloaded and extracted by the `download_div2k()` function:
```python
hr_folder, lr_folder = download_div2k("div2k")
```

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- PIL (Pillow)
- numpy
- matplotlib (for visualization)
- Standard Python libraries: os, random, math, urllib.request, zipfile

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/super-resolution-project.git
cd super-resolution-project

# Install dependencies
pip install torch torchvision pillow numpy matplotlib
```

## Usage

### Training CycleGAN

```python
python CycleGAN/CycleGAN_w_v1/CycleGAN.py
```

This will:
1. Download the DIV2K dataset if not already present
2. Set up the CycleGAN model with appropriate generators and discriminators
3. Train the model for 10 epochs (default)
4. Save the trained models as:
   - `G_LR2HR.pth`: Generator for LR to HR conversion
   - `F_HR2LR.pth`: Generator for HR to LR conversion
   - `D_HR.pth`: Discriminator for HR images
   - `D_LR.pth`: Discriminator for LR images

### Configuration

You can modify the training parameters in the `train_cyclegan_sr` function call:

```python
train_cyclegan_sr(
    lr_folder=lr_folder,
    hr_folder=hr_folder,
    batch_size=4,         # Adjust based on GPU memory
    n_epochs=10,          # Increase for better results
    lambda_cycle=10.0,    # Weight for cycle consistency loss
    lambda_identity=0.0,  # Weight for identity mapping loss
    device="cuda"         # Use "cpu" if no GPU is available
)
```

## Implementation Details

### CycleGAN Architecture

#### Generator G (LR → HR)
```
GeneratorSR(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): ReLU(inplace=True)
  )
  (res_blocks): Sequential(
    (0): ResidualBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    ... [6 residual blocks total]
  )
  (up1): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): PixelShuffle(upscale_factor=2)
    (2): ReLU(inplace=True)
  )
  (up2): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): PixelShuffle(upscale_factor=2)
    (2): ReLU(inplace=True)
  )
  (conv_out): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
  (tanh): Tanh()
)
```

#### Generator F (HR → LR)
```
GeneratorDown(
  (down): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (5): ReLU(inplace=True)
  )
  (res_blocks): Sequential(
    (0): ResidualBlock(...)
    (1): ResidualBlock(...)
  )
  (out_conv): Conv2d(256, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
  (tanh): Tanh()
)
```

#### Discriminator
```
Discriminator(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): LeakyReLU(negative_slope=0.2, inplace=True)
    (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (5): LeakyReLU(negative_slope=0.2, inplace=True)
    (6): Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
  )
)
```

### Training Process

The CycleGAN training process follows these steps for each batch:

1. **Generator Update**:
   - Generate fake HR images from real LR images: `fake_B = G(real_A)`
   - Reconstruct LR images: `recov_A = F(fake_B)`
   - Generate fake LR images from real HR images: `fake_A = F(real_B)`
   - Reconstruct HR images: `recov_B = G(fake_A)`
   - Compute adversarial losses for both generators
   - Compute cycle consistency losses
   - Optionally compute identity mapping losses
   - Update generator parameters

2. **Discriminator Updates**:
   - Update HR discriminator using real and fake HR images
   - Update LR discriminator using real and fake LR images

## Model Comparison

| Model | Strengths | Weaknesses | Best Use Case |
|-------|-----------|------------|---------------|
| CycleGAN | Works with unpaired data, preserves content structure | Less sharp details than perceptual methods | When paired training data is unavailable |
| ESRGAN | Highest perceptual quality, photorealistic textures | Can hallucinate details, computationally intensive | When visual quality is the priority |
| SRVAE | Probabilistic approach, multiple possible outputs | Less sharp than adversarial methods | When diversity of solutions is desired |
| CVAE | Controlled generation, good balance of fidelity and diversity | Complex training dynamics | When controllable generation is needed |

## Results

The models produce different types of results:

- **CycleGAN**: Focuses on structure preservation with cycle consistency
- **ESRGAN**: Produces the most visually pleasing results with fine texture details
- **SRVAE/CVAE**: Generate diverse possible HR outputs for a single LR input

## Extending the Project

Possible extensions:
- Add validation during training to monitor progress
- Implement early stopping based on validation metrics
- Add more sophisticated loss functions (perceptual loss, etc.)
- Experiment with different network architectures
- Add inference code for testing on new images
- Implement metrics for quantitative evaluation (PSNR, SSIM, LPIPS)
- Create a unified interface for comparing different models
- Add a web demo with Gradio or Streamlit

## License

[Specify your license here]

## Acknowledgements

- DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- CycleGAN paper: https://arxiv.org/abs/1703.10593
- ESRGAN paper: https://arxiv.org/abs/1809.00219
- VAE paper: https://arxiv.org/abs/1312.6114
Project

This repository contains multiple implementations of state-of-the-art deep learning models for image super-resolution, including CycleGAN, ESRGAN, SRVAE, and CVAE approaches.

## Project Overview

Image super-resolution (SR) is the process of recovering high-resolution (HR) images from low-resolution (LR) inputs. This project implements several cutting-edge deep learning approaches to SR with a focus on 4x upscaling.

## Models

### 1. CycleGAN for Super-Resolution

The CycleGAN implementation uses unpaired learning to translate between low-resolution and high-resolution image domains.

**Implementation Details:**
- **Architecture:**
  - Generator G (LR→HR): Uses 6 residual blocks and 2 pixel shuffle upsampling layers (2x each) for 4x total upscaling
  - Generator F (HR→LR): Uses downsampling convolutions and 2 residual blocks for 4x downscaling
  - Discriminators: PatchGAN-style with 4 convolutional layers
- **Loss Functions:**
  - Adversarial Loss: MSE between discriminator predictions and target labels
  - Cycle Consistency Loss: L1 distance between original and reconstructed images (weighted by λ_cycle=10.0)
  - Optional Identity Loss: L1 distance between input and output when input is from target domain
- **Training Details:**
  - Batch size: 4
  - Learning rate: 2e-4 with Adam optimizer (β1=0.5, β2=0.999)
  - Random horizontal and vertical flips for data augmentation
  - Images resized to 256×256 during training
  - Weights initialized with normal distribution (scale=0.02)

**Code Structure:**
```
CycleGAN/
├── CycleGAN_w_v1/
│   └── CycleGAN.py  # Main implementation with dataset, models, and training
├── CycleGAN_w_v2/
│   └── CycleGAN2.py # Enhanced version
└── CycleGAN_w_v3/
    └── CycleGAN5_2.py # Further improvements
```

### 2. ESRGAN (Enhanced Super-Resolution GAN)

ESRGAN improves upon the original SRGAN with architectural enhancements and a more sophisticated perceptual loss.

**Implementation Details:**
- **Architecture:**
  - **Generator**: Based on the RRDB (Residual in Residual Dense Block) network
    - Initial convolutional layer: 3→64 channels
    - Main body: 23 RRDB blocks with dense connections and channel growth of 32
    - Trunk convolution: 64→64 channels
    - Two upsampling blocks using nearest-neighbor interpolation followed by convolution
    - Final layers: HR convolution and output convolution (64→3 channels)
    - Activation: LeakyReLU with negative slope of 0.2
  - **RRDB Block Structure**:
    - Each RRDB contains multiple dense blocks
    - Dense connections within blocks for better gradient flow
    - Residual scaling for training stability
    - Residual connections at both block and sub-block levels

**Network Architecture:**
```
GeneratorRRDB(
  (conv_first): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (body): Sequential(
    (0): RRDB(
      (dense_blocks): Sequential(...)
      (conv_last): Conv2d(...)
    )
    ... [23 RRDB blocks total]
  )
  (trunk_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (upconv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (upconv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (HR_conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_last): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
)
```

**Forward Pass:**
1. Initial feature extraction with `conv_first`
2. Process through 23 RRDB blocks
3. Apply trunk convolution and add residual connection
4. Upsample 2x using nearest interpolation + convolution + LeakyReLU (twice for 4x total)
5. Final convolutions to produce the HR output

### 3. CVAE (Conditional Variational Autoencoder)

CVAE extends the VAE approach by conditioning the generation process on the input LR image.

**Implementation Details:**
- **Architecture:**
  - **Encoder**:
    - 4 convolutional layers with increasing channel depth (3→64→128→256→512)
    - LeakyReLU activations and batch normalization
    - Skip connections saved for decoder
    - Flattened output projected to latent space (mu and logvar)
    - Latent dimension: 256
  - **Decoder**:
    - Linear layer to reshape latent vector
    - Transposed convolutions for upsampling
    - Skip connections from encoder concatenated at each level
    - Additional upsampling using PixelShuffle for efficient 4x upscaling
    - Final convolution to produce RGB output

**Network Architecture:**
```
CVAE(
  (enc_conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2)
  )
  (enc_conv2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(128)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (enc_conv3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(256)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (enc_conv4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(512)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (flatten): Flatten()
  (fc_mu): Linear(in_features=8192, out_features=256)
  (fc_logvar): Linear(in_features=8192, out_features=256)
  (decoder_input): Linear(in_features=256, out_features=8192)
  
  (dec_conv1): Sequential(
    (0): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(256)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (dec_conv2): Sequential(
    (0): ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(128)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (dec_conv3): Sequential(
    (0): ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(64)
    (2): LeakyReLU(negative_slope=0.2)
  )
  (upsample1): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): PixelShuffle(upscale_factor=2)
    (2): BatchNorm2d(64)
    (3): LeakyReLU(negative_slope=0.2)
  )
  (upsample2): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): PixelShuffle(upscale_factor=2)
    (2): BatchNorm2d(64)
    (3): LeakyReLU(negative_slope=0.2)
  )
  (final_conv): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (sigmoid): Sigmoid()
)
```

**Key Operations:**
1. **Encode**: Process LR image through encoder, compute latent parameters (mu, logvar)
2. **Reparameterize**: Sample from latent distribution using reparameterization trick
3. **Decode**: Process latent vector through decoder with skip connections from encoder
4. **Forward**: Combine encode, reparameterize, and decode operations
5. **Sample**: Generate multiple HR versions by sampling from latent space

**Loss Functions:**
- Reconstruction Loss: L1 or MSE between generated and target HR images
- KL Divergence: Regularization term to ensure latent space follows normal distribution
- Combined loss with weighting factor for KL term

### 4. SRVAE (Super-Resolution Variational Autoencoder)

SRVAE uses a probabilistic approach to super-resolution by modeling the distribution of high-resolution images.

**Implementation Details:**
- Similar to CVAE but with key differences in the conditioning mechanism
- Focuses on modeling the distribution of possible HR images for a given LR input
- Uses a simpler architecture compared to CVAE with fewer skip connections
- Emphasizes the probabilistic nature of the super-resolution task

## Dataset

The project uses the DIV2K dataset (DIVerse 2K resolution high-quality images):
- 800 high-resolution training images
- LR versions created using bicubic downsampling with a scale factor of 4x

The dataset is automatically downloaded and extracted by the `download_div2k()` function:
```python
hr_folder, lr_folder = download_div2k("div2k")
```

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- PIL (Pillow)
- numpy
- matplotlib (for visualization)
- Standard Python libraries: os, random, math, urllib.request, zipfile

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/super-resolution-project.git
cd super-resolution-project

# Install dependencies
pip install torch torchvision pillow numpy matplotlib
```

## Usage

### Training CycleGAN

```python
python CycleGAN/CycleGAN_w_v1/CycleGAN.py
```

This will:
1. Download the DIV2K dataset if not already present
2. Set up the CycleGAN model with appropriate generators and discriminators
3. Train the model for 10 epochs (default)
4. Save the trained models as:
   - `G_LR2HR.pth`: Generator for LR to HR conversion
   - `F_HR2LR.pth`: Generator for HR to LR conversion
   - `D_HR.pth`: Discriminator for HR images
   - `D_LR.pth`: Discriminator for LR images

### Training ESRGAN

```python
python ESRGAN/train_esrgan.py
```

### Training CVAE

```python
python CVAE/train_cvae.py
```

### Configuration

You can modify the training parameters in the `train_cyclegan_sr` function call:

```python
train_cyclegan_sr(
    lr_folder=lr_folder,
    hr_folder=hr_folder,
    batch_size=4,         # Adjust based on GPU memory
    n_epochs=10,          # Increase for better results
    lambda_cycle=10.0,    # Weight for cycle consistency loss
    lambda_identity=0.0,  # Weight for identity mapping loss
    device="cuda"         # Use "cpu" if no GPU is available
)
```

## Implementation Details

### CycleGAN Architecture

#### Generator G (LR → HR)
```
GeneratorSR(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): ReLU(inplace=True)
  )
  (res_blocks): Sequential(
    (0): ResidualBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    ... [6 residual blocks total]
  )
  (up1): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): PixelShuffle(upscale_factor=2)
    (2): ReLU(inplace=True)
  )
  (up2): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): PixelShuffle(upscale_factor=2)
    (2): ReLU(inplace=True)
  )
  (conv_out): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
  (tanh): Tanh()
)
```

#### Generator F (HR → LR)
```
GeneratorDown(
  (down): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (5): ReLU(inplace=True)
  )
  (res_blocks): Sequential(
    (0): ResidualBlock(...)
    (1): ResidualBlock(...)
  )
  (out_conv): Conv2d(256, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
  (tanh): Tanh()
)
```

#### Discriminator
```
Discriminator(
  (model): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): LeakyReLU(negative_slope=0.2, inplace=True)
    (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (5): LeakyReLU(negative_slope=0.2, inplace=True)
    (6): Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
  )
)
```

## Model Comparison

| Model | Strengths | Weaknesses | Best Use Case |
|-------|-----------|------------|---------------|
| CycleGAN | Works with unpaired data, preserves content structure | Less sharp details than perceptual methods | When paired training data is unavailable |
| ESRGAN | Highest perceptual quality, photorealistic textures | Can hallucinate details, computationally intensive | When visual quality is the priority |
| CVAE | Controlled generation, good balance of fidelity and diversity | Complex training dynamics | When controllable generation is needed |
| SRVAE | Probabilistic approach, multiple possible outputs | Less sharp than adversarial methods | When diversity of solutions is desired |

## Results

The models produce different types of results:

- **CycleGAN**: Focuses on structure preservation with cycle consistency
- **ESRGAN**: Produces the most visually pleasing results with fine texture details
- **CVAE/SRVAE**: Generate diverse possible HR outputs for a single LR input

## Extending the Project

Possible extensions:
- Add validation during training to monitor progress
- Implement early stopping based on validation metrics
- Add more sophisticated loss functions (perceptual loss, etc.)
- Experiment with different network architectures
- Add inference code for testing on new images
- Implement metrics for quantitative evaluation (PSNR, SSIM, LPIPS)
- Create a unified interface for comparing different models
- Add a web demo with Gradio or Streamlit


## Acknowledgements

- DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- CycleGAN paper: https://arxiv.org/abs/1703.10593
- ESRGAN paper: https://arxiv.org/abs/1809.00219
- VAE paper: https://arxiv.org/abs/1312.6114
