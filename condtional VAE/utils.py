import os
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir="checkpoints"):
    """
    Save training checkpoint
    
    Args:
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state to save
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"cvae_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load training checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        model: Model with loaded weights
        optimizer: Optimizer with loaded state
        epoch: Epoch number when checkpoint was saved
        loss: Loss value when checkpoint was saved
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return model, optimizer, epoch, loss


# PSNR calculation function
def calculate_psnr(sr_img, hr_img):
    """
    Calculate PSNR between super-resolved and high-resolution images
    
    Args:
        sr_img: Super-resolution image tensor
        hr_img: High-resolution image tensor
        
    Returns:
        psnr: PSNR value in dB
    """
    mse = torch.mean((sr_img - hr_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


# Function to create Gaussian window for SSIM calculation
def create_gaussian_window(window_size, sigma):
    """
    Create a Gaussian window for SSIM calculation
    
    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
        
    Returns:
        window: 2D Gaussian window
    """
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                          for x in range(window_size)])
    # Normalize
    gauss = gauss / gauss.sum()
    
    # Create 2D window (outer product)
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    return _2D_window


# SSIM calculation function
def calculate_ssim(sr_img, hr_img, window_size=11, sigma=1.5, L=1.0):
    """
    Calculate SSIM (Structural Similarity Index) between super-resolved and high-resolution images
    
    Args:
        sr_img: Super-resolution image tensor (B×C×H×W or C×H×W)
        hr_img: High-resolution image tensor (B×C×H×W or C×H×W)
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
        L: Dynamic range of pixel values (1.0 for normalized images)
        
    Returns:
        ssim: SSIM value (averaged over batch if input is batched)
    """
    # Handle both batched and single image inputs
    is_batched = sr_img.dim() == 4
    
    if is_batched:
        batch_size = sr_img.size(0)
        ssim_values = []
        
        # Process each item in the batch separately
        for i in range(batch_size):
            ssim_val = calculate_ssim(sr_img[i], hr_img[i], window_size, sigma, L)
            ssim_values.append(ssim_val)
        
        # Return average SSIM across the batch
        return sum(ssim_values) / len(ssim_values)
    
    # Process single image (non-batched)
    if sr_img.dim() != 3 or hr_img.dim() != 3:
        raise ValueError("Images should be 3D tensors (C×H×W) for non-batched processing")
        
    # Constants for stability
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    # Create Gaussian window
    window = create_gaussian_window(window_size, sigma).to(sr_img.device)
    
    # Calculate SSIM per channel and average
    num_channels = sr_img.size(0)
    ssim_value = 0.0
    
    for i in range(num_channels):
        # Get current channel
        sr_img_channel = sr_img[i].unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, H, W]
        hr_img_channel = hr_img[i].unsqueeze(0).unsqueeze(0)
        
        # Apply Gaussian filter
        mu1 = F.conv2d(sr_img_channel, window, padding=window_size//2)
        mu2 = F.conv2d(hr_img_channel, window, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(sr_img_channel**2, window, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(hr_img_channel**2, window, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(sr_img_channel * hr_img_channel, window, padding=window_size//2) - mu1_mu2
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        # Average SSIM for current channel
        ssim_value += torch.mean(ssim_map)
    
    # Return average SSIM across channels
    return (ssim_value / num_channels).item()