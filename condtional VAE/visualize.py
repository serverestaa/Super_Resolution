import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils import calculate_psnr, calculate_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_side_by_side_comparison(lr_img, sr_img, hr_img, save_path):
    """
    Create a side-by-side comparison of LR, SR, and HR images
    
    Args:
        lr_img: Low-resolution image tensor (CxHxW)
        sr_img: Super-resolution image tensor (CxHxW)
        hr_img: High-resolution image tensor (CxHxW)
        save_path: Path to save the comparison image
    """
    # Convert to numpy for matplotlib
    lr_np = lr_img.numpy()
    sr_np = sr_img.numpy()
    hr_np = hr_img.numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(np.transpose(lr_np, (1, 2, 0)))
    axes[0].set_title("LR Input")
    axes[0].axis("off")
    
    axes[1].imshow(np.transpose(sr_np, (1, 2, 0)))
    axes[1].set_title("SR Output")
    axes[1].axis("off")
    
    axes[2].imshow(np.transpose(hr_np, (1, 2, 0)))
    axes[2].set_title("HR Ground Truth")
    axes[2].axis("off")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_comparison_image(lr_img, sr_img, hr_img, psnr, ssim, save_path):
    """
    Create a side-by-side comparison image with metrics
    
    Args:
        lr_img: Low-resolution image (numpy array, CxHxW)
        sr_img: Super-resolution image (numpy array, CxHxW)
        hr_img: High-resolution image (numpy array, CxHxW)
        psnr: PSNR value between SR and HR
        ssim: SSIM value between SR and HR
        save_path: Path to save the comparison image
    """
    # Convert from CxHxW to HxWxC format for matplotlib
    lr_img = np.transpose(lr_img, (1, 2, 0))
    sr_img = np.transpose(sr_img, (1, 2, 0))
    hr_img = np.transpose(hr_img, (1, 2, 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(lr_img)
    axes[0].set_title("LR Input")
    axes[0].axis("off")
    
    axes[1].imshow(sr_img)
    axes[1].set_title(f"SR Output\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    axes[1].axis("off")
    
    axes[2].imshow(hr_img)
    axes[2].set_title("HR Ground Truth")
    axes[2].axis("off")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_comparison_images(model, dataloader, output_dir="comparison_results", num_samples=None):
    """
    Generate and save comparison images (LR, SR, HR) for all validation images
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        output_dir: Directory to save comparison images
        num_samples: Number of samples to process (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Create subdirectories
    lr_dir = os.path.join(output_dir, "LR")
    sr_dir = os.path.join(output_dir, "SR")
    hr_dir = os.path.join(output_dir, "HR")
    comparison_dir = os.path.join(output_dir, "LR_SR_HR")
    
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"Generating comparison images for validation set...")
    
    with torch.no_grad():
        img_count = 0
        for lr_imgs, hr_imgs in dataloader:
            if num_samples is not None and img_count >= num_samples:
                break
                
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate super-resolved images
            sr_imgs, _, _ = model(lr_imgs)
            
            # Denormalize for saving (from [-1, 1] to [0, 1])
            lr_imgs_norm = (lr_imgs + 1) / 2
            sr_imgs_norm = (sr_imgs + 1) / 2
            hr_imgs_norm = (hr_imgs + 1) / 2
            
            # Process each image in the batch
            for i in range(lr_imgs.size(0)):
                if num_samples is not None and img_count >= num_samples:
                    break
                
                # Save individual images
                lr_path = os.path.join(lr_dir, f"img_{img_count:04d}_lr.png")
                sr_path = os.path.join(sr_dir, f"img_{img_count:04d}_sr.png")
                hr_path = os.path.join(hr_dir, f"img_{img_count:04d}_hr.png")
                
                save_image(lr_imgs_norm[i], lr_path)
                save_image(sr_imgs_norm[i], sr_path)
                save_image(hr_imgs_norm[i], hr_path)
                
                # Calculate metrics
                psnr_val = calculate_psnr(sr_imgs_norm[i], hr_imgs_norm[i])
                ssim_val = calculate_ssim(sr_imgs_norm[i], hr_imgs_norm[i])
                
                # Create side-by-side comparison image
                comparison_path = os.path.join(comparison_dir, f"img_{img_count:04d}_comparison.png")
                create_comparison_image(
                    lr_imgs_norm[i].cpu().numpy(),
                    sr_imgs_norm[i].cpu().numpy(),
                    hr_imgs_norm[i].cpu().numpy(),
                    psnr_val, ssim_val, comparison_path
                )
                
                img_count += 1
                
                # Print progress
                if img_count % 10 == 0:
                    print(f"Processed {img_count} images")
    
    print(f"Generated comparison images for {img_count} validation samples in {output_dir}")


def generate_all_sr_images(model, dataloader, output_dir="super_res_outputs"):
    """
    Generate and save super-resolution images for a dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader with low-resolution images
        output_dir: Directory to save generated images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        count = 0
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            
            # Generate super-resolved images
            recon_imgs, _, _ = model(lr_imgs)
            
            # Denormalize for saving
            recon_imgs = recon_imgs * 0.5 + 0.5
            lr_imgs = lr_imgs * 0.5 + 0.5
            hr_imgs = hr_imgs * 0.5 + 0.5
            
            # Save each image
            for j in range(recon_imgs.size(0)):
                sr_path = os.path.join(output_dir, f"sr_{count:03d}.png")
                lr_path = os.path.join(output_dir, f"lr_{count:03d}.png")
                hr_path = os.path.join(output_dir, f"hr_{count:03d}.png")
                save_image(recon_imgs[j], sr_path)
                save_image(lr_imgs[j].cpu(), lr_path)
                save_image(hr_imgs[j], hr_path)
                
                # Create combined comparison image (LR | SR | HR)
                comparison_path = os.path.join(output_dir, f"comparison_{count:03d}.png")
                create_side_by_side_comparison(
                    lr_imgs[j].cpu(), 
                    recon_imgs[j].cpu(), 
                    hr_imgs[j], 
                    comparison_path
                )
                
                count += 1
    
    print(f"Saved {count} sets of images to '{output_dir}'.")