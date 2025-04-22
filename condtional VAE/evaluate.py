import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from utils import calculate_psnr, calculate_ssim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, eval_dataloader, output_dir="evaluation_results"):
    """
    Evaluate model performance on a dataset
    
    Args:
        model: Trained model to evaluate
        eval_dataloader: DataLoader with test/validation data
        output_dir: Directory to save evaluation results
        
    Returns:
        avg_psnr: Average PSNR value
        avg_ssim: Average SSIM value
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    psnr_values = []
    ssim_values = []
    
    def denormalize(tensor):
        """Convert images from normalized [-1,1] back to [0,1]."""
        return tensor * 0.5 + 0.5
    
    with torch.no_grad():
        # Use a small batch for detailed visualization
        lr_batch, hr_batch = next(iter(eval_dataloader))
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        
        # Generate super-resolved images
        recon_batch, _, _ = model(lr_batch)
        
        # Move to CPU and denormalize for metrics and visualization
        lr_batch_cpu = denormalize(lr_batch.cpu())
        hr_batch_cpu = denormalize(hr_batch.cpu())
        recon_batch_cpu = denormalize(recon_batch.cpu())
        
        # Compute metrics for each image in the batch
        for i in range(lr_batch.size(0)):
            # Compute PSNR
            psnr = calculate_psnr(recon_batch_cpu[i], hr_batch_cpu[i])
            psnr_values.append(psnr)
            
            # Compute SSIM
            ssim_val = calculate_ssim(recon_batch_cpu[i], hr_batch_cpu[i])
            ssim_values.append(ssim_val)
            
            # Save individual examples
            save_image(lr_batch_cpu[i], os.path.join(output_dir, f"sample_{i}_lr.png"))
            save_image(recon_batch_cpu[i], os.path.join(output_dir, f"sample_{i}_sr.png"))
            save_image(hr_batch_cpu[i], os.path.join(output_dir, f"sample_{i}_hr.png"))
        
        # Calculate average metrics
        avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
        avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
        
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Display comparison grid for first few images
        num_samples = min(5, lr_batch.size(0))
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 8))
        
        for i in range(num_samples):
            # LR images
            axes[0, i].imshow(np.transpose(lr_batch_cpu[i].numpy(), (1, 2, 0)))
            axes[0, i].set_title("LR Input")
            axes[0, i].axis("off")
            
            # Super-resolved images
            axes[1, i].imshow(np.transpose(recon_batch_cpu[i].numpy(), (1, 2, 0)))
            axes[1, i].set_title(f"SR (PSNR: {psnr_values[i]:.2f})")
            axes[1, i].axis("off")
            
            # Ground truth HR images
            axes[2, i].imshow(np.transpose(hr_batch_cpu[i].numpy(), (1, 2, 0)))
            axes[2, i].set_title("HR Ground Truth")
            axes[2, i].axis("off")
        
        plt.suptitle(f"Super-resolution Results (Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_grid.png"), dpi=300)
        plt.close()
    
    # Full evaluation on the entire dataset
    full_psnr_values = []
    full_ssim_values = []
    
    print("Evaluating on full dataset...")
    with torch.no_grad():
        for lr_imgs, hr_imgs in eval_dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate SR images
            sr_imgs, _, _ = model(lr_imgs)
            
            # Denormalize
            lr_imgs_norm = denormalize(lr_imgs)
            sr_imgs_norm = denormalize(sr_imgs)
            hr_imgs_norm = denormalize(hr_imgs)
            
            # Calculate metrics
            for i in range(lr_imgs.size(0)):
                psnr = calculate_psnr(sr_imgs_norm[i], hr_imgs_norm[i])
                ssim_val = calculate_ssim(sr_imgs_norm[i], hr_imgs_norm[i])
                
                full_psnr_values.append(psnr)
                full_ssim_values.append(ssim_val)
    
    # Calculate overall metrics
    overall_avg_psnr = sum(full_psnr_values) / len(full_psnr_values) if full_psnr_values else 0
    overall_avg_ssim = sum(full_ssim_values) / len(full_ssim_values) if full_ssim_values else 0
    
    print(f"Overall Average PSNR: {overall_avg_psnr:.2f} dB")
    print(f"Overall Average SSIM: {overall_avg_ssim:.4f}")
    
    # Save metrics to a file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Average PSNR: {overall_avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {overall_avg_ssim:.6f}\n")
    
    return overall_avg_psnr, overall_avg_ssim