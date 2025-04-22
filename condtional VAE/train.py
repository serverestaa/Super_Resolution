import os
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from loss import loss_function
from utils import calculate_psnr, calculate_ssim, save_checkpoint

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataloader, val_dataloader, num_epochs=20, checkpoint_dir="checkpoints",
               lr=1e-4, kl_weight=0.001, perceptual_weight=0.1):
    """
    Train the CVAE model
    
    Args:
        model: CVAE model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        num_epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        lr: Learning rate
        kl_weight: Weight for KL divergence loss term
        perceptual_weight: Weight for perceptual loss term
        
    Returns:
        model: Trained model
        training_log: Dictionary containing training metrics
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Initialize VGG perceptual loss
    from model import VGGPerceptualLoss
    vgg_loss = VGGPerceptualLoss().to(device)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize log data
    training_log = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_psnr': [],
        'val_psnr': [],
        'train_ssim': [],
        'val_ssim': []
    }
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_loss_components = {'recon': 0.0, 'kl': 0.0, 'perceptual': 0.0, 'total': 0.0}
        train_psnr_sum = 0.0
        train_ssim_sum = 0.0
        batch_count = 0
        
        # Create tqdm progress bar for training
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for lr_imgs, hr_imgs in train_pbar:
            # Move tensors to the configured device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_imgs, mu, logvar = model(lr_imgs)
            
            # Calculate loss
            loss, loss_components = loss_function(recon_imgs, hr_imgs, mu, logvar, vgg_loss, 
                                                lr_imgs, kl_weight, perceptual_weight)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_train_loss += loss_components['total']
            for k in loss_components:
                train_loss_components[k] += loss_components[k]
            
            # Calculate PSNR for monitoring
            with torch.no_grad():
                train_psnr = calculate_psnr(recon_imgs.detach(), hr_imgs)
                train_psnr_sum += train_psnr
                
                train_ssim = calculate_ssim(recon_imgs.detach(), hr_imgs)
                train_ssim_sum += train_ssim
            
            batch_count += 1
            
            # Update progress bar with current metrics
            train_pbar.set_postfix({
                'loss': f"{loss_components['total']:.4f}", 
                'PSNR': f"{train_psnr:.2f}", 
                'SSIM': f"{train_ssim:.4f}"
            })
        
        # Calculate average training metrics
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')
        avg_train_psnr = train_psnr_sum / batch_count if batch_count > 0 else 0
        avg_train_ssim = train_ssim_sum / batch_count if batch_count > 0 else 0
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_loss_components = {'recon': 0.0, 'kl': 0.0, 'perceptual': 0.0, 'total': 0.0}
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_batch_count = 0
        
        # Create tqdm progress bar for validation
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_pbar:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                # Forward pass
                recon_imgs, mu, logvar = model(lr_imgs)
                
                # Calculate validation loss
                val_loss, val_loss_comps = loss_function(recon_imgs, hr_imgs, mu, logvar, vgg_loss,
                                                      lr_imgs, kl_weight, perceptual_weight)
                
                # Accumulate metrics
                total_val_loss += val_loss_comps['total']
                for k in val_loss_comps:
                    val_loss_components[k] += val_loss_comps[k]
                
                # Calculate PSNR
                val_psnr = calculate_psnr(recon_imgs.detach(), hr_imgs)
                val_psnr_sum += val_psnr
                
                val_ssim = calculate_ssim(recon_imgs.detach(), hr_imgs)
                val_ssim_sum += val_ssim
                
                val_batch_count += 1
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'loss': f"{val_loss_comps['total']:.4f}", 
                    'PSNR': f"{val_psnr:.2f}", 
                    'SSIM': f"{val_ssim:.4f}"
                })
        
        # Calculate average validation metrics
        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        avg_val_psnr = val_psnr_sum / val_batch_count if val_batch_count > 0 else 0
        avg_val_ssim = val_ssim_sum / val_batch_count if val_batch_count > 0 else 0
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch data
        training_log['epochs'].append(epoch + 1)
        training_log['train_losses'].append(avg_train_loss)
        training_log['val_losses'].append(avg_val_loss)
        training_log['train_psnr'].append(avg_train_psnr)
        training_log['val_psnr'].append(avg_val_psnr)
        training_log['train_ssim'].append(avg_train_ssim)
        training_log['val_ssim'].append(avg_val_ssim)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Train - Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.2f}, SSIM: {avg_train_ssim:.4f}")
        print(f"Valid - Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}")
        
        # Detailed loss breakdown
        print(f"Training loss components - Recon: {train_loss_components['recon']/batch_count:.4f}, "
              f"KL: {train_loss_components['kl']/batch_count:.4f}, "
              f"Perceptual: {train_loss_components['perceptual']/batch_count:.4f}")
        
        # Save a checkpoint every epoch
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            save_checkpoint(epoch + 1, model, optimizer, avg_val_loss, checkpoint_dir)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Plot training progress
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plot_training_progress(training_log, epoch, checkpoint_dir)
        
        print("-" * 80)  # Separator between epochs
    
    return model, training_log


def plot_training_progress(training_log, epoch, checkpoint_dir):
    """
    Plot training progress curves
    
    Args:
        training_log: Dictionary containing training metrics
        epoch: Current epoch number
        checkpoint_dir: Directory to save plots
    """
    plt.figure(figsize=(18, 5))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(training_log['epochs'], training_log['train_losses'], 'b-', label='Training Loss')
    plt.plot(training_log['epochs'], training_log['val_losses'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot PSNR curves
    plt.subplot(1, 3, 2)
    plt.plot(training_log['epochs'], training_log['train_psnr'], 'b-', label='Training PSNR')
    plt.plot(training_log['epochs'], training_log['val_psnr'], 'r-', label='Validation PSNR')
    plt.title('Training and Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    
    # Plot SSIM curves
    plt.subplot(1, 3, 3)
    plt.plot(training_log['epochs'], training_log['train_ssim'], 'b-', label='Training SSIM')
    plt.plot(training_log['epochs'], training_log['val_ssim'], 'r-', label='Validation SSIM')
    plt.title('Training and Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'training_progress_epoch_{epoch+1}.png'))
    plt.close()