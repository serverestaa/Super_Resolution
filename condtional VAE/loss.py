import torch
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, vgg_loss, lr_img, kl_weight=0.001, perceptual_weight=0.1):
    """
    loss function with reconstruction loss, perceptual loss, and KL divergence
    
    Args:
        recon_x: Reconstructed (super-resolved) image
        x: Target high-resolution image
        mu: Mean of the latent distribution
        logvar: Log variance of the latent distribution
        vgg_loss: VGG perceptual loss module
        lr_img: Low-resolution input image
        kl_weight: Weight for the KL divergence term
        perceptual_weight: Weight for the perceptual loss term
        
    Returns:
        total_loss: Combined loss value
        loss_components: Dictionary with individual loss components
    """
    # Reconstruction loss (L1 for better edge preservation)
    recon_loss = F.l1_loss(recon_x, x)
    
    # Perceptual loss using VGG
    percep_loss = vgg_loss(recon_x, x)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with weights
    total_loss = recon_loss + kl_weight * kl_loss + perceptual_weight * percep_loss
    
    # Return individual losses for monitoring
    return total_loss, {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item(),
        'perceptual': percep_loss.item()
    }