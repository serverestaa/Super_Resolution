import os
import torch
import argparse
from data import load_dataset
from model import CVAE
from train import train_model
from evaluate import evaluate_model
from visualize import generate_comparison_images
from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    parser = argparse.ArgumentParser(description="CVAE Super Resolution")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "visualize"],
                       help="Operation mode: train, test, or visualize")
    parser.add_argument("--root_dir", type=str, default="DIV2K",
                       help="Path to dataset root directory")
    parser.add_argument("--train_lr", type=str, default="DIV2K_train_LR_bicubic",
                       help="Training LR folder name")
    parser.add_argument("--train_hr", type=str, default="DIV2K_train_HR",
                       help="Training HR folder name")
    parser.add_argument("--valid_lr", type=str, default="DIV2K_valid_LR_bicubic",
                       help="Validation LR folder name")
    parser.add_argument("--valid_hr", type=str, default="DIV2K_valid_HR",
                       help="Validation HR folder name")
    parser.add_argument("--hr_patch_size", type=int, default=128,
                       help="Size of HR patches")
    parser.add_argument("--scale", type=int, default=4,
                       help="Super-resolution scale factor")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.001,
                       help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.1,
                       help="Perceptual loss weight")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to load checkpoint (for testing/visualizing)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to visualize (None for all)")
    
    args = parser.parse_args()
    
    # 1. Load datasets
    train_dataloader, val_dataloader, test_dataloader = load_dataset(
        root_dir=args.root_dir,
        train_lr_folder=args.train_lr,
        train_hr_folder=args.train_hr,
        valid_lr_folder=args.valid_lr,
        valid_hr_folder=args.valid_hr,
        hr_patch_size=args.hr_patch_size,
        scale=args.scale,
        batch_size=args.batch_size
    )
    
    if train_dataloader is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # 2. Initialize model
    model = CVAE(latent_dim=256).to(device)
    
    # 3. Run in specified mode
    if args.mode == "train":
        print("Starting training...")
        model, training_log = train_model(
            model=model, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            num_epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            lr=args.lr,
            kl_weight=args.kl_weight,
            perceptual_weight=args.perceptual_weight
        )
        
        # Evaluate the best model
        print("Loading best model for evaluation...")
        best_model = CVAE(latent_dim=256).to(device)
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"), map_location=device)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Using validation set for evaluation
        avg_psnr, avg_ssim = evaluate_model(best_model, val_dataloader, 
                                          os.path.join(args.output_dir, "evaluation"))
        
        # Report final results
        print(f"Model training completed.")
        print(f"Final average PSNR: {avg_psnr:.2f} dB")
        print(f"Final average SSIM: {avg_ssim:.4f}")
        
    elif args.mode == "test":
        # Load checkpoint if specified
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model, optimizer, epoch, loss = load_checkpoint(model, optimizer, args.checkpoint)
        else:
            print("No checkpoint specified. Using best model.")
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        avg_psnr, avg_ssim = evaluate_model(model, test_dataloader, 
                                          os.path.join(args.output_dir, "test_results"))
        
    elif args.mode == "visualize":
        # Load checkpoint if specified
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model, optimizer, epoch, loss = load_checkpoint(model, optimizer, args.checkpoint)
        else:
            print("No checkpoint specified. Using best model.")
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate comparison images
        generate_comparison_images(model, val_dataloader, 
                                 os.path.join(args.output_dir, "visualizations"),
                                 args.num_samples)


if __name__ == "__main__":
    main()