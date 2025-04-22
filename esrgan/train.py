import os
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from datasets.dataset import DIV2KDataset
from models.esrgan_model import GeneratorRRDB, Discriminator, VGGFeatureExtractor
from utils.metrics import psnr, ssim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',         type=str,   default='data')
    parser.add_argument('--batch_size',        type=int,   default=16)
    parser.add_argument('--hr_patch_size',     type=int,   default=128)
    parser.add_argument('--scale',             type=int,   default=4)
    parser.add_argument('--pixel_iters',       type=int,   default=30000)
    parser.add_argument('--adv_iters',         type=int,   default=60000)
    parser.add_argument('--num_train_images',  type=int,   default=None,
                        help='Limit the number of training images (for faster runs)')
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--beta1',             type=float, default=0.9)
    parser.add_argument('--beta2',             type=float, default=0.999)
    parser.add_argument('--val_interval',      type=int,   default=1000)
    parser.add_argument('--save_interval',     type=int,   default=5000)
    parser.add_argument('--print_interval',    type=int,   default=100)
    parser.add_argument('--checkpoint_dir',    type=str,   default='checkpoints')
    parser.add_argument('--log_dir',           type=str,   default='logs')
    parser.add_argument('--resume',            action='store_true')
    return parser.parse_args()

def validate(generator, val_loader, device):
    generator.eval()
    total_psnr, total_ssim = 0.0, 0.0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = generator(lr).clamp(0,1)
            sr_np = sr.squeeze().cpu().numpy().transpose(1,2,0)
            hr_np = hr.squeeze().cpu().numpy().transpose(1,2,0)
            total_psnr += psnr(sr_np, hr_np)
            total_ssim += ssim(sr_np, hr_np)
    n = len(val_loader)
    return total_psnr / n, total_ssim / n

def main():
    args = parse_args()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Detected {torch.cuda.device_count()} CUDA GPU(s):")
        for idx in range(torch.cuda.device_count()):
            print(f"  GPU {idx}: {torch.cuda.get_device_name(idx)}")
        print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
 
        x = torch.randn(1, 3, args.hr_patch_size, args.hr_patch_size).to(device)
        print(f"Dummy tensor allocated on device: {x.device}")
    cudnn.benchmark = True
 
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
 
    full_train_ds = DIV2KDataset(
        root_dir=args.data_root,
        lr_folder='DIV2K_train_LR_bicubic/X4',
        hr_folder='DIV2K_train_HR',
        hr_patch_size=args.hr_patch_size,
        scale=args.scale
    )
 
    if args.num_train_images is not None and args.num_train_images < len(full_train_ds):
        indices = random.sample(range(len(full_train_ds)), args.num_train_images)
        train_ds = Subset(full_train_ds, indices)
        print(f"Using subset of training data: {len(train_ds)} images")
    else:
        train_ds = full_train_ds

    val_ds = DIV2KDataset(
        root_dir=args.data_root,
        lr_folder='DIV2K_valid_LR_bicubic/X4',
        hr_folder='DIV2K_valid_HR',
        hr_patch_size=args.hr_patch_size,
        scale=args.scale
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True
    )
 
    netG = GeneratorRRDB().to(device)
    netD = Discriminator().to(device)
    vgg  = VGGFeatureExtractor(feature_layer=35).to(device)
 
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print("Wrapping models with DataParallel")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
 
    pixel_criterion      = nn.L1Loss().to(device)
    perceptual_criterion = nn.L1Loss().to(device)
    gan_criterion        = nn.MSELoss().to(device)
 
    optim_G = torch.optim.Adam(
        netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    optim_D = torch.optim.Adam(
        netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    scheduler_G = torch.optim.lr_scheduler.StepLR(
        optim_G, step_size=200000, gamma=0.5
    )
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optim_D, step_size=200000, gamma=0.5
    )
 
    scaler_G = GradScaler()
    scaler_D = GradScaler()
 
    start_iter = 0
    if args.resume:
        ckpts = sorted(os.listdir(args.checkpoint_dir))
        if ckpts:
            latest = ckpts[-1]
            print(f"Resuming from checkpoint {latest}")
            ckpt = torch.load(os.path.join(args.checkpoint_dir, latest), map_location=device)
            netG.load_state_dict(ckpt['netG'])
            netD.load_state_dict(ckpt.get('netD', netD.state_dict()))
            optim_G.load_state_dict(ckpt['optim_G'])
            optim_D.load_state_dict(ckpt['optim_D'])
            start_iter = ckpt.get('iter', 0)
 
    print("Starting pixel-loss pre-training...")
    netD.eval()
    loader_iter = iter(train_loader)
    for it in trange(start_iter, args.pixel_iters, desc="Pixel-pretrain"):
        try:
            lr, hr = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            lr, hr = next(loader_iter)
        lr, hr = lr.to(device), hr.to(device)

        optim_G.zero_grad()
        with autocast():
            sr = netG(lr)
            loss_pix = pixel_criterion(sr, hr)
        scaler_G.scale(loss_pix).backward()
        scaler_G.step(optim_G)
        scaler_G.update()
        scheduler_G.step()

        if (it + 1) % args.print_interval == 0:
            print(f"[Pre-train] Iter {it+1}/{args.pixel_iters}  pix_loss={loss_pix.item():.4f}")

        if (it + 1) % args.val_interval == 0:
            avg_psnr, avg_ssim = validate(netG, val_loader, device)
            writer.add_scalar('Pixel/PSNR', avg_psnr, it+1)
            writer.add_scalar('Pixel/SSIM', avg_ssim, it+1)
            writer.add_scalar('Pixel/Loss', loss_pix.item(), it+1)

        if (it + 1) % args.save_interval == 0:
            path = os.path.join(args.checkpoint_dir, f'G_pre_{it+1}.pth')
            torch.save({
                'netG': netG.state_dict(),
                'optim_G': optim_G.state_dict(),
                'iter': it+1
            }, path)
            print(f"Saved pretrain checkpoint: {path}")
 
    print("Starting adversarial training...")
    netD.train()
    loader_iter = iter(train_loader)
    total_iters = args.pixel_iters + args.adv_iters
    for it in trange(args.pixel_iters, total_iters, desc="Adv-train"):
        try:
            lr, hr = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            lr, hr = next(loader_iter)
        lr, hr = lr.to(device), hr.to(device)
 
        optim_D.zero_grad()
        with autocast():
            fake = netG(lr).detach()
            pred_real = netD(hr)
            pred_fake = netD(fake)
            loss_D = 0.5 * (
                gan_criterion(pred_real, torch.ones_like(pred_real)) +
                gan_criterion(pred_fake, torch.zeros_like(pred_fake))
            )
        scaler_D.scale(loss_D).backward()
        scaler_D.step(optim_D)
        scaler_D.update()
        scheduler_D.step()
 
        optim_G.zero_grad()
        with autocast():
            fake = netG(lr)
            pred_fake = netD(fake)
            loss_G_gan = gan_criterion(pred_fake, torch.ones_like(pred_fake))
            loss_G_pix = pixel_criterion(fake, hr)
            feat_sr = vgg(fake)
            feat_hr = vgg(hr)
            loss_G_per = perceptual_criterion(feat_sr, feat_hr)
            loss_G = loss_G_pix + 0.01 * loss_G_per + 1e-3 * loss_G_gan
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optim_G)
        scaler_G.update()
        scheduler_G.step()

        if (it + 1) % args.print_interval == 0:
            print(f"[Adv-train] Iter {it+1}/{total_iters}  loss_G={loss_G.item():.4f}  loss_D={loss_D.item():.4f}")

        if (it + 1) % args.val_interval == 0:
            avg_psnr, avg_ssim = validate(netG, val_loader, device)
            writer.add_scalar('Adv/PSNR', avg_psnr, it+1)
            writer.add_scalar('Adv/SSIM', avg_ssim, it+1)
            writer.add_scalar('Adv/Loss_G', loss_G.item(), it+1)
            writer.add_scalar('Adv/Loss_D', loss_D.item(), it+1)

        if (it + 1) % args.save_interval == 0:
            path = os.path.join(args.checkpoint_dir, f'G_adv_{it+1}.pth')
            torch.save({
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'iter': it+1
            }, path)
            print(f"Saved adversarial checkpoint: {path}")

    writer.close()

if __name__ == '__main__':
    main()
