import os
import argparse
from collections import OrderedDict

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage

from models.esrgan_model import GeneratorRRDB
from utils.metrics import psnr, ssim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',   type=str, required=True,
                        help='root dir containing LR and HR folders')
    parser.add_argument('--lr_folder',   type=str, default='div2k_lr_x4',
                        help='LR folder name under data_root')
    parser.add_argument('--hr_folder',   type=str, default='div2k_hr',
                        help='HR folder name under data_root')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='path to generator checkpoint (.pth)')
    parser.add_argument('--output_dir',  type=str, default='results',
                        help='directory to save SR images')
    parser.add_argument('--device',      type=str, default='cuda',
                        help='device, e.g. cuda, cuda:1 or cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
 
    netG = GeneratorRRDB().to(device)
 
    ckpt = torch.load(args.checkpoint, map_location=device)
    print("Checkpoint keys:", list(ckpt.keys()))
 
    sd = ckpt.get('netG', ckpt)   
   
    clean_sd = OrderedDict()
    for k, v in sd.items():
        name = k
        if name.startswith('module.'):
            name = name[len('module.'):]
        clean_sd[name] = v
 
    msg = netG.load_state_dict(clean_sd, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    netG.eval()

    to_tensor = ToTensor()
    to_image  = ToPILImage()

    psnr_vals, ssim_vals = [], []

    lr_dir = os.path.join(args.data_root, args.lr_folder)
    hr_dir = os.path.join(args.data_root, args.hr_folder)
    img_names = sorted(f for f in os.listdir(lr_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    for name in img_names:

        lr = Image.open(os.path.join(lr_dir, name)).convert('RGB')
        lr_t = to_tensor(lr).unsqueeze(0).to(device)
 
        with torch.no_grad():
            sr_t = netG(lr_t).clamp(0, 1)
        sr_img = to_image(sr_t.squeeze().cpu())
        sr_img.save(os.path.join(args.output_dir, name))
 
        hr_path = os.path.join(hr_dir, name)
        if os.path.exists(hr_path):
            hr = Image.open(hr_path).convert('RGB')
            if hr.size != sr_img.size:
                hr = hr.resize(sr_img.size, Image.BICUBIC)
            sr_np = np.array(sr_img).astype(np.float32) / 255.0
            hr_np = np.array(hr).astype(np.float32) / 255.0
            psnr_vals.append(psnr(sr_np, hr_np))
            ssim_vals.append(ssim(sr_np, hr_np))

    if psnr_vals:
        print(f'Average PSNR: {np.mean(psnr_vals):.4f} dB')
        print(f'Average SSIM: {np.mean(ssim_vals):.4f}')

if __name__ == '__main__':
    main()
