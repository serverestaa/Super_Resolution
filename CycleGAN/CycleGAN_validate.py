import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from CycleGAN5_2 import GeneratorLR2HR
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

def validate_and_visualize_psnr_ssim(G, val_lr_folder, val_hr_folder, device="cuda",
                                     n_samples=4, out_file="results/val_samples.png"):


    os.makedirs(os.path.dirname(out_file), exist_ok=True)  
    G.eval().to(device)

    def normalize_name(filename: str) -> str:
        name = os.path.splitext(filename)[0]  
        if name.endswith('x4'):
            name = name[:-2]
        return name

    files_lr = sorted([f for f in os.listdir(val_lr_folder) if f.lower().endswith(('png','jpg','jpeg'))])
    files_hr = sorted([f for f in os.listdir(val_hr_folder) if f.lower().endswith(('png','jpg','jpeg'))])

    dict_lr = {normalize_name(f): os.path.join(val_lr_folder, f) for f in files_lr}
    dict_hr = {normalize_name(f): os.path.join(val_hr_folder, f) for f in files_hr}

    common_keys = sorted(set(dict_lr.keys()).intersection(dict_hr.keys()))
    if not common_keys:
        print("Нет общих файлов между LR и HR папками!")
        return

     #PIL->Tensor  [-1,1]
    transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    vis_lr = []
    vis_fake = []
    vis_hr = []

    for idx, key in enumerate(tqdm(common_keys, desc="Validation"), start=1):
        path_lr = dict_lr[key]
        path_hr = dict_hr[key]

        lr_img = Image.open(path_lr).convert('RGB')
        hr_img = Image.open(path_hr).convert('RGB')

        lr_tensor = transform_(lr_img).unsqueeze(0).to(device)  # (1,3,H,W)
        hr_tensor = transform_(hr_img).unsqueeze(0).to(device)  # (1,3,H,W)

        with torch.no_grad():
            fake_hr = G(lr_tensor)  # (1,3,H',W')
        #  [-1,1] -> [0,1]
        real_hr_01 = 0.5*(hr_tensor + 1.0)
        fake_hr_01 = 0.5*(fake_hr + 1.0)

        # skimage (C,H,W) -> (H,W,C)
        real_hr_np = real_hr_01[0].permute(1,2,0).cpu().numpy()
        fake_hr_np = fake_hr_01[0].permute(1,2,0).cpu().numpy()


        psnr_val = psnr_fn(real_hr_np, fake_hr_np, data_range=1.0)
        ssim_val = ssim_fn(real_hr_np, fake_hr_np, data_range=1.0, multichannel = True)
        total_psnr += psnr_val
        total_ssim += ssim_val
        count += 1

        if len(vis_lr) < n_samples:
            vis_lr.append(lr_tensor.cpu()[0])       # (3,H,W) [-1,1]
            vis_fake.append(fake_hr.cpu()[0])       # (3,H,W)
            vis_hr.append(hr_tensor.cpu()[0])       # (3,H,W)

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"Validation on {count} pairs:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    if vis_lr:
        rows = len(vis_lr)
        cols = 3  # LR, FakeHR, RealHR
        plt.figure(figsize=(9, 3*rows))
        for i in range(rows):
            # LR
            lr_img_01 = 0.5*(vis_lr[i] + 1.0)
            lr_img_np = lr_img_01.permute(1,2,0).numpy()
            plt.subplot(rows, cols, i*cols + 1)
            plt.imshow(lr_img_np, interpolation='none')
            plt.title("LR")
            plt.axis("off")

            # Fake HR
            fake_hr_01 = 0.5*(vis_fake[i] + 1.0)
            fake_hr_np = fake_hr_01.permute(1,2,0).numpy()
            plt.subplot(rows, cols, i*cols + 2)
            plt.imshow(fake_hr_np, interpolation='none')
            plt.title("Fake HR")
            plt.axis("off")
            # Real HR
            real_hr_01 = 0.5*(vis_hr[i] + 1.0)
            real_hr_np = real_hr_01.permute(1,2,0).numpy()
            plt.subplot(rows, cols, i*cols + 3)
            plt.imshow(real_hr_np, interpolation='none')
            plt.title("Real HR")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(out_file)
        plt.show()

    G.train() 


if __name__ == "__main__":
    G = GeneratorLR2HR()  
    G.load_state_dict(torch.load("Cycle_GAN5_2_w/generator_G_LR2HR_x4.pth", map_location="cuda"))  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    validate_and_visualize_psnr_ssim(
        G,
        val_lr_folder="DIV2K/DIV2K_valid_LR_bicubic_X4\X4",
        val_hr_folder="DIV2K/DIV2K_valid_HR",
        device=device,
        n_samples=4,
        out_file="results/val_samples.png"
    )