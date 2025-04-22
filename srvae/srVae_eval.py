
import math, pathlib, torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.utils     as vutils
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ───── paths & params ──────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parent
CKPT      = ROOT / "srVAE_4x.pth"
HR_DIR    = ROOT / "ML Train and Valid data" / "Val" / "DIV2K_valid_HR"
LR_DIR    = ROOT / "ML Train and Valid data" / "Val" / "DIV2K_valid_LR_bicubic" / "X4"
IMAGE_IDS = list(range(801, 811))        # 10 samples
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── import architecture & load checkpoint ──────────────────────────────
from srVAE_final import SRVAE            # ← change if your module name differs
model = SRVAE().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint from {CKPT}\nDevice: {DEVICE}")

# ───── helper: modern / legacy SSIM kwarg ---------------------------------
def ssim(a, b):
    kw = dict(data_range=1.0)
    if "channel_axis" in structural_similarity.__code__.co_varnames:
        kw["channel_axis"] = -1
    else:
        kw["multichannel"] = True
    return structural_similarity(a, b, **kw)

# ───── evaluation loop ────────────────────────────────────────────────────
to_tensor = T.ToTensor()

tot_psnr = tot_ssim = 0.0
show_triplets = []       # store first three (LR_up, SR, HR) for plotting

with torch.no_grad():
    for i, idx in enumerate(IMAGE_IDS):
        hr_path = HR_DIR / f"{idx:04d}.png"
        lr_path = LR_DIR / f"{idx:04d}x4.png"

        hr = to_tensor(Image.open(hr_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        lr = to_tensor(Image.open(lr_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        sr = model(lr).clamp(0, 1)

        # upscale LR with bicubic only for visual comparison (not used for metric)
        lr_up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic",
                                                align_corners=False)

        # metrics
        sr_np = sr.squeeze(0).permute(1,2,0).cpu().numpy()
        hr_np = hr.squeeze(0).permute(1,2,0).cpu().numpy()
        psnr  = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
        ssim_ = ssim(hr_np, sr_np)

        print(f"{idx:04d}:  PSNR {psnr:6.2f} dB   SSIM {ssim_:6.4f}")
        tot_psnr += psnr
        tot_ssim += ssim_

        # keep first three triplets for plotting
        if len(show_triplets) < 3:
            show_triplets.append((lr_up.squeeze(0), sr.squeeze(0), hr.squeeze(0)))

# averages
n = len(IMAGE_IDS)
print(f"\nAverage ({n} images)  ►  PSNR {tot_psnr/n:6.2f} dB   SSIM {tot_ssim/n:6.4f}")

# ───── plot the first 3 LR/SR/HR triplets ─────────────────────────────────
titles = ["LR (bicubic ↑4×)", "SR (srVAE)", "HR (GT)"]
fig, axes = plt.subplots(len(show_triplets), 3, figsize=(9, 3*len(show_triplets)))
if len(show_triplets) == 1:
    axes = axes[None, :]  # make it 2‑D for consistent indexing

for r, (lr_im, sr_im, hr_im) in enumerate(show_triplets):
    for c, im in enumerate([lr_im, sr_im, hr_im]):
        axes[r, c].imshow(im.permute(1,2,0).cpu())
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(titles[c], fontsize=10)

fig.tight_layout()
plt.show()
