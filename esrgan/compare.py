import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


lr_folder = 'data/Set14_LR'     
sr_folder = 'test_results_run2'      
hr_folder = 'data/Set14_HR'         
out_path  = 'comparisonnn_run2.png'   
extension = 'png'           
max_images = 3
wspace = 0.3   
hspace = 0.5    

def sorted_images(folder):
    return sorted(glob.glob(os.path.join(folder, f'*.{extension}')))

lr_list = sorted_images(lr_folder)[:max_images]
sr_list = sorted_images(sr_folder)[:max_images]
hr_list = sorted_images(hr_folder)[:max_images]

assert len(lr_list) == len(sr_list) == len(hr_list) == max_images, \
       f"Need at least {max_images} images in each folder"

n = max_images
h, w = cv2.imread(hr_list[0]).shape[:2]
 
fig, axes = plt.subplots(
    n, 3,
    figsize=(3 * 5, n * 5),
    squeeze=False,
    gridspec_kw={'wspace': wspace, 'hspace': hspace}
)

titles = ['LR (bicubic ×4)', 'SR (ESRGAN)', 'HR (GT)']
for col in range(3):
    axes[0, col].set_title(titles[col], fontsize=16)

for i, (lr_p, sr_p, hr_p) in enumerate(zip(lr_list, sr_list, hr_list)):
    for j, path in enumerate([lr_p, sr_p, hr_p]):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        ax = axes[i, j]
        ax.imshow(img)
        ax.axis('off')
 

plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved comparison of {n} images to {out_path}")