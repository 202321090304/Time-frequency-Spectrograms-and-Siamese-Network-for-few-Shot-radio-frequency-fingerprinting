import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from rf_dataset import SPDataset

train_dataset = SPDataset(
    data_dir='/disk/datasets/rf_data/newspectrum/SelectAB/train',
    transform=transforms.ToTensor(),
    data_type='test'
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

mean = 0.0
std = 0.0
num_samples = 0

for data, _ in train_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)

    mean += data.mean(dim=2).sum(dim=0)
    std  += data.std(dim=2).sum(dim=0)
    num_samples += batch_samples

mean /= num_samples
std /= num_samples

print(f"Mean: {mean}")
print(f"Std:  {std}")

num_to_show = 5
random_indices = random.sample(range(len(train_dataset)), num_to_show)

os.makedirs('visual_random', exist_ok=True)

for i, idx in enumerate(random_indices):
    img_tensor, label = train_dataset[idx]  
    pil_img = to_pil_image(img_tensor)

    save_path = os.path.join('visual_random', f"random_idx_{idx}_label_{label}.png")
    pil_img.save(save_path)

print(f"随机抽取的 {num_to_show} 张图片已保存到 'visual_random' 文件夹。")
