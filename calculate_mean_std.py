import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import CifarDataset

# 定义转换为Tensor的转换
transform = transforms.Compose([transforms.ToTensor()])


def calculate_mean_std(loader):
    # 通过迭代整个数据集来计算平均值和标准差
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # 批次中的样本数
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    return mean, std

mean, std = calculate_mean_std(CifarDataset.train_dataloader)

print(f'Mean: {mean}')
print(f'Std: {std}')
