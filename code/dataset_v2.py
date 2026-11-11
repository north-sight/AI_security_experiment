import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_data_v2(data_dir=None, seed=1024):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # 划分训练集 (55,000) 与 验证集 (5,000)，使用随机种子 1024
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(train_full, [55000, 5000], generator=generator)

    # 从 10,000 张测试集中按类别分层无放回抽样 200 张/类，共 2,000 张
    rng = np.random.RandomState(seed)
    test_targets = np.array(test_full.targets)
    sampled_indices = []
    for digit in range(10):
        digit_indices = np.where(test_targets == digit)[0]
        chosen = rng.choice(digit_indices, size=200, replace=False)
        sampled_indices.extend(chosen)
    
    # 按官方测试集原始索引升序排序
    sampled_indices = sorted(sampled_indices)
    test_subset = Subset(test_full, sampled_indices)

    return train_set, val_set, test_full, test_subset, sampled_indices

if __name__ == '__main__':
    train_set, val_set, test_full, test_subset, sampled_indices = get_mnist_data_v2()
    print(f"Dataset v2 initialized. Train: {len(train_set)}, Val: {len(val_set)}, Test sample: {len(test_subset)}")
    print(f"First 10 sample indices: {sampled_indices[:10]}")
