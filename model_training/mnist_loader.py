import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_mnist_loaders(batch_size=256, num_workers=5):
    """获取MNIST数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为[0,1]范围的张量
    ])
    
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 