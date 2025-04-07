import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import datasets

class HandwrittenDataset(Dataset):
    def __init__(self, root_dir, transform=None, noise_level=0.05):
        self.root_dir = root_dir
        self.transform = transform
        self.noise_level = noise_level
        self.samples = []
        
        # 遍历目录结构：root_dir/类别/图片文件
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    self.samples.append((os.path.join(label_dir, img_file), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载并预处理图像
        image = Image.open(img_path).convert('L')
        image = transforms.Resize((64, 64))(transforms.ToTensor()(image))
        
        # 标准化到[-1,1]范围
        image = (image - 0.5) * 2
        
        # 数据增强（训练时随机应用）
        if self.transform:
            image = self.transform(image)
            
        # 添加随机噪声
        noise = torch.randn_like(image) * self.noise_level
        noisy_image = image + noise
        
        # 生成one-hot标签
        c = torch.zeros(10)
        c[label] = 1.0
        
        return {
            'x0': image,
            'noisy': noisy_image,
            'label': c
        }

# 数据增强变换组合
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
])

# 示例用法
if __name__ == '__main__':
    dataset = HandwrittenDataset(
        root_dir='d:/Code_Project/Python Project/Handwritten_digital_image_generation_system_diffusion_model/dataset/Training',
        transform=train_transform,
        noise_level=0.1
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 验证数据加载
    batch = next(iter(loader))
    print(f'Image shape: {batch["x0"].shape}')
    print(f'Label shape: {batch["label"].shape}')

def get_mnist_loaders(batch_size=64):
    # 修改数据预处理
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 将图像标准化到[-1, 1]范围
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
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader