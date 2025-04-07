import torch
from unet import UNet

def print_tensor_info(tensor, name):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Channels: {tensor.shape[1]}")
    print(f"  Spatial size: {tensor.shape[2]}x{tensor.shape[3]}")
    print("-" * 50)

def test_dimensions():
    """测试UNet模型的维度"""
    # 测试参数
    batch_size = 4
    channels = 1
    height = 28
    width = 28
    n_classes = 10
    
    # 创建模型
    model = UNet(in_channels=channels, n_feat=128, n_classes=n_classes)
    
    # 创建测试输入
    x = torch.randn(batch_size, channels, height, width)
    t = torch.tensor([0.5]).float()
    c = torch.randint(0, n_classes, (batch_size,))
    
    # 前向传播
    output = model(x, t, c)
    
    # 检查输出维度
    expected_shape = (batch_size, channels, height, width)
    assert output.shape == expected_shape, f"输出维度错误: 期望 {expected_shape}, 得到 {output.shape}"
    print("维度测试通过!")

if __name__ == '__main__':
    test_dimensions() 