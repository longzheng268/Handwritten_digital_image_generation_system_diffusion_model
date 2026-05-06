from setuptools import setup, find_packages

# 训练配置参数
TRAINING_CONFIG = {
    #######################
    # 基础训练参数
    #######################
    'n_epoch': 201,          # 总训练轮数
    'batch_size': 256,      # 批次大小
    'base_lr': 1e-4,        # 初始学习率
    'num_workers': 5,       # 数据加载的工作进程数
    
    #######################
    # 模型架构参数
    #######################
    'n_feat': 128,          # 基础特征通道数 (128够用，256更好但更慢)
    'n_classes': 10,        # 类别数量 (MNIST为10)
    'in_channels': 1,       # 输入通道数 (MNIST为灰度图，所以是1)
    'drop_prob': 0.1,       # Dropout概率
    
    #######################
    # 扩散模型参数
    #######################
    'T': 400,              # 扩散步数 (原始论文用500，这里用400加快训练)
    'beta_start': 1e-4,    # beta调度的起始值
    'beta_end': 0.02,      # beta调度的结束值

    #######################
    # 采样和生成参数
    #######################
    # 生成引导强度列表，用于控制条件生成的强度：
    # - 0.0: 完全无条件生成
    # - 0.5: 轻微的条件引导
    # - 2.0: 强条件引导
    'guidance_scales': [0.0, 0.5, 2.0],
    
    #######################
    # 保存和检查点参数
    #######################
    'save_model': True,     # 是否保存模型
    'save_every': 1,        # 每隔多少个epoch保存一次模型
}

# 项目依赖配置
setup(
    name="handwritten_digit_generation",
    version="0.1",
    description="使用扩散模型生成手写数字的项目",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        'torch>=2.0.0',        # PyTorch深度学习框架
        'torchvision>=0.15.0', # PyTorch图像工具
        'tqdm>=4.65.0',        # 进度条显示
        'flask>=2.0.0',        # Web应用框架
        'numpy>=1.24.0',       # 数值计算
        'pillow>=9.0.0',       # 图像处理
        'matplotlib>=3.5.0',   # 绘图和动画
        'tensorboard>=2.12.0', # 训练可视化
    ],
) 