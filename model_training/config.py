# 训练相关参数
TRAINING_CONFIG = {
    # 基础训练参数
    'n_epoch': 20,
    'batch_size': 256,
    'base_lr': 1e-4,
    'num_workers': 5,
    
    # 模型参数
    'n_feat': 128,  # 128 ok, 256 better (but slower)
    'n_classes': 10,
    'in_channels': 1,
    'drop_prob': 0.1,
    
    # 扩散模型参数
    'T': 400,  # 原始是500，这里用400
    'beta_start': 1e-4,
    'beta_end': 0.02,

    # 采样参数组合列表 [guidance_scale]
    'guidance_scales': [0.0, 0.5, 2.0],  # strength of generative guidance
    
    # 保存相关
    'save_model': True,
    'save_every': 1,  # 每几个epoch保存一次
} 