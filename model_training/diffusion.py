import torch
import torch.nn.functional as F
from tqdm import tqdm

# 在文件开头定义device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Diffusion使用设备: {device}")

class DiffusionProcess(torch.nn.Module):  # 继承自nn.Module
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super().__init__()
        self.device = device
        self.model = nn_model
        self.T = n_T
        
        # 设置beta调度
        self.register_buffer('betas', torch.linspace(betas[0], betas[1], n_T))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    def forward(self, x, c):
        """训练时的前向传播"""
        t = torch.randint(0, self.T, (x.shape[0],), device=self.device)
        noise = torch.randn_like(x)
        
        # 计算噪声图像
        x_t = (
            self.sqrt_alphas_cumprod[t, None, None, None] * x +
            self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )
        
        # 预测噪声
        predicted_noise = self.model(x_t, t/self.T, c)
        
        # 返回MSE损失
        return F.mse_loss(predicted_noise, noise)

    def sample(self, n_sample, size, device, guide_w=0.0):
        """采样生成图像"""
        self.eval()
        with torch.no_grad():
            x = torch.randn((n_sample, *size)).to(device)
            x_store = []  # 存储生成过程
            
            # 为每个数字类别生成相等数量的样本
            # 修改标签生成方式，确保每行都是0-9
            c = torch.tensor(
                [i % 10 for i in range(n_sample)],  # 每10个数字循环一次0-9
                device=device
            ).long()
            
            for i in tqdm(reversed(range(self.T)), desc='sampling loop time step'):
                t = torch.full((n_sample,), i, device=device)
                t_is = t/self.T
                
                # 分类器引导
                pred_noise = self.model(x, t_is, c)
                if guide_w > 0:
                    # 无条件生成
                    uncond_pred = self.model(x, t_is, torch.zeros_like(c))
                    # 应用引导
                    pred_noise = uncond_pred + guide_w * (pred_noise - uncond_pred)
                
                # 计算均值
                alpha = self.alphas[i]
                alpha_hat = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise
                ) + torch.sqrt(beta) * noise
                
                if i % 20 == 0:
                    x_store.append(x.clone())
                    
            x_store = torch.stack(x_store)
            return x, x_store

def extract(a, t, x_shape):
    """从a中提取适当的索引以匹配x_shape"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)