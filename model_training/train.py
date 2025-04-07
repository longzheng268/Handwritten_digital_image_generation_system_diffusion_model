import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys
import multiprocessing
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from unet import UNet
from diffusion import DiffusionProcess
from config import TRAINING_CONFIG as cfg

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练使用设备: {device}")

def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 创建保存目录
    save_dir = os.path.join(project_root, 'samples')
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 初始化模型和扩散过程
    model = UNet(
        in_channels=cfg['in_channels'],
        n_feat=cfg['n_feat'],
        n_classes=cfg['n_classes'],
        drop_prob=cfg['drop_prob']
    )

    diffusion = DiffusionProcess(
        nn_model=model,
        betas=(cfg['beta_start'], cfg['beta_end']),
        n_T=cfg['T'],
        device=device,
        drop_prob=cfg['drop_prob']
    ).to(device)

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor()  # MNIST已经归一化到0-1
    ])

    dataset = MNIST(
        os.path.join(project_root, "data"), 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )

    # 优化器
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg['base_lr'])

    # 初始化 TensorBoard
    log_dir = os.path.join(project_root, 'logs', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 记录配置参数
    writer.add_text('Config', str(cfg))

    # 训练循环
    global_step = 0  # 添加全局步数计数器
    for ep in range(cfg['n_epoch']):
        print(f'epoch {ep}')
        diffusion.train()
        
        # 线性学习率衰减
        current_lr = cfg['base_lr'] * (1 - ep/cfg['n_epoch'])
        optimizer.param_groups[0]['lr'] = current_lr
        writer.add_scalar('Training/learning_rate', current_lr, ep)

        total_loss = 0
        pbar = tqdm(dataloader)
        loss_ema = None
        
        for x, c in pbar:
            optimizer.zero_grad()
            x = x.to(device)
            c = c.to(device)
            
            loss = diffusion(x, c)
            loss.backward()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            total_loss += loss.item()
            
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optimizer.step()

            # 记录训练指标
            writer.add_scalar('Training/batch_loss', loss.item(), global_step)
            writer.add_scalar('Training/loss_ema', loss_ema, global_step)
            
            # 每100个批次记录一次详细信息
            if global_step % 100 == 0:
                writer.add_scalar('Training/epoch', ep, global_step)
                writer.add_scalar('Training/batch', global_step % len(dataloader), global_step)
            
            global_step += 1

        # 记录每个epoch的平均损失
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Training/epoch_avg_loss', avg_loss, ep)
        
        # 记录到控制台
        print(f'Epoch {ep} Average Loss: {avg_loss:.4f}')

        # 评估和保存样本
        diffusion.eval()
        with torch.no_grad():
            n_sample = 4 * cfg['n_classes']
            for w in cfg['guidance_scales']:
                x_gen, x_gen_store = diffusion.sample(
                    n_sample,
                    (1, 28, 28),
                    device,
                    guide_w=w
                )

                # 添加真实图像在底部
                x_real = torch.zeros_like(x_gen)
                for k in range(cfg['n_classes']):
                    for j in range(int(n_sample/cfg['n_classes'])):
                        try:
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*cfg['n_classes'])] = x[idx]

                # 为每一帧创建完整的图像网格
                for i, frame in enumerate(x_gen_store):
                    # 将生成的图像和真实图像拼接（都在GPU上）
                    x_all = torch.cat([frame, x_real])
                    # 转换到CPU并进行后处理
                    grid = make_grid(x_all.cpu()*-1 + 1, nrow=10)
                    
                    # 添加到TensorBoard
                    writer.add_image(f'Generation_Process/w{w}', grid, global_step=i)
                
                # 保存最终结果
                final_grid = make_grid(torch.cat([x_gen, x_real]).cpu()*-1 + 1, nrow=10)
                save_path = os.path.join(
                    save_dir, 
                    f"image_ep{ep}_w{w}_feat{cfg['n_feat']}_T{cfg['T']}_loss{loss_ema:.4f}.png"
                )
                save_image(final_grid, save_path)
                print(f'saved image at {save_path}')

                # GIF生成
                if ep % 5 == 0 or ep == cfg['n_epoch']-1:
                    from matplotlib import pyplot as plt
                    from matplotlib.animation import FuncAnimation, PillowWriter
                    
                    # 将数据移到CPU并转换为numpy数组
                    x_gen_store_cpu = x_gen_store.cpu()
                    
                    fig, axs = plt.subplots(
                        nrows=int(n_sample/cfg['n_classes']),
                        ncols=cfg['n_classes'],
                        figsize=(8,3)
                    )
                    
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/cfg['n_classes'])):
                            for col in range(cfg['n_classes']):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(
                                    axs[row, col].imshow(
                                        -x_gen_store[i,(row*cfg['n_classes'])+col,0],
                                        cmap='gray',
                                        vmin=(-x_gen_store[i]).min(),
                                        vmax=(-x_gen_store[i]).max()
                                    )
                                )
                        return plots
                    
                    ani = FuncAnimation(
                        fig,
                        animate_diff,
                        fargs=[x_gen_store_cpu],  # 使用CPU数据
                        interval=200,
                        blit=False,
                        repeat=True,
                        frames=x_gen_store.shape[0]
                    )
                    
                    gif_path = os.path.join(save_dir, f"gif_ep{ep}_w{w}.gif")
                    ani.save(gif_path, dpi=100, writer=PillowWriter(fps=5))
                    print(f'saved gif at {gif_path}')
                    plt.close(fig)

        # 保存模型检查点
        if cfg['save_model'] and ep % cfg['save_every'] == 0:
            model_path = os.path.join(
                model_dir,
                f"model_ep{ep}_feat{cfg['n_feat']}_T{cfg['T']}_loss{loss_ema:.4f}.pth"
            )
            checkpoint = {
                'epoch': ep,
                'model': diffusion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss_ema,
                'config': cfg
            }
            torch.save(checkpoint, model_path)
            print(f'saved model at {model_path}')

    writer.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()