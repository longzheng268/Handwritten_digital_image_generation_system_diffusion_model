import sys
import os
import glob
import io

# === 路径设置（与 app.py 一致）===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
model_training_dir = os.path.join(project_root, 'model_training')
if model_training_dir not in sys.path:
    sys.path.append(model_training_dir)

import streamlit as st
import torch
from torchvision.utils import save_image
from PIL import Image
from model_training.diffusion import DiffusionProcess
from model_training.unet import UNet
from config import TRAINING_CONFIG as cfg

# === 设备 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_available_models():
    """获取 models 目录下所有 .pth 文件"""
    model_files = glob.glob(os.path.join(project_root, 'models', '*.pth'))
    return sorted([os.path.basename(f) for f in model_files])


@st.cache_resource
def load_model(model_name):
    """加载指定模型（带缓存，避免重复加载）"""
    model_path = os.path.join(project_root, 'models', model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

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

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    diffusion.load_state_dict(checkpoint['model'])
    diffusion.eval()

    return diffusion


def generate_digit_image(diffusion, digit, guide_w):
    """生成手写数字图像（逻辑与 app.py 一致）"""
    n_sample = 40  # 4行 x 10列
    with torch.no_grad():
        x_gen, x_store = diffusion.sample(
            n_sample=n_sample,
            size=(1, 28, 28),
            device=device,
            guide_w=guide_w
        )

    # 选择对应数字的图像
    selected_image = x_gen[digit::10][0]
    img_tensor = selected_image.cpu() * -1 + 1

    return img_tensor, x_store


def tensor_to_pil(img_tensor):
    """将 tensor 转为 PIL 图片（内存中完成，无需写磁盘）"""
    buf = io.BytesIO()
    save_image(img_tensor, buf, format='png', normalize=True)
    buf.seek(0)
    return Image.open(buf)


# === Streamlit 页面 ===

st.set_page_config(
    page_title="手写数字生成系统",
    page_icon="✏️",
    layout="centered"
)

st.title("手写数字生成系统")
st.caption("基于扩散模型 (DDPM) 的手写数字图片生成 | MNIST 数据集")

# === 侧边栏控件 ===
with st.sidebar:
    st.header("生成参数")

    digit = st.number_input(
        "请输入数字 (0-9)",
        min_value=0, max_value=9, value=5, step=1
    )

    models = get_available_models()
    if not models:
        st.error("未找到模型文件！请确保 models/ 目录下有 .pth 文件。")
        st.stop()
    model_name = st.selectbox("选择模型", options=models)

    guide_w = st.slider(
        "引导强度 (Guidance Scale)",
        min_value=0.0, max_value=5.0, step=0.1, value=2.0
    )

    st.divider()
    show_process = st.checkbox("显示扩散去噪过程", value=False)

# === 生成 ===
if st.button("生成手写数字", type="primary", use_container_width=True):
    with st.spinner(f"正在生成数字 {digit}，请稍候..."):
        try:
            diffusion = load_model(model_name)
            img_tensor, x_store = generate_digit_image(diffusion, digit, guide_w)
            pil_image = tensor_to_pil(img_tensor)

            st.image(pil_image, caption=f"生成的数字 {digit}", width=200)

            # 可选：展示扩散去噪过程
            if show_process:
                st.subheader("扩散去噪过程")
                n_frames = x_store.shape[0]
                frame_idx = st.slider(
                    "去噪步骤", 0, n_frames - 1, n_frames - 1
                )
                frame_tensor = x_store[frame_idx, digit::10][0].cpu() * -1 + 1
                frame_pil = tensor_to_pil(frame_tensor)
                st.image(
                    frame_pil,
                    caption=f"步骤 {(frame_idx + 1) * 20} / {cfg['T']}",
                    width=200
                )

        except FileNotFoundError as e:
            st.error(f"模型文件未找到: {str(e)}")
        except Exception as e:
            st.error(f"生成失败: {str(e)}")

# === 页脚 ===
st.markdown("---")
st.caption("基于 DDPM (Denoising Diffusion Probabilistic Model) | U-Net + Classifier-Free Guidance")
