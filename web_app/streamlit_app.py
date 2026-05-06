import sys
import os
import glob
import io
import re

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


def extract_numbers_locally(text):
    """本地提取文本中的数字"""
    arabic_numbers = re.findall(r'\d+', text)

    cn_num_map = {
        '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
    }

    en_num_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }

    chinese_numbers = []
    for cn_num, value in cn_num_map.items():
        if cn_num in text:
            chinese_numbers.append(str(value))

    english_numbers = []
    text_lower = text.lower()
    for en_num, value in en_num_map.items():
        if en_num in text_lower:
            english_numbers.append(str(value))

    all_numbers = arabic_numbers + chinese_numbers + english_numbers
    return ','.join(all_numbers)


def extract_numbers_from_text(text):
    """使用LLM从文本中提取数字（通过环境变量配置）"""
    api_key = os.environ.get('LLM_API_KEY', '')
    base_url = os.environ.get('LLM_BASE_URL', 'https://api.ppinfra.com/v3/openai')

    if not api_key:
        return extract_numbers_locally(text)

    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)

        prompt = f"""请从以下文本中提取所有数字，包括：
        1. 阿拉伯数字（如1, 2, 3）
        2. 中文数字（如一, 二, 三, 十, 百）
        3. 英文数字（如one, two, three）

        文本内容："{text}"

        请将所有识别到的数字转换为阿拉伯数字，并用逗号分隔返回。
        例如，如果文本中有"twelve", "两个"和"5"，则返回"12,2,5"。
        """

        response = client.chat.completions.create(
            model="deepseek/deepseek-v3/community",
            messages=[
                {"role": "system", "content": "你是一个专门提取文本中数字的助手，负责将各种形式的数字转换为阿拉伯数字。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0
        )

        extracted = response.choices[0].message.content.strip()
        if not extracted or extracted.strip() == '':
            return extract_numbers_locally(text)
        return extracted
    except Exception:
        return extract_numbers_locally(text)


# === 初始化 session_state ===
if 'history' not in st.session_state:
    st.session_state.history = {}

# === Streamlit 页面 ===

st.set_page_config(
    page_title="手写数字生成系统",
    page_icon="✏️",
    layout="wide"
)

st.title("手写数字生成系统")
st.caption("基于扩散模型 (DDPM) 的手写数字图片生成 | MNIST 数据集")

# === 侧边栏控件 ===
with st.sidebar:
    st.header("生成参数")

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

# === 主区域：双栏布局 ===
col_left, col_right = st.columns(2)

# 左栏：手写数字生成
with col_left:
    st.subheader("手写数字生成")
    digit_input = st.text_input("请输入数字", placeholder="例如：20020315")

    if st.button("生成", type="primary", use_container_width=True):
        if not digit_input or not digit_input.strip():
            st.warning("请输入数字")
        elif not digit_input.strip().isdigit():
            st.warning("请只输入数字 0-9")
        else:
            digits = list(digit_input.strip())
            total = len(digits)
            progress_bar = st.progress(0)
            results = []

            for i, digit_char in enumerate(digits):
                digit = int(digit_char)
                progress_bar.progress((i + 1) / total, text=f"正在生成: {i + 1}/{total}")

                try:
                    diffusion = load_model(model_name)
                    img_tensor, x_store = generate_digit_image(diffusion, digit, guide_w)
                    pil_image = tensor_to_pil(img_tensor)
                    results.append((digit, pil_image))

                    # 记录历史
                    st.session_state.history[digit] = st.session_state.history.get(digit, 0) + 1

                except Exception as e:
                    st.error(f"生成数字 {digit} 失败: {str(e)}")

            progress_bar.empty()

            # 网格展示结果
            if results:
                st.subheader("生成结果")
                cols = st.columns(min(len(results), 10))
                for idx, (digit, img) in enumerate(results):
                    with cols[idx % len(cols)]:
                        st.image(img, caption=f"数字 {digit}", width=120)

                # 可选：展示最后一个数字的扩散过程
                if show_process and len(results) > 0:
                    st.subheader("扩散去噪过程")
                    n_frames = x_store.shape[0]
                    frame_idx = st.slider(
                        "去噪步骤", 0, n_frames - 1, n_frames - 1
                    )
                    last_digit = results[-1][0]
                    frame_tensor = x_store[frame_idx, last_digit::10][0].cpu() * -1 + 1
                    frame_pil = tensor_to_pil(frame_tensor)
                    st.image(
                        frame_pil,
                        caption=f"步骤 {(frame_idx + 1) * 20} / {cfg['T']}",
                        width=200
                    )

# 右栏：文本数字提取
with col_right:
    st.subheader("文本数字提取")
    text_input = st.text_area(
        "请输入文本",
        placeholder="例如：'我有三个苹果和五个梨子，总共8个水果'"
    )

    if st.button("提取数字", use_container_width=True):
        if not text_input or not text_input.strip():
            st.warning("请输入文本")
        else:
            extracted_numbers = extract_numbers_from_text(text_input)

            if not extracted_numbers or extracted_numbers.strip() == '':
                st.info("未能从文本中提取到任何数字")
            else:
                numbers = [n.strip() for n in extracted_numbers.split(',') if n.strip()]
                if len(numbers) == 0:
                    st.info("未能从文本中提取到任何数字")
                else:
                    st.success(f"从文本中提取到 {len(numbers)} 个数字:")
                    st.write(" ".join([f"`{num}`" for num in numbers]))

                    # 记录历史
                    for num_str in numbers:
                        try:
                            num = int(num_str)
                            if 0 <= num <= 9:
                                st.session_state.history[num] = st.session_state.history.get(num, 0) + 1
                        except ValueError:
                            pass

# === 历史记录 ===
if st.session_state.history:
    st.markdown("---")
    st.subheader("数字生成历史")

    sorted_history = sorted(st.session_state.history.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(len(sorted_history))
    for idx, (digit, count) in enumerate(sorted_history):
        with cols[idx]:
            st.metric(label=f"数字 {digit}", value=f"{count} 次")

# === 页脚 ===
st.markdown("---")
st.caption("基于 DDPM (Denoising Diffusion Probabilistic Model) | U-Net + Classifier-Free Guidance")
