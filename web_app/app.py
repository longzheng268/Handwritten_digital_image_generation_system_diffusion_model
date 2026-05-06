import sys
import os
import glob
from collections import defaultdict
from openai import OpenAI
import re

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 添加项目根目录和model_training目录到Python路径
if project_root not in sys.path:
    sys.path.append(project_root)
model_training_dir = os.path.join(project_root, 'model_training')
if model_training_dir not in sys.path:
    sys.path.append(model_training_dir)

from flask import Flask, render_template, request, send_from_directory, jsonify
import torch
from torchvision.utils import save_image
from datetime import datetime
from model_training.diffusion import DiffusionProcess
from model_training.unet import UNet
from config import TRAINING_CONFIG as cfg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'static', 'images')

# 在应用启动时创建目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 在app实例后添加
digit_history = defaultdict(int)

def get_available_models():
    """获取models目录下所有.pth文件"""
    model_files = glob.glob(os.path.join(project_root, 'models', '*.pth'))
    return [os.path.basename(f) for f in model_files]

def load_model(model_name):
    """加载指定模型"""
    model_path = os.path.join(project_root, 'models', model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    # 初始化模型
    model = UNet(
        in_channels=cfg['in_channels'],
        n_feat=cfg['n_feat'],
        n_classes=cfg['n_classes'],
        drop_prob=cfg['drop_prob']
    )

    # 初始化扩散过程
    diffusion = DiffusionProcess(
        nn_model=model,
        betas=(cfg['beta_start'], cfg['beta_end']),
        n_T=cfg['T'],
        device=device,
        drop_prob=cfg['drop_prob']
    ).to(device)

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    diffusion.load_state_dict(checkpoint['model'])
    diffusion.eval()

    return diffusion

# 打印调试信息
print("Current directory:", current_dir)
print("Project root:", project_root)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models')
def get_models():
    models = get_available_models()
    return jsonify({'models': models})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    digit = int(data['digit'])
    model_name = data.get('model', 'model_19.pth')
    guide_w = float(data.get('guidance_scale', 0.5))

    if 0 <= digit <= 9:
        try:
            diffusion = load_model(model_name)

            # 生成4x10的数字矩阵
            n_sample = 40  # 4行10列
            with torch.no_grad():
                x_gen, _ = diffusion.sample(
                    n_sample=n_sample,
                    size=(1, 28, 28),
                    device=device,
                    guide_w=guide_w
                )

            # 选择对应数字的图像（从每组0-9中选择第digit个）
            selected_image = x_gen[digit::10][0]  # 取第一组中的对应数字

            # 保存生成的图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{timestamp}.png')

            # 转换图像范围并保存单个数字图像
            save_image(selected_image.cpu()*-1 + 1, img_path, normalize=True)

            return jsonify({
                'status': 'success',
                'image_url': f'/images/{timestamp}.png'
            })

        except Exception as e:
            print(f"生成错误: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            })

    return jsonify({
        'status': 'error',
        'message': '请输入0-9之间的数字'
    })

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/record_digit', methods=['POST'])
def record_digit():
    data = request.json
    digit = data.get('digit')
    if digit is not None:
        digit_history[digit] += 1
    return jsonify({'status': 'success'})

@app.route('/get_history')
def get_history():
    formatted_history = {str(k): v for k, v in digit_history.items()}
    return jsonify({
        'history': formatted_history,
        'total': sum(digit_history.values())
    })

def extract_numbers_from_text(text):
    """使用LLM从文本中提取数字（通过环境变量配置API Key和Base URL）"""
    api_key = os.environ.get('LLM_API_KEY', '')
    base_url = os.environ.get('LLM_BASE_URL', 'https://api.ppinfra.com/v3/openai')

    if not api_key:
        print("未配置 LLM_API_KEY 环境变量，使用本地提取")
        return extract_numbers_locally(text)

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

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
        print(f"API返回结果: {extracted}")
        return extracted
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        numbers = re.findall(r'\d+', text)
        return ','.join(numbers)

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

@app.route('/extract_numbers', methods=['POST'])
def extract_numbers():
    """从文本中提取数字的API端点"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'status': 'error', 'message': '请输入文本'})

    try:
        extracted_numbers = extract_numbers_from_text(text)
        if not extracted_numbers or extracted_numbers.strip() == '':
            print("API提取失败，切换到本地提取")
            extracted_numbers = extract_numbers_locally(text)
    except Exception as e:
        print(f"提取错误: {str(e)}")
        extracted_numbers = extract_numbers_locally(text)

    if extracted_numbers:
        for num_str in extracted_numbers.split(','):
            try:
                num = int(num_str.strip())
                if 0 <= num <= 9:
                    digit_history[num] += 1
            except ValueError:
                pass

    return jsonify({
        'status': 'success',
        'extracted_numbers': extracted_numbers
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
