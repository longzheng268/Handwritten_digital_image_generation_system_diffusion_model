import sys
import os
import glob

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
from config import TRAINING_CONFIG as cfg  # 改用config.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'static', 'images')

# 在应用启动时创建目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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

if __name__ == '__main__':
    app.run(debug=True) 