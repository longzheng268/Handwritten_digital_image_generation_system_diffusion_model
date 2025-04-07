// 页面加载时获取可用模型
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/get_models');
        const data = await response.json();
        
        const select = document.getElementById('model-select');
        select.innerHTML = data.models.map(model => 
            `<option value="${model}">${model}</option>`
        ).join('');
    } catch (error) {
        console.error('Error loading models:', error);
        alert('加载模型列表失败');
    }

    // 添加滑块值的实时更新
    const guidanceSlider = document.getElementById('guidance-scale');
    const guidanceValue = document.getElementById('guidance-value');
    guidanceSlider.addEventListener('input', () => {
        guidanceValue.textContent = guidanceSlider.value;
    });

    const stepsSlider = document.getElementById('steps');
    const stepsValue = document.getElementById('steps-value');
    stepsSlider.addEventListener('input', () => {
        stepsValue.textContent = stepsSlider.value;
    });
});

document.getElementById('generate-btn').addEventListener('click', async () => {
    const btn = document.getElementById('generate-btn');
    const digit = document.getElementById('digit').value;
    const model = document.getElementById('model-select').value;
    const guidanceScale = document.getElementById('guidance-scale').value;
    const steps = document.getElementById('steps').value;

    if (!digit || digit < 0 || digit > 9) {
        alert('请输入0-9之间的数字');
        return;
    }

    try {
        btn.disabled = true;
        btn.textContent = '生成中...';

        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                digit: parseInt(digit),
                model: model,
                guidance_scale: parseFloat(guidanceScale),
                steps: parseInt(steps)
            }),
        });

        const data = await response.json();
        if (data.status === 'success') {
            const resultDiv = document.getElementById('result-image');
            resultDiv.innerHTML = `<img src="${data.image_url}" alt="Generated Digit">`;
        } else {
            alert(data.message || '生成失败');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('生成过程中出现错误');
    } finally {
        btn.disabled = false;
        btn.textContent = '生成';
    }
}); 