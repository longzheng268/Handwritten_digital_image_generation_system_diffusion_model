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

    // 初始化时加载历史
    updateHistory();
});

document.getElementById('generate-btn').addEventListener('click', async () => {
    const btn = document.getElementById('generate-btn');
    const digitInput = document.getElementById('digit').value;
    const model = document.getElementById('model-select').value;
    const guidanceScale = document.getElementById('guidance-scale').value;
    const steps = document.getElementById('steps').value;

    if (!digitInput || digitInput.trim() === '') {
        alert('请输入数字');
        return;
    }
    
    // 检查输入是否为数字
    if (!/^\d+$/.test(digitInput)) {
        alert('请只输入数字 0-9');
        return;
    }
    
    try {
        btn.disabled = true;
        btn.textContent = '生成中...';
        
        // 清空结果区域
        document.getElementById('result-image').innerHTML = '';
        document.getElementById('results-grid').innerHTML = '';
        
        // 准备队列
        const digits = digitInput.split('');
        const totalDigits = digits.length;
        
        // 显示进度区域
        const progressContainer = document.getElementById('generation-progress');
        const progressBar = document.getElementById('progress-bar');
        const currentDigitSpan = document.getElementById('current-digit');
        const totalDigitsSpan = document.getElementById('total-digits');
        
        progressContainer.style.display = 'block';
        totalDigitsSpan.textContent = totalDigits;
        
        // 创建结果网格
        const resultsGrid = document.getElementById('results-grid');
        
        // 逐个处理数字
        for (let i = 0; i < digits.length; i++) {
            const digit = digits[i];
            
            // 更新进度
            currentDigitSpan.textContent = i + 1;
            progressBar.style.width = `${((i + 1) / totalDigits) * 100}%`;
            
            try {
                // 生成当前数字
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
                    // 创建结果项目
                    const resultItem = document.createElement('div');
                    resultItem.className = 'result-item';
                    resultItem.innerHTML = `
                        <img src="${data.image_url}" alt="Generated ${digit}">
                        <div class="result-digit">数字 ${digit}</div>
                    `;
                    
                    // 添加到网格
                    resultsGrid.appendChild(resultItem);
                    
                    // 记录数字
                    await fetch('/record_digit', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            digit: digit
                        }),
                    });
                    
                    // 即时更新历史记录
                    updateHistory();
                    
                    // 滚动到最新结果
                    resultItem.scrollIntoView({ behavior: 'smooth', block: 'end' });
                } else {
                    console.error(`生成数字 ${digit} 失败:`, data.message);
                    // 继续处理下一个数字
                }
            } catch (error) {
                console.error(`处理数字 ${digit} 时出错:`, error);
                // 继续处理下一个数字
            }
        }
        
        // 所有数字处理完成
        progressContainer.style.display = 'none';
        
    } catch (error) {
        console.error('整体处理出错:', error);
        alert('生成过程中出现错误');
    } finally {
        btn.disabled = false;
        btn.textContent = '生成';
    }
});

// 修改数字提取处理部分
document.getElementById('extract-btn')?.addEventListener('click', async () => {
    const textInput = document.getElementById('text-input').value;
    const resultDiv = document.getElementById('extraction-result');
    const btn = document.getElementById('extract-btn');
    
    if (!textInput) {
        alert('请输入文本');
        return;
    }
    
    try {
        btn.disabled = true;
        btn.textContent = '提取中...';
        resultDiv.innerHTML = '<p>正在分析文本...</p>';
        
        const response = await fetch('/extract_numbers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: textInput }),
        });
        
        const data = await response.json();
        console.log("API返回:", data);  // 添加调试信息
        
        if (data.status === 'success') {
            if (!data.extracted_numbers || data.extracted_numbers.trim() === '') {
                resultDiv.innerHTML = `<p>未能从文本中提取到任何数字</p>`;
                return;
            }
            
            const numbers = data.extracted_numbers.split(',').filter(n => n.trim() !== '');
            
            if (numbers.length === 0) {
                resultDiv.innerHTML = `<p>未能从文本中提取到任何数字</p>`;
            } else {
                // 显示提取结果
                resultDiv.innerHTML = `
                    <p>从文本中提取到 ${numbers.length} 个数字:</p>
                    <div>
                        ${numbers.map(num => `<span class="extracted-number">${num.trim()}</span>`).join('')}
                    </div>
                `;
            }
            
            // 更新历史记录
            updateHistory();
        } else {
            resultDiv.innerHTML = `<p>提取失败: ${data.message || '未知错误'}</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerHTML = '<p>提取过程中出现错误，请查看控制台</p>';
    } finally {
        btn.disabled = false;
        btn.textContent = '提取数字';
    }
});

// 添加历史更新函数
async function updateHistory() {
    try {
        const response = await fetch('/get_history');
        const data = await response.json();
        
        // 更新列表，使用更美观的布局
        const list = document.getElementById('history-list');
        
        if (Object.keys(data.history).length === 0) {
            list.innerHTML = '<div class="empty-history">暂无生成历史</div>';
            return;
        }
        
        list.innerHTML = Object.entries(data.history)
            .sort((a, b) => b[1] - a[1])
            .map(([digit, count]) => `
                <div class="history-item">
                    <div class="digit">${digit}</div>
                    <div class="count-bar" style="width: ${Math.min(100, count*10)}%;">${count}</div>
                </div>
            `).join('');
        
    } catch (error) {
        console.error('Error fetching history:', error);
    }
} 