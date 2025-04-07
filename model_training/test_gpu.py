import torch

def test_gpu():
    if torch.cuda.is_available():
        print(f"找到GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        
        # 简单矩阵乘法测试
        a = torch.randn(10000, 10000).cuda()
        b = torch.randn(10000, 10000).cuda()
        c = torch.matmul(a, b)
        print("GPU矩阵乘法测试成功！数值为：\n",c,"\n")
    else:
        print("未找到可用的GPU，将使用CPU")

if __name__ == "__main__":
    test_gpu() 