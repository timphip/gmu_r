import torch

torch.manual_seed(0)

W = torch.tensor([0.146, 0.154, 0.247])

s = 0.1
noise_std = 0.02

def quantize(W, s):
    return torch.round(W / s) * s

noise = torch.randn_like(W) * noise_std
W_noise = W + noise

QW = quantize(W, s)
QW_noise = quantize(W_noise, s)

print("原权重：", W)
print("随机噪声：", noise)
print("扰动后权重：", W_noise)
print("原量化结果：", QW)
print("扰动后量化结果：", QW_noise)
print("改变数量：", (QW != QW_noise).sum().item())












