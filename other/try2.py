import torch

torch.manual_seed(0)

W = torch.randn(1000, 1000) * 0.1
Z = torch.randn_like(W)

FP4 = [-1, -.6667, -.5, -.3333, -.25, -.1667, -.0052,
       0, .0052, .1667, .25, .3333, .5, .6667, 1]

NF4 = [-1, -.6962, -.5251, -.3949, -.2844, -.1848, -.0911,
       0, .0796, .1609, .2461, .3379, .4407, .5626, .7230, 1]

s_int8 = W.abs().max().clamp_min(1e-8) / 127
s_4bit = W.abs().max().clamp_min(1e-8)

def int8_code(W, s):
    return torch.round(W / s).clamp(-127, 127).to(torch.int8)

def codebook_code(W, values, scale):
    codebook = torch.tensor(values, device=W.device, dtype=W.dtype)
    normalized = W / scale
    return (normalized[..., None] - codebook).abs().argmin(-1)

QW_INT8 = int8_code(W, s_int8)
QW_FP4 = codebook_code(W, FP4, s_4bit)
QW_NF4 = codebook_code(W, NF4, s_4bit)

for ratio in [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
    noise_std = ratio * s_int8
    W_noise = W + Z * noise_std

    noise_INT8 = int8_code(W_noise, s_int8)
    noise_FP4 = codebook_code(W_noise, FP4, s_4bit)
    noise_NF4 = codebook_code(W_noise, NF4, s_4bit)

    changed_INT8 = (QW_INT8 != noise_INT8).float().mean() * 100
    changed_FP4 = (QW_FP4 != noise_FP4).float().mean() * 100
    changed_NF4 = (QW_NF4 != noise_NF4).float().mean() * 100

    print(
        f"噪声/INT8步长={ratio:.2f}, "
        f"INT8={changed_INT8:.2f}%, "
        f"FP4={changed_FP4:.2f}%, "
        f"NF4={changed_NF4:.2f}%"
    )