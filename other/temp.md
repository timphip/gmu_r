import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
W = torch.randn(4, 8, device=device) * 0.1

FP4 = [-1, -.6667, -.5, -.3333, -.25, -.1667, -.0052,
       0, .0052, .1667, .25, .3333, .5, .6667, 1]

NF4 = [-1, -.6962, -.5251, -.3949, -.2844, -.1848, -.0911,
       0, .0796, .1609, .2461, .3379, .4407, .5626, .7230, 1]

def quantize_int8(W):
    s = W.abs().max() / 127
    return s * torch.round(W / s).clamp(-127, 127)

def quantize_codebook(W, values):
    codebook = torch.tensor(values, device=W.device)
    s = W.abs().max().clamp_min(1e-8)
    normalized = W / s
    index = (normalized[..., None] - codebook).abs().argmin(-1)
    return codebook[index] * s

results = {
    "INT8": quantize_int8(W),
    "FP4": quantize_codebook(W, FP4),
    "NF4": quantize_codebook(W, NF4),
}

print("原始权重：", W[0].cpu())

for name, QW in results.items():
    error = (W - QW).abs().mean()
    print(f"\n{name}量化：", QW[0].cpu())
    print(f"{name}平均误差：{error.item():.8f}")




---
---


torch.manual_seed(0)
W = torch.randn(4, 8)
它让每次运行时，随机生成的这个 \(4\times8\) 矩阵 \(W\) 数值相同。
(4, 8)决定矩阵形状：4行、8列。
manual_seed(0)决定其中随机数的生成结果。






对于**对称量化**，确实可以这样替换：

```python
# 对称 INT8
s = W.abs().max() / 127

# 对称 INT4
s = W.abs().max() / 7
```

通用公式是：

\[
s=\frac{\max|W|}{2^{N-1}-1}
\]

但非对称 UINT8不能只把 `127`换成 `255`，通常是：

\[
s=\frac{W_{\max}-W_{\min}}{255}
\]

并且还需要一个 `zero_point`，把原来的零映射到 `[0,255]`中的适当位置。

因此：

- INT8 → INT4：主要替换 `127→7`。
- 对称 → 非对称：整个公式都会变化，不只是分母。


---
---


两个函数的**总任务相同**：

\[
W\rightarrow计算s\rightarrow W/s\rightarrow选择离散值\rightarrow Q(W)
\]

区别在于“如何选择离散值”。

### `quantize_int8`

它是**均匀量化**：

```python
s = W.abs().max() / 127
QW = s * round(W / s)
```

\(W/s\)只能变成 `-127～127`之间的整数，相邻量化值距离相同。这里的 \(s\)就是相邻量化值的间隔。

### `quantize_codebook`

它用于 FP4/NF4等**非均匀量化**：

```python
s = W.abs().max()
normalized = W / s
```

然后不使用 `round()`，而是从 codebook中寻找距离最近的数。相邻 codebook值之间的距离可以不同。这里的 \(s\)主要负责把权重缩放到约 \([-1,1]\)，不一定是相邻值的间隔。

你现在需要回答下面四个关键问题：

1. 输入是什么？  
   输入是连续的全精度权重 \(W\)，最终想得到什么 \(Q(W)\)？

2. 为什么计算 \(s\)？  
   \(s\)是量化间隔，还是仅用于把 \(W\)缩放到 codebook范围？

3. 谁决定 \(W/s\)最后落在哪里？  
   INT8使用 `round()`；FP4/NF4使用“寻找最近的 codebook值”。

4. 为什么最后还要乘回 \(s\)？  
   因为 \(W/s\)只是归一化的中间值，乘回 \(s\)才能回到原始权重的数值尺度。

抓住这一条主线即可：

> 两个函数都是“缩放—选择离散值—还原”，只是 INT8选择整数，FP4/NF4选择 codebook。









