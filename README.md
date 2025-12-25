# DiP: Taming Diffusion Models in Pixel Space

[English](#english) | [中文](#中文)

---

## English

### Overview

Unofficial implementation of **DiP** based on the paper:

📄 **[DiP: Taming Diffusion Models in Pixel Space](https://arxiv.org/abs/2511.18822)**

### Key Features

- **Patch Detailer Head**: Lightweight U-Net for local texture refinement
- **Global Semantic Injection**: DiT features concatenated at U-Net bottleneck (1×1)
- **Minimal Overhead**: Only +0.3% parameters over baseline DiT
- **State-of-the-art**: FID 1.79 on ImageNet 256×256, 10× faster than PixelFlow

### Quick Start

```python
from dip_model import DiP

# Create model
model = DiP(
    input_size=256,
    patch_size=16,
    hidden_size=1152,
    patch_depth=26,
    num_classes=1000,
)

# Forward
x = torch.randn(2, 3, 256, 256)
t = torch.randint(0, 1000, (2,))
y = torch.randint(0, 1000, (2,))
noise_pred = model(x, t, y)
```

### Requirements

```
torch
einops
timm
numpy
```

### Contact

If you have any questions or suggestions, feel free to reach out!

---

## 中文

### 概述

基于论文的 **DiP** 非官方实现：

📄 **[DiP: Taming Diffusion Models in Pixel Space](https://arxiv.org/abs/2511.18822)**

### 核心特性

- **Patch Detailer Head**：轻量级 U-Net 用于局部纹理细化
- **全局语义注入**：DiT 特征在 U-Net bottleneck (1×1) 处 concat
- **极小开销**：相比 DiT 仅增加 0.3% 参数
- **SOTA 性能**：ImageNet 256×256 FID 1.79，比 PixelFlow 快 10 倍

### 快速开始

```python
from dip_model import DiP

# 创建模型
model = DiP(
    input_size=256,
    patch_size=16,
    hidden_size=1152,
    patch_depth=26,
    num_classes=1000,
)

# 前向传播
x = torch.randn(2, 3, 256, 256)
t = torch.randint(0, 1000, (2,))
y = torch.randint(0, 1000, (2,))
noise_pred = model(x, t, y)
```

### 依赖

```
torch
einops
timm
numpy
```

### 联系方式

如有任何问题或建议，欢迎随时联系我！

---
