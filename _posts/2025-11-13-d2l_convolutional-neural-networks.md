---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 6. 卷积神经网络"
date: 2025-11-13
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

本章开始我们将介绍构成所有卷积网络主干的基本元素。这包括卷积层本身、填充和步幅等细节、用于聚合相邻区域信息的池化层、各层多通道的使用，以及对现代卷积网络架构的仔细讨论。本章最后将完整实现 LeNet 网络——这是现代深度学习兴起前首个成功部署的卷积网络<!-- more -->

## 6.1. 从全连接层到卷积

全连接层处理高维感知数据（如图像）时，参数规模过大（如 100 万像素输入配 1000 隐藏单元需 10^9 参数），难以实现

图像存在可利用的结构，**卷积神经网络（CNNs）**通过利用这些结构解决上述问题

### 6.1.1. 关键设计原则

1. **平移不变性**：网络对相同图像块的响应不应受其位置影响，输入平移时隐藏表示也相应平移

2. **局部性**：网络早期层应关注局部区域，远距离区域内容不影响当前局部表示，后续可聚合局部表示

### 6.1.2. 卷积与相关概念

数学卷积定义（二维离散）：$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b)$

卷积层实际为**互相关（cross-correlation）**操作，与数学卷积的区别为索引方式，可通过调整核实现等价

### 6.1.3. 核心特性

卷积层参数远少于全连接层，仅与核大小、通道数相关

隐藏表示为三阶张量（高度、宽度、通道），通道也称为特征图

归纳偏置：依赖平移不变性和局部性，符合实际时泛化好，否则可能难以拟合数据

## 6.2. 图像卷积

### 6.2.1. 互相关运算

卷积层实际执行的是互相关运算

二维互相关：输入 tensor 与核 tensor 滑动窗口元素相乘求和

输出尺寸公式：$(n_h-k_h+1) \times (n_w-k_w+1)$，其中 $n_h \times n_w$ 为输入尺寸，$k_h \times k_w$ 为核尺寸

实现函数：

```py
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```

### 6.2.2. 卷积层

组成：互相关运算 + 标量偏置

参数：核（weight）和偏置（bias），训练时随机初始化

PyTorch 自定义实现：

```py
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

### 6.2.3. 核学习

可通过数据学习核参数，使用平方误差损失

PyTorch 内置卷积层：`nn.Conv2d(in_channels, out_channels, kernel_size, ...)`

输入输出格式：(批量大小，通道数，高度，宽度)

### 6.2.4. 互相关和卷积

严格卷积是对核进行水平和垂直翻转后再做互相关

深度学习中因核是学习得到，两者效果一致，通常统称卷积

### 6.2.5. 特征映射和感受野

**特征映射（feature map）**：卷积层输出，代表空间维度上的学习表征

**感受野（receptive field）**：某层元素在所有前层中影响其计算的元素集合

深层网络可获得更大感受野

## 6.3. 填充和步幅

### 6.3.1. 核心概念

**填充（padding）**：在输入图像边界添加额外像素（通常为 0），解决卷积后边界信息丢失及输出尺寸缩小问题

**步幅（stride）**：卷积窗口滑动时的步长，用于控制输出尺寸缩减程度，可提高计算效率

### 6.3.2. 计算公式

1. 输出尺寸计算：

    - 输入尺寸：$n_h \times n_w$

    - 核尺寸：$k_h \times k_w$

    - 填充：$p_h$（高度方向总填充）、$p_w$（宽度方向总填充）

    - 步幅：$s_h$（高度方向步幅）、$s_w$（宽度方向步幅）

    - 输出尺寸：$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor$

2. 等尺寸输出条件：

    当 $p_h = k_h - 1$ 且 $p_w = k_w - 1$ 时，输入输出尺寸相同（假设步幅为 1）

### 6.3.3. PyTorch 实现要点

卷积层类：`nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=1)`

填充参数：`padding` 表示单侧填充数（总填充为 2×padding）

步幅参数：`stride` 表示滑动步长

多维度设置： `kernel_size`、`padding`、`stride` 可设为元组 `(h, w)` 分别指定高度和宽度方向参数

## 6.4. 多输入多输出通道

含多通道时，输入和隐藏表示为三维张量（如 RGB 图像为 $3\times h\times w$），通道维度为额外维度

卷积操作需考虑输入通道数（$c_i$）和输出通道数（$c_o$）

### 6.4.1. 多输入通道

输入通道数为 $c_i$ 时，卷积核需同样具有 $c_i$ 个输入通道

卷积核形状为 $c_i\times k_h\times k_w$（$k_h,k_w$ 为核高宽）

计算方式：对每个通道的输入与对应通道的核做互相关，再将 $c_i$ 个结果求和

```py
# 多输入通道互相关
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

### 6.4.2. 多输出通道

输出通道数为 $c_o$ 时，需 $c_o$ 个卷积核（每个对应一个输出通道）

卷积核总形状为 $c_o\times c_i\times k_h\times k_w$

计算方式：每个输出通道由对应的核与输入做多输入通道卷积，结果堆叠为 $c_o$ 个通道

```py
# 多输入多输出通道互相关
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

### 6.4.3. $1\times 1$ 卷积层

核大小为 $1\times 1$，仅在通道维度计算

等价于在每个像素位置应用全连接层（权重共享）

作用：调整通道数、控制模型复杂度

参数数量：$c_o\times c_i$（含偏置）

```py
# 1x1卷积等价实现
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```

## 6.5. 汇聚层

**作用**

- 降低卷积层对位置的敏感性

- 对表示进行空间下采样，逐步减少空间分辨率，聚合信息使高层网络节点对更大输入区域敏感

### 6.5.1. 主要类型

1. **最大汇聚层（maximum pooling）**：计算池化窗口内元素的最大值

2. **平均汇聚层（average pooling）**：计算池化窗口内元素的平均值

3. **关键特性**

    - 无参数（无卷积核）

    - 操作确定性

    - 池化窗口按步长滑动，每个位置输出单值

### 6.5.2. 填充和步幅

可通过调整填充和步幅改变输出形状

与卷积层类似，用于控制输出尺寸

### 6.5.3. 多通道处理

对每个输入通道单独池化

输出通道数与输入通道数相同

### 6.5.4. PyTorch 实现要点

- 最大汇聚层：`nn.MaxPool2d(kernel_size, stride=None, padding=0)`

    - `kernel_size`：池化窗口大小

    - `stride`：步长，默认与 `kernel_size` 相同

    - `padding`：填充大小

- 平均汇聚层：`nn.AvgPool2d`（参数类似）

## 6.6. 卷积神经网络（LeNet）

### 6.6.1. 模型结构

- **组成部分**：2 个卷积块 + 3 个全连接层

- **卷积块结构**：

    - 第 1 块：`Conv2d(1, 6, kernel_size=5, padding=2)` → `Sigmoid` → `AvgPool2d(kernel_size=2, stride=2)`

    - 第 2 块：`Conv2d(6, 16, kernel_size=5)` → `Sigmoid` → `AvgPool2d(kernel_size=2, stride=2)`

- **全连接层**：

    - `Flatten()` → `Linear(16×5×5, 120)` → `Sigmoid`

    - `Linear(120, 84)` → `Sigmoid`

    - `Linear(84, 10)`（输出 10 类）

**关键操作**

- 卷积层使用 5×5 核，激活函数为 Sigmoid（受限于当时技术）

- 池化层为 2×2 平均池化，步长 2，降维 4 倍

- 输入需通过 `Reshape` 转为 (1, 28, 28) 格式

### 6.6.2. 模型训练

数据集：Fashion-MNIST

优化器：SGD

损失函数：`CrossEntropyLoss`

初始化：Xavier 均匀初始化

典型参数：批量大小 256，学习率 0.9，迭代 10 轮

训练流程：数据移至 GPU → 前向传播 → 计算损失 → 反向传播 → 参数更新
