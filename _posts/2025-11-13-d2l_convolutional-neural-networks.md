---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 6. 卷积神经网络"
date: 2025-11-13
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

本章开始我们将介绍构成所有卷积网络主干的基本元素。这包括卷积层本身、填充和步幅等细节、用于聚合相邻区域信息的池化层、各层多通道的使用，以及对现代卷积网络架构的仔细讨论。本章最后将完整实现 LeNet 网络——这是现代深度学习兴起前首个成功部署的卷积网络。<!-- more -->

## 6.1. 从全连接层到卷积

全连接层处理高维感知数据（如图像）时，参数规模过大（如 100 万像素输入配 1000 隐藏单元需 10^9 参数），难以实现

图像存在可利用的结构，**卷积神经网络（CNNs）**通过利用这些结构解决上述问题

### 6.1.1. 关键设计原则

1. **平移不变性**：网络对相同图像块的响应不应受其位置影响，输入平移时隐藏表示也相应平移

2. **局部性**：网络早期层应关注局部区域，远距离区域内容不影响当前局部表示，后续可聚合局部表示

### 6.1.2. 卷积层的数学表达

单通道：$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}$，$\mathbf{V}$ 为卷积核（滤波器）

多通道：$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c}$，$d$ 为输出通道索引

### 6.1.3. 卷积与相关概念

数学卷积定义（二维离散）：$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b)$

卷积层实际为**互相关（cross-correlation）**操作，与数学卷积的区别为索引方式，可通过调整核实现等价

### 6.1.4. 核心特性

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

