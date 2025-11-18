---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 7. 现代卷积神经网络"
date: 2025-11-17
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

本章将介绍现代的卷积神经网络架构，许多现代卷积神经网络的研究都是建立在这些模型的基础上的。本章中的每一个模型都曾一度占据主导地位，其中许多模型都是 ImageNet 竞赛的优胜者<!-- more -->

## 7.1. 深度卷积神经网络（AlexNet）

### 7.1.1. 背景

2012 年由 Alex Krizhevsky 等人提出，在 ImageNet 挑战赛中表现优异，推动深度学习在计算机视觉领域的发展

突破点：证明学习到的特征可超越手工设计特征，打破传统计算机视觉范式

关键支撑：大规模数据集（ImageNet）和 GPU 硬件加速

### 7.1.2. AlexNet

#### 7.1.2.1. 与 LeNet 对比

更深：8 层结构（5 个卷积层 + 2 个全连接隐藏层 + 1 个全连接输出层）

更宽：卷积通道数远超 LeNet

激活函数：使用 ReLU 替代 sigmoid

正则化：引入 dropout（LeNet 仅用权重衰减）

#### 7.1.2.2. 核心架构（PyTorch 实现）

```py
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)  # 针对Fashion-MNIST的10类输出
)
```

#### 7.1.2.3. 关键设计

卷积层：首次使用 11×11 大卷积核（适应高分辨率图像），后续逐步减小为 5×5、3×3

池化层：3×3 窗口，步长 2，用于降低维度

全连接层：两个 4096 维的大尺寸层，配合 dropout (0.5) 抑制过拟合

数据增强：通过翻转、裁剪、颜色变化等提升模型鲁棒性

### 7.1.3. 训练 AlexNet

输入：Fashion-MNIST 图像需 resize 至 224×224

批次大小：128

学习率：0.01

优化器：默认使用 SGD（结合 d2l.train_ch6 训练函数）

## 7.2. 使用块的网络（VGG）

### 7.2.1. VGG 块

每个块由多个卷积层 + 1 个最大池化层组成

卷积层：$3×3$ 核，padding=1（保持分辨率），ReLU 激活

池化层：$2×2$ 核，stride=2（分辨率减半）

PyTorch 实现：

```py
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

### 7.2.2. VGG 网络

分为两部分：卷积部分 + 全连接部分

卷积部分：堆叠多个 VGG 块

- VGG-11 的卷积架构：`conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))`

- 共 8 个卷积层（按元组中数字累加）

全连接部分：3 个全连接层（与 AlexNet 类似）

- 含 Dropout (0.5) 防止过拟合

- 输出层为 10 类（针对 Fashion-MNIST）

PyTorch 实现：

```py
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
```

### 7.2.3. 训练模型

为降低计算量，可缩小通道数（如除以 4）：`small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]`

输入图像大小：224×224

超参数：学习率 0.05，轮次 10，批大小 128

## 7.3. 网络中的网络（NiN）

### 7.3.1. NiN 块

结构：1 个普通卷积层 + 2 个 1×1 卷积层（均带 ReLU 激活）

1×1 卷积作用：实现像素级全连接，增加非线性表达能力

```py
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

### 7.3.2. NiN 模型

整体结构：

- 多个 NiN 块与最大池化层交替（池化层：3×3 窗口，步长 2）

- 包含一个 Dropout 层（dropout=0.5）

- 最后用 NiN 块将通道数转为类别数（10 类）

- 用自适应平均池化（`AdaptiveAvgPool2d((1, 1))`）替代全连接层

- 最终展平为二维输出（批量大小 × 类别数）

```py
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())
```

### 7.3.3. 训练模型

学习率 0.1，轮次 10，批量大小 128

训练数据：Fashion-MNIST（resize 至 224×224）

## 7.4. 含并行连结的网络（GoogLeNet）

### 7.4.1. Inception 块

包含 4 条并行路径：

1. 1×1 卷积层（直接输出）

2. 1×1 卷积 → 3×3 卷积（降维后提取中等尺度特征）

3. 1×1 卷积 → 5×5 卷积（降维后提取大尺度特征）

4. 3×3 最大池化 → 1×1 卷积（池化后调整通道）

所有路径保持输入输出的高宽一致（通过 padding）

输出沿通道维度拼接

```py
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(** kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)  # 路径1
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)  # 路径2-1
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)  # 路径2-2
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)  # 路径3-1
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)  # 路径3-2
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 路径4-1
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)  # 路径4-2

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 通道维度拼接
```

### 7.4.2. GoogLeNet 模型

b1 模块：7×7 卷积（64 通道）→ ReLU → 3×3 最大池化

b2 模块：1×1 卷积（64 通道）→ ReLU → 3×3 卷积（192 通道）→ 3×3 最大池化

b3 模块：2 个 Inception 块 → 3×3 最大池化

b4 模块：5 个 Inception 块 → 3×3 最大池化

b5 模块：2 个 Inception 块 → 自适应最大池化 → 展平

```py
net = nn.Sequential(
    b1, b2, b3, b4, b5,  # 上述模块
    nn.Linear(1024, 10)  # 最终分类层
)
```

### 7.4.3. 训练模型

输入图像尺寸：96×96（简化计算）

优化参数：学习率 0.1，批次大小 128，训练 10 轮

数据集：Fashion-MNIST（resize 为 96×96）

## 7.5. 批量规范化

### 7.5.1. 核心概念

批量规范化：训练深层网络时，对各层输入进行标准化处理，加速收敛，稳定训练

操作：对小批量数据进行均值和方差标准化，再应用可学习的拉伸（$\gamma$）和偏移（$\beta$）参数

公式：

$$
\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}
$$

其中 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ 为小批量均值，$\hat{\boldsymbol{\sigma}}_\mathcal{B}$ 为小批量标准差（加 $\epsilon$ 防除零）

### 7.5.2. 关键特性

训练/预测模式差异：

- 训练：用当前小批量均值和方差

- 预测：用训练过程中累积的移动平均均值和移动平均方差

正则化作用：小批量统计带来的噪声可减少过拟合

批量大小影响：需足够大（通常 50~100），否则效果差

### 7.5.3. 实现细节

全连接层：在特征维度计算均值和方差，形状为 (1, num_features)

卷积层：在通道维度计算均值和方差（含所有空间位置），形状为 (1, num_features, 1, 1)

PyTorch 实现：

- 自定义层：`class BatchNorm(nn.Module)`，含 `gamma`、`beta`、`moving_mean`、`moving_var` 参数

- 框架 API：`nn.BatchNorm1d(num_features)`（全连接层）、`nn.BatchNorm2d(num_features)`（卷积层）

## 7.6. 残差网络（ResNet）

### 7.6.1. 核心思想

深层网络需保证函数类嵌套性（$\mathcal{F} \subseteq \mathcal{F}'$），确保增加层数能提升性能

核心创新：残差块（residual block），使新增层易于拟合恒等映射（$f(\mathbf{x}) = \mathbf{x}$）

残差映射（$f(\mathbf{x}) - \mathbf{x}$）比直接拟合映射更易优化

### 7.6.2. 残差块

结构：2 个 $3 \times 3$ 卷积层，每层后接批量规范化和 ReLU

跨层残差连接：输入直接加在第二个卷积层输出前，再经 ReLU

通道数变化时：用 $1 \times 1$ 卷积调整输入形状后再相加

```py
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

### 7.6.3. ResNet 模型结构

1. 初始层：$7 \times 7$ 卷积（64 通道，步幅 2）→ BatchNorm → ReLU → $3 \times 3$ 最大池化（步幅 2）

2. 4 个残差模块：

    - 每个模块含多个残差块，通道数依次为 64、128、256、512

    - 非首个模块的第一个残差块用 $1 \times 1$ 卷积翻倍通道数并减半尺寸

3. 输出层：全局平均池化 → 全连接层（10 类输出）

```py
# 模块构建
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 完整网络（ResNet-18）
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

## 7.7. 稠密连接网络（DenseNet）

### 7.7.1. 核心思想

与 ResNet 的残差连接（相加）不同，DenseNet 采用稠密连接，通过通道维度上的连结融合特征

函数映射形式：$\mathbf{x} \to [\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})]), \ldots]$

### 7.7.2. 主要组件

1. 卷积块

    ```py
    def conv_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        )
    ```

2. **稠密块（dense block）**

    ```py
    class DenseBlock(nn.Module):
        def __init__(self, num_convs, input_channels, num_channels):
            super().__init__()
            layer = []
            for i in range(num_convs):
                layer.append(conv_block(num_channels*i + input_channels, num_channels))
            self.net = nn.Sequential(*layer)
        def forward(self, X):
            for blk in self.net:
                Y = blk(X)
                X = torch.cat((X, Y), dim=1)  # 通道维度连结
            return X
    ```

3. **过渡层（transition layer）**

    控制模型复杂度，减少通道数并减半空间维度

    ```py
    def transition_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    ```

### 7.7.3. 网络结构

1. 初始模块：7×7 卷积（64 通道）+ 3×3 最大池化

2. 4 个稠密块（每个含 4 个卷积层，增长率 32）

3. 稠密块间通过过渡层连接（通道数减半）

4. 最终：全局平均池化 + 全连接层（10 类输出）

### 7.7.4. 训练配置

学习率 0.1，轮次 10，批次大小 256

输入图像大小调整为 96×96
