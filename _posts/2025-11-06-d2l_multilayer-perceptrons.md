---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 4. 多层感知机"
date: 2025-11-06
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

最简单的深度网络称为多层感知机。多层感知机由多层神经元组成，每一层与它的上一层相连，从中接收输入；同时每一层也与它的下一层相连，影响当前层的神经元<!-- more -->

## 4.1. 多层感知机

### 4.1.1. 隐藏层

#### 4.1.1.1. 核心概念

**多层感知机（multilayer perceptron）**通过引入一个或多个隐藏层克服线性模型的局限性，可处理更一般的函数关系类型

架构：堆叠多个全连接层，前 L-1 层作为特征表示，最后一层作为线性预测器

隐藏层输出称为**隐藏变量（hidden variable）**

#### 4.1.1.2. 关键公式

含隐藏层的 MLP 计算（无激活函数）：

$$
\begin{split}\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}\\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
\end{aligned}\end{split}
$$

  - 此形式等价于线性模型，无实际增益

含激活函数的 MLP 计算：

$$
\begin{split}\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
\end{aligned}\end{split}
$$

  - $\sigma$ 为非线性激活函数，使模型摆脱线性限制

### 4.1.2. 激活函数

#### 4.1.2.1. ReLU 函数

定义：$\operatorname{ReLU}(x) = \max(x, 0)$

导数：输入为负时 0，输入为正时 1，0 处取左导数 0

PyTorch 实现：`torch.relu(x)`

#### 4.1.2.2. sigmoid 函数

定义：$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}$

导数：$\operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right)$

PyTorch 实现：`torch.sigmoid(x)`

#### 4.1.2.3. tanh 函数

定义：$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}$

导数：$1 - \operatorname{tanh}^2(x)$

PyTorch 实现：`torch.tanh(x)`

## 4.2. 多层感知机的从零开始实现

多层感知机包含输入层、隐藏层、输出层，此处实现含 1 个隐藏层

输入：Fashion-MNIST 数据集（28×28 灰度图，展平为 784 维特征）

输出：10 个类别（对应 10 种服饰）

### 4.2.1. 初始化模型参数

输入维度 `num_inputs=784`，输出维度 `num_outputs=10`，隐藏层维度 `num_hiddens=256`

权重参数：`W1`（784×256）、`W2`（256×10），随机初始化（缩放 0.01）

偏置参数：`b1`（256 维）、`b2`（10 维），初始化为 0

所有参数需计算梯度（`requires_grad=True`）

### 4.2.2. 激活函数

实现 ReLU 函数：`relu(X) = max(X, torch.zeros_like(X))`

### 4.2.3. 模型

输入展平：`X = X.reshape((-1, num_inputs))`

计算：隐藏层 `H = relu(X@W1 + b1)`，输出层 `output = H@W2 + b2`

### 4.2.4. 损失函数

使用 `nn.CrossEntropyLoss()`

### 4.2.5. 训练

优化器：随机梯度下降（SGD），学习率 `lr=0.1`

训练轮次 `num_epochs=10`

调用 `d2l.train_ch3` 进行训练

调用 `d2l.predict_ch3` 在测试集上预测

## 4.3. 多层感知机的简洁实现

### 4.3.1. 模型

网络结构：

  ```py
  net = nn.Sequential(nn.Flatten(),  # 展平输入
                      nn.Linear(784, 256),  # 隐藏层：784→256
                      nn.ReLU(),  # ReLU激活函数
                      nn.Linear(256, 10))  # 输出层：256→10
  ```

权重初始化：

  ```py
  def init_weights(m):
      if type(m) == nn.Linear:
          nn.init.normal_(m.weight, std=0.01)
  net.apply(init_weights)
  ```

训练配置:

  - 超参数：批量大小 256，学习率 0.1，训练轮次 10

  - 损失函数：`nn.CrossEntropyLoss()`

  - 优化器：`torch.optim.SGD(net.parameters(), lr=lr)`

训练过程:

  - 数据加载：`d2l.load_data_fashion_mnist(batch_size)`

  - 训练函数：`d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)`

## 4.4. 模型选择、欠拟合和过拟合

