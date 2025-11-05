---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 3. 线性神经网络"
date: 2025-11-03
tags: [AI, notes]
toc: true
comments: true
author: pianfan
---

在深入探讨深度神经网络之前，我们需要先掌握神经网络训练的基础知识，包括定义简单神经网络架构、处理数据、指定损失函数以及训练模型<!-- more -->

## 3.1. 线性回归

**回归（regression）**：建模自变量与因变量之间的关系，机器学习中主要用于预测数值型目标

### 3.1.1. 线性回归的基本元素

**线性回归（linear regression）**：假设自变量与因变量间存在线性关系，即目标可表示为特征的加权和加偏置

#### 3.1.1.1. 线性模型

- 单样本预测：$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$，其中 $\mathbf{w}$ 为权重向量，$b$ 为偏置

- 批量预测：${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b$，$\mathbf{X}$ 为设计矩阵 $(n \times d)$

#### 3.1.1.2. 损失函数

- 单个样本损失：$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$（平方误差）

- 总体损失：$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b)$

#### 3.1.1.3. 解析解

**解析解（analytical solution）**：$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}$（将偏置并入权重向量时）

#### 3.1.1.4. 随机梯度下降

**小批量随机梯度下降（minibatch stochastic gradient descent）**：

  - 迭代更新：$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{\vert\mathcal{B}\vert} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)$

  - 其中 $\eta$ 为**学习率（learning rate）**，$\vert\mathcal{B}\vert$ 为**批量大小（batch size）**，均为**超参数（hyperparameter）**

#### 3.1.1.5. 用模型进行预测

预测过程：使用学习到的 $\hat{\mathbf{w}}$，$\hat{b}$ 计算 $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$

### 3.1.2. 矢量化加速

矢量化计算可大幅提升效率，避免显式循环

### 3.1.3. 正态分布与平方损失

假设噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$时，最小化平方损失等价于最大化似然估计

正态分布概率密度：$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right)$

### 3.1.4. 从线性回归到深度网络

#### 3.1.4.1. 神经网络图

- 线性回归可视为单神经元神经网络

- 属于全连接层结构

- 不计输入层时，层数为 1

## 3.2. 线性回归的从零开始实现

### 3.2.1. 生成数据集

人工数据集基于线性模型生成：$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon$

$\epsilon$ 为服从均值 0、标准差 0.01 的正态分布噪声

```py
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
```

### 3.2.2. 读取数据集

实现小批量随机读取

```py
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

### 3.2.3. 初始化模型参数

权重随机初始化（正态分布，均值 0，标准差 0.01）

偏置初始化为 0

需记录梯度用于更新

```py
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

### 3.2.4. 定义模型

线性回归模型

```py
def linreg(X, w, b):
    return torch.matmul(X, w) + b
```

### 3.2.5. 定义损失函数

采用平方损失

```py
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### 3.2.6. 定义优化算法

小批量随机梯度下降

```py
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

### 3.2.7. 训练

迭代过程：

  1. 读取小批量数据
  2. 计算模型预测与损失
  3. 反向传播计算梯度
  4. 梯度下降更新参数

```py
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

## 3.3. 线性回归的简洁实现

### 3.3.1. 生成数据集

使用 `d2l.synthetic_data` 生成符合线性关系的数据集，包含特征 `features` 和标签 `labels`

### 3.3.2. 读取数据集

定义 `load_array` 函数，利用 `torch.utils.data.TensorDataset` 和 `DataLoader` 构建数据迭代器

可指定 `batch_size` 和是否打乱数据（`shuffle`）

### 3.3.3. 定义模型

用 `nn.Sequential` 构建模型容器，包含一个全连接层 `nn.Linear(2, 1)`（输入特征数为 2，输出为 1）

### 3.3.4. 初始化模型参数

访问层参数：`net[0].weight.data`（权重）、`net[0].bias.data`（偏置）

权重初始化：`normal_(0, 0.01)`（均值 0，标准差 0.01 的正态分布）

偏置初始化：`fill_(0)`（初始化为 0）

### 3.3.5. 定义损失函数

使用 `nn.MSELoss`（均方误差损失）

### 3.3.6. 定义优化算法

采用随机梯度下降：`torch.optim.SGD(net.parameters(), lr=0.03)`，学习率 `lr=0.03`

### 3.3.7. 训练

1. 迭代指定轮次（`num_epochs`）

2. 每轮遍历所有迷你批次：

    - 前向传播：计算预测值 `net(X)` 和损失 `l = loss(net(X), y)`
    - 梯度清零：`trainer.zero_grad()`
    - 反向传播：`l.backward()`
    - 参数更新：`trainer.step()`

3. 每轮结束计算总损失并输出

训练后获取权重 `net[0].weight.data` 和偏置 `net[0].bias.data`，与真实值比较误差

## 3.4. softmax 回归

### 3.4.1. 分类问题

目标：解决“哪一个”的问题，输出类别归属

标签表示：采用**独热编码（one-hot encoding）**，即对于 q 个类别，标签向量中仅对应类别位置为 1，其余为 0

### 3.4.2. 网络架构

单层神经网络，属于线性模型

输出层为全连接层，每个类别对应一个**仿射函数（affine function）**

数学表达：$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$，其中 $\mathbf{W}$ 为权重矩阵，$\mathbf{b}$ 为偏置向量

### 3.4.3. softmax 运算

作用：将 logits 转换为概率分布（非负且和为 1）

公式：$\hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$

特性：不改变 logits 的排序

### 3.4.4. 小批量样本的矢量化

矩阵形式：$\mathbf{O} = \mathbf{X} \mathbf{W} + \mathbf{b}$，$\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{O})$

维度：$\mathbf{X} \in \mathbb{R}^{n \times d}$，$\mathbf{W} \in \mathbb{R}^{d \times q}$，$\mathbf{b} \in \mathbb{R}^{1\times q}$，输出 $\hat{\mathbf{Y}} \in \mathbb{R}^{n \times q}$

### 3.4.5. 损失函数

**交叉熵损失（cross-entropy loss）**：$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j$

简化形式（代入 softmax 后）：$l = \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j$

梯度：$\partial_{o_j} l = \mathrm{softmax}(\mathbf{o})_j - y_j$

### 3.4.6. 信息论基础

**熵（entropy）**：$H[p] = \sum_j - p(j) \log p(j)$，表示编码数据的最小信息量

交叉熵：$H(p, q)$ 表示用分布 q 编码来自 p 的数据的期望信息量，$H(p, q) \ge H(p)$

### 3.4.7. 模型预测和评估

预测：选择概率最高的类别

**精度（accuracy）**：正确预测数与预测总数之间的比率

## 3.5. 图像分类数据集

选用 Fashion-MNIST 替代 MNIST 作为基准数据集，因 MNIST 过于简单，不适合区分模型强弱

### 3.5.1. 数据集组成

包含 10 个类别：t-shirt、trouser、pullover、dress、coat、sandal、shirt、sneaker、bag、ankle boot

训练集：60000 张图像（每个类别 6000 张）

测试集：10000 张图像（每个类别 1000 张）

图像规格：28×28 像素，灰度图（单通道）

### 3.5.2. 数据加载

1. 数据转换

    ```py
    trans = transforms.ToTensor()  # 转换为32位浮点张量，像素值归一化到[0,1]
    ```

2. 加载数据集

    ```py
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    ```

3. 数据迭代器

    ```py
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers())
    ```

    - `batch_size`：每次读取的批量大小
    - `shuffle=True`：训练集随机打乱
    - 多进程读取：通常使用 4 个进程

### 3.5.3. 核心函数

1. 标签转换函数

    ```py
    def get_fashion_mnist_labels(labels):  # 将数字标签转换为文本标签
    ```

2. 数据加载封装函数

    ```py
    def load_data_fashion_mnist(batch_size, resize=None):  # 下载并加载数据，支持图像resize
    ```

    - 返回训练集和测试集的迭代器
    - 可通过 `resize` 参数调整图像尺寸

## 3.6. softmax 回归的从零开始实现

使用 Fashion-MNIST 数据集

批量大小设为 256：batch_size = 256

加载数据迭代器：`train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)`

### 3.6.1. 初始化模型参数

输入维度：784（28×28 图像展平）

输出维度：10（10 个类别）

权重 `W`：形状 (784, 10)，用均值 0、标准差 0.01 的正态分布初始化，需计算梯度

偏置 `b`：形状 (10)，初始化为 0，需计算梯度

```py
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

### 3.6.2. 定义 softmax 操作

公式：$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}$

实现：先对输入元素指数化，再按行求和得到归一化常数，最后每行元素除以对应归一化常数

```py
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
```

### 3.6.3. 定义模型

将输入展平为向量，再通过矩阵乘法和偏置相加得到 logits，最后应用 softmax

```py
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

### 3.6.4. 定义损失函数

取真实标签对应的预测概率的负对数

```py
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
```

### 3.6.5. 分类精度

计算预测类别（取每行最大概率索引）与真实标签匹配的比例

```py
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

计算模型在数据集上的精度

```py
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设为评估模式
    metric = Accumulator(2)  # 正确预测数、总预测数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

### 3.6.6. 训练

1. 单轮训练：迭代训练数据，计算梯度并更新参数，记录损失和精度

    ```py
    def train_epoch_ch3(net, train_iter, loss, updater):
        if isinstance(net, torch.nn.Module):
            net.train()  # 设为训练模式
        metric = Accumulator(3)  # 训练损失总和、训练精度总和、样本数
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]
    ```

2. 多轮训练：迭代多个 epoch，每轮结束评估模型在测试集上的精度并可视化

    ```py
    def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
    ```

3. 优化器：使用小批量随机梯度下降，学习率 0.1

    ```py
    lr = 0.1
    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)
    ```

4. 训练配置：迭代 10 个 epoch

    ```py
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    ```

### 3.6.7. 预测

对测试集图像进行预测并对比真实标签

```py
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
```

## 3.7. softmax 回归的简洁实现

批量大小设为 256

使用 `d2l.load_data_fashion_mnist` 加载 Fashion-MNIST 数据集，获取训练迭代器 `train_iter` 和测试迭代器 `test_iter`

### 3.7.1. 初始化模型参数

网络为 `Sequential`，包含展平层（`nn.Flatten()`）和全连接层（`nn.Linear(784, 10)`）

权重初始化：使用正态分布（`nn.init.normal_`），标准差 0.01

### 3.7.2. 重新审视 Softmax 的实现

采用 `nn.CrossEntropyLoss()`，内部结合 softmax 和交叉熵计算，避免数值稳定性问题

### 3.7.3. 优化算法

使用小批量随机梯度下降，学习率 0.1

通过 `torch.optim.SGD(net.parameters(), lr=0.1)` 定义优化器

### 3.7.4. 训练

训练轮次 `num_epochs = 10`

调用 `d2l.train_ch3` 函数进行模型训练，输入网络、数据迭代器、损失函数、轮次和优化器
