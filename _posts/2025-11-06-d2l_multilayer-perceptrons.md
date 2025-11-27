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

#### 4.1.1.2. 公式

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

实现：`torch.relu(x)`

#### 4.1.2.2. sigmoid 函数

定义：$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}$

导数：$\operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right)$

实现：`torch.sigmoid(x)`

#### 4.1.2.3. tanh 函数

定义：$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}$

导数：$1 - \operatorname{tanh}^2(x)$

实现：`torch.tanh(x)`

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

**核心目标**：发现具有**泛化能力**的模式

### 4.4.1. 训练误差和泛化误差

**训练误差（training error）**：模型在训练数据集上的误差

**泛化误差（generalization error）**：模型在来自相同分布的无限新数据上的期望误差（无法精确计算，需用测试集估计）

#### 4.4.1.1. 统计学习理论

**i.i.d. 假设**：训练数据和测试数据独立同分布

训练误差会收敛到泛化误差（Glivenko-Cantelli 定理）

#### 4.4.1.2. 模型复杂性

影响模型复杂性（模型泛化）的因素：

1. 可调整参数数量（**自由度**）
2. 参数取值范围
3. 训练迭代次数
4. 训练样本数量

### 4.4.2. 模型选择

**验证集（validation set）**：用于模型选择，避免使用测试集进行选择（防止过拟合测试集）

**K 折交叉验证**：训练数据稀缺时使用，将数据分为 K 个子集，轮流用 K-1 个子集训练，1 个子集验证，结果取平均

### 4.4.3. 欠拟合还是过拟合？

**欠拟合（underfitting）**：模型过于简单，无法捕捉数据模式，训练误差和泛化误差都较高且差距小

**过拟合（overfitting）**：模型过度拟合训练数据，训练误差远低于泛化误差

**正则化（regularization）**：对抗过拟合的技术

### 4.4.4. 多项式回归

生成数据：基于三次多项式 $y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon$

关键函数：

`evaluate_loss(net, data_iter, loss)`：评估模型在数据集上的损失

`train(...)`：训练模型，使用 MSELoss 和 SGD 优化器

#### 4.4.4.1. 三种拟合情况

**正常拟合**（3 阶多项式）：训练和测试损失均低，参数接近真实值

**欠拟合**（线性函数）：训练损失难降低，无法拟合非线性模式

**过拟合**（高阶多项式）：训练损失低但测试损失高，受噪声影响大

## 4.5. 权重衰减

**权重衰减（weight decay）**：常用的参数化机器学习模型正则化技术

核心思想：通过在损失函数中添加权重向量的 $L_2$ 范数作为惩罚项，限制权重大小，降低模型复杂度，缓解过拟合

原始损失函数（线性回归示例）：$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2$

带 $L_2$ 正则化的损失函数：$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2$，其中 $\lambda$ 为非负**正则化常数**，控制惩罚强度

小批量随机梯度下降更新公式：

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)
\end{aligned}
$$

- 权重在更新时被向零方向衰减

### 4.5.1. 从零开始实现

```py
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
# 训练时将惩罚项加入损失
l = loss(net(X), y) + lambd * l2_penalty(w)
```

### 4.5.2. 简洁实现

```py
trainer = torch.optim.SGD([
    {"params": net[0].weight, 'weight_decay': wd},  # 对权重应用衰减
    {"params": net[0].bias}  # 不对偏置应用衰减
], lr=lr)
```

## 4.6. 暂退法（Dropout）

### 4.6.1. 基本概念

一种用于改善深度神经网络泛化能力的正则化技术

训练时在每层前向传播中随机“丢弃”部分神经元（置零）

通过打破神经元间的共适应减少过拟合

### 4.6.2. 数学原理

给定丢弃概率 $p$，每个激活值 $h$ 变为随机变量 $h'$：

$$
\begin{split}\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 概率为 } p \\
    \frac{h}{1-p} & \text{ 概率为 } 1 - p
\end{cases}
\end{aligned}\end{split}
$$

设计保证 $E[h'] = h$（期望不变）

### 4.6.3. 实践要点

仅在训练时启用，测试时禁用

不同层可设置不同丢弃概率，通常输入层概率较低

测试时无需规范化处理

### 4.6.4. 实现

1. 手动实现 dropout 层：

    ```py
    def dropout_layer(X, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1:
            return torch.zeros_like(X)
        if dropout == 0:
            return X
        mask = (torch.rand(X.shape) > dropout).float()
        return mask * X / (1.0 - dropout)
    ```

2. 含 dropout 的网络定义：

    ```py
    class Net(nn.Module):
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
            super(Net, self).__init__()
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()

        def forward(self, X):
            H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
            if self.training:  # 仅训练时应用
                H1 = dropout_layer(H1, dropout1)
            H2 = self.relu(self.lin2(H1))
            if self.training:
                H2 = dropout_layer(H2, dropout2)
            return self.lin3(H2)
    ```

3. 简洁实现（使用内置层）：

    ```py
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(dropout1),  # 内置dropout层
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, 10)
    )
    ```

## 4.7. 前向传播、反向传播和计算图

**前向传播（forward propagation）**：按从输入层到输出层的顺序计算并存储神经网络的中间变量（含输出）

**反向传播（backward propagation）**：按从输出到输入层的反向顺序，根据微积分链式法则计算神经网络参数的梯度，存储所需中间变量（偏导数）

**计算图**：可视化计算中算子与变量的依赖关系，方块表示变量，圆圈表示算子

### 4.7.1. 前向传播关键公式

隐藏层中间变量：$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x}$

隐藏层激活向量：$\mathbf{h}= \phi (\mathbf{z})$（$\phi$ 为激活函数）

输出层变量：$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}$

单个样本损失项：$L = l(\mathbf{o}, y)$（$l$ 为损失函数，$y$ 为标签）

$L_2$ 正则化项：$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right)$

正则化损失（目标函数）：$J = L + s$

### 4.7.2. 反向传播梯度计算

目标函数对损失项和正则项梯度：$\frac{\partial J}{\partial L} = 1,\;\frac{\partial J}{\partial s} = 1$

目标函数对输出层变量梯度：$\frac{\partial J}{\partial \mathbf{o}} = \frac{\partial L}{\partial \mathbf{o}}$

正则项对参数梯度：$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)},\;\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}$

目标函数对输出层权重梯度：$\frac{\partial J}{\partial \mathbf{W}^{(2)}} = \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}$

目标函数对隐藏层输出梯度：$\frac{\partial J}{\partial \mathbf{h}} = {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}$

目标函数对隐藏层中间变量梯度：$\frac{\partial J}{\partial \mathbf{z}} = \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right)$（$\odot$ 为元素乘法）

目标函数对输入层权重梯度：$\frac{\partial J}{\partial \mathbf{W}^{(1)}} = \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}$

### 4.7.3. 神经网络训练特点

前向传播与反向传播相互依赖，训练时交替进行

反向传播复用前向传播存储的中间值，减少重复计算

训练比预测需要更多内存（需保留中间值直至反向传播完成）

中间值大小与网络层数和批量大小大致成正比，深层网络 + 大批量易导致内存不足

## 4.8. 数值稳定性和模型初始化

### 4.8.1. 梯度消失和梯度爆炸

深层网络中，梯度是多层矩阵乘积与梯度向量的结果，易因矩阵特征值乘积过大/过小导致梯度爆炸/消失，影响优化稳定性

**对称性问题**：神经网络参数化存在对称性（如 MLP 隐藏单元的排列对称），若初始化不当（如参数全为相同值），会导致隐藏单元功能等价，无法发挥网络表达能力

**激活函数影响**:

- Sigmoid 函数梯度在输入过大或过小时会消失，深层网络中易导致梯度传递中断

- ReLU 能缓解梯度消失问题，加速收敛，成为实践中的默认选择

### 4.8.2. 参数初始化

1. **默认初始化**：框架采用默认随机初始化（如正态分布），适用于中等规模问题

2. **Xavier 初始化**：

    目标：使各层输出方差不受输入数量影响，梯度方差不受输出数量影响

    高斯分布：均值 0，方差 $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$（$n_{in}$ 为输入数，$n_{out}$ 为输出数）

    均匀分布：$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right)$

## 4.9. 环境和分布偏移

**分布偏移**：训练数据分布 $p_S(\mathbf{x},y)$与测试数据分布 $p_T(\mathbf{x},y)$ 不同，可能导致模型部署失败

**真实风险**：模型在真实数据分布上的期望损失 $E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)]$

**经验风险**：训练数据上的平均损失 $\frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i)$，用于近似真实风险

**经验风险最小化**：通过最小化经验风险训练模型的策略

### 4.9.1. 分布偏移的类型

1. **协变量偏移（covariate shift）**

    - 输入分布变化（$p_S(\mathbf{x}) \neq p_T(\mathbf{x})$）

    - 条件分布不变（$p_S(y \mid \mathbf{x}) = p_T(y \mid \mathbf{x})$）

    - 适用于 $\mathbf{x}$ 导致 $y$ 的场景

2. **标签偏移（label shift）**

    - 标签边缘分布变化（$p_S(y) \neq p_T(y)$）

    - 类条件分布不变（$p_S(\mathbf{x} \mid y) = p_T(\mathbf{x} \mid y)$）

    - 适用于 $y$ 导致 $\mathbf{x}$ 的场景

3. **概念偏移（concept shift）**

    - 标签定义本身发生变化（$p(y \mid \mathbf{x})$ 改变）

    - 常随时间/地理等因素逐渐发生

### 4.9.2. 分布偏移纠正

1. **协变量偏移纠正**

    - 计算权重 $\beta_i = \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}$（目标/源分布密度比）

    - 通过逻辑回归训练二分类器区分源/目标分布估计 $h(\mathbf{x})$，得 $\beta_i = \exp(h(\mathbf{x}_i))$

    - 加权经验风险最小化

2. **标签偏移纠正**

    - 计算权重 $\beta_i = \frac{p(y_i)}{q(y_i)}$（目标/源标签分布比）

    - 利用混淆矩阵 $\mathbf{C}$ 和测试集预测均值 $\mu(\hat{\mathbf{y}})$ 求解 $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$

    - 应用加权经验风险最小化

3. **概念偏移纠正**

    - 多采用增量更新策略，利用新数据微调模型而非从头训练

### 4.9.3. 学习问题分类

**批处理学习**：用固定训练集训练模型后部署，不再更新

**在线学习**：数据逐次到达，模型持续更新

**老虎机（bandits）**：有限动作集的在线学习特例

**控制**：环境会记忆并基于历史行为响应

**强化学习**：环境可能合作或对抗，存在复杂交互
