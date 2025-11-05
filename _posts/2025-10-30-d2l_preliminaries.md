---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 2. 预备知识"
date: 2025-10-30
tags: [AI, notes]
toc: true
comments: true
author: pianfan
---

要学习深度学习，首先需要先掌握一些基本技能，如数据处理、线性代数、微积分和概率<!-- more -->

## 2.1. 数据操作

**张量（tensor）**：由数值组成的多维数组。单轴张量对应数学中的**向量（vector）**；双轴张量对应数学中的**矩阵（matrix）**

**张量类**：PyTorch 中为 `Tensor`，支持 GPU 加速计算和自动微分，适用于深度学习

### 2.1.1. 入门

张量的**元素（element）**：张量中的每个值

```py
import torch
x = torch.arange(12)  # 创建包含前12个整数的行向量
```

张量的**形状（shape）**：张量沿每个轴的长度，可通过 `shape` 属性访问

```py
x.shape  # 获取张量形状
```

张量的**大小（size）**：张量中元素的总数（形状各元素的乘积），PyTorch 中通过 `numel()` 方法获取

```py
x.numel()  # 获取张量大小
```

`reshape` 函数：改变张量形状而不改变元素数量和值，可通过 `-1` 自动计算某一维度

```py
X = x.reshape(3, 4)  # 转换为3行4列的矩阵，等价于x.reshape(-1,4)或x.reshape(3,-1)
```

张量初始化方式：

```py
torch.zeros((2, 3, 4))  # 全0张量，形状为(2,3,4)
torch.ones((2, 3, 4))   # 全1张量，形状为(2,3,4)
torch.randn(3, 4)       # 元素从标准高斯分布采样，形状为(3,4)
torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])  # 从列表初始化张量
```

### 2.1.2. 运算符

**按元素（elementwise）**运算：对张量每个元素应用标量运算，适用于相同形状的张量。常见算术运算符（`+`、`-`、`*`、`/`、`**`）及指数函数 `torch.exp()` 均支持按元素运算

**连结（concatenate）**：将多个张量沿指定轴拼接为更大张量

```py
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0)  # 沿轴0（行）拼接
torch.cat((X, Y), dim=1)  # 沿轴1（列）拼接
```

**逻辑运算符**：构建二元张量，如 `X == Y` 会生成一个新张量，元素为 1（相等）或 0（不等）

张量求和：对所有元素求和得到单元素张量

```py
X.sum()  # 求张量所有元素的和
```

### 2.1.3. 广播机制

**广播机制（broadcasting mechanism）**处理不同形状张量的按元素运算，步骤为：

  1. 扩展一个或两个张量（复制元素），使形状相同
  2. 执行按元素操作

通常沿长度为 1 的轴广播

### 2.1.4. 索引和切片

通过索引访问元素：首元素索引为 0，尾元素索引为 -1；可通过范围指定选取元素（如 `X[1:3]` 选取第二和第三个元素），也可指定索引修改元素值

### 2.1.5. 节省内存

避免不必要的内存分配，可使用切片或复合赋值运算符进行原地更新

```py
Z = torch.zeros_like(Y)
Z[:] = X + Y  # 切片方式原地更新Z
X += Y        # 复合赋值运算符原地更新X
```

### 2.1.6. 转换为其他 Python 对象

- 张量与 NumPy 数组转换：`X.numpy()`（张量转 NumPy 数组）、`torch.tensor(A)`（NumPy 数组转张量），转换后共享内存

- 大小为 1 的张量转 Python 标量：使用 `item()` 函数或强制类型转换（如 `float(a)`、`int(a)`）

## 2.2. 数据预处理

为了能用深度学习解决现实世界的问题，我们常从预处理原始数据开始，而非直接使用已准备好的张量格式数据

### 2.2.1. 读取数据集

可通过 pandas 的 `read_csv` 函数加载 CSV 格式的原始数据集

```py
import os
import pandas as pd

# 创建示例数据文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取数据
data = pd.read_csv(data_file)
```

### 2.2.2. 处理缺失值

处理缺失数据的典型方法有插值法（用替代值弥补）和删除法（忽略缺失值）

对于数值型缺失值，可采用插值法，如用所在列的均值替换

```py
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())  # 用均值填充数值型缺失值
```

对于类别型或离散型缺失值，可将“NaN”视为一个类别，通过 `pd.get_dummies` 进行独热编码转换

```py
inputs = pd.get_dummies(inputs, dummy_na=True)  # 处理类别型缺失值
```

### 2.2.3. 转换为张量格式

当数据均为数值类型时，可转换为张量格式，以便利用张量相关功能进行后续处理

```py
import torch

X = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)
```

## 2.3. 线性代数

本节介绍线性代数中的基本数学对象、算术和运算，以下用数学符号和 PyTorch 代码实现表示

### 2.3.1. 标量

- **标量（scalar）**：仅包含一个数值，可用只有一个元素的张量表示

- 标量**变量（variable）**：表示未知的标量值，由普通小写字母表示（例如，$x$、$y$、$z$）

- $\mathbb{R}$ 表示所有实值标量的集合，$x \in \mathbb{R}$ 表示 $x$ 是实值标量

### 2.3.2. 向量

- 向量：标量值组成的列表，元素（分量）为标量，可用一维张量表示，数学中记为粗体小写符号（例如，$\mathbf{x}$、$\mathbf{y}$、$\mathbf{z}$）

- 元素引用：通过下标引用，如第 $i$ 个元素表示为 $x_i$（标量，不加粗）。默认列向量，数学表示为：

$$
\begin{split}\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},\end{split}
$$

#### 2.3.2.1. 长度、维度和形状

- 维度：向量的长度，$n$ 维向量 $\mathbf{x}$ 可表示为 $\mathbf{x} \in \mathbb{R}^n$

- 长度访问：`len(x)`（`x` 为向量张量）

- 形状：张量沿各轴的长度，一维张量形状为单元素元组，如 `x.shape`（`x` 为向量张量）

### 2.3.3. 矩阵

- 矩阵：二维数组，数学中记为粗体大写字母（例如，$\mathbf{X}$、$\mathbf{Y}$、$\mathbf{Z}$），代码中为二维张量

- 数学表示：$\mathbf{A} \in \mathbb{R}^{m \times n}$ 表示 $m$ 行 $n$ 列实值矩阵。元素 $a_{ij}$ 位于第 $i$ 行第 $j$ 列：

$$
\begin{split}\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.\end{split}
$$

- 形状：$(m, n)$ 或 $m \times n$，行列数相同的为**方阵（square matrix）**

- 创建：`A = torch.arange(20, dtype=torch.float32).reshape(5, 4)`

- 元素访问：通过行索引 $i$ 和列索引 $j$，如 $[\mathbf{A}]_{ij}$

- **转置（transpose）**：交换行列，记为 $\mathbf{A}^\top$，代码中为 `A.T`，满足 $(\mathbf{A}^\top)^\top = \mathbf{A}$、$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$

- **对称矩阵（symmetric matrix）**：方阵中 $\mathbf{A} = \mathbf{A}^\top$

### 2.3.4. 张量

- 张量：描述任意数量轴的 $n$ 维数组，数学中用特殊字体大写字母表示（如，$\mathsf{X}$、$\mathsf{Y}$、$\mathsf{Z}$）

### 2.3.5. 张量算法的基本性质

- 同形状张量按元素二元运算结果形状不变

- **Hadamard积（Hadamard product）**：两矩阵按元素乘法，记为 $\odot$：

$$
\begin{split}\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.\end{split}
$$

- 标量与张量运算：不改变张量形状，每个元素与标量运算

### 2.3.6. 降维

- 求和：`sum()`，默认沿所有轴降为标量，可指定轴，如 `A.sum(dim=0)`（沿轴 0 求和）、`A.sum(dim=[0, 1])`（沿轴 0 和 1 求和）

- 平均值：`mean()`，或总和除以元素总数（`A.sum() / A.numel()`），可指定轴

#### 2.3.6.1. 非降维求和

- 保持轴数：`sum(dim=..., keepdims=True)`

- 累积和：`cumsum(dim=...)`，不降低维度

### 2.3.7. 点积（Dot Product）

- 定义：两向量 $\mathbf{x},\mathbf{y}\in\mathbb{R}^d$ 的点积为 $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$，代码中为 `torch.dot(x, y)`，等价于 `torch.sum(x * y)`

### 2.3.8. 矩阵-向量积

- 定义：矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 与向量 $\mathbf{x} \in \mathbb{R}^n$ 的积为长度 $m$ 的列向量，第 $i$ 元素为 $\mathbf{a}^\top_i \mathbf{x}$（$\mathbf{a}^\top_i$ 为 $\mathbf{A}$ 第 $i$ 行）：

$$
\begin{split}\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.\end{split}
$$

- 代码：`torch.mv(A, x)`，要求 $\mathbf{A}$ 列数与 $\mathbf{x}$ 长度相同

### 2.3.9. 矩阵-矩阵乘法

- 定义：矩阵 $\mathbf{A} \in \mathbb{R}^{n \times k}$ 与 $\mathbf{B} \in \mathbb{R}^{k \times m}$ 的积 $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{n \times m}$，元素 $c_{ij} = \mathbf{a}^\top_i \mathbf{b}_j$（$\mathbf{a}^\top_i$ 为 $\mathbf{A}$ 第 $i$ 行，$\mathbf{b}_j$ 为 $\mathbf{B}$ 第 $j$ 列）

- 代码：`torch.mm(A, B)`

### 2.3.10. 范数

- 性质：$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x})$；$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y})$；$
f(\mathbf{x}) \geq 0$；$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0$

- $L_2$ 范数：$\Vert\mathbf{x}\Vert_2 = \sqrt{\sum_{i=1}^n x_i^2}$，代码 `torch.norm(u)`

- $L_1$ 范数：$\Vert\mathbf{x}\Vert_1 = \sum_{i=1}^n \vert x_i \vert$，代码 `torch.abs(u).sum()`

- $L_p$ 范数：$\Vert\mathbf{x}\Vert_p = \left(\sum_{i=1}^n \vert x_i \vert^p \right)^{1/p}$

- Frobenius 范数（矩阵）：$\Vert\mathbf{X}\Vert_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}$，代码 `torch.norm(matrix)`

## 2.4. 微积分

**逼近法（method of exhaustion）**

**积分学（integral calculus）**

**微分学（differential calculus）**

**损失函数（loss function）**

拟合模型的任务可分解为两个关键问题：

  - **优化（optimization）**：用模型拟合观测数据的过程

  - **泛化（generalization）**：指导生成有效性超出训练数据集的模型的数学原理与实践智慧

### 2.4.1. 导数和微分

深度学习中，通常选择对模型参数可微的损失函数。即对于每个参数，当参数发生无穷小量的增减时，能确定损失增减的速率

设函数 $f: \mathbb{R} \rightarrow \mathbb{R}$，其输入和输出均为标量。若 $f$ 的**导数**存在，定义为：

$$
f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.
$$

若 $f'(a)$ 存在，则 $f$ 在 $a$ 处**可微（differentiable）**；若 $f$ 在某区间内每一点都可微，则该函数在该区间可微。导数 $f'(x)$ 可解释为 $f(x)$ 相对于 $x$ 的**瞬时（instantaneous）变化率**

对于 $y=f(x)$（$x$ 为自变量，$y$ 为因变量），以下导数表示等价：

$$
f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),
$$

其中 $\frac{d}{dx}$ 和 $D$ 为**微分**运算符，表示微分操作。常见函数的微分规则：

  - $DC = 0$（$C$ 为常数）
  - $Dx^n = nx^{n-1}$（**幂律（power rule）**，$n$ 为任意实数）
  - $De^x = e^x$
  - $D\ln(x) = 1/x$

设函数 $f$、$g$ 均可微，$C$ 为常数，微分法则如下：

**常数相乘法则**

$$
\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),
$$

**加法法则**

$$
\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),
$$

**乘法法则**

$$
\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],
$$

**除法法则**

$$
\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.
$$

### 2.4.2. 偏导数

**多元函数（multivariate function）**

设 $y = f(x_1, x_2, \ldots, x_n)$ 为 $n$ 变量函数，$y$ 关于第 $i$ 个参数 $x_i$ 的**偏导数（partial derivative）**为：

$$
\frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.
$$

计算 $\frac{\partial y}{\partial x_i}$ 时，可将 $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ 视为常数，求 $y$ 对 $x_i$ 的导数。偏导数的等价表示：

$$
\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.
$$

### 2.4.3. 梯度

多元函数对所有变量的偏导数可组成该函数的**梯度（gradient）**向量。设函数 $f:\mathbb{R}^n\rightarrow\mathbb{R}$，输入为 $n$ 维向量 $\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，输出为标量，则 $f(\mathbf{x})$ 相对于 $\mathbf{x}$ 的梯度为含 $n$ 个偏导数的向量：

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,
$$

无歧义时可简写为 $\nabla f(\mathbf{x})$

设 $\mathbf{x}$ 为 $n$ 维向量，多元函数微分常用规则：

  - 对 $\mathbf{A} \in \mathbb{R}^{m \times n}$，$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
  - 对 $\mathbf{A} \in \mathbb{R}^{n \times m}$，$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$
  - 对 $\mathbf{A} \in \mathbb{R}^{n \times n}$，$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
  - $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$

对任意矩阵 $\mathbf{X}$，$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。梯度在深度学习优化算法设计中很有用

### 2.4.4. 链式法则

深度学习中的多元函数常为**复合（composite）**函数，链式法则可用于其微分

单变量函数场景：若 $y=f(u)$ 和 $u=g(x)$ 均可微，则：

$$
\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.
$$

多变量函数场景：设可微函数 $y$ 依赖变量 $u_1, u_2, \ldots, u_m$，每个可微函数 $u_i$ 依赖变量 $x_1, x_2, \ldots, x_n$（即 $y$ 是 $x_1, x_2， \ldots, x_n$ 的函数），则对任意 $i = 1, 2, \ldots, n$：

$$
\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i} + \frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i} + \cdots + \frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}
$$

## 2.5. 自动微分

深度学习框架通过自动计算导数（即**自动微分**）加快求导过程。系统会根据模型构建**计算图（computational graph）**，跟踪数据通过哪些操作组合产生输出，进而通过**反向传播**（跟踪计算图并填充各参数的偏导数）自动计算梯度

### 2.5.1. 基本流程

1. 创建需求导的变量并指定追踪梯度（如 `x = torch.arange(4.0, requires_grad=True)`）

2. 计算目标值（如 `y = 2 * torch.dot(x, x)`）

3. 调用反向传播函数（`y.backward()`）计算梯度，结果存储在变量的 `grad` 属性中

4. 多次计算梯度时，需先用 `x.grad.zero_()` 清除之前累积的梯度

### 2.5.2. 非标量变量的反向传播

- 非标量变量调用 `backward()` 时，需传入 `gradient` 参数（通常为全 1 向量），实际是计算各元素偏导数的和

- 示例：`y = x * x` 求导时，可通过 `y.sum().backward()` 实现

### 2.5.3. 分离计算

- 使用 `detach()` 方法可将变量从计算图中分离，得到的新变量值与原变量相同，但梯度不会反向流经该变量

- 示例：`u = y.detach()` 后，计算 `z = u * x` 的梯度时，`u` 被视为常数

## 2.6. 概率

机器学习的核心是做出预测。概率是用于描述确定程度的灵活语言，可有效应用于广泛领域

### 2.6.1. 基本概率论

- **大数定律（law of large numbers）**：随着试验次数增加，事件发生的频率会逐渐接近其真实概率

- **抽样（sampling）**：从概率分布中抽取样本的过程

- **分布（distribution）**：对**随机变量（random variable）**取值概率的分配

- **多项分布（multinomial distribution）**：为离散选择分配概率的分布

#### 2.6.1.1. 概率论公理

- **样本空间（结果空间）**：随机试验所有可能结果的集合，记为 $\mathcal{S}$

- **结果（outcome）**：样本空间中的元素

- **事件（event）**：样本空间的子集，即一组可能的结果

- **概率（probability）**：将事件映射到实数的函数，满足：

  - 对任意事件 $\mathcal{A}$，$P(\mathcal{A}) \geq 0$

  - 整个样本空间的概率为 $1$，即 $P(\mathcal{S}) = 1$

#### 2.6.1.2. 随机变量

- 随机变量是在随机试验中取值不确定的量

- $P(X=a)$ 表示随机变量 $X$ 取 $a$ 值的概率；$P(X)$ 表示 $X$ 的分布；$P(a)$ 可简化表示随机变量取 $a$ 的概率

- **离散（discrete）**随机变量和**连续（continuous）**随机变量的区别：连续随机变量取特定值的概率为 0，需用**密度（density）**描述取值可能性，区间内取值概率非零

### 2.6.2. 处理多个随机变量

#### 2.6.2.1. 联合概率

**联合概率（joint probability）**$P(A=a, B=b)$ 表示 $A=a$ 与 $B=b$ 同时发生的概率

#### 2.6.2.2. 条件概率

**条件概率（conditional probability）**$P(B=b \mid A=a)$ 表示在 $A=a$ 发生的条件下 $B=b$ 发生的概率，定义为 $\frac{P(A=a, B=b)}{P(A=a)}$

#### 2.6.2.3. 贝叶斯定理

根据**乘法法则（multiplication rule）**$P(A, B) = P(B \mid A) P(A) = P(A \mid B) P(B)$，若 $P(B)>0$，则 $P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$。其中 $P(A, B)$ 为联合分布，$P(A \mid B)$ 为条件分布

#### 2.6.2.4. 边际化

- **求和法则（边际化）**：$P(B) = \sum_{A} P(A, B)$，由此得到的概率或分布称为**边际概率（marginal probability）**或**边际分布（marginal distribution）**

#### 2.6.2.5. 独立性

- **独立（independence）**：若 $P(B \mid A) = P(B)$（或 $P(A \mid B) = P(A)$，或 $P(A, B) = P(A)P(B)$），则 $A$ 与 $B$ 独立，记为 $A \perp B$

- **条件独立**：给定 $C$ 时，若 $P(A, B \mid C) = P(A \mid C)P(B \mid C)$，则 $A$ 与 $B$ 条件独立，记为 $A \perp B \mid C$

### 2.6.3. 期望和方差

- **期望（expectation）**：随机变量 $X$ 的期望 $E[X] = \sum_{x} x P(X = x)$；函数 $f(x)$ 的期望 $E_{x \sim P}[f(x)] = \sum_x f(x) P(x)$

- **方差**：衡量随机变量与期望的偏离程度，$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2$。其平方根为**标准差（standard deviation）**

- 随机变量函数的方差：$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right]$
