---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 2. 预备知识"
date: 2025-10-31
tags: [AI, notes]
toc: true
comments: true
author: pianfan
---

要学习深度学习，首先需要先掌握一些基本技能。 如数据处理、线性代数、微积分和概率。<!-- more -->

## 2.1. 数据操作
*张量*（tensor）：一个由数值组成的数组，这个数组可能有多个维度。具有一个轴的张量对应数学上的*向量*（vector）； 具有两个轴的张量对应数学上的*矩阵*（matrix）。

*张量类*：在 PyTorch 中为 `Tensor`。

### 2.1.1. 入门
张量的*元素*（element）：张量中的每个值。
```py
import torch

# 使用 arange 创建一个行向量 x，这个行向量包含以 0 开始的前 12 个整数
x = torch.arange(12)
```

张量的*形状*（shape）：张量沿每个轴的长度。
```py
# 可以通过张量的 shape 属性来访问张量的形状 
x.shape
```

张量的*大小*（size）：张量中元素的总数，即形状的所有元素乘积。
```py
# PyTorch 中调用 numel 方法来获取张量的大小
x.numel()
```

reshape 函数：改变一个张量的形状而不改变元素数量和元素值。
```py
# 把张量 x 从形状为(12,)的行向量转换为形状为(3,4)的矩阵
X = x.reshape(3, 4)
```

不需要通过手动指定每个维度来改变形状，可以通过 -1 来调用自动计算出维度的功能。即可以用 `x.reshape(-1, 4)` 或 `x.reshape(3, -1)` 来取代 `x.reshape(3, 4)`。

有时，我们希望使用全 0、全 1、其他常量，或者从特定分布中随机采样的数字来初始化矩阵。代码如下：
```py
# 创建一个形状为(2,3,4)的张量，其中所有元素都设置为 0
torch.zeros((2, 3, 4))

# 创建一个形状为(2,3,4)的张量，其中所有元素都设置为 1
torch.ones((2, 3, 4))

# 创建一个形状为(3,4)的张量，其中的每个元素都从均值为 0、标准差为 1 的标准高斯分布（正态分布）中随机采样
torch.randn(3, 4)

# 通过提供包含数值的 Python 列表（或嵌套列表）为张量中的每个元素赋予确定值
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

### 2.1.2. 运算符
*按元素*（elementwise）运算：将标准标量运算符应用于数组的每个元素。对于将两个数组作为输入的函数，按元素运算将二元运算符应用于两个数组中的每对位置对应的元素。

*一元*标量运算符：通过符号 $f: \mathbb{R} \rightarrow \mathbb{R}$ 表示，这意味着该函数从任何实数（$\mathbb{R}$）映射到另一个实数。

*二元*标量运算符：通过符号 $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$ 表示，这意味着该函数接收两个输入，并产生一个输出。

对于任意具有相同形状的张量，常见的标准算术运算符（+、-、*、/ 和 **）或指数函数计算（`torch.exp()`）都可以被升级为按元素运算。

*连结*（concatenate）：把多个张量端对端地叠起来形成一个更大的张量。
```py
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 沿行（轴-0，形状的第一个元素）连结
torch.cat((X, Y), dim=0)
# 按列（轴-1，形状的第二个元素）连结
torch.cat((X, Y), dim=1)
```

*逻辑运算符*：可用来构建二元张量。以 `X == Y` 为例：对于每个位置，如果 X 和 Y 在该位置相等，则新张量中相应项的值为 1，否则为 0。

对张量中的所有元素进行求和，会产生一个单元素张量。
```py
X.sum()
```

### 2.1.3. 广播机制
*广播机制*（broadcasting mechanism）的工作方式：
  1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
  2. 对生成的数组执行按元素操作。

在大多数情况下，将沿着数组中长度为 1 的轴进行广播，如下例子：
```py
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
# 由于 a 和 b 形状不匹配，如果让它们相加，两个矩阵将广播为更大的 3 X 2 矩阵：矩阵 a 将复制列，矩阵 b 将复制行，然后再按元素相加
a + b
```

### 2.1.4. 索引和切片
就像在任何其他 Python 数组中一样，张量中的元素可以通过索引访问：第一个元素的索引是 0，最后一个元素索引是 -1；可以指定范围以包含第一个元素和最后一个之前的元素。
```py
# 选择第二个和第三个元素
X[1:3]
```

### 2.1.5. 节省内存
运行一些操作可能会导致为新结果分配内存。
```py
before = id(Y)  # id()函数提供内存中引用对象的确切地址
Y = Y + X  # Python 首先计算 Y + X，为结果分配新的内存，然后使 Y 指向内存中的这个新位置
# 运行 Y = Y + X 后，id(Y)会指向另一个位置
id(Y) == before  # 结果为 False
```

要想原地更新，可以使用切片表示法或复合赋值运算符（如 +=）将操作的结果分配给先前分配的数组，例如：
```py
Z = torch.zeros_like(Y)  # 创建一个新的矩阵 Z，其形状与 Y 相同，使用 zeros_like 来分配一个全 0 的块
print('id(Z):', id(Z))
Z[:] = X + Y  # 这里 Z 使用了切片表示法
print('id(Z):', id(Z))  # 结果将与上次输出相同

before = id(X)
X += Y  # 使用复合赋值运算符取代 X = X + Y
id(X) == before  # 结果为 True
```

### 2.1.6. 转换为其他 Python 对象
将深度学习框架定义的张量转换为 NumPy 张量（ndarray）很容易，反之也同样容易。torch 张量和 numpy 数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。
```py
A = X.numpy()  # A 的类型为 numpy.ndarray
B = torch.tensor(A)  # B 的类型为 torch.Tensor
type(A), type(B)
```

要将大小为 1 的张量转换为 Python 标量，可以调用 item 函数或强制类型转换。
```py
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)  # 结果为 (tensor([3.5000]), 3.5, 3.5, 3)
```

### 2.1.7. 小结
- 深度学习存储和操作数据的主要接口是张量（$n$ 维数组）。它提供了各种功能，包括基本数学运算、广播、索引、切片、内存节省和转换其他 Python 对象。

## 2.2. 数据预处理
为了能用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始，而不是从那些准备好的张量格式数据开始。

### 2.2.1. 读取数据集
举一个例子，我们首先创建一个人工数据集，并存储在 CSV（逗号分隔值）文件 ../data/house_tiny.csv 中。以其他格式存储的数据也可以通过类似的方式进行处理。下面我们将数据集按行写入 CSV 文件中。
```py
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```
要从创建的 CSV 文件中加载原始数据集，我们导入 pandas 包并调用 read_csv 函数。
```py
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```
运行结果如下：
```
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
```

### 2.2.2. 处理缺失值
注意，“NaN”项代表缺失值。为了处理缺失的数据，典型的方法包括*插值法*和*删除法*，其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。

在本例中，我们考虑插值法。
```py
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # 通过位置索引 iloc，将 data 分成 inputs 和 outputs，其中前者为 data 的前两列，而后者为 data 的最后一列
inputs = inputs.fillna(inputs.select_dtypes(include='number').mean())  # 对于 inputs 中缺少的数值，用同一列的均值替换“NaN”项
print(inputs)
```
运行结果如下：
```
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

对于 inputs 中的类别值或离散值，我们将“NaN”视为一个类别。由于“Alley”列只存在两种类型的类别值“Pave”和“NaN”，pandas 可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
```py
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```
运行结果如下：
```
   NumRooms  Alley_Pave  Alley_nan
0       3.0        True      False
1       2.0       False       True
2       4.0       False       True
3       3.0       False       True
```

### 2.2.3. 转换为张量格式
现在 inputs 和 outputs 中的所有条目都是数值类型（True 和 False 等同于 1 和 0），它们可以转换为张量格式。
```py
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y
```

### 2.2.4. 小结
- pandas 软件包是 Python 中常用的数据分析工具，pandas 可以与张量兼容。

- 用 pandas 处理缺失的数据时，我们可根据情况选择用插值法和删除法。

## 2.3. 线性代数
本节将介绍线性代数中的基本数学对象、算术和运算，并用数学符号和相应的代码实现来表示它们。

### 2.3.1. 标量
*标量*（scalar）：仅包含一个数值，可用只有一个元素的张量表示。

标量*变量*（variable）：表示未知的标量值，由普通小写字母表示（例如，$x$、$y$ 和 $z$）。

### 2.3.2. 向量
向量：可视为标量值组成的列表，这些标量值被称为向量的*元素*（element）或*分量*（component）。在数学表示法中，向量通常记为粗体、小写的符号（例如，$\mathbf{x}$、$\mathbf{y}$ 和 $\mathbf{z}$）。向量可用一维张量表示。

我们可以使用下标来引用向量的任一元素，例如可以通过 $x_i$ 来引用第 $i$ 个元素。注意，元素 $x_i$ 是一个标量，所以我们在引用它时不会加粗。大量文献认为列向量是向量的默认方向，在本书中也是如此。在数学中，向量 $\mathbf{x}$ 可以写为：
$$
\begin{split}\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},\end{split}
$$
其中 $x_1,\ldots,x_n$ 是向量的元素。在代码中，我们通过张量的索引来访问任一元素。

#### 2.3.2.1. 长度、维度和形状
向量只是一个数字数组，就像每个数组都有一个长度一样，每个向量也是如此。在数学表示法中，如果我们想说一个向量 $\mathbf{x}$ 由 $n$ 个实值标量组成，可以将其表示为 $\mathbf{x}\in\mathbb{R}^n$。向量的长度通常称为向量的*维度*（dimension）。

我们可以通过调用 Python 的内置 len() 函数或张量的 .shape 属性来访问向量的长度。形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。对于只有一个轴的张量，形状只有一个元素。
```py
x = torch.arange(4)
len(x)  # 结果为 4
x.shape  # 结果为 torch.Size([4])
```

### 2.3.3. 矩阵
*矩阵*（matrix）：通常用粗体、大写字母来表示（例如，$\mathbf{X}$、$\mathbf{Y}$ 和 $\mathbf{Z}$），在代码中表示为具有两个轴的张量。

数学表示法使用 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 来表示矩阵，其由 $m$ 行和 $n$ 列的实值标量组成。我们可以将任意矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 视为一个表格，其中每个元素 $a_{ij}$ 属于第 $i$ 行第 $j$ 列：
$$
\begin{split}\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.\end{split}
$$

对于任意 $\mathbf{A} \in \mathbb{R}^{m \times n}$，$\mathbf{A}$ 的形状是（$m$, $n$）或 $m \times n$。当矩阵具有相同数量的行和列时，被称为*方阵*（square matrix）。

当调用函数来实例化张量时，我们可以通过指定两个分量 $m$ 和 $n$ 来创建一个形状为 $m \times n$ 的矩阵。
```py
A = torch.arange(20).reshape(5, 4)
```

我们可以通过行索引（$i$）和列索引（$j$）来访问矩阵中的标量元素 $a_{ij}$，例如 $[\mathbf{A}]_{ij}$。为了表示起来简单，只有在必要时才会插入逗号以分隔行列索引。

矩阵的*转置*（transpose）：交换矩阵的行和列，通常用 $\mathbf{a}^\top$ 来表示。在代码中使用 `A.T` 访问矩阵的转置。

*对称矩阵*（symmetric matrix）：方阵的一种特殊类型，对于一个对称矩阵 $\mathbf{A}$，满足 $\mathbf{A} = \mathbf{A}^\top$。

### 2.3.4. 张量
张量：描述具有任意数量轴的 $n$ 维数组的通用方法。用特殊字体的大写字母表示（例如，$\mathsf{X}$、$\mathsf{Y}$ 和 $\mathsf{Z}$）。

### 2.3.5. 张量算法的基本性质
给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法。
```py
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将 A 的一个副本分配给 B
A, A + B
```

*Hadamard积*（Hadamard product）：两个矩阵的按元素乘法（数学符号 $\odot$）。
$$
\begin{split}\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.\end{split}
$$

将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
```py
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

### 2.3.6. 降维
可以对任意张量进行的一个有用的操作是计算其元素的和。数学表示法使用 $\sum$ 符号表示求和。在代码中可以调用计算求和的函数 `sum()`。

默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。我们还可以指定张量沿哪一个轴来通过求和降低维度。以矩阵为例：
```py
# 通过求和所有行的元素来降维（轴 0）
A_sum_dim0 = A.sum(dim=0)
# 通过求和所有列的元素来降维（轴 1）
A_sum_dim1 = A.sum(dim=1)
# 沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和
A.sum(dim=[0, 1])  # 结果和 A.sum() 相同
```

可以通过将总和除以元素总数来计算*平均值*（mean 或 average）。在代码中，我们可以调用函数来计算任意形状张量的平均值。
```py
A.mean(), A.sum() / A.numel()
```

同样，计算平均值的函数也可以沿指定轴降低张量的维度。
```py
A.mean(dim=0), A.sum(dim=0) / A.shape[0]
```

#### 2.3.6.1. 非降维求和
在调用函数来计算总和或均值时，可以通过指定 `keepdims=True` 来保持轴数不变。
```py
sum_A = A.sum(dim=1, keepdims=True)
```

如果我们想沿某个轴计算元素的累积总和，可以调用 cumsum 函数。此函数不会沿任何轴降低输入张量的维度。
```py
A.cumsum(dim=0)
```

### 2.3.7. 点积（Dot Product）
给定两个向量 $\mathbf{x},\mathbf{y}\in\mathbb{R}^d$，它们的*点积*（dot product）$\mathbf{x}^\top\mathbf{y}$（或 $\langle\mathbf{x},\mathbf{y}\rangle$）是相同位置的按元素乘积的和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。
```py
torch.dot(x, y)
```

我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积：
```py
torch.sum(x * y)
```

给定一组由向量 $\mathbf{x} \in \mathbb{R}^d$ 表示的值，和一组由 $\mathbf{w} \in \mathbb{R}^d$ 表示的权重。$\mathbf{x}$ 中的值根据权重 $\mathbf{w}$ 的加权和，可以表示为点积 $\mathbf{x}^\top \mathbf{w}$。当权重为非负数且和为 1（即 $\left(\sum_{i=1}^{d}{w_i}=1\right)$）时，点积表示*加权平均*（weighted average）。将两个向量规范化得到单位长度后，点积表示它们夹角的余弦。

### 2.3.8. 矩阵-向量积
*矩阵-向量积*（matrix-vector product）

定义一个矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 和一个向量 $\mathbf{x} \in \mathbb{R}^n$，让我们将矩阵 $\mathbf{A}$ 用它的行向量表示：
$$
\begin{split}\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},\end{split}
$$
其中每个 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ 都是行向量，表示矩阵的第 $i$ 行。矩阵向量积 $\mathbf{A}\mathbf{x}$ 是一个长度为 $m$ 的列向量，其第 $i$ 个元素是点积 $\mathbf{a}^\top_i \mathbf{x}$：
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

在代码中使用张量表示矩阵-向量积，我们使用 `mv` 函数。注意，`A` 的列维数（沿轴 1 的长度）必须与 `x` 的维数（其长度）相同。
```py
torch.mv(A, x)
```

### 2.3.9. 矩阵-矩阵乘法
**矩阵-矩阵乘法**（matrix-matrix multiplication），可简称为**矩阵乘法**

假设有两个矩阵 $\mathbf{A} \in \mathbb{R}^{n \times k}$ 和 $\mathbf{B} \in \mathbb{R}^{k \times m}$：
$$
\begin{split}\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.\end{split}
$$

用行向量 $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ 表示矩阵 $\mathbf{A}$ 的第 $i$ 行，并让列向量 $\mathbf{b}_{j} \in \mathbb{R}^k$ 作为矩阵 $\mathbf{B}$ 的第 $j$ 列。要生成矩阵积 $\mathbf{C} = \mathbf{A}\mathbf{B}$，最简单的方法是考虑 $\mathbf{A}$ 的行向量和 $\mathbf{B}$ 的列向量：
$$
\begin{split}\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.\end{split}
$$

当我们简单地将每个元素 $c_{ij}$ 计算为点积 $\mathbf{a}^\top_i \mathbf{b}_j$：
$$
\begin{split}\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.\end{split}
$$
我们可以将矩阵-矩阵乘法 $\mathbf{AB}$ 看作简单地执行 $m$ 次矩阵-向量积，并将结果拼接在一起，形成一个 $n \times m$ 矩阵。在代码中，使用 `mm` 函数执行矩阵乘法。
```py
torch.mm(A, B)
```

### 2.3.10. 范数
向量*范数*（norm）：将向量映射到标量的函数 $f$。

给定任意向量 $\mathbf{x}$，向量范数要满足一些属性。第一个性质是：如果我们按常数因子 $\alpha$ 缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放：
$$
f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).
$$

第二个性质是熟悉的三角不等式：
$$
f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).
$$

第三个性质简单地说范数必须是非负的：
$$
f(\mathbf{x}) \geq 0.
$$

范数听起来很像距离的度量。欧几里得距离和毕达哥拉斯定理中的非负性概念和三角不等式可能会给出一些启发。事实上，欧几里得距离是一个 $L_2$ 范数：假设 $n$ 维向量 $\mathbf{x}$ 中的元素是 $x_1,\ldots,x_n$，其 $L_2$ 范数是向量元素平方和的平方根：
$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},
$$
其中，在 $L_2$ 范数中常常省略下标 $2$，也就是说 $\Vert\mathbf{x}\Vert$ 等同于 $\Vert\mathbf{x}\Vert_2$。在代码中，我们可以按如下方式计算向量的 $L_2$ 范数。
```py
u = torch.tensor([3.0, -4.0])
torch.norm(u)  # 结果为 tensor(5.)
```

深度学习中更经常使用 $L_2$ 范数的平方，也会经常遇到 $L_1$ 范数，它表示为向量元素的绝对值之和：
$$
\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.
$$

与 $L_2$ 范数相比，$L_1$ 范数受异常值的影响较小。为了计算 $L_1$ 范数，我们将绝对值函数和按元素求和组合起来。
```py
torch.abs(u).sum()  # 结果为 tensor(7.)
```

一般的，$L_p$ 范数可表示为：
$$
\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.
$$

类似于向量的 $L_2$ 范数，矩阵 $\mathbf{X} \in \mathbb{R}^{m \times n}$ 的 *Frobenius 范数*（Frobenius norm）是矩阵元素平方和的平方根：
$$
\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.
$$
Frobenius 范数满足向量范数的所有性质，它就像是矩阵形向量的 $L_2$ 范数。调用以下函数将计算矩阵的 Frobenius 范数。
```py
torch.norm(torch.ones((4, 9)))  # 结果为 tensor(6.)
```

#### 2.3.10.1. 范数和目标
在深度学习中，我们经常试图解决优化问题：最大化分配给观测数据的概率；最小化预测和真实观测之间的距离。用向量表示物品，以便最小化相似项目之间的距离，最大化不同项目之间的距离。目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。

### 2.3.11. 关于线性代数的更多信息
线性代数还有很多，其中很多数学对于机器学习非常有用。例如，矩阵可以分解为因子，这些分解可以显示真实世界数据集中的低维结构。机器学习的整个子领域都侧重于使用矩阵分解及其向高阶张量的泛化，来发现数据集中的结构并解决预测问题。

### 2.3.12. 小结
- 标量、向量、矩阵和张量是线性代数中的基本数学对象。

- 向量泛化自标量，矩阵泛化自向量。

- 标量、向量、矩阵和张量分别具有零、一、二和任意数量的轴。

- 一个张量可以通过 `sum` 和 `mean` 沿指定的轴降低维度。

- 两个矩阵的按元素乘法被称为他们的 Hadamard 积。它与矩阵乘法不同。

- 在深度学习中，我们经常使用范数，如 $L_1$ 范数、$L_2$ 范数和 Frobenius 范数。

- 我们可以对标量、向量、矩阵和张量执行各种操作。

## 2.4. 微积分
*逼近法*（method of exhaustion）

*积分*（integral calculus）

*微分*（differential calculus）

*损失函数*（loss function）

我们可以将拟合模型的任务分解为两个关键问题：
  - *优化*（optimization）：用模型拟合观测数据的过程；

  - *泛化*（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

### 2.4.1. 导数和微分
在深度学习中，我们通常选择对于模型参数可微的损失函数。简而言之，对于每个参数，如果我们把这个参数*增加*或*减少*一个无穷小的量，可以知道损失会以多快的速度增加或减少。

假设我们有一个函数 $f: \mathbb{R} \rightarrow \mathbb{R}$，其输入和输出都是标量。如果 $f$ 的*导数*存在，这个极限被定义为
$$
f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.
$$

如果 $f'(a)$ 存在，则称 $f$ 在 $a$ 处是*可微*（differentiable）的。如果 $f$ 在一个区间内的每个数上都是可微的，则此函数在此区间中是可微的。我们可以将导数 $f'(x)$ 解释为 $f(x)$ 相对于 $x$ 的*瞬时*（instantaneous）变化率。

给定 $y=f(x)$，其中 $x$ 和 $y$ 分别是函数 $f$ 的自变量和因变量。以下表达式是等价的：
$$
f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),
$$
其中符号 $\frac{d}{dx}$ 和 $D$ 是*微分*运算符，表示*微分*操作。我们可以使用以下规则来对常见函数求微分：
  - $DC = 0$（$C$ 是一个常数）
  - $Dx^n = nx^{n-1}$（*幂律*（power rule），$n$ 是任意实数）
  - $De^x = e^x$
  - $D\ln(x) = 1/x$

假设函数 $f$ 和 $g$ 都是可微的，$C$ 是一个常数，有以下法则：

*常数相乘法则*
$$
\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),
$$

*加法法则*
$$
\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),
$$

*乘法法则*
$$
\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],
$$

*除法法则*
$$
\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.
$$

### 2.4.2. 偏导数
*多元函数*（multivariate function）

设 $y = f(x_1, x_2, \ldots, x_n)$ 是一个具有 $n$ 个变量的函数。$y$ 关于第 $i$ 个参数 $x_i$ 的*偏导数*（partial derivative）为：
$$
\frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.
$$

为了计算 $\frac{\partial y}{\partial x_i}$，我们可以简单地将 $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ 看作常数，并计算 $y$ 关于 $x_i$ 的导数。对于偏导数的表示，以下是等价的：
$$
\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.
$$

### 2.4.3. 梯度
我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的*梯度*（gradient）向量。具体而言，设函数 $f:\mathbb{R}^n\rightarrow\mathbb{R}$ 的输入是一个 $n$ 维向量 $\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，并且输出是一个标量。函数 $f(\mathbf{x})$ 相对于 $\mathbf{x}$ 的梯度是一个包含 $n$ 个偏导数的向量：
$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,
$$
其中 $\nabla_{\mathbf{x}} f(\mathbf{x})$ 通常在没有歧义时被 $\nabla f(\mathbf{x})$ 取代。

假设 $\mathbf{x}$ 为 $n$ 维向量，在微分多元函数时经常使用以下规则：
  - 对于所有 $\mathbf{A} \in \mathbb{R}^{m \times n}$，都有 $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
  - 对于所有 $\mathbf{A} \in \mathbb{R}^{n \times m}$，都有 $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$
  - 对于所有 $\mathbf{A} \in \mathbb{R}^{n \times n}$，都有 $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
  - $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$

同样，对于任何矩阵 $\mathbf{X}$，都有 $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。正如我们之后将看到的，梯度对于设计深度学习中的优化算法有很大用处。

### 2.4.4. 链式法则
在深度学习中，多元函数通常是*复合*（composite）的，所以难以应用上述任何规则来微分这些函数。幸运的是，链式法则可以被用来微分复合函数。

让我们先考虑单变量函数。假设函数 $y=f(u)$ 和 $u=g(x)$ 都是可微的，根据链式法则：
$$
\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.
$$

现在考虑更一般的场景，即函数具有任意数量的变量的情况。假设可微分函数 $y$ 有变量 $u_1, u_2, \ldots, u_m$，其中每个可微分函数 $u_i$ 都有变量 $x_1, x_2, \ldots, x_n$。注意，$y$ 是 $x_1, x_2， \ldots, x_n$ 的函数。对于任意 $i = 1, 2, \ldots, n$，链式法则给出：
$$
\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i} + \frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i} + \cdots + \frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}
$$

### 2.4.5. 小结
- 微分和积分是微积分的两个分支，前者可以应用于深度学习中的优化问题。

- 导数可以被解释为函数相对于其变量的瞬时变化率，它也是函数曲线的切线的斜率。

- 梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数。

- 链式法则可以用来微分复合函数。

## 2.5. 自动微分
深度学习框架通过自动计算导数，即*自动微分*（automatic differentiation）来加快求导。实际上，根据设计好的模型，系统会构建一个*计算图*（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。这里，*反向传播*（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

### 2.5.1. 一个简单的例子
作为一个演示例子，假设我们想对函数 $y=2\mathbf{x}^{\top}\mathbf{x}$ 关于列向量 $\mathbf{x}$ 求导。首先，我们创建变量 `x` 并为其分配一个初始值。
```py
import torch

x = torch.arange(4.0)
```

在我们计算 $y$ 关于 $\mathbf{x}$ 的梯度之前，需要一个地方来存储梯度。重要的是，我们不会在每次对一个参数求导时都分配新的内存。注意，一个标量函数关于向量 $\mathbf{x}$ 的梯度是向量，并且与 $\mathbf{x}$ 具有相同的形状。
```py
x.requires_grad_(True)  # 等价于 x = torch.arange(4.0, requires_grad=True)
x.grad  # 默认值是 None
```

现在计算 $y$。
```py
y = 2 * torch.dot(x, x)
y  # 结果为 tensor(28., grad_fn=<MulBackward0>)
```

接下来，通过调用反向传播函数来自动计算 `y` 关于 `x` 每个分量的梯度。
```py
y.backward()
x.grad  # 结果为 tensor([ 0.,  4.,  8., 12.])，即 4x
```

现在计算 `x` 的另一个函数。
```py
# 在默认情况下，PyTorch 会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad  # 结果为 tensor([1., 1., 1., 1.])
```

### 2.5.2. 非标量变量的反向传播
当 `y` 不是标量时，向量 `y` 关于向量 `x` 的导数的最自然解释是一个矩阵。对于高阶和高维的 `y` 和 `x`，求导的结果可以是一个高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习（包括深度学习）中，但当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。这里，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。
```py
# 对非标量调用 backward 需要传入一个 gradient 参数，该参数指定微分函数关于 self 的梯度
# 本例只想求偏导数的和，所以传递一个 1 的梯度是合适的
x.grad.zero_()
y = x * x
y.sum().backward()  # 等价于 y.backward(torch.ones(len(x)))
x.grad  # 结果为 tensor([0., 2., 4., 6.])
```

### 2.5.3. 分离计算
有时，我们希望将某些计算移动到记录的计算图之外。例如，假设 `y` 是作为 `x` 的函数计算的，而 `z` 则是作为 `y` 和 `x` 的函数计算的。想象一下，我们想计算 `z` 关于 `x` 的梯度，但由于某种原因，希望将 `y` 视为一个常数，并且只考虑到 `x` 在 `y` 被计算后发挥的作用。

这里可以分离 `y` 来返回一个新变量 `u`，该变量与 `y` 具有相同的值，但丢弃计算图中如何计算 `y` 的任何信息。换句话说，梯度不会向后流经 `u` 到 `x`。因此，下面的反向传播函数计算 `z=u*x` 关于 `x` 的偏导数，同时将 `u` 作为常数处理，而不是 `z=x*x*x` 关于 `x` 的偏导数。
```py
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u  # 结果为 tensor([True, True, True, True])
```

由于记录了 `y` 的计算结果，我们可以随后在 `y` 上调用反向传播，得到 `y=x*x` 关于的 `x` 的导数，即 `2*x`。
```py
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x  # 结果为 tensor([True, True, True, True])
```

### 2.5.4. Python 控制流的梯度计算
使用自动微分的一个好处是：即使构建函数的计算图需要通过 Python 控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。在下面的代码中，`while` 循环的迭代次数和 `if` 语句的结果都取决于输入 `a` 的值。
```py
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

让我们计算梯度。
```py
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

我们现在可以分析上面定义的 `f` 函数。请注意，它在其输入 `a` 中是分段线性的。换言之，对于任何 `a`，存在某个常量标量 `k`，使得 `f(a)=k*a`，其中 `k` 的值取决于输入 `a`，因此可以用 `d/a` 验证梯度是否正确。
```py
a.grad == d / a  # 结果为 tensor(True)
```

### 2.5.5. 小结
- 深度学习框架可以自动计算导数：我们首先将梯度附加到想要对其计算偏导数的变量上，然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度。

## 2.6. 概率
简单地说，机器学习就是做出预测。概率是一种灵活的语言，用于说明我们的确定程度，并且它可以有效地应用于广泛的领域中。

### 2.6.1. 基本概率论
*大数定律*（law of large numbers）

*抽样*（sampling）：从概率分布中抽取样本的过程。

*分布*（distribution）：可看作对事件的概率分配。更正式的定义是*随机变量*（random variable）获得某一值的概率。

*多项分布*（multinomial distribution）：将概率分配给一些离散选择的分布。

#### 2.6.1.1. 概率论公理
*样本空间*（sample space），或称*结果空间*（outcome space）

*结果*（outcome）：样本空间中的每个元素。

*事件*（event）：一组给定样本空间的随机结果。

*概率*（probability）可以被认为是将集合映射到真实值的函数。在给定的样本空间 $\mathcal{S}$ 中，事件 $\mathcal{A}$ 的概率，表示为 $P(\mathcal{A})$，满足以下属性：
  - 对于任意事件 $\mathcal{A}$，其概率从不会是负数，即 $P(\mathcal{A}) \geq 0$；
  - 整个样本空间的概率为 $1$，即 $P(\mathcal{S}) = 1$；
  - 对于*互斥*（mutually exclusive）事件，它们的交集为空，即它们在任何一次试验中都不会同时发生。

#### 2.6.1.2. 随机变量
考虑一个随机变量 $X$，通过 $P(X=a)$，我们区分了随机变量 $X$ 和 $X$ 可以采取的值（例如 $a$）。为了简化符号，一方面，我们可以将 $P(X)$ 表示为随机变量 $X$ 上的分布。另一方面，我们可以简单用 $P(a)$ 表示随机变量取值 $a$ 的概率。

请注意，*离散*（discrete）随机变量和*连续*（continuous）随机变量之间存在微妙的区别。在连续随机变量情况下，我们将取某个数值的可能性量化为*密度*（density），随机变量恰好为某一个值的概率为 0，但密度不是 0。

### 2.6.2. 处理多个随机变量
很多时候，我们会考虑多个随机变量。当我们处理多个随机变量时，会有若干个变量是我们感兴趣的。

#### 2.6.2.1. 联合概率
*联合概率*（joint probability）：多个随机变量同时满足的概率。表示为 $P(A=a,B=b)$。

#### 2.6.2.2. 条件概率
*条件概率*（conditional probability）：在 $A=a$ 已发生的前提下，$B=b$ 的概率表示为 $P(B=b \mid A=a)$，它等于 $\frac{P(A=a, B=b)}{P(A=a)}$。

#### 2.6.2.3. 贝叶斯定理
*Bayes 定理*（Bayes’ theorem）

根据*乘法法则*（multiplication rule），$P(A, B) = P(B \mid A) P(A)$。根据对称性，可得 $P(A, B) = P(A \mid B) P(B)$。假设 $P(B)>0$，容易得出
$$
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.
$$
请注意，这里我们使用紧凑的表示法：其中 $P(A, B)$ 是一个联合分布，$P(A \mid B)$ 是一个条件分布。这种分布可以在给定值 $A = a, B=b$ 上进行求值。

#### 2.6.2.4. 边际化
*求和法则*（sum rule）：
$$
P(B) = \sum_{A} P(A, B),
$$
这也称为*边际化*（marginalization）。边际化结果的概率或分布称为*边际概率*（marginal probability）或*边际分布*（marginal distribution）。

#### 2.6.2.5. 独立性
*依赖*（dependence）

*独立*（independence）：事件 $A$ 的发生跟 $B$ 事件的发生无关。表述为 $A \perp B$。

由于 $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ 等价于 $P(A, B) = P(A)P(B)$，因此两个随机变量是独立的**当且仅当**两个随机变量的联合分布是其各自分布的乘积。同样地，给定另一个随机变量 $C$ 时，两个随机变量 $A$ 和 $B$ 是*条件独立*的（conditionally independent）**当且仅当** $P(A, B \mid C) = P(A \mid C)P(B \mid C)$。这个情况表示为 $A \perp B \mid C$。

### 2.6.3. 期望和方差
一个随机变量 $X$ 的*期望*（expectation，或平均值（average））表示为
$$
E[X] = \sum_{x} x P(X = x).
$$

当函数 $f(x)$ 的输入是从分布 $P$ 中抽取的随机变量时，$f(x)$ 的期望值为
$$
E_{x \sim P}[f(x)] = \sum_x f(x) P(x).
$$

在许多情况下，我们希望衡量随机变量 $X$ 与其期望值的偏置。这可以通过方差来量化
$$
\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.
$$
方差的平方根被称为*标准差*（standard deviation）。

随机变量函数的方差衡量的是：当从该随机变量分布中采样不同值 $x$ 时，函数值偏离该函数的期望的程度：
$$
\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].
$$

### 2.6.4. 小结
- 我们可以从概率分布中采样。

- 我们可以使用联合分布、条件分布、Bayes 定理、边缘化和独立性假设来分析多个随机变量。

- 期望和方差为概率分布的关键特征的概括提供了实用的度量形式。
