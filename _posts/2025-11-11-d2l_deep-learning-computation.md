---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 5. 深度学习计算"
date: 2025-11-11
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

在本章中，我们将深入探索深度学习计算的关键组件，即模型构建、参数访问与初始化、设计自定义层和块、将模型读写到磁盘，以及利用 GPU 实现显著的加速<!-- more -->

## 5.1. 层和块

**块（block）**：由 `nn.Module` 类表示，可描述单个层、多层组件或整个模型

块的关键功能：

- 接收输入数据（前向传播函数参数）
- 生成输出（前向传播函数返回值）
- 自动计算输出对输入的梯度（反向传播）
- 存储和访问必要参数
- 初始化模型参数

### 5.1.1. 自定义块

继承 `nn.Module` 类

必须实现：

- `__init__`：初始化父类及定义层（调用 `super().__init__()`）
- `forward`：定义前向传播逻辑

```py
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)     # 输出层
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

### 5.1.2. 顺序块

`nn.Sequential`：按顺序组合多个块

内部维护 `_modules` 有序字典存储子块

前向传播按添加顺序依次执行各块

```py
# 自定义简化版Sequential
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
```

### 5.1.3. 在前向传播函数中执行代码

前向传播可包含任意 Python 控制流

可定义非可学习的常量参数（`requires_grad=False`）

块可嵌套组合，形成复杂架构

## 5.2. 参数管理

### 5.2.1. 参数访问

单层参数：通过 `net[层索引].state_dict()` 访问

参数值获取：`net[层索引].weight.data`（权重）、`net[层索引].bias.data`（偏置）

梯度访问：`net[层索引].weight.grad`（未反向传播时为 `None`）

所有参数：

- `net.named_parameters()`：返回 (参数名，参数) 迭代器
- `net.state_dict()`：返回包含所有参数的字典

嵌套块参数：通过嵌套索引访问，如 `rgnet[0][1][0].bias.data`

### 5.2.2. 参数初始化

1. 内置初始化：

    - 通过自定义初始化函数结合 `net.apply(func)` 应用

    - 常用方法：`nn.init.normal_(m.weight, mean=0, std=0.01)`（正态分布）、`nn.init.zeros_(m.bias)`（零初始化）、`nn.init.constant_(m.weight, 1)`（常数初始化）、`nn.init.xavier_uniform_(m.weight)`（Xavier 初始化）

2. 自定义初始化：

    - 定义函数对 `nn.Linear` 类型模块进行参数设置

    - 可直接修改参数值：`net[0].weight.data[:] += 1`

### 5.2.3. 参数绑定

实现方式：将同一层实例多次加入网络，如共享 `shared = nn.Linear(8, 8)`

特性：共享层参数为同一对象，修改一处则多处同步变化

梯度处理：反向传播时共享参数的梯度会累加

## 5.3. 自定义层

### 5.3.1. 不带参数的层

继承 `nn.Module` 类

需实现 `__init__` 方法（调用父类初始化）和 `forward` 方法（定义前向传播逻辑）

```py
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()
```

可作为组件融入复杂模型（如 `nn.Sequential` 中）

### 5.3.2. 带参数的层

继承 `nn.Module` 类

参数通过 `nn.Parameter` 定义（自动纳入模型参数管理）

需实现 `__init__` （定义参数）和 `forward`（定义计算逻辑）

```py
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

可像内置层一样用于构建模型（如 `nn.Sequential` 组合）

## 5.4. 读写文件

### 5.4.1. 加载和保存张量

单个张量：使用 `torch.save(x, '文件名')` 保存，`torch.load('文件名')` 加载

张量列表：`torch.save([x, y], '文件名')` 保存，`torch.load('文件名')` 加载（返回列表）

张量字典：`torch.save({'key': tensor}, '文件名')` 保存，`torch.load('文件名')` 加载（返回字典）

### 5.4.2. 加载和保存模型参数

保存参数：`torch.save(net.state_dict(), '文件名')`（仅保存参数，不包含架构）

加载参数：

1. 先实例化与原模型相同架构的模型（如 `clone = MLP()`）
2. 加载参数：`clone.load_state_dict(torch.load("文件名"))`
3. 切换评估模式：`clone.eval()`

## 5.5. GPU

### 5.5.1. 计算设备

设备表示：CPU 为 `torch.device('cpu')`，GPU 为 `torch.cuda.device('cuda')` 或 `torch.cuda.device('cuda:i')`（i 为 GPU 编号，从 0 开始）

设备查询：

- 可用 GPU 数量：`torch.cuda.device_count()`

- 设备获取函数：

    ```py
    def try_gpu(i=0):  # 若存在则返回gpu(i)，否则返回cpu
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    def try_all_gpus():  # 返回所有可用GPU，无则返回[cpu()]
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        return devices if devices else [torch.device('cpu')]
    ```

### 5.5.2. 张量与 GPU

默认张量创建在 CPU 上，可通过 `x.device` 查询设备

多设备操作要求张量在同一设备

GPU 存储：创建时指定设备，如 `X = torch.ones(2, 3, device=try_gpu())`

复制操作：

- 跨设备传输：`Z = X.cuda(1)`（将 X 复制到第 2 个 GPU）
- 同一设备复制：`Z.cuda(1)` 若 Z 已在 cuda:1，则返回自身不复制

注意：设备间数据传输缓慢，应尽量避免

### 5.5.3. 神经网络与GPU

模型部署到 GPU：`net = net.to(device=try_gpu())`

输入张量在 GPU 时，模型计算在同一 GPU 进行

模型参数设备查询：`net[0].weight.data.device`
