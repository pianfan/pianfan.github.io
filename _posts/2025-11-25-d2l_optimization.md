---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 11. 优化算法"
date: 2025-11-25
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

优化算法对于深度学习非常重要。一方面，优化算法的性能直接影响模型的训练效率。另一方面，理解不同优化算法的原理及其超参数的作用将使我们能够以有针对性的方式调整超参数，以提高深度学习模型的性能<!-- more -->

## 11.1. 优化和深度学习

深度学习先定义损失函数（优化中称为目标函数），通过优化算法最小化损失

### 11.1.1. 优化目标

优化目标：最小化训练误差（基于训练数据集的损失）

深度学习目标：最小化泛化误差（整个数据群的预期损失），需关注过拟合

### 11.1.2. 优化挑战

1. **局部最小值**：目标函数在某点的值小于附近所有点的值，可能非全局最小；优化可能收敛到局部最小值

2. **鞍点（saddle point）**：梯度为零但既非局部最小值也非全局最小值的点；高维中更常见，其 Hessian 矩阵（也称黑塞矩阵）特征值有正有负

3. **梯度消失**：梯度接近零导致优化停滞（如 tanh 在 x=4 处梯度约 0.0013）；ReLU 激活函数缓解了此问题

## 11.2. 凸性（convexity）

### 11.2.1. 定义

1. **凸集（convex sets）**：向量空间中集合 $\mathcal{X}$，对任意 $a, b \in \mathcal{X}$ 和 $\lambda \in [0, 1]$，有 $\lambda  a + (1-\lambda)  b \in \mathcal{X}$

    - 凸集的交集 $\cap_i \mathcal{X}_i$ 是凸集，其并集不一定是凸集

2. **凸函数（convex functions）**：给定凸集 $\mathcal{X}$，若对任意 $x, x' \in \mathcal{X}$ 和 $\lambda \in [0, 1]$，函数 $f: \mathcal{X} \to \mathbb{R}$ 为凸函数，有 $\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x')$

### 11.2.2. 性质

1. **詹森不等式（Jensen’s inequality）**：对凸函数 $f$，有 $\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)$（$\alpha_i$ 非负且 $\sum_i \alpha_i = 1$），及 $E_X[f(X)] \geq f\left(E_X[X]\right)$

2. **凸性和二阶导数**：

    一维二次可微函数：凸函数当且仅当 $f''(x) \geq 0$

    多维二次可微函数：凸函数当且仅当 Hessian 矩阵 $\nabla^2 f \succeq 0$（半正定）

    多维函数凸性等价于：对任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$，$g(z) = f(z\mathbf{x} + (1-z)\mathbf{y})$（$z \in [0,1]$）是凸函数

### 11.2.3. 约束（constraints）

1. **约束优化（constrained optimization）问题**：形式为 $\mathop{\mathrm{minimize~}}_{\mathbf{x}} f(\mathbf{x}), \text{ subject to } c_i(\mathbf{x}) \leq 0$（$i = 1, \dots, n$）

2. **拉格朗日函数**：$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x})$（$\alpha_i \geq 0$），为鞍点优化问题（对 $\alpha_i$ 最大化，对 $\mathbf{x}$ 最小化）

3. **惩罚**：通过添加惩罚近似满足约束

4. **投影（projections）**：凸集 $\mathcal{X}$ 上的投影定义为 $\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|$

## 11.3. 梯度下降

**基本原理**

- 基于泰勒展开：对函数 $f(x)$，一阶近似为 $f(x + \epsilon) \approx f(x) + \epsilon f'(x)$

- 核心更新规则：沿负梯度方向更新参数以减小目标函数值

    - 一维：$x \leftarrow x - \eta f'(x)$（$\eta$ 为学习率）

    - 多元：$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$（$\nabla f(\mathbf{x})$ 为梯度向量，含各偏导数）

### 11.3.1. 一维梯度下降

关键要素：目标函数 $f(x)$、其梯度 $f'(x)$、学习率 $\eta$

**学习率（learning rate）**影响：

- 过小：收敛缓慢，需更多迭代

- 过大：可能超出最优解，导致发散或振荡

存在局部极小值问题，可能收敛到非全局最优

### 11.3.2. 多元梯度下降

梯度向量：$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_d}\bigg]^\top$

核心流程：通过迭代应用多元更新规则，使 $\mathbf{x}$ 逼近最小值

### 11.3.3. 自适应方法

1. **预处理（preconditioning）**：为每个变量（参数坐标）选择不同学习率，解决尺度不匹配问题，是随机梯度下降优化算法的创新动力之一

2. **牛顿法**：

    利用二阶泰勒展开：$f(\mathbf{x} + \boldsymbol{\epsilon}) \approx f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \mathbf{H} \boldsymbol{\epsilon}$（$\mathbf{H}$ 为 Hessian 矩阵）

    更新规则：$\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x})$

    注意：非凸问题中 Hessian 可能为负，需调整（如取绝对值、引入学习率）

3. **梯度下降和线搜索**：沿梯度方向通过二分搜索找最优学习率 $\eta$，收敛快但因需全数据集评估，深度学习中成本过高不常用

## 11.4. 随机梯度下降

**基本定义**

- 目标函数：训练数据集各样本损失函数的平均值，即 $f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x})$，其中 $f_i(\mathbf{x})$ 为第 $i$ 个样本的损失函数，$\mathbf{x}$ 为参数向量

- 目标函数梯度：$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x})$

### 11.4.1. SGD 核心思想

每次迭代随机均匀采样一个样本索引 $i$，用该样本的梯度 $\nabla f_i(\mathbf{x})$ 更新参数，更新规则：$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x})$，其中 $\eta$ 为学习率

优势：单轮迭代计算代价从梯度下降的 $\mathcal{O}(n)$ 降至 $\mathcal{O}(1)$

无偏性：随机梯度是完整梯度的无偏估计，即 $\mathbb{E}_i \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x})$

### 11.4.2. 动态学习率

需动态调整学习率 $\eta(t)$ 以平衡收敛速度与稳定性，常见策略：

- **分段常数（piecewise constant）**：$\eta(t) = \eta_i$（当 $t_i \leq t \leq t_{i+1}$ 时）

- **指数衰减（exponential decay）**：$\eta(t) = \eta_0 \cdot e^{-\lambda t}$

- **多项式衰减（polynomial decay）**：$\eta(t) = \eta_0 \cdot (\beta t + 1)^{-\alpha}$（常用 $\alpha = 0.5$）

### 11.4.3. 凸目标收敛性分析

对凸函数，SGD 可证明收敛到最优解，收敛速率为 $\mathcal{O}(1/\sqrt{T})$（$T$ 为迭代次数）

关键：学习率需随时间衰减，收敛性依赖随机梯度范数上界 $L$ 及初始参数与最优解距离 $r$

### 11.4.4. 有限样本实践

实际中采用无替换采样（遍历所有样本一次），比有替换采样方差更小、数据效率更高

多轮训练时，每次遍历数据集采用不同随机顺序

## 11.5. 小批量随机梯度下降

### 11.5.1. 核心概念

是梯度下降（全量数据）与随机梯度下降（单样本）的折中，平衡计算效率与统计效率

计算效率：依托向量化，减少框架开销，利用 CPU/GPU 缓存和内存 locality

统计效率：梯度基于小批量计算（$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$），方差随批量大小增大而降低

小批量选择：需足够大以保证计算效率，且适合 GPU 内存

### 11.5.2. 实现

1. **SGD 优化器实现**：

    ```py
    def sgd(params, states, hyperparams):
        for p in params:
            p.data.sub_(hyperparams['lr'] * p.grad)  # 参数更新：w = w - lr*grad
            p.grad.data.zero_()  # 梯度清零
    ```

2. **训练函数（从零实现）**：

    初始化：随机初始化权重 $w$（正态分布，std=0.01）、偏置 $b$（0），定义线性回归网络和平方损失

    迭代过程：遍历 epochs 和小批量，计算损失均值→反向传播求梯度→SGD 更新参数→记录损失和时间

    关键代码片段：

    ```py
    def train_ch11(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2):
        w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1), requires_grad=True)
        b = torch.zeros((1), requires_grad=True)
        net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
        for _ in range(num_epochs):
            for X, y in data_iter:
                l = loss(net(X), y).mean()
                l.backward()
                trainer_fn([w, b], states, hyperparams)  # 更新参数
    ```

3. **简洁实现**：

    网络：`nn.Sequential(nn.Linear(5, 1))`，权重初始化用 `torch.nn.init.normal_`

    优化器：`torch.optim.SGD`，损失函数 `nn.MSELoss`（需注意与 L2 损失的系数差异）

    迭代过程：清零梯度→前向计算→损失反向传播→优化器更新参数

    关键代码片段：

    ```py
    def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
        net = nn.Sequential(nn.Linear(5, 1))
        net.apply(lambda m: torch.nn.init.normal_(m.weight, std=0.01) if type(m)==nn.Linear else None)
        optimizer = trainer_fn(net.parameters(), **hyperparams)
        loss = nn.MSELoss(reduction='none')
        for _ in range(num_epochs):
            for X, y in data_iter:
                optimizer.zero_grad()
                out = net(X)
                l = loss(out, y.reshape(out.shape)).mean()
                l.backward()
                optimizer.step()
    ```

## 11.6. 动量法（momentum）

### 11.6.1. 基础

**动量法定义**：通过维护动量变量 $\mathbf{v}$（过去梯度的**泄漏平均值（leaky average）**）更新参数，公式：

$$
\begin{split}\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}\end{split}
$$

**特殊情况**：$\beta = 0$ 时退化为普通梯度下降

**动量含义**：$\mathbf{v}_t$ 是过去梯度的加权和，$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$，$\beta$ 越大，平均过去梯度的范围越广

### 11.6.2. 理论分析

- **二次凸函数**：

    - 函数形式：$h(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{Q}\mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$（$\mathbf{Q}$ 正定），最小值点 $\mathbf{x}^* = -\mathbf{Q}^{-1}\mathbf{c}$

    - 梯度：$\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$

    - 变量替换：$\mathbf{z} = \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$，简化为 $h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'$（$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda}\mathbf{O}$，$\mathbf{O}$ 正交，$\boldsymbol{\Lambda}$ 为特征值对角阵）

    - 动量更新（坐标级）：

        $$
        \begin{split}\begin{aligned}
        \mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
        \mathbf{z}_t & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}
        \end{aligned}\end{split}
        $$

- **标量函数**：

    - 函数 $f(x) = \frac{\lambda}{2}x^2$，梯度下降更新：$x_{t+1} = (1 - \eta\lambda)x_t$，收敛条件 $|1 - \eta \lambda| < 1$

    - 动量法更新矩阵：$\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_t \\ x_t \end{bmatrix}$，收敛条件 $0 < \eta \lambda < 2 + 2 \beta$（比梯度下降的 $0 < \eta\lambda < 2$ 范围更大）

### 11.6.3. 实际实验

- **从零实现**：

    - 初始化动量状态：`v_w = d2l.zeros((feature_dim, 1))`，`v_b = d2l.zeros(1)`

    - 动量更新函数：

        ```py
        def sgd_momentum(params, states, hyperparams):
            for p, v in zip(params, states):
                with torch.no_grad():
                    v[:] = hyperparams['momentum'] * v + p.grad
                    p[:] -= hyperparams['lr'] * v
                p.grad.data.zero_()
        ```

- **简洁实现**：使用 `torch.optim.SGD`，参数含 `lr`（学习率）和 `momentum`（动量系数）：

    ```py
    trainer = torch.optim.SGD
    d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
    ```

## 11.7. AdaGrad 算法

### 11.7.1. 核心思想

针对稀疏特征（偶尔出现的特征）优化学习率：为每个参数坐标动态调整学习率，使常见特征学习率下降较快，不常见特征学习率下降较慢

用过去梯度的平方和替代简单计数器，自动根据梯度大小调整学习率：梯度大的坐标学习率衰减更显著，梯度小的坐标学习率衰减更平缓

### 11.7.2. 公式

梯度计算：$\mathbf{g}_t = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w}))$

梯度平方累加：$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$（按坐标平方累加）

参数更新：$\mathbf{w}_t = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t$（$\eta$ 为学习率，$\epsilon$ 为数值稳定项，避免除零）

初始化：$\mathbf{s}_0 = \mathbf{0}$

### 11.7.3. 预处理视角

用梯度平方和作为黑塞矩阵（Hessian）对角线的代理，缓解优化问题的条件数（最大/最小特征值比）问题，无需计算二阶导数（避免 $\mathcal{O}(d^2)$ 的计算开销）

### 11.7.4. 实现

1. **从零开始实现**

    - 状态初始化：为权重和偏置分别维护累加变量

        ```py
        def init_adagrad_states(feature_dim):
            s_w = torch.zeros((feature_dim, 1))
            s_b = torch.zeros(1)
            return (s_w, s_b)
        ```

    - 更新函数：累加梯度平方，按公式更新参数并清零梯度

        ```py
        def adagrad(params, states, hyperparams):
            eps = 1e-6
            for p, s in zip(params, states):
                with torch.no_grad():
                    s[:] += torch.square(p.grad)
                    p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
                p.grad.data.zero_()
        ```

2. **简洁实现**：使用 PyTorch 内置优化器

    ```py
    trainer = torch.optim.Adagrad
    # 训练时传入参数，如学习率lr=0.1
    ```

## 11.8. RMSProp 算法

**提出背景**

- 解决 Adagrad 中学习率随时间过度衰减（$\mathcal{O}(t^{-\frac{1}{2}})$）的问题，适用于非凸问题（如深度学习）

- 保留 Adagrad 的坐标自适应特性，分离速率调度与坐标自适应学习率

### 11.8.1. 公式

1. 状态更新（泄漏平均梯度平方）：

    $$
    \begin{split}\begin{aligned}
        \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2
    \end{aligned}\end{split}
    $$

    （$\gamma$ 为衰减系数，控制历史梯度平方的记忆程度）

2. 参数更新：

    $$
    \begin{split}\begin{aligned}
        \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t
    \end{aligned}\end{split}
    $$

    （$\eta$ 为学习率，$\epsilon \approx 10^{-6}$ 避免除零或步长过大）

### 11.8.2. 实现

1. 状态初始化：

    ```py
    def init_rmsprop_states(feature_dim):
        s_w = torch.zeros((feature_dim, 1))  # 权重的梯度平方状态
        s_b = torch.zeros(1)  # 偏置的梯度平方状态
        return (s_w, s_b)
    ```

2. 算法实现：

    ```py
    def rmsprop(params, states, hyperparams):
        gamma, eps = hyperparams['gamma'], 1e-6
        for p, s in zip(params, states):
            with torch.no_grad():
                s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)  # 更新状态
                p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)  # 更新参数
            p.grad.data.zero_()  # 清零梯度
    ```

3. 简洁实现（框架内置）：

    ```py
    trainer = torch.optim.RMSprop
    # 关键参数：lr（学习率）、alpha（对应γ，默认0.9）
    d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9}, data_iter)
    ```

## 11.9. Adadelta

Adadelta 是 AdaGrad 的变体，减少学习率对坐标的适应性

广义上无学习率，使用参数变化量校准未来变化，由 Zeiler 于 2012 年提出

### 11.9.1. 公式

**状态变量**：2 个，$\mathbf{s}_t$（梯度二阶矩的泄露平均值）、$\Delta\mathbf{x}_t$（参数变化二阶矩的泄露平均值）

**更新公式**：

- $\mathbf{s}_t = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2$（$\rho$ 为超参数）

- 调整梯度：

    $$
    \mathbf{g}_t' = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t
    $$

    （$\epsilon$ 为小值，如 1e-5，保证数值稳定）

- 参数更新：$\mathbf{x}_t = \mathbf{x}_{t-1} - \mathbf{g}_t'$

- $\Delta \mathbf{x}_t = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2$

### 11.9.2. 实现

**状态初始化**：

```py
def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))
```

**优化过程**：

```py
def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)  # 更新s_t
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad  # 计算g_t'
            p[:] -= g  # 更新参数
            delta[:] = rho * delta + (1 - rho) * g * g  # 更新Δx_t
        p.grad.data.zero_()  # 清零梯度
```

**简洁实现**：使用 `torch.optim.Adadelta`，核心超参数 `rho`（常用 0.9）

## 11.10. Adam 算法

结合 SGD、动量法、Adagrad、RMSProp 的优点，是高效的深度学习优化算法

使用指数加权移动平均（EWMA）估计梯度的动量（一阶矩）和二次矩（二阶矩）

### 11.10.1. 公式

1. **状态变量更新**（$\beta_1 = 0.9$，$\beta_2 = 0.999$）：

    - 动量估计：$\mathbf{v}_t \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t$

    - 二次矩估计：$\mathbf{s}_t \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$

2. **偏差校正**（解决初始值为 0 的偏差）：

    - $\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t}$

    - $\hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}$

3. **参数更新**（$\epsilon = 10^{-6}$）：

    - $\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$

### 11.10.2. 实现

1. **状态初始化**：

    ```py
    def init_adam_states(feature_dim):
        v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)  # 动量状态
        s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)  # 二次矩状态
        return ((v_w, s_w), (v_b, s_b))
    ```

2. **Adam 更新函数**：

    ```py
    def adam(params, states, hyperparams):
        beta1, beta2, eps = 0.9, 0.999, 1e-6
        for p, (v, s) in zip(params, states):
            with torch.no_grad():
                # 更新动量和二次矩
                v[:] = beta1 * v + (1 - beta1) * p.grad
                s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
                # 偏差校正
                v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
                s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
                # 参数更新
                p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
            p.grad.data.zero_()  # 清零梯度
        hyperparams['t'] += 1  # 时间步递增
    ```

3. **框架内置调用**：

    ```py
    trainer = torch.optim.Adam
    d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
    ```

### 11.10.3. Yogi（Adam 改进）

**问题**：Adam 中 $\mathbf{s}_t$ 可能因梯度方差大或稀疏更新而爆炸

**改进**：修改二次矩更新，避免更新幅度依赖偏差量：

- $\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$

**实现**：修改 `s` 的更新逻辑，其余同 Adam

## 11.11. 学习率调度器

### 11.11.1. 一个简单的问题

- 训练函数：`train(net, train_iter, test_iter, num_epochs, loss, trainer, device, scheduler=None)`

    - 功能：处理训练循环，包括前向传播、损失计算、反向传播、参数更新

    - 调度器应用：若存在调度器，内置调度器调用 `scheduler.step()`；自定义调度器通过 `trainer.param_groups[0]['lr'] = scheduler(epoch)` 更新学习率

### 11.11.2. 学习率调度器

- 调整方式：可在迭代轮数或小批量后动态调整学习率

- 手动设置：`trainer.param_groups[0]["lr"] = 新学习率`

- 调度器类：通过 `__call__` 方法根据更新次数返回对应学习率，如 `SquareRootScheduler`（$\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$）

### 11.11.3. 策略

#### 11.11.3.1. 单因子调度器

- 机制：乘法衰减，$\eta_{t+1} \leftarrow \max(\eta_{\min}, \eta_t \cdot \alpha)$（$\alpha \in (0, 1)$）

- 实现：`FactorScheduler` 类，含参数 `factor`（衰减因子）、`stop_factor_lr`（最小学习率）、`base_lr`（初始学习率）

#### 11.11.3.2. 多因子调度器

- 机制：分段常数，在指定步骤（$s$）按因子衰减，$\eta_{t+1} \leftarrow \eta_t \cdot \alpha$（$t \in s$）

- PyTorch 内置：`MultiStepLR(trainer, milestones=[...], gamma=0.5)`，`milestones` 为衰减步骤，`gamma` 为衰减因子

#### 11.11.3.3. 余弦调度器

- 机制：按余弦函数调整学习率，常结合预热

- 应用：在部分计算机视觉问题中表现良好

#### 11.11.3.4. 预热

- 机制：优化初期线性增加学习率至初始最大值，再按调度器衰减

- 作用：防止因初始参数随机导致的优化发散，适用于高级网络
