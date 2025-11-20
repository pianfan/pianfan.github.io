---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 8. 循环神经网络"
date: 2025-11-19
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

如果说卷积神经网络可以有效地处理空间信息，那么本章的**循环神经网络（recurrent neural network，RNN）**则可以更好地处理序列信息。循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出<!-- more -->

## 8.1. 序列模型

**内插法（interpolation）**：在现有观测值之间估计

**外推法（extrapolation）**：对超出已知观测范围预测（更难）

### 8.1.1. 统计工具

**自回归模型（autoregressive models）**：使用过去 τ 个时间步的观测值 $x_{t-1},...,x_{t-τ}$ 预测 $x_t$，参数数量固定

**隐变量自回归模型（latent autoregressive models）**：保留过去观测的总结 $h_t$，通过 

$$
\hat{x}_t = P(x_t \mid h_t)\\
h_t = g(h_{t-1}, x_{t-1})
$$

更新

**马尔可夫模型（first-order Markov model）**：满足马尔可夫条件（仅需近期历史），一阶模型满足 $P(x_1,...,x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1})$（$P(x_1 \mid x_0) = P(x_1)$）

**因果关系**：时间具有方向性，未来不影响过去，$P(x_{t+1} \mid x_t)$ 比 $P(x_t \mid x_{t+1})$ 更容易解释

### 8.1.2. 训练

1. **数据生成**：生成带噪声的正弦序列作为示例数据

2. **特征标签构造**：基于 τ，构建特征和标签

    $$
    \mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]\\
    y_t = x_t
    $$

3. **模型架构**：简单多层感知机（2 个全连接层 + ReLU 激活）

    ```py
    def get_net():
        net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
        net.apply(init_weights)  # Xavier初始化
        return net
    ```

4. **损失函数**：均方误差损失（MSELoss）

5. **训练过程**：使用 Adam 优化器，迭代训练并计算损失

### 8.1.3. 预测

1. **单步预测（one-step-ahead prediction）**：直接预测下一个时间步的值

2. **$k$ 步预测（$k$-step-ahead prediction）**：使用自身预测结果作为输入进行后续预测

3. 预测误差随步数增加而累积

## 8.2. 文本预处理

### 8.2.1. 核心步骤

1. 将文本作为字符串加载到内存

2. 将字符串拆分为词元（单词或字符）

3. 建立词表，映射词元到数字索引

4. 将文本转换为数字索引序列

### 8.2.2. 关键函数与类

1. **读取数据集**

    ```py
    import collections
    from d2l import torch as d2l
    import re

    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                    '090b5e7e70c295757f55df93cb0a180b9691891a')

    def read_time_machine():
        with open(d2l.download('time_machine'), 'r') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    ```

2. **词元化**

    ```py
    def tokenize(lines, token='word'):
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('错误：未知词元类型：' + token)
    ```

3. **词表（Vocab 类）**

    - 功能：将词元映射到数字索引

    - 包含未知词元（`<unk>`）及可选保留词元（`<pad>`、`<bos>`、`<eos>`等）

    - 按词元频率排序，可通过 `min_freq` 过滤低频词元

    ```py
    class Vocab:
        def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
            # 初始化逻辑
        def __len__(self):
            return len(self.idx_to_token)
        def __getitem__(self, tokens):
            # 词元转索引
        def to_tokens(self, indices):
            # 索引转词元
    ```

4. **整合函数**

    ```py
    def load_corpus_time_machine(max_tokens=-1):
        lines = read_time_machine()
        tokens = tokenize(lines, 'char')
        vocab = Vocab(tokens)
        corpus = [vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab
    ```

### 8.2.3. 核心概念

**词元（token）**：文本的基本单位（单词或字符）

**语料（corpus）**：训练集中的所有文档集合

**词表（vocabulary）**：词元与数字索引的映射字典

## 8.3. 语言模型和数据集

**语言模型（language model）目标**：估计文本序列的联合概率 $P(x_1, x_2, \ldots, x_T)$

**联合概率分解**：$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \ldots, x_{t-1})$

### 8.3.1. 自然语言统计

**词频特性**：遵循齐普夫定律

- 公式：$n_i \propto \frac{1}{i^\alpha}$ 或 $\log n_i = -\alpha \log i + c$

**n-gram 模型**：

- 一元语法（unigram）：单个词概率

- 二元语法（bigram）：连续两个词的条件概率

- 三元语法（trigram）：连续三个词的条件概率

**n-gram 频率特性**：均遵循齐普夫定律，指数随 n 增大而减小

### 8.3.2. 长序列数据读取

**随机采样**：

- 从随机起始索引获取子序列

- 相邻批量的子序列在原始序列中不相邻

**顺序分区**：

- 从固定偏移（可随机选择初始偏移）开始连续划分

- 相邻批量的子序列在原始序列中相邻

**数据加载类**：

```py
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        self.data_iter_fn = seq_data_iter_random if use_random_iter else seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

**数据加载函数**：

```py
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

### 8.3.3. 关键问题与解决方案

**低频序列问题**：长序列组合罕见，传统计数方法效果差

**拉普拉斯平滑（Laplace smoothing）**：为计数加小常数，解决零概率问题

- 示例：$\hat{P}(x) = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}$

**马尔可夫假设**：通过截断依赖简化计算（n-gram 模型基础）

## 8.4. 循环神经网络

**隐状态（hidden state）模型**：用隐状态 $h_{t-1}$ 存储序列到 $t-1$ 的信息，近似条件概率 $P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1})$

**隐状态计算**：$h_t = f(x_{t}, h_{t-1})$，其中 $f$ 为映射函数

**隐藏层与隐状态区别**：隐藏层是输入到输出路径上的隐藏层；隐状态是给定步骤的输入，仅通过先前时间步数据计算

### 8.4.1. 无隐状态的神经网络

隐藏层计算：$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h)$，$\mathbf{X} \in \mathbb{R}^{n \times d}$，$\mathbf{H} \in \mathbb{R}^{n \times h}$

输出层计算：$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q$，$\mathbf{O} \in \mathbb{R}^{n \times q}$

分类问题可用 $\text{softmax}(\mathbf{O})$ 计算概率分布

### 8.4.2. 有隐状态的循环神经网络

**隐状态计算**：

$$
\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h)
$$

其中

$$
\mathbf{X}_t \in \mathbb{R}^{n \times d}, \mathbf{H}_t \in \mathbb{R}^{n \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}
$$

**输出计算**：

$$
\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q
$$

**参数特性**：参数在不同时间步共享，参数数量不随时间步增加

### 8.4.3. 基于循环神经网络的字符级语言模型

词元化为字符，原始序列移位一个词元作为标签

训练时对每个时间步输出做 softmax，用交叉熵损失计算误差

时间步 $t$ 的输出 $\mathbf{O}_t$ 由截至 $t$ 的序列信息决定

### 8.4.4. 困惑度（Perplexity）

定义：$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right)$

意义：衡量语言模型质量，表示下一个词元实际选择数的调和平均数

极端情况：完美模型 perplexity=1；最差模型 perplexity=∞；均匀分布模型 perplexity = 词表大小

## 8.5. 循环神经网络的从零开始实现

### 8.5.1. 数据准备

加载数据集：使用 `d2l.load_data_time_machine` 获取批量数据 `train_iter` 和词汇表 `vocab`，参数包括 `batch_size` 和 `num_steps`

### 8.5.2. 初始化模型参数

函数 `get_params(vocab_size, num_hiddens, device)`：初始化隐藏层参数（`W_xh`、`W_hh`、`b_h`）和输出层参数（`W_hq`、`b_q`），参数服从正态分布（均值 0，标准差 0.01），并附加梯度

### 8.5.3. 循环神经网络模型

1. 状态初始化

    - 函数 `init_rnn_state(batch_size, num_hiddens, device)`：返回形状为（批量大小，隐藏单元数）的零张量作为初始隐状态

2. 前向计算

    - 函数 `rnn(inputs, state, params)`：按时间步更新隐状态 `H`（使用 `tanh` 激活函数），计算输出 `Y`，返回所有时间步输出拼接结果和最终隐状态

3. 模型类 `RNNModelScratch`

    - `__init__`：存储词汇表大小、隐藏单元数、参数，初始化状态函数和前向函数

    - `__call__`：对输入 `X` 进行独热编码后，调用前向函数计算输出和状态

    - `begin_state`：初始化隐状态

### 8.5.4. 训练

1. 训练周期函数 `train_epoch_ch8`

    - 初始化状态和计时器，使用 `d2l.Accumulator` 记录训练损失和词元数量

    - 遍历训练数据，根据是否随机抽样初始化或分离隐状态梯度

    - 计算预测值 `y_hat` 和损失 `l`，反向传播并裁剪梯度（`grad_clipping(net, 1)`），更新参数

    - 返回困惑度（`math.exp(总损失/总词元数)`）和速度（词元/秒）

2. 训练函数 `train_ch8`

    - 使用 `nn.CrossEntropyLoss` 作为损失函数，`d2l.Animator` 可视化困惑度

    - 初始化优化器（`torch.optim.SGD` 或自定义 `sgd`）

    - 训练多个周期，定期输出预测结果并可视化，最终输出困惑度、速度和预测示例

### 8.5.5. 关键技术

梯度裁剪：将梯度投影到半径为 1 的球内，防止梯度爆炸（公式：$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}$）

隐状态处理：顺序划分时需分离隐状态梯度以减少计算量，随机抽样时重新初始化状态

### 8.5.6. 预测

函数 `predict_ch8`：根据前缀生成指定长度的后续文本，基于模型输出概率分布预测下一个字符

## 8.6. 循环神经网络的简洁实现

**数据准备**

- 加载时光机器数据集，设置批量大小 `batch_size=32`，时间步长 `num_steps=35`

- 获取数据迭代器 `train_iter` 和词汇表 `vocab`

### 8.6.1. 定义模型

1. **RNN 层**：使用 `nn.RNN`，参数为 `(词汇表大小, 隐藏单元数)`

    ```py
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    ```

2. **隐状态初始化**：形状为 `(隐藏层数, 批量大小, 隐藏单元数)`

    ```py
    state = torch.zeros((1, batch_size, num_hiddens))
    ```

3. 完整模型类 `RNNModel`

    - 包含 `rnn` 层和输出层 `linear`

    - 前向传播：输入经独热编码后传入 `RNN` 层，输出经全连接层转换

    - 隐状态处理：支持单向/双向及 LSTM/GRU 不同类型

### 8.6.2. 训练与预测

初始随机权重模型预测效果差

训练超参数：轮次 `num_epochs=500`，学习率 `lr=1`

调用 `d2l.train_ch8` 训练，`d2l.predict_ch8` 预测

框架高级 API 实现比从零开始实现训练更快

## 8.7. 通过时间反向传播

BPTT 是反向传播在循环神经网络（RNN）中的特定应用，通过展开时间步计算梯度

核心是基于链式法则，对 RNN 的计算图按时间步展开，获取变量与参数的依赖关系并计算梯度

### 8.7.1. 循环神经网络的梯度分析

**梯度计算关键**

简化模型中，隐状态和输出定义：

$$
\begin{split}\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h)\\o_t &= g(h_t, w_o)\end{aligned}\end{split}
$$

目标函数关于隐藏层参数 $w_h$ 的梯度：

$$
\begin{split}\begin{aligned}\frac{\partial L}{\partial w_h} & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}\end{aligned}\end{split}
$$

隐状态对参数的梯度存在递归关系：

$$
\frac{\partial h_t}{\partial w_h}= \frac{\partial f}{\partial w_h} +\frac{\partial f}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}
$$

#### 8.7.1.1. 截断时间步

对长序列，在 $\tau$ 步后截断梯度计算（终止 $\partial h_{t-\tau}/\partial w_h$ 后的求和）

是对真实梯度的近似，聚焦短期影响，偏向简单稳定模型

实现方式：在 PyTorch 中通过 `detach()` 方法分离梯度

#### 8.7.1.2. 梯度问题

长序列中，$\mathbf{W}_{hh}^\top$ 的高次幂导致：

- 特征值 > 1：梯度爆炸

- 特征值 < 1：梯度消失

解决方式：截断时间步、梯度裁剪、使用 LSTM 等高级模型

### 8.7.2. 通过时间反向传播的细节

隐状态梯度递归计算：

$$
\frac{\partial L}{\partial \mathbf{h}_t} = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}
$$

参数梯度计算：

$$
\begin{split}\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top
\end{aligned}\end{split}
$$

计算时缓存中间值（如 $\partial L/\partial \mathbf{h}_t$）以提高效率
