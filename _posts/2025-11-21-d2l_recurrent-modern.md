---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 9. 现代循环神经网络"
date: 2025-11-21
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

我们将引入两个广泛使用的网络，即**门控循环单元（gated recurrent unit，GRU）**和**长短期记忆网络（long short-term memory，LSTM）**。然后，我们将基于一个单向隐藏层来扩展循环神经网络架构<!-- more -->

## 9.1. 门控循环单元（GRU）

解决 RNN 中梯度消失/爆炸问题，更好处理长序列依赖

比 LSTM 结构更简单，计算更快，性能相当

### 9.1.1. 门控机制

1. **重置门（$\mathbf{R}_t$）**

    公式：

    $$
    \mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r)
    $$

    作用：控制保留多少过去状态，帮助捕获短期依赖

2. **更新门（$\mathbf{Z}_t$）**

    公式：

    $$
    \mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z)
    $$

    作用：控制新旧状态的融合比例，帮助捕获长期依赖

3. **候选隐状态（$\tilde{\mathbf{H}}_t$）**

    公式：

    $$
    \tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + (\mathbf{R}_t \odot \mathbf{H}_{t-1}) \mathbf{W}_{hh} + \mathbf{b}_h)
    $$

4. **最终隐状态（$\mathbf{H}_t$）**

    公式：

    $$
    \mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t
    $$

### 9.1.2. PyTorch 实现要点

1. **参数初始化**

    ```py
    def get_params(vocab_size, num_hiddens, device):
        # 定义重置门、更新门、候选隐状态和输出层参数
        # 使用正态分布初始化权重，偏置初始化为0
    ```

2. **状态初始化**

    ```py
    def init_gru_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device), )
    ```

3. **GRU 计算过程**

    ```py
    def gru(inputs, state, params):
        # 依次计算更新门、重置门、候选隐状态和最终隐状态
        # 输出层计算并返回结果
    ```

4. **简洁实现**

    ```py
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    ```

## 9.2. 长短期记忆网络（LSTM）

解决隐变量模型中长时信息保存和短时输入缺失问题

### 9.2.1. 门控记忆元

引入**记忆元（memory cell）**记录额外信息，通过门控机制控制记忆元

#### 9.2.1.1. 门控机制

1. **输入门（$\mathbf{I}_t$）**：控制新数据进入记忆元的量

    $$
    \mathbf{I}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i)
    $$

2. **遗忘门（$\mathbf{F}_t$）**：控制保留过去记忆元内容的量

    $$
    \mathbf{F}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f)
    $$

3. **输出门（$\mathbf{O}_t$）**：控制从记忆元读取信息的量

    $$
    \mathbf{O}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o)
    $$

#### 9.2.1.2. 记忆元计算

1. **候选记忆元（$\tilde{C}_t$）**

    $$
    \tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c)
    $$

2. **记忆元更新**

    $$
    \mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t
    $$

3. **隐状态计算**

    $$
    \mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)
    $$

### 9.2.2. PyTorch 实现要点

1. **参数初始化**

    ```py
    def get_lstm_params(vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size
        # 定义参数初始化函数及三门参数、候选记忆元参数、输出层参数
    ```

2. **状态初始化**

    ```py
    def init_lstm_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),
                torch.zeros((batch_size, num_hiddens), device=device))
    ```

3. **LSTM 计算**

    ```py
    def lstm(inputs, state, params):
        # 实现输入门、遗忘门、输出门计算，更新记忆元和隐状态，生成输出
    ```

4. **简洁实现**

    ```py
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    ```

## 9.3. 深度循环神经网络

深度循环神经网络通过堆叠多层循环层构成，隐状态同时传递到当前层下一时间步和下一层当前时间步

隐藏层数量 $L$ 和隐藏单元数量 $h$ 为超参数

### 9.3.1. 函数依赖关系

设时间步 $t$ 输入为 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$，第 $l$ 层隐状态为 $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$

层间关系：

$$
\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)} + \mathbf{b}_h^{(l)})
$$

其中

$$
\mathbf{H}_t^{(0)} = \mathbf{X}_t
$$

输出层计算：

$$
\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q
$$

### 9.3.2. PyTorch 实现要点

1. 数据加载

    ```py
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    ```

2. 模型定义

    ```py
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = d2l.try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)  # 指定num_layers设置层数
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    ```

3. 训练配置

    ```py
    num_epochs, lr = 500, 2
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    ```

## 9.4. 双向循环神经网络

双向循环神经网络通过同时使用前向和后向的循环层，使每个时间步的隐状态能结合序列中过去和未来的信息

与**隐马尔可夫模型（hidden Markov model，HMM）**的前向-后向递归类似，但作为通用可学习函数存在

### 9.4.1. 结构定义

前向隐状态更新：

$$
\overrightarrow{\mathbf{H}}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)})
$$

后向隐状态更新：

$$
\overleftarrow{\mathbf{H}}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)})
$$

隐状态拼接：$\mathbf{H}_t$ 由 $\overrightarrow{\mathbf{H}}_t$ 和 $\overleftarrow{\mathbf{H}}_t$ 拼接而成（维度为 $n \times 2h$）

输出层计算：

$$
\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q
$$

### 9.4.2. 关键特性

利用双向上下文信息，适用于序列编码、缺失词填充、命名实体识别等任务

不适合下一 token 预测（测试时无法获取未来信息）

计算成本高：前向传播需双向递归，反向传播依赖前向结果，梯度链长

### 9.4.3. PyTorch 实现要点

定义双向 LSTM：`nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)`

模型训练需注意：不适用于语言模型等预测未来的任务，否则会导致生成结果不佳

## 9.5. 机器翻译与数据集

**机器翻译（machine translation）**：将序列从一种语言自动翻译成另一种语言

**序列转换模型（sequence transduction）**：将输入序列转换为输出序列的模型，是机器翻译的核心

**神经机器翻译（neural machine translation）**：基于神经网络的端到端学习方法，区别于传统**统计机器翻译（statistical machine translation）**

### 9.5.1. 数据集处理

1. **数据集特点**：由**源语言（source language）**和**目标语言（target language）**的文本序列对组成

2. **下载**：使用 Tatoeba 项目的英－法双语句子对数据集

3. **预处理步骤**：

    - 替换不间断空格为普通空格
    - 大写转小写
    - 单词与标点间插入空格

### 9.5.2. 核心操作

1. **词元化**：

    - 采用单词级词元化（词或标点为单位）

    - 生成源语言和目标语言两个词元列表

2. **词表构建**：

    - 为源语言和目标语言分别构建词表

    - 处理低频词（出现 < 2 次）为未知词（`<unk>`）

    - 包含特殊词元：填充（`<pad>`）、开始（`<bos>`）、结束（`<eos>`）

3. **序列处理**：

    - 截断（长序列取前 `num_steps` 个词元）

    - 填充（短序列补 `<pad>` 至 `num_steps` 长度）

    - 每个序列末尾添加 `<eos>` 标记

4. **数据加载**：

    - 转换为小批量数据

    - 记录序列有效长度（排除填充词元）

### 9.5.3. PyTorch 相关代码框架

```py
# 数据读取与预处理
raw_text = read_data_nmt()
text = preprocess_nmt(raw_text)

# 词元化
source, target = tokenize_nmt(text, num_examples)

# 构建词表
src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

# 加载数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
```

### 9.5.4. 关键函数

- `read_data_nmt()`：载入数据集

- `preprocess_nmt()`：预处理文本

- `tokenize_nmt()`：词元化处理

- `truncate_pad()`：截断或填充序列

- `build_array_nmt()`：转换文本序列为小批量

- `load_data_nmt()`：返回数据迭代器和词表

## 9.6. 编码器-解码器架构

用于处理输入和输出均为长度可变序列的场景

包含两个核心组件：**编码器（encoder）**和**解码器（decoder）**

- 编码器：将长度可变的输入序列转换为固定形状的编码状态

- 解码器：将固定形状的编码状态映射为长度可变的输出序列

### 9.6.1. PyTorch 实现接口

**编码器**

```py
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

**解码器**

```py
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

**编码器－解码器组合**

```py
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

### 9.6.2. 关键流程

1. 编码器处理输入序列 `enc_X` 得到编码输出 `enc_outputs`

2. 解码器通过 `init_state` 将编码输出转换为初始状态 `dec_state`

3. 解码器基于输入 `dec_X` 和状态 `dec_state` 生成输出序列

## 9.7. 序列到序列学习（seq2seq）

用于处理输入输出均为变长序列的任务（如机器翻译）

基于编码器－解码器架构，使用两个 RNN 分别实现

### 9.7.1. 编码器

功能：将变长输入序列转换为固定形状的隐状态（编码输入信息）

实现：

```py
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    def forward(self, X, *args):
        X = self.embedding(X).permute(1, 0, 2)  # (num_steps, batch_size, embed_size)
        output, state = self.rnn(X)  # output: (num_steps, batch_size, num_hiddens); state: (num_layers, batch_size, num_hiddens)
        return output, state
```

### 9.7.2. 解码器

功能：基于编码器输出的隐状态和已生成的词元预测下一个词元

实现关键：

- 以 `<bos>` 作为初始输入

- 用编码器最终隐状态初始化解码器隐状态

- 输出形状为 `(batch_size, num_steps, vocab_size)`

### 9.7.3. 训练

**强制教学（teacher forcing）**：将原始输出序列（不含 `<eos>`）与 `<bos>` 拼接作为解码器输入

训练函数核心步骤：

```py
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    # 初始化权重、优化器、损失函数（带遮蔽的Softmax交叉熵）
    for epoch in range(num_epochs):
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            # 反向传播、梯度裁剪、参数更新
```

### 9.7.4. 预测

过程：编码器处理输入→解码器从 `<bos>` 开始，迭代预测下一词元直至 `<eos>` 或达到最大长度

关键代码：

```py
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.eval()
    # 编码输入序列
    # 解码器迭代生成输出序列
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)  # 取概率最高的词元作为下一输入
        pred = dec_X.squeeze(dim=0).item()
        if pred == tgt_vocab['<eos>']: break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

### 9.7.5. 评估

采用 **BLEU（bilingual evaluation understudy）**指标：通过匹配预测序列与标签序列的 n-gram 计算得分

### 9.7.6. 关键组件

**嵌入层（embedding layer）**：将词元索引转换为特征向量

门控循环单元（GRU）：作为编码器和解码器的核心循环网络

遮蔽（mask）：过滤无关计算（如计算损失时忽略填充词元）

## 9.8. 束搜索

目标：从所有可能的输出序列（$\mathcal{O}(\vert\mathcal{Y}\vert^{T'})$ 种，$\vert\mathcal{Y}\vert$ 为词表大小，$T'$ 为最大长度）中寻找理想输出

输出序列需考虑 `<eos>` 终止符，其后部分会被丢弃

### 9.8.1. 序列搜索策略

1. **贪心搜索（greedy search）**

    策略：每个时间步 $t'$ 选择条件概率最高的词元：

    $$
    y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c})
    $$

    计算量：$\mathcal{O}(\vert\mathcal{Y}\vert T')$

    缺点：无法保证得到最优序列（最优序列需最大化 $\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$）

2. **穷举搜索（exhaustive search）**

    策略：列举所有可能序列，选择条件概率最高的序列

    计算量：$\mathcal{O}(\vert\mathcal{Y}\vert^{T'})$（计算成本极高）

    优点：能获得最优序列

3. **束搜索（beam search）**

    策略：

    - 超参数：束宽 $k$

    - 时间步 1：选择 $k$ 个最高条件概率的词元作为候选序列起点

    - 后续时间步：基于上一步的 $k$ 个候选，从 $k\vert\mathcal{Y}\vert$ 个可能选择中保留 $k$ 个最高条件概率的候选序列

    评分公式：$\frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$（$L$ 为序列长度，$\alpha$ 通常取 0.75，用于惩罚长序列）

    计算量：$\mathcal{O}(k\vert\mathcal{Y}\vert T')$（介于贪心和穷举之间）

    特点：贪心搜索是束宽 $k=1$ 的特殊情况，通过调整 $k$ 权衡正确率和计算代价
