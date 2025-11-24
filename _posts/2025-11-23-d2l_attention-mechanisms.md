---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之 10. 注意力机制"
date: 2025-11-23
tags: [AI, notes]
toc: true
comments: true
author: Pianfan
---

意识的聚集和专注使灵长类动物能够在复杂的视觉环境中将注意力引向感兴趣的物体，例如猎物和天敌。只关注一小部分信息的能力对进化更加有意义，使人类得以生存和成功<!-- more -->

### 10.1. 注意力提示

注意力是稀缺、有价值的资源，存在机会成本

处于“注意力经济”时代，注意力被视为可交换的稀缺商品

### 10.1.1. 生物学中的注意力提示

**双组件（two-component）**框架（威廉·詹姆斯提出）：

- 非自主性提示：基于环境中物体的突出性和易见性

- 自主性提示：受认知和意识控制，由主观意愿推动

### 10.1.2. 查询、键和值

注意力机制与全连接层/汇聚层的区别：包含自主性提示

核心要素：

- **查询（query）**：自主性提示
- **键（key）**：非自主性提示（与值配对）
- **值（value）**：感官输入（sensory inputs，例如中间特征表示）
- **注意力汇聚（attention pooling）**：将查询与键匹配，引导选择对应的值

### 10.1.3. 注意力的可视化

注意力汇聚是加权平均总和，权重通过查询与不同键计算得出

PyTorch 相关代码：

```py
# 显示矩阵热图函数
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
```

## 10.2. 注意力汇聚：Nadaraya-Watson 核回归

**注意力汇聚**：对值的加权平均。**注意力权重（attention weight）**由查询与键的关系决定，满足非负且和为 1 的概率分布特性

**Nadaraya-Watson 核回归（Nadaraya-Watson kernel regression）**：基于注意力机制的机器学习模型，公式为：

$$
f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i
$$

其中 $\alpha(x, x_i)$ 为注意力权重，$(x_i, y_i)$ 为键值对，$x$ 为查询

### 10.2.1. 非参数注意力汇聚

**高斯核（Gaussian kernel）**：$K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2})$

**注意力权重计算**：

$$
\alpha(x, x_i) = \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right)
$$

**实现关键**：

- 将查询重复为与键同形状（`repeat_interleave`）

- 用 `softmax` 计算权重（`dim=1`）

- 加权平均通过矩阵乘法实现

### 10.2.2. 带参数注意力汇聚

**改进公式**：引入可学习参数 $w$

$$
\alpha(x, x_i) = \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right)
$$

#### 10.2.2.1. 批量矩阵乘法

用于高效计算小批量数据的加权平均

PyTorch 中使用 `torch.bmm`，要求输入形状为 $(n,a,b)$ 和 $(n,b,c)$，输出为 $(n,a,c)$

#### 10.2.2.2. 定义模型

```py
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
    
    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)
```

#### 10.2.2.3. 训练

损失函数：均方误差（`nn.MSELoss`）

优化器：随机梯度下降（`torch.optim.SGD`）

## 10.3. 注意力评分函数

注意力汇聚输出：值的加权和，即 $f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$

注意力权重：由**注意力评分函数（attention scoring function）**经 softmax 得到，即 $\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))}$，其中 $a$ 为评分函数

### 10.3.1. 掩蔽 softmax 操作（masked softmax operation）

功能：过滤超出有效长度的位置，使这些位置在 softmax 计算中输出为 0

PyTorch 实现：

```py
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

### 10.3.2. 加性注意力（additive attention）

适用场景：查询和键为不同长度的矢量

评分函数：$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k)$，其中 $\mathbf W_q\in\mathbb R^{h\times q}$、$\mathbf W_k\in\mathbb R^{h\times k}$、$\mathbf w_v\in\mathbb R^{h}$

PyTorch 实现：

```py
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

### 10.3.3. 缩放点积注意力（scaled dot-product attention）

适用场景：查询和键为相同长度的矢量（长度为 $d$）

评分函数：$a(\mathbf q, \mathbf k) = \mathbf{q}^\top \mathbf{k}  /\sqrt{d}$

批量计算：$\mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V$，其中 $\mathbf Q\in\mathbb R^{n\times d}$、$\mathbf K\in\mathbb R^{m\times d}$、$\mathbf V\in\mathbb R^{m\times v}$

PyTorch 实现：

```py
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

## 10.4. Bahdanau 注意力

**背景**：解决传统 Seq2Seq 模型中每个解码步骤使用相同上下文变量的局限，通过注意力机制在解码时选择性关注输入序列相关部分

**核心思想**：将上下文变量视为注意力池化的输出，使每个解码时间步 $t'$ 的上下文变量 $\mathbf{c}_{t'}$ 动态变化

### 10.4.1. 模型公式

解码时间步 $t'$ 的上下文变量计算：

$$
\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t
$$

- $\mathbf{s}_{t' - 1}$：解码器 $t' - 1$ 时刻隐状态（查询）

- $\mathbf{h}_t$：编码器 $t$ 时刻隐状态（键和值）

- $\alpha$：通过加性注意力打分函数计算的注意力权重

### 10.4.2. 定义注意力解码器

1. **注意力解码器基类**

    ```py
    class AttentionDecoder(d2l.Decoder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        @property
        def attention_weights(self):
            raise NotImplementedError
    ```

2. **Bahdanau 注意力解码器**

    ```py
    class Seq2SeqAttentionDecoder(AttentionDecoder):
        def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
            super().__init__(**kwargs)
            self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
            self.dense = nn.Linear(num_hiddens, vocab_size)
        
        # 初始化状态：编码器输出、隐状态和有效长度
        # 前向传播：将注意力输出与嵌入输入拼接后输入GRU
    ```

3. **核心机制**：每个解码步使用上一时刻解码器隐状态作为查询，计算与编码器所有隐状态的注意力，生成动态上下文变量

### 10.4.3. 训练

训练流程与基础 Seq2Seq 类似，但因注意力机制计算量增加，训练速度较慢

可通过可视化注意力权重观察解码时对输入序列的关注区域

评估指标使用 BLEU 分数衡量翻译质量

## 10.5. 多头注意力

**多头注意力（multihead attention）**通过多个注意力头并行计算，融合不同子空间的表示，捕获序列中多种范围的依赖关系

每个**头（head）**学习查询、键、值的不同**子空间表示（representation subspaces）**，最终拼接结果并通过线性变换输出

### 10.5.1. 模型

每个注意力头 $h_i$：$\mathbf{h}_i = f(\mathbf{W}_i^{(q)}\mathbf{q}, \mathbf{W}_i^{(k)}\mathbf{k}, \mathbf{W}_i^{(v)}\mathbf{v}) \in \mathbb{R}^{p_v}$

其中参数：$\mathbf{W}_i^{(q)} \in \mathbb{R}^{p_q \times d_q}$，$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$，$\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$

最终输出：$\mathbf{W}_o \begin{bmatrix}\mathbf{h}_1\\\vdots\\\mathbf{h}_h\end{bmatrix} \in \mathbb{R}^{p_o}$，$\mathbf{W}_o \in \mathbb{R}^{p_o \times hp_v}$

### 10.5.2. 实现

1. **MultiHeadAttention 类**

    包含参数：num_heads（头数）、attention（缩放点积注意力）、W_q/W_k/W_v/W_o（线性层）

    前向传播步骤：

    - 对查询、键、值进行线性变换
    - 通过 transpose_qkv 转换形状以并行计算多头
    - 应用注意力机制
    - 通过 transpose_output 还原形状并经 W_o 输出

2. **关键变换函数**

    transpose_qkv：将输入形状从 (batch_size, 序列长度，num_hiddens) 转换为 (batch_size*num_heads, 序列长度，num_hiddens/num_heads)，实现多头并行

    transpose_output：逆转 transpose_qkv 的操作，拼接多头结果

3. **参数设置**

    通常设 $p_q = p_k = p_v = p_o/h$，避免计算和参数代价激增

    num_hiddens 为输出特征维度

**输入输出形状**

输入：queries/keys/values 为 (batch_size, 序列长度，num_hiddens)；valid_lens 为 (batch_size,) 或 (batch_size, 查询数)

输出：(batch_size, 查询数，num_hiddens)

## 10.6. 自注意力和位置编码

### 10.6.1. 自注意力（self-attention）

定义：查询、键、值来自同一输入序列的注意力机制

公式：$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$，其中 $f$ 为注意力汇聚函数

输入输出形状：(批量大小，序列长度，隐藏维度)，输入输出形状相同

PyTorch 实现示例：

```py
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

### 10.6.2. 比较卷积神经网络、循环神经网络和自注意力

| 架构 | 计算复杂度 | 顺序操作数 | 最大路径长度 |
| --- | --- | --- | --- |
| CNN（核大小 k）| $\mathcal{O}(knd^2)$ | $\mathcal{O}(1)$ | $\mathcal{O}(n/k)$ |
| RNN | $\mathcal{O}(nd^2)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ |
| 自注意力 | $\mathcal{O}(n^2d)$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ |

### 10.6.3. 位置编码（positional encoding）

作用：为自注意力注入序列位置信息（解决自注意力并行计算丢失的顺序信息）

固定位置编码公式：

$$
\begin{split}\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right)\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right)\end{aligned}\end{split}
$$

PyTorch 实现：

```py
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

特性：

- 包含绝对位置信息（编码维度上频率单调降低）

- 包含相对位置信息（位置偏移 $\delta$ 可通过线性投影表示）

## 10.7. Transformer

### 10.7.1. 模型架构

基于编码器－解码器架构，完全依赖注意力机制，无卷积或循环层

输入/输出序列嵌入需加位置编码后分别输入编码器/解码器

### 10.7.2. 关键组件

1. **残差连接（residual connection）与层规范化（layer normalization）**

    层规范化基于特征维度，适用于变长序列

    残差连接形式：`x + sublayer(x)`，紧随层规范化

2. **编码器**

    堆叠 `num_layers` 个 `EncoderBlock`

    每个 `EncoderBlock` 包含：

    - **多头自注意力（multi-head self-attention）**子层
    - **基于位置的前馈网络（positionwise feed-forward network）**子层
    - 两个子层均带残差连接和层规范化

    输入**嵌入（embedding）**乘嵌入维度平方根后与位置编码相加

3. **解码器**

    堆叠 `num_layers` 个 `DecoderBlock`

    每个 `DecoderBlock` 包含：

    - **掩蔽（masked）**多头自注意力子层（确保自回归属性）
    - **编码器－解码器注意力（encoder-decoder attention）**子层（查询来自解码器，键/值来自编码器）
    - 基于位置的前馈网络子层
    - 三个子层均带残差连接和层规范化

4. **注意力机制**

    多头自注意力：并行执行多个缩放点积注意力
    编码器自注意力：查询、键、值均来自前一层编码器输出
    解码器自注意力：仅允许关注当前位置及之前位置
    编码器－解码器注意力：查询来自解码器，键/值来自编码器输出

### 10.7.3. 核心实现（PyTorch）

`TransformerEncoder`：包含嵌入层、位置编码、编码器块序列

`TransformerDecoder`：包含嵌入层、位置编码、解码器块序列

编码器/解码器特征维度均为 `num_hiddens`，确保注意力计算和残差连接兼容

### 10.7.4. 训练要点

超参数：隐藏层维度 `num_hiddens`、层数 `num_layers`、注意力头数 `num_heads`、前馈网络维度 `ffn_num_hiddens` 等

训练过程使用序列到序列学习框架

可通过注意力权重可视化分析模型关注重点

### 10.7.5. 特点

并行计算能力强，最大路径长度短

残差连接和层规范化是训练深层模型的关键

位置编码提供序列位置信息，与嵌入表示相加

可单独使用编码器或解码器适用于不同任务
