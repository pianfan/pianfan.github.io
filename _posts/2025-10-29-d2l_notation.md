---
layout: post
title: "《动手学深度学习（第二版）》学习笔记之符号"
date: 2025-10-29
tags: [AI, notes]
toc: true
comments: true
author: pianfan
---

本书中使用的符号概述如下。

## 数字
- $x$：标量
- $\mathbf{x}$：向量
- $\mathbf{X}$：矩阵
- $\mathsf{X}$：张量
- $\mathbf{I}$：单位矩阵
- $x_i$, $[\mathbf{x}]_i$：向量 $\mathbf{x}$ 第 $i$ 个元素
- $x_{ij}$, $[\mathbf{X}]_{ij}$：矩阵 $\mathbf{X}$ 第 $i$ 行第 $j$ 列的元素

## 集合论
- $\mathcal{X}$：集合
- $\mathbb{Z}$：整数集合
- $\mathbb{R}$：实数集合
- $\mathbb{R}^n$：$n$ 维实数向量集合
- $\mathbb{R}^{a \times b}$：包含 $a$ 行和 $b$ 列的实数矩阵集合
- $\mathcal{A} \cup \mathcal{B}$：集合 $\mathcal{A}$ 和 $\mathcal{B}$ 的并集
- $\mathcal{A} \cap \mathcal{B}$：集合 $\mathcal{A}$ 和 $\mathcal{B}$ 的交集
- $\mathcal{A} \setminus \mathcal{B}$：集合 $\mathcal{A}$ 与集合 $\mathcal{B}$ 相减，$\mathcal{B}$ 关于 $\mathcal{A}$ 的相对补集

## 函数和运算符
- $f(\cdot)$：函数
- $\log(\cdot)$：自然对数
- $\exp(\cdot)$：指数函数
- $\mathbf{1}_\mathcal{X}$：指示函数
- $\mathbf{(\cdot)}^\top$：向量或矩阵的转置
- $\mathbf{X}^{-1}$：矩阵的逆
- $\odot$：按元素相乘
- $[\cdot, \cdot]$：连结
- $|\mathcal{X}|$：集合的基数
- $\|\cdot\|_p$：$L_p$ 正则
- $\|\cdot\|$：$L_2$ 正则
- $\langle\mathbf{x}, \mathbf{y}\rangle$：向量 $\mathbf{x}$ 和 $\mathbf{y}$ 的点积
- $\sum$：连加
- $\prod$：连乘
- $\stackrel{\mathrm{def}}{=}$：定义

## 微积分
- $\frac{dy}{dx}$：$y$ 关于 $x$ 的导数
- $\frac{\partial y}{\partial x}$：$y$ 关于 $x$ 的偏导数
- $\nabla_{\mathbf{x}} y$：$y$ 关于 $\mathbf{x}$ 的梯度
- $\int_a^b f(x)\;dx$：$f$ 在 $a$ 到 $b$ 区间上关于 $x$ 的定积分
- $\int f(x)\;dx$：$f$ 关于 $x$ 的不定积分

## 概率与信息论
- $P(\cdot)$：概率分布
- $z \sim P$：随机变量 $z$ 具有概率分布 $P$
- $P(X \mid Y)$：$X \mid Y$ 的条件概率
- $p(x)$：概率密度函数
- $E_x[f(x)]$：函数 $f$ 对 $x$ 的数学期望
- $X \perp Y$：随机变量 $X$ 和 $Y$ 是独立的
- $X \perp Y \mid Z$：随机变量 $X$ 和 $Y$ 在给定随机变量 $Z$ 的条件下是独立的
- $\mathrm{Var}(X)$：随机变量 $X$ 的方差
- $\sigma_X$：随机变量 $X$ 的标准差
- $\mathrm{Cov}(X, Y)$：随机变量 $X$ 和 $Y$ 的协方差
- $\rho(X, Y)$：随机变量 $X$ 和 $Y$ 的相关性
- $H(X)$：随机变量 $X$ 的熵
- $D_{\mathrm{KL}}(P\|Q)$：$P$ 和 $Q$ 的 KL-散度

## 复杂度
- $\mathcal{O}$：大 O 标记

---

以上为对[原文内容](https://zh.d2l.ai/chapter_notation/index.html)的搬运，markdown 公式参考的[本书 GitHub 源码](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_notation/index.md?plain=1)。

推荐一个[在线 LaTeX 公式编辑器](https://www.latexlive.com/)，提供了很全的符号表示和公式模板，还支持转化为 MathML、图片等各种格式。其帮助文档的[“2 数学公式编辑 Displaying a formula”部分](https://www.latexlive.com/help#d11)给出了常用数学符号及公式的 LaTeX 语法。
