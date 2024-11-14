---
layout: post
title: "P5课上指令攻略"
date:   2024-11-12
tags: [CO,P5]
comments: true
author: BUAA-Yzx2023
---


<!-- more -->

首先建议设计文档里有一份顶层架构图，能够反映流水线运作原理，尤其是各级之间的转发。而画这个图的过程，本身也是对代码的数据通路debug的过程。

课上指令通常为三类：计算类、跳转类、访存类。

### 一、计算类

通常是R型指令，`GPR[rs]`与`GPR[rt]`运算，结果存入寄存器 `rd` 中。

一般只需要修改 ALU 即可，复杂的运算可在顺序块`always @ (*)`中实现，注意循环时计数变量要初始化

### 二、跳转类

情况多种，主要体现在以下三个方面：

- 有/无 条件 跳转

  在CMP模块中处理好 isBranch 信号即可

- 有/无 条件 链接
  - 无条件链接 ：通常是链接 `PC+8` 到  `$ra` 。容易实现，与写寄存器相关的信号同jal指令即可。
  - 有条件链接 ：即条件不成立时，不链接。存在两种解决方案：
    1. 在D级引入新的写寄存器信号 `RFWr_new` ，根据CMP模块的条件返回值对其赋值，然后将其流水至W级，代替原有的写寄存器信号。要注意当指令不是此指令时， `RFWr_new` 信号应等于原来的 `RFWr` 信号。个人认为这种方法较为麻烦，不建议采用。
    2. 注意到，当被写的寄存器为 `$zero` 时，将不会执行写入操作。于是，可以改变被写寄存器的控制信号，使写入的寄存器为`$zero`

- 条件不成立时，是/否 清空延迟槽

  清空延迟槽，即废除延迟槽指令，不再执行，等价于将延迟槽指令替换成`nop`。

  课下的beq指令是不清空的，而需要清空时，我们将 F_to_D 流水线寄存器清零即可（注意这和阻塞的区别）。这需要一个从CMP发往 F_to_D 的控制信号`D_clr`。
  
  注意其置1条件 ：（指令为此跳转指令） &&  （跳转条件不成立） &&  (**此时非阻塞**)  。一定要注意，阻塞时是不可清零的，因为此时还未获取最新的`GPR[rs]`和`GPR[rt]`，跳转条件成立与否未知。



本次课上Q2的指令`JTL`，属于条件跳转、无条件链接、清空延迟槽

| jtl<br>100110 | rs   | rt   | offset |
| ------------- | ---- | ---- | ------ |
| 6             | 5    | 5    | 16     |

GTL语言大致如下：

```GTL
I：  
	 temp1 ← GPR[rs]+GPR[rt]
	 temp2 ← PC+4+sign_ext(offset||0^2)
	 GPR[31] ← PC+8
	 
II： 
	 if(temp2<temp1):
	 	PC ← temp2
	 else
	 	PC ← temp1
	 	NullifyCurrentInstruction()
⁡
注： 1.NullifyCurrentInstruction() 的意思是清空延迟槽
	2.相加不考虑溢出
	3.比较为无符号比较
```

笔者在实现时，在CMP模块中增加了`[31:0]jump`，专门用来计算`JTL`指令的跳转地址，并接到NPC模块上。同时将控制信号NPCOp拓展了一位，但是笔者最开始粗心忘记了在NPC模块也将其接口拓宽一位，导致WA了一会。



### 三、访存类

通常为条件访存，`lw`的变式指令。需要改动的点主要是以下两个方面：

- 条件：

  课下的`lw`指令计算出基地址后无条件写寄存器，而课上指令一般会加以条件。条件可能会用到 `rt` 甚至别的寄存器。

  如果只是用到 `rt`，那么我们只添加一个该条件的组合逻辑即可，因为课下已经实现了此处 `rt` 的转发；

  如果是别的寄存器，譬如`$ra`，那么需要在GRF模块增加一个 `raData` 的输出端口，并将其流水至M级，同时增加与它相关的暂停、转发逻辑。

- 访存：

  通常是根据条件，将数据写入指定寄存器。而这个指定寄存器大概率不再是`lw`的 `rt` 寄存器。

  此时也需要改变数据通路，实现方式比较容易，只需在M级更新被写寄存器的编号，即增加`A3M_new`的组合逻辑，然后令其替代原本的`A3M`即可。下面将以本次的课上题目为例进行说明。



本次课上Q3的指令`LWOC`

| lwoc<br>111110 | base | rt   | offset |
| -------------- | ---- | ---- | ------ |
| 6              | 5    | 5    | 16     |

其RTL语言大致如下

```
I:
    Addr ← GPR[base] + sign_ext(offset)
    word ← Memory[Addr]
    Re ← word3..0
    condition ← (word<0x80000000)
II:
	if(condition):
		GPR[Re] ← word
	else :
		GPR[rt] ← word

```

本指令比较简单，无条件取字，仅需修改被写入的寄存器

首先我们在M级增加`A3M_new`的组合逻辑。（这里DM_resultM在有些指令时可能是不定值，但此时`InstrOpM`一定不是`LWOC`，故不会影响条件判断）

 <img src="./../AppData/Roaming/Typora/typora-user-images/image-20241112112139936.png" alt="image-20241112112139936" style="zoom:50%;" />

然后将 **暴力转发**、**暂停判断**、**MW 流水线寄存器**里所有的 `A3M` 替换成 `A3M_new` 即可

- 暴力转发：

   <img src="./../AppData/Roaming/Typora/typora-user-images/image-20241112112644435.png" alt="image-20241112112644435" style="zoom:50%;" />

- 暂停判断：

   <img src="./../AppData/Roaming/Typora/typora-user-images/image-20241112112730755.png" alt="image-20241112112730755" style="zoom:50%;" />

- MW 流水线寄存器：

   <img src="./../AppData/Roaming/Typora/typora-user-images/image-20241112112810863.png" alt="image-20241112112810863" style="zoom:50%;" />

 

