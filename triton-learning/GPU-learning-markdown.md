# GPU 架构与 CUDA 编程核心机制学习笔记

## 1. 硬件与软件层级架构 (Hardware & Software Hierarchy)

### SM (Hardware) vs Block (Software)
*   **多对一关系**：一个物理 SM (Streaming Multiprocessor) 可以同时驻留多个逻辑 Block。
    *   *目的*：为了**延迟掩盖 (Latency Hiding)**。当 Block A 在等内存数据时，SM 立即切换去执行 Block B 的指令，不让计算核心闲着。
*   **不可拆分性**：一个 Block 必须完整地在一个 SM 内执行，绝不能拆分到两个 SM。

### 重点：Block 间隔离 vs Block 内协作 (用户答疑)
*   **Block 之间：老死不相往来**
    *   **隔离**：Block A 和 Block B 拥有各自独立的 Shared Memory 空间。Block A 无法直接去搬运或访问 Block B 的数据。
    *   *比喻*：两栋不同的大楼，门禁卡不通用。
    *   **结论**：Block 之间不能合作搬运同一个 Tile。
*   **Block 内部：Warp 间紧密协作**
    *   **协作**：同一个 Block 内包含多个 Warp（如 Warp 0, Warp 1...）。它们共享同一块 Shared Memory。
    *   **分工**：当 Block 需要搬运一个大的 Tile（如 1024 个元素）到 Shared Memory 时，任务会被瓜分。
        *   Warp 0 搬运第 0-31 个元素。
        *   Warp 1 搬运第 32-63 个元素。
    *   **同步**：搬运完成后，必须使用 `__syncthreads()` 栅栏，确保所有 Warp 都搬完了，才能进入下一步计算。

---

## 2. 内存层级与 SIMT 执行原理 (Memory & SIMT)

### 内存分类速查
| 内存类型 | 物理位置 | 访问权限 | 速度 | 典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| **Registers** | SM 内部 | 单个线程私有 | **最快 (0延迟)** | 存放 Result Tile (累加结果) |
| **Shared Memory** | SM 内部 | 同 Block 共享 | **极快** | 存放 Loading Tile (中转数据) |
| **Global Memory** | 显存 (DRAM) | 全局共享 | **慢 (高延迟)** | 存放原始大矩阵 A, B |

### 重点：什么是“单指令多线程 (SIMT)”？(用户答疑)
你可能会疑惑：*“如果指令只有一条，大家动作必须一样，我怎么让 Thread 0 搬 A，Thread 1 搬 B？”*

*   **核心逻辑**：指令是**“公式”**，而线程 ID 是**“变量”**。
*   **代码视角**：你写下的代码 `B[threadIdx.x]`。
*   **硬件执行视角**：
    *   硬件广播了一条指令：`Load_Address = Base_Address + (My_ID * 4)`。
    *   **Thread 0** 执行：`Base + 0` -> 读取地址 `0x1000`。
    *   **Thread 1** 执行：`Base + 4` -> 读取地址 `0x1004`。
*   **结论**：大家执行的是**同一个逻辑公式**，但因为代入的 `My_ID` 不同，所以产生了**并排访问**的效果。

### 重点：为什么是 `+4` 而不是 `+1`？(用户答疑)
*   **软件视角 (C语言)**：`Array[0]` 到 `Array[1]`，下标确实是 **+1**。
*   **硬件视角 (物理内存)**：内存是按**字节 (Byte)** 编号的。
    *   一个 `float` 占据 **4 Bytes**。
    *   为了拿到下一个 float，物理地址必须跳过 4 个房间。
    *   这就是为什么 Thread 0 读 `Addr`，Thread 1 读 `Addr + 4`。
*   **合并访问 (Coalesced Access)**：当 32 个线程请求的地址是 `X, X+4, X+8...` 这样紧密排列时，GPU 内存控制器会将其合并为一个大的 **Burst Transaction**，效率 100%。

---

## 3. 矩阵乘法 (GEMM) 优化策略 (核心深度解析)

### 背景与冲突 (The Conflict)
我们要计算 $C = A \times B$。假设矩阵巨大（如 $2048 \times 2048$），无法放入片上内存。
*   **数学规则**：$C_{ij} = \text{Row}_i(A) \cdot \text{Col}_j(B)$。我们需要 **A 的一行** 和 **B 的一列** 进行点积。
*   **存储现实**：C/C++ 中矩阵是 **行存储 (Row-Major)** 的。
    *   $A$ 的一行：内存连续 -> **好读** (合并访问)。
    *   $B$ 的一列：内存跨度极大 (Stride = Width) -> **难读** (未合并访问，带宽浪费严重)。
*   **矛盾**：如果直接从 Global Memory 读 B 的一列进行计算，速度会慢几十倍。

### 解决方案：基于 Shared Memory 的分块 (Tiling)
我们引入 Shared Memory (SMEM) 作为“高速中转站”，利用它来**解耦**“搬运时的形状”和“计算时的形状”。

#### Step 1: 循环切分 (The Loop)
由于 A 和 B 的行/列太长（K 维度），SMEM 装不下。我们将长条切成小段。
*   **Block 的任务**：负责计算 $C$ 的一个小方块（Result Tile，如 $32 \times 32$）。
*   **流程**：在一个循环中，每次只处理一小段数据（Loading Tile，如 $32 \times 32$）。

#### Step 2: 搬运阶段 (Loading Phase) —— 关键优化
**目标**：把 B 的一个小方块搬进 SMEM。
*   **策略**：虽然逻辑上我们需要 B 的列，但在**搬运时刻**，我们**“挂羊头卖狗肉”**。
*   **操作**：我们让 Warp **横着** 去读 B 的这个方块。
    *   让 `threadIdx.x` 对应 B 的 **列索引**。
    *   **效果**：从 Global Memory 读取时，构成了**合并访问**（Coalesced），带宽跑满。
*   **结果**：数据高效地进入了 SMEM。

#### Step 3: 计算阶段 (Compute Phase)
**目标**：利用 SMEM 里的数据进行点积。
*   **环境**：数据已经在 SMEM 里了。
*   **操作**：线程现在可以**“竖着”** 去读 SMEM 里的 B（按列取值）。
*   **原因**：SMEM 是片上内存，随机访问极快（没有 Burst Mode 限制）。虽然竖着读可能有 Bank Conflict，但相比 Global Memory 的跨步访问，性能提升是巨大的。

### 总结：Tile 的双重身份
| 概念 | 身份 | 存储位置 | 访问模式 |
| :--- | :--- | :--- | :--- |
| **Loading Tile** | **原料** (Loop 中不断更新) | **Shared Memory** | **搬运时横着搬** (为了 Global Memory 带宽)<br>**计算时竖着读** (为了数学逻辑) |
| **Result Tile** | **产物** (Block 最终目标) | **Registers** | **静态驻留** (累加期间一直握在手里，不碰显存) |

---

## 4. 极端情况：Split-K (进阶)

*   **场景**：**瘦长矩阵** (GEMV) 或 **Rank 极小** 的乘法（如 $2048 \times 1$）。
*   **问题**：结果矩阵 $C$ 只有 1 个元素（或很少）。如果按标准做法，只有 1 个 Block 有活干，其他 80 个 SM 围观。GPU 利用率接近 0。
*   **对策 (Split-K)**：
    *   打破“一个 Block 算一个结果”的惯例。
    *   将那 2048 的累加长度切成 100 份。
    *   派 100 个 Block 协作，每人算一份部分和。
    *   **代价**：最后必须通过 Global Memory 的 **原子加法 (Atomic Add)** 来合并结果，开销较大。
    *   **收益**：相比于核心闲置，这个代价是值得的。

---


## 6. Debugging (调试指南)

GPU 编程难在**异步**与**黑盒**。
1.  **强行同步**：
    *   设置环境变量 `CUDA_LAUNCH_BLOCKING=1`。
    *   效果：让 GPU 变成串行执行，报错时能定位到具体的 CPU 代码行。
2.  **内存越界检查**：
    *   使用 `compute-sanitizer` 工具。
    *   效果：精确捕捉 Illegal Memory Access。
3.  **Kernel 内部打印**：
    *   在核函数内使用 `if (tid==0) printf(...)`。
    *   注意：不要让所有线程都打印，否则缓冲区会爆。
4.  **IDE 级调试**：
    *   使用 Nsight Compute (Ncu) 或 Nsight VSE。
    *   能力：支持断点、单步执行、查看寄存器值。