# GPU 高性能内存搬运与指令执行机制详解

## 1. 核心硬件约束：显存带宽的“最低消费”

在优化 GPU 程序时，必须时刻牢记两个物理层面的数字。这两个数字决定了你的代码是“满载运行”还是“空转浪费”。

### 1.1 宏观层级：128 Bytes (显存事务)
这是 **显存控制器 (Memory Controller)** 与 **L2 Cache** 之间传输数据的最小单位，俗称“显存大巴车”。

*   **物理根源**：**Burst Mode (突发模式)**。显存颗粒（DRAM）寻址慢、传输快。一旦寻址，它会强制连续吐出数据（物理上通常是 32 Bytes，逻辑上 GPU 倾向于凑满 L2 Cache Line 的 128 Bytes）。
*   **规则**：哪怕你只需要 4 Bytes 的数据，大巴车出车一次也是 128 Bytes。如果剩下的 124 Bytes 没人要，就是**带宽浪费**。

### 1.2 微观层级：128 Bits (线程向量化)
这是 **单个线程 (Thread)** 一条汇编指令能处理的最大数据宽度。

*   **对应指令**：`LD.E.128` (Load 128-bit)。
*   **常见类型**：`float4` (16 Bytes), `int4`。
*   **规则**：为了减少指令发射次数，提高流水线效率，应尽可能让一个线程一次搬更多数据。

---

## 2. 基础优化：合并访问 (Coalesced Access)

这是 CUDA 编程的第一天条。它的目标是**填满“128 Bytes 大巴车”**。

### 2.1 场景：标准 `float` 读取
假设一个 Warp (32 Threads) 读取连续的 `float` 数组。

*   **数据需求**：
    $$ 32 \text{ Threads} \times 4 \text{ Bytes/Thread} = 128 \text{ Bytes} $$
*   **硬件行为**：
    *   内存控制器看到请求地址连续（如 `0x0000` 到 `0x0080`）。
    *   正好是一辆大巴车的容量。
    *   **发起 1 次 Memory Transaction**。
*   **效率**：**100%**。

### 2.2 细节：为什么地址是 `+4`？
在 C 语言中我们写 `A[idx]` 和 `A[idx+1]`，但在物理地址上，这是 **+4**。
*   Thread 0 读取地址 `Base + 0`
*   Thread 1 读取地址 `Base + 4` (因为 1个 float = 4 Bytes)
*   ...
*   Thread 31 读取地址 `Base + 124`

**结论**：只要 Thread $N$ 读 `Addr`，Thread $N+1$ 读 `Addr + sizeof(Type)`，就是合并访问。

---

## 3. 进阶优化：向量化加载 (Vectorized Load)

这是让性能起飞的关键。目标是**让每个线程“暴食”，减少指令数**。

### 3.1 场景：使用 `float4` 读取
假设我们强制转换指针 `reinterpret_cast<float4*>(ptr)`，让每个线程一次搬运 4 个 float。

*   **单线程食量**：
    $$ 1 \text{ float4} = 4 \times 4 \text{ Bytes} = 16 \text{ Bytes} = \mathbf{128 \text{ Bits}} $$
    *(注：这里吃满了线程的 128-bit 限制)*

*   **Warp 总需求**：
    $$ 32 \text{ Threads} \times 16 \text{ Bytes} = \mathbf{512 \text{ Bytes}} $$

*   **硬件行为**：
    *   总需求 512 Bytes。
    *   大巴车容量 128 Bytes。
    *   内存控制器连续发起：
        $$ 512 / 128 = \mathbf{4 \text{ 次 Transactions}} $$

*   **收益分析**：
    *   **带宽**：依然是 100% 满载（4 辆车都是满的）。
    *   **指令吞吐**：原本需要 4 条 `LD.32` 指令搬运这些数据，现在只需要 **1 条** `LD.128` 指令。
    *   **结论**：计算单元压力减小 4 倍，非常高效。

---

## 4. 实战应用：GEMM 中的 Tile 搬运

在矩阵乘法中，我们需要把一大块数据（Tile）从 Global Memory 搬运到 Shared Memory。

### 4.1 核心冲突
*   **逻辑需求**：我们需要矩阵 B 的 **一列**。
*   **物理存储**：矩阵 B 是 **行存储** 的。一列数据在内存中是跳跃的（Stride = Width）。

### 4.2 解决方案：挂羊头卖狗肉
为了满足 **合并访问**，我们在搬运阶段完全无视逻辑需求。

1.  **Block 内分工**：
    假设 Tile 大小是 $32 \times 32$ (1024 个元素)，Block 内有 4 个 Warps。
    *   Warp 0 搬运第 0~255 个数据。
    *   Warp 1 搬运第 256~511 个数据。
    *   ...
    *   **大家合作**，瓜分这 1024 个搬运任务。

2.  **横向读取 (Coalescing)**：
    所有线程的 `threadIdx.x` 映射到矩阵的 **列索引**（连续地址）。
    *   *动作*：Warp 就像一把宽扫帚，横着把数据从 Global Memory 扫进来。
    *   *结果*：带宽利用率 100%。

3.  **存入 Shared Memory**：
    数据一旦进入 Shared Memory（高速片上内存），我们就可以随心所欲了。
    *   *计算时*：线程可以**竖着**去读 Shared Memory 里的数据。

---

## 5. Triton vs CUDA：算账时刻

这是理解 Triton **Block-Level 编程** 最精彩的例子。

### 5.1 场景参数
我们使用 Triton 编写一个 Kernel：
*   **数据类型**：`bf16` (BFloat16, 2 Bytes)。
*   **数据块大小**：`BLOCK_SIZE = 1024` (我们要处理 1024 个元素)。
*   **计算资源**：`num_warps = 4`。

### 5.2 表面矛盾
*   元素数量：1024 个。
*   线程数量：$4 \times 32 = 128$ 个。
*   *疑问：128 个线程怎么处理 1024 个数据？*

### 5.3 深度计算 (The "Perfect Match")
Triton 编译器会在后台自动进行**向量化分配**。

1.  **每个线程的任务量**：
    $$ \frac{1024 \text{ Elements}}{128 \text{ Threads}} = \mathbf{8 \text{ Elements/Thread}} $$

2.  **计算数据位宽 (关键一步)**：
    每个线程要搬运 8 个 `bf16`。
    $$ 8 \times \text{sizeof(bf16)} = 8 \times 16 \text{ bits} = \mathbf{128 \text{ bits}} $$

### 5.4 结论
*   **128 bits** 正好是 NVIDIA GPU 线程向量化加载的物理上限 (`LD.128`)。
*   Triton 编译器看到这个配置，直接生成了**最高效的向量化汇编代码**。
*   **Block-Level 哲学**：你只需要告诉编译器“我要搬 1024 个数”，编译器自己会去凑这“128 bits”的黄金比例，而不需要像 CUDA 那样手写 `float4`。

---

## 6. 总结对比：视角差异

| 维度 | CUDA (Thread-Level) | Triton (Block-Level) |
| :--- | :--- | :--- |
| **编程主体** | **士兵 (Thread)** | **排长 (Block)** |
| **数据视角** | 标量 (Scalar, 如 `float`) | 张量/块 (Tensor/Tile, 如 `1024xfloat`) |
| **优化方式** | 手动写 `float4`，手动算对齐 | 编译器自动推导向量化 (Vectorization) |
| **搬运协作** | 手动计算 `threadIdx` + `offset` | 自动生成多 Warp 协作代码 |
| **分支逻辑** | 显式处理 `Mask` (谓词) | 编译器处理 `Mask` |

**一句话总结**：
CUDA 的代码是写给**线程**看的，但优化是为了迎合**Warp**；Triton 的代码是写给**数据块**看的，优化由**编译器**自动映射到 Warp 和 Thread 指令。