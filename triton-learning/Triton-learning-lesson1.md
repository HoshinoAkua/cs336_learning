# 📘 Triton 学习笔记：从入门到硬件原理

**日期**：2026-01-15
**主题**：Triton 编程模型、编译原理与硬件映射

---

## Part 1: Triton 的核心语法与启动机制

### 1.1 核心概念：Block-based Programming
Triton 与 CUDA C++ 最大的区别在于抽象层级：
*   **CUDA (SIMT)**：你编写的是**单个线程 (Thread)** 的逻辑。你需要关心 `threadIdx`，手动计算标量索引。
*   **Triton (Block-based)**：你编写的是**数据块 (Block)** 的逻辑。你操作的是 `Tensor`，编译器负责将其切分给底层的线程。

### 1.2 启动语法：`kernel[grid](args)`

```python
# 示例
grid_fn = lambda meta: (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']), )
add_kernel[grid_fn](x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE=1024)
```

#### 🧐 你的疑惑与思考
> **Q: 为什么语法写成 `add_kernel[grid](**fun_args)`？这看起来很奇怪。**

**解析**：
这是利用 Python 对象模型实现的 DSL（领域特定语言）设计，目的是将**并行配置**与**数据传输**解耦。
1.  **`add_kernel`**: 经过 `@triton.jit` 修饰后，它是一个 `JITFunction` 对象。
2.  **`[grid]` (Launcher)**: 调用了 `__getitem__`。它负责计算**“我要启动多少个 Block”**。这是**配置层**。
3.  **`(args)` (Call)**: 调用了 `__call__`。它负责传递**指针、常量和参数**。这是**数据层**。

> **Q: 为什么社区习惯用 `meta` 这个名字？我觉得 `args` 更直观。**

**解析**：
*   这纯粹是惯例。`grid_fn` 接收的是一个包含所有参数的**字典 (Dict)**。
*   Python 中 `*args` 通常指代位置参数的**元组 (Tuple)**。为了避免混淆（让读代码的人不误以为可以用下标访问），社区使用 `meta`（Metadata，元数据）来强调这是用于计算 Grid 配置的信息。
*   **结论**：你可以用 `params` 或 `kwargs`，但为了合群，用 `meta` 更好。

> **Q: 为什么要返回一个 Tuple `(x, )` 而不是直接返回 Int？**

**解析**：
*   **硬件统一性**：GPU 的 Grid 硬件上支持 3 个维度 `(x, y, z)`。Triton 为了接口统一，强制要求返回一个序列。
*   **Python 陷阱**：`(10)` 是整数 `10`，`(10, )` 才是元组。如果漏掉逗号，Python 会传递整数导致 Triton 报错。

> **Q: 这个 `grid` 是 Index（第几个）吗？**

**解析**：
*   **绝对不是 Index，是 Count（总量）。**
*   **Grid (Python端)**：是**边界**。告诉 GPU 驱动：“给我发射 10 个 Block！”
*   **Program ID (Kernel端)**：是**坐标**。Kernel 醒来后问 `tl.program_id(0)`：“我是这 10 个里面的第几个？”

## Part1.5 triton kernel 的两步走

`kernel[grid](**kwargs)` 的方式, 本质上是执行了两个函数:
```Python
class KernelLauncher:
    def __init__(self, jit_func, grid):
        self.jit_func = jit_func
        self.grid = grid  # 可能是 tuple，也可能是 lambda

    def __call__(self, *args, **kwargs):
        # 1. 提取 constexpr 参数，组成 META 字典
        #    例如: args里有 BLOCK_SIZE=1024，这里提取出来
        meta = self._extract_meta(kwargs)
        
        # 2. 如果 grid 是个函数，现在调用它来计算真正的 grid 形状
        if callable(self.grid):
            real_grid = self.grid(meta) 
        else:
            real_grid = self.grid
            
        # 3. 检查缓存，是否需要编译
        #    注意：BLOCK_SIZE=1024 变了会导致重新编译
        compiled_kernel = self.jit_func.compile_or_load(meta, *args)
        
        # 4. 真正的 GPU 发射
        print(f"Running kernel on GPU: Grid={real_grid}")
        compiled_kernel.launch(real_grid, *args)

class JITFunction:
    def __init__(self, func):
        self.func = func
        self.cache = {} # 缓存编译好的内核

    def __getitem__(self, grid):
        # step 1: 仅仅是包装，不做重计算
        return KernelLauncher(self, grid)

# 使用装饰器
# @triton.jit 相当于: 
# add_kernel = JITFunction(original_add_kernel_func)
```

当你写 `add_kernel[grid](x, y, ...)` 时，Python 的执行顺序是：
1. 先执行 `add_kernel[grid]` -> 触发 `JITFunction.__getitem__` -> 返回一个 KernelLauncher 对象。
2. 再执行 `KernelLauncher_Object(x, y, ...)` -> 触发 `__call__` -> 真正启动 GPU 内核。
---

## Part 2: 编译原理 —— `constexpr` 与 JIT 缓存

这是 Triton 性能优化的核心，也是最容易踩坑的地方。

### 2.1 静态 vs 动态参数

| 特性 | **`tl.constexpr` (编译时常量)** | **普通参数 (运行时变量)** |
| :--- | :--- | :--- |
| **典型用途** | `BLOCK_SIZE`, `BLOCK_M`, `num_stages` | `n_elements`, `x_ptr`, `learning_rate` |
| **编译器行为** | 值被直接“烧录”进机器码；决定寄存器分配。 | 被视为寄存器变量；生成通用指令。 |
| **变动后果** | **触发重编译 (Re-compile)**。耗时，占显存。 | **零开销**。复用缓存 (Cache Hit)。 |

### 2.2 深入理解 `BLOCK_SIZE`

#### 🧐 你的疑惑与思考
> **Q: 为什么 `BLOCK_SIZE` 必须是 `constexpr`？Triton 不能自己看输入值识别吗？**

**深度解析**：
这涉及到 GPU 的底层架构。
1.  **寄存器分配 (Register Allocation)**：GPU 并没有“动态内存分配”给局部变量。编译器必须在生成二进制代码（PTX/SASS）时，精确算出这个 Kernel 需要多少个寄存器。
2.  **形状决定资源**：`BLOCK_SIZE` 决定了 `tl.arange` 生成的 Tensor 长度，进而决定了需要多少寄存器来存这些数。
3.  **特化 (Specialization)**：如果你不加 `constexpr`，编译器无法确定资源需求，就无法生成代码。标记它，就是在告诉编译器：“**请专门为 1024 这个大小定制一份最高效的代码。**”

> **Q: 那 `n_elements` 需要声明为 `constexpr` 吗？**

**回答**：**绝对不要！**

> **Q: 为什么？是因为没有哪个 Tensor 的形状依赖于 `n_elements` 吗？比如 `y = a + b + 1`？**

**深度解析（你的顿悟点）**：
*   **Triton 是“窗口式”处理**：Kernel 内部永远只拿着 `BLOCK_SIZE` (比如 1024) 这么大的勺子（Tensor）在操作。
*   **大海与勺子**：`n_elements` 是大海的总水量。我们在 Kernel 里不需要创建一个“大海那么大”的 Tensor。我们只需要知道“什么时候停止舀水”（Mask 边界判断）。
*   **标量广播**：`+1` 操作中，`1` 是标量，会自动广播成 `(BLOCK_SIZE,)` 的形状，与 `n_elements` 无关。
*   **结论**：`n_elements` 不影响 Kernel 内部 Tensor 的物理尺寸，所以它可以是动态的。

---

## Part 3: 并行模型与硬件映射 (解耦机制)

这是 Triton 最“黑科技”的地方。它将逻辑块与物理线程解耦了。

### 3.1 动态 Grid 与 静态 Block

> **Q: 这样说来，启动几个 block 是动态的，但是 block 的大小是静态的？**

**总结**：
*   **Block Size (静态)**：这是**模具**。编译时决定了桌子造多大。改了要重造（重编译）。
*   **Grid Size (动态)**：这是**产量**。运行时决定了招多少人来用这个桌子。随便改，零成本。
*   **补充**: 回顾之前的GPU课程, Triton中的BLOCK和GPU中的BLOCK指的是一种东西. BLOCK是一个逻辑划分, 它的内部有物理分割出来的wrap, 每个wrap有32个thread. 在cuda代码中, 我们需要手动管理wrap/thread(给每个thread写代码, 但是由wrap打包执行), 在triton中, 我们需要指定block. 所以Triton可以看作BLOCK-Level单指令多线程代码.

### 3.2 物理映射：Warps 与 自动向量化

> **Q: 一个 block=1024 就是分配了 32 个 wraps 吗？**

**回答**：
**不一定，通常更少。** 这是 Triton 性能优越的关键。
*   **CUDA 思维**：1 Thread 处理 1 Element。1024 Elements = 1024 Threads (32 Warps)。
*   **Triton 思维**：**Thread Coarsening (线程粗化)**。
    *   Triton 倾向于让更少的线程（例如 4 Warps = 128 Threads）去处理 1024 个数据。
    *   **结果**：每个线程处理 $1024 \div 128 = 8$ 个数据。

#### 🧪 性能案例：BF16 的 Warp 选择

> **Q: 假如我的数值是 bf16, 是不是 num_warps=4 反而是最高性能的做法?**

**深度推导（但正确）**：

1.  **硬件目标**：NVIDIA GPU 最高效的内存加载指令是 **128-bit (16 Bytes)**。
2.  **数据大小**：BF16 = 2 Bytes。
3.  **凑单计算**：要凑齐 16 Bytes，一个线程需要一次性搬运 $16 \div 2 = \mathbf{8}$ **个元素**。
4.  **推演**：
    *   **方案 B (Warps=4 / Threads=128)**：$1024 \div 128 = 8$ 个元素/线程。
        *   $8 \times 2B = 16B$。**完美触发** 128-bit 向量化加载 (`ld.global.v4` 等效指令)。

**结论**：Triton 通过调整 `num_warps`，让每个线程负责更多的数据，从而自动实现**向量化内存访问**。这是手动写 CUDA 很难做到的优化。


## Part 3.5: GPU 访问细节
请参考文档 `GPU-Access-Detail.md`

---

## Part 4: 实现细节与底层

### 4.1 `tl.arange` 的本质

> **Q: `offsets = block_start + tl.arange(...)` 是编译时就创建好了吗？**

**解析**：
*   **它是指令，不是数组。**
*   Triton 编译器看到 `tl.arange(0, 1024)`，不会在内存里存一个 `[0, 1, ..., 1023]` 的数组。
*   它会生成类似 `val = base + threadIdx.x` 的 PTX 指令。
*   **意义**：
    1.  **零内存开销**：不占 DRAM/SRAM。
    2.  **编译期优化**：编译器看到连续的索引，就能大胆地将多次 `load` 合并为一次 Vector Load。

### 4.2 C++ CUDA vs Triton 对比

*   **CUDA (C++)**:
    *   **微观管理**：你需要指挥**每一个士兵**（Thread）。
    *   **手动挡**：手动计算索引，手动写 `if` 边界检查，手动把 `float` 拼成 `float4` 来优化访存。
    *   **心智负担**：高。
*   **Triton (Python)**:
    *   **宏观指挥**：你指挥**一个班组**（Block）。
    *   **自动挡**：你只管写 `x + y`。编译器自动分析 `BLOCK_SIZE`，自动决定用多少 Warps，自动把数据切分成 128-bit 的块喂给硬件。
    *   **心智负担**：低。

---

## 📝 终极心法 (Takeaway)

1.  **世界观**：用 `BLOCK_SIZE` 的勺子（编译期定死的形状），去舀 `n_elements` 的大海（运行期动态的大小）。
2.  **避坑**：
    *   `BLOCK_SIZE` 必须是 `constexpr`（为了寄存器）。
    *   `n_elements` 千万别是 `constexpr`（为了缓存复用）。
3.  **性能直觉**：
    *   不要迷信“线程越多越好”。
    *   对于小数据类型（FP16/BF16），**更少的 Warps** 往往意味着**更好的向量化**效率。
    *   拿不准就用 `@triton.autotune`。


## 附录: 加法实现的Cuda和Triton对比

### C++
```cpp
// add.cu
#include <cuda_runtime.h>

// 1. Kernel 定义
// __global__ 表示这是一个在 GPU 上跑的函数
__global__ void add_kernel_cuda(float* x, float* y, float* out, int n) {
    // A. 计算全局索引 (Global Index)
    // "我是第几个线程？" = (我所在的 block ID * 一个 block 有多少线程) + 我在 block 里的 ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // B. 边界检查 (Masking 的手动版)
    // 必须写 if，否则最后一个 block 多出来的线程会越界访问内存导致崩溃
    if (tid < n) {
        // C. 加载 & 计算 & 存储 (Scalar 操作)
        // 这里的代码是标量操作：一次处理 1 个 float
        out[tid] = x[tid] + y[tid];
    }
}

// 2. Host 端调用 (C++ 这边很繁琐)
void launch_add(float* d_x, float* d_y, float* d_out, int n) {
    int threads_per_block = 256; // 对应 Triton 的 num_warps * 32
    // 计算 grid 大小 (向上取整)
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 启动 Kernel
    // <<<Grid, Block>>> 语法
    add_kernel_cuda<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, d_out, n);
}
```

### Triton
```Python
import triton
import triton.language as tl

# 1. Kernel 定义
@triton.jit
def add_kernel_triton(
    x_ptr, y_ptr, out_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr # 对应 CUDA 的 threads_per_block * 向量化系数
):
    # A. 计算 Block 的起始位置
    # 只需要知道我是第几个 Program (blockIdx)，不需要知道 threadIdx
    pid = tl.program_id(axis=0)
    
    # B. 生成偏移量向量 (Vectorized Offsets)
    # 这里生成的是一个 [0, 1, ..., BLOCK_SIZE-1] 的向量
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # C. 边界掩码 (Mask)
    # Triton 要求显式生成 mask 向量
    mask = offsets < n_elements

    # D. 加载 & 计算 & 存储 (Vector 操作)
    # 一次加载 BLOCK_SIZE 个数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

# 2. Host 端调用
# grid = (blocks_per_grid, )
# num_warps 控制了物理线程数，但你在 kernel 里不用管它
add_kernel_triton[grid](x, y, out, n, BLOCK_SIZE=1024)