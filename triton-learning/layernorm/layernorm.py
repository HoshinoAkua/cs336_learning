import triton
import torch
import triton.language as tl 

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


@triton.jit
def _layernorm_forward_triton(x_ptr, y_ptr, mean_ptr, rstd_ptr,
                              w_ptr, b_ptr, 
                              x_stride0, N, 
                              y_stride0, 
                              eps, BLOCK_SIZE:tl.constexpr,
                              ):
    row_idx = tl.program_id(axis=0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x_ptrs = x_ptr + row_idx * x_stride0 + offset 
    data = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(data) / N
    diff = tl.where(mask, data - mean, 0.0)
    var = tl.sum(diff * diff) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)
    data = (data - mean) * rstd
    w_ptrs = w_ptr + offset
    b_ptrs = b_ptr + offset
    weight = tl.load(w_ptrs, mask=mask, other=0.0)
    bias = tl.load(b_ptrs, mask=mask, other=0.0)
    
    data = data * weight + bias
    
    y_ptrs = y_ptr + row_idx * y_stride0 + offset
    tl.store(y_ptrs, data, mask=mask)



@triton.jit

def _layernorm_backward_dLdx(dy_ptr, dx_ptr, x_ptr, 
                             dy_stride0, dx_stride0, x_stride0,
                             w_ptr,rstd_ptr, mean_ptr, N,
                             lock_ptr, 
                             dw_inter_ptr, db_inter_ptr, 
                             BLOCK_SIZE:tl.constexpr,
                             GROUP_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    dy_ptrs = dy_ptr + row_idx * dy_stride0 + offset
    dx_ptrs = dx_ptr + row_idx * dx_stride0 + offset
    x_ptrs = x_ptr + row_idx * x_stride0 + offset

    mean_ptr = mean_ptr + row_idx
    rstd_ptr = rstd_ptr + row_idx
    w_ptrs = w_ptr + offset

    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr).to(tl.float32)
    rstd = tl.load(rstd_ptr).to(tl.float32)

    x_normlized = tl.where(mask, (x-mean) * rstd, 0.)

    dx = dy * weight * rstd - 1/N * (tl.sum(weight * rstd * dy)) - 1/N * rstd * x_normlized * (tl.sum(weight * x_normlized * dy))
    tl.store(dx_ptrs, dx, mask=mask)


    dw_inter = (x_normlized * dy).to(weight.dtype)
    db_inter = (dy).to(weight.dtype)

    # 自旋锁
    # 整个lock一共长为 [2*GROUP_SIZE]
    lock_id = row_idx % GROUP_SIZE
    lock_ptr = lock_ptr + lock_id
    count_ptr = lock_ptr + GROUP_SIZE

    # complete necessary computation before go to lock
    dw_inter_ptrs = dw_inter_ptr + lock_id * N + offset
    db_inter_ptrs = db_inter_ptr + lock_id * N + offset



    while (tl.atomic_cas(lock_ptr, 0, 1) == 1): 
        # 原理是lock_ptr的数值只有0，1两种状态
        # 当输入是0的时候，即当前解锁状态，那么程序继续进行，同时把数值转为1，保证上锁
        # 为了让其跳出while循环，设置 == 1这个判断条件
        pass
    count = tl.load(count_ptr)
    if count == 0:
        # 第一次写入，是覆盖
        tl.atomic_xchg(count_ptr, 1)

    else:
        # 如果是第二次
        dw_inter += tl.load(dw_inter_ptrs, mask=mask, other=0.0)
        db_inter += tl.load(db_inter_ptrs, mask=mask, other=0.0)
    

    tl.store(dw_inter_ptrs, dw_inter, mask=mask)
    tl.store(db_inter_ptrs, db_inter, mask=mask)

    tl.atomic_xchg(lock_ptr, 0)
    

@triton.jit
def _layernorm_backward_dLdw_legacy(dw_inter_ptr, db_inter_ptr, 
                                    dw_ptr, db_ptr, N, 
                                    dw_count_ptr, db_count_ptr, dw_lock_ptr,
                                    GROUP_SIZE, # 该参数仅用于计算bias grad
                                    BLOCK_SIZE:tl.constexpr):
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    dw_inter_ptrs = dw_inter_ptr + row_idx * N + offsets
    db_inter_ptrs = db_inter_ptr + row_idx * N + offsets
    dw_inter_data = tl.load(dw_inter_ptrs, mask=mask, other=0.0)
    
    dw_ptrs = dw_ptr + offsets
    db_ptrs = db_ptr + offsets

    # db_lock_state = tl.load(db_lock_ptr)
    db_count_state = tl.atomic_cas(db_count_ptr, 0, 1)
    if db_count_state == 0:
        db_data = tl.load(db_inter_ptrs, mask=mask, other=0.0) * GROUP_SIZE
        tl.store(db_ptrs, value=db_data, mask=mask)
    
    while (tl.atomic_cas(dw_lock_ptr, 0 , 1) == 1):
        pass
    
    dw_count_state = tl.load(dw_count_ptr)
    
    if dw_count_state != 0:
        dw_data = dw_inter_data + tl.load(dw_ptrs, mask=mask, other=0.0)
    else:
        dw_data = dw_inter_data
        tl.atomic_xchg(dw_count_ptr, 1)
    
    tl.store(dw_ptrs, dw_data, mask=mask)
    tl.atomic_xchg(dw_lock_ptr,0)

@triton.jit        
def _layernorm_backward_dwdb(dw_ptr, db_ptr, dw_inter_ptr, db_inter_ptr,
                             GROUP_SIZE, N,
                             BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr):
    
    pid = tl.program_id(axis=0)
    # 为了实现并行, 我们读取[:, pid * BLOCK_SIZE_N : (pid+1) * BLOCK_SIZE_N] 这一部分的数据, 然后直接用tl.sum求和
    # 但是实际上在求和的过程中, 我们不会直接完全载入所有行, 而是先载入一个 BLOCK_SIZE_M \times BLOCK_SIZE_N 的tile
    # 然后使用for loop循环求和
    
    '''
    读取矩阵的时候, 分为三层: 
    1. 读取坐标 (所在行, 所在列), 这是为了计算mask所用的
        比如在第 0 个循环, 第 0 个 pid, 矩阵块对应的坐标为: (tl.arange(0, BLOCK_SIZE_M)[:,None] , tl.arange(0, BLOCK_SIZE_N) -> [[(0,0),...,(0, BLOCK_SIZE_N-1)],...,[(BLOCK_SIZE_M-1,0),..., (BLOCK_SIZE_M-1, BLOCK_SIZE_N-1)]]
        而在第 i 个循环, 第 j 个 pid, 矩阵的坐标为: ((tl.arange(0, BLOCK_SIZE_M)[:, None] + i * BLOCK_SIZE_M) , (tl.arange(0, BLOCK_SIZE_N)[None, :] + j * BLOCK_SIZE_N)) -> 终点坐标 = 初始坐标 + 偏移
        在计算mask的时候, 我们需要让 坐标 < length
        
    2. 读取内存偏移: 内存偏移的本质是为了把一个二元元组降维成一维的情况
        还是以第 i 个循环, 第 j 个 pid, 矩阵的偏移举例子: 内存偏移的计算就是 ((tl.arange(0, BLOCK_SIZE_M)[:, None] + i * BLOCK_SIZE_M)) * stride_0 + (tl.arange(0, BLOCK_SIZE_N) + j * BLOCK_SIZE_N)) * stride_1, 即行坐标 * stride0 + 列坐标 * stride1
    
    3. 读取内存:
        这就是内存偏移 + 初始内存就行
    '''



    row_idx = tl.arange(0, BLOCK_SIZE_M)[:,None]
    col_idx = tl.arange(0,BLOCK_SIZE_N)[None, :] + pid * BLOCK_SIZE_N
    
    sum_acc_dw = tl.zeros(shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    sum_acc_db = tl.zeros(shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in tl.range(0, GROUP_SIZE, BLOCK_SIZE_M):
        
        mask = (row_idx < GROUP_SIZE) & (col_idx < N)
        offsets = row_idx * N + col_idx * 1
        sum_acc_dw += tl.load(dw_inter_ptr + offsets, mask=mask, other=0.0)
        sum_acc_db += tl.load(db_inter_ptr + offsets, mask=mask, other=0.0)
        row_idx += BLOCK_SIZE_M
    dw = tl.sum(sum_acc_dw, axis=0)
    db = tl.sum(sum_acc_db, axis=0)
    
    col_idx = tl.arange(0,BLOCK_SIZE_N) + pid * BLOCK_SIZE_N
    dw_ptrs = dw_ptr + col_idx * 1
    db_ptrs = db_ptr + col_idx * 1
    dw_mask = col_idx < N

    tl.store(dw_ptrs, dw, mask=dw_mask)
    tl.store(db_ptrs, db, mask=dw_mask)


    




class LayerNorm(torch.autograd.Function):
  
    @staticmethod
    def forward(ctx,
                x:torch.Tensor,
                weight: torch.Tensor,
                bias:torch.Tensor, 
                eps: float):
        oshape = x.shape
        M, N = x.reshape(-1, oshape[-1]).shape #保证x是row major的
        mean = torch.zeros(size = (M,), device=x.device, dtype=torch.float32)
        rstd = torch.zeros_like(mean)
        out = torch.zeros_like(x)
        FUSED_KERNEL_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = triton.next_power_of_2(N)

        if FUSED_KERNEL_SIZE < N:
          raise ValueError("The feature size is toooooo large for our triton kernel, use the partial layernorm kernel")
        grid = (M, 1, 1)
        x_stride0 = x.stride(0)
        out_stride0 = out.stride(0)

        _layernorm_forward_triton[grid](x, out, mean, rstd, weight, 
                                        bias, x_stride0, N,
                                        out_stride0, eps, BLOCK_SIZE)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        x, weight, mean, rstd = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        eps = ctx.eps

        GROUP_SIZE = 32
        M, N = x.shape
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        dw_inter = torch.zeros((GROUP_SIZE, N), dtype=weight.dtype, device=weight.device)
        db_inter = torch.zeros((GROUP_SIZE, N), dtype=weight.dtype, device=weight.device)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(weight)

        lock = torch.zeros((2 * GROUP_SIZE,),dtype=torch.int32, device=x.device)


        grad_x = torch.zeros_like(x)
        grid = (M,1,1)
        _layernorm_backward_dLdx[grid](grad_out, grad_x, x, 
                                       grad_out.stride(0), grad_x.stride(0), x.stride(0),
                                       weight, rstd, mean, N,
                                       lock, dw_inter, db_inter, 
                                       BLOCK_SIZE, GROUP_SIZE)

        grid_fn = lambda meta : (triton.cdiv(N, meta["BLOCK_SIZE_N"]),1,1)
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 128
        _layernorm_backward_dwdb[grid_fn](
            dw, db, dw_inter, db_inter,
            min(M, GROUP_SIZE), N, BLOCK_SIZE_M, BLOCK_SIZE_N
        )

        return grad_x, dw, db, None


layernorm = LayerNorm.apply

def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    weight = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    y_tri = layernorm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
    
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0) 
    print("Passed fwd")
    
    # 求导
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    for _ in [x, weight, bias]:
        _.grad = None
    
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    torch.testing.assert_close(dx_tri, dx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dw_tri, dw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(db_tri, db_ref, atol=1e-2, rtol=0)
    
    print("Passed bwd")

if __name__ == "__main__":
    test_layernorm_kernel(256, 1024, dtype=torch.float32)

