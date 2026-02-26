import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")



properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
TOTAL_REGS_PER_SM = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties['max_shared_mem']
WARP_SIZE = properties["warpSize"]

autotune_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32,}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32,}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32,}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32,}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32,}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32,}, num_stages=5, num_warps=2)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_nn_kernel(a_ptr, 
                    b_ptr, 
                    c_ptr, 
                    a_stride0,
                    a_stride1,
                    b_stride0,
                    b_stride1,
                    c_stride0,
                    c_stride1,
                    M,
                    N,
                    K,
                    BLOCK_M:tl.constexpr, 
                    BLOCK_N:tl.constexpr, 
                    BLOCK_K:tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    a_row_idx = (tl.arange(0,BLOCK_M) + rid * BLOCK_M) % M
    b_col_idx = (tl.arange(0,BLOCK_N) + cid * BLOCK_N) % N
    
    k_idx = tl.arange(0,BLOCK_K)
    tl.assume(M % BLOCK_M == 0)
    tl.assume(N % BLOCK_N == 0)
    a_ptrs = a_ptr + a_row_idx[:,None] * a_stride0 + k_idx[None,:] * a_stride1
    b_ptrs = b_ptr + b_col_idx[None,:] * b_stride1 + k_idx[:,None] * b_stride0
    c_data = tl.zeros(shape=(BLOCK_M, BLOCK_N), dtype=tl.float32)
    c_ptrs = c_ptr + a_row_idx[:,None] * c_stride0 + b_col_idx[None,:] * c_stride1
    for k in tl.range(0,tl.cdiv(K, BLOCK_K)):
      k_remains = K - k * BLOCK_K
      mask = k_remains > k_idx

      a_data = tl.load(a_ptrs, mask=mask[None,:],other=0.0)
      b_data = tl.load(b_ptrs, mask=mask[:,None],other=0.0)

      c_data = tl.dot(a_data, b_data, c_data)

      a_ptrs += BLOCK_K * a_stride1
      b_ptrs += BLOCK_K * b_stride0
    
    c_mask = (a_row_idx[:,None] < M ) & (b_col_idx[None,:] < N)
    tl.store(c_ptrs, mask = c_mask, value=c_data)



def triton_matmul(a, b):
  M,Ka = a.shape
  Kb,N = b.shape
  assert Ka == Kb
  
  out = torch.zeros(size=(M,N), dtype=a.dtype, device=DEVICE)

  grid_fn = lambda meta:(triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), 1)
  matmul_nn_kernel[grid_fn](a, b, out,
                            a.stride(0),
                            a.stride(1),
                            b.stride(0),
                            b.stride(1),
                            out.stride(0),
                            out.stride(1),
                            M,N,Ka)
  return out








def test(size:tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
  a = torch.randn(size, device=device)
  b = torch.randn(size, device=device).T.contiguous()

  ref = torch.matmul(a, b)
  tri = triton_matmul(a, b)
  torch.testing.assert_close(ref, tri, atol=atol, rtol=rtol)
  print("PASS")

if __name__ == "__main__":
  test(size=(512,1024))


  