import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")



properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
TOTAL_REGS_PER_SM = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties['max_shared_mem']
WARP_SIZE = properties["warpSize"]


# 为了更有效率, 
def matmul_kernel(a_ptr, 
                  b_ptr, 
                  c_ptr, 
                  BLOCKSIZE_M:tl.constexpr, 
                  BLOCKSIZE_N:tl.constexpr, 
                  BLOCKSIZE_K:tl.constexpr):
    rid = tl.program_id(axis=0)
    
    



def triton_matmul(a, b):
  M,Ka = a.shape
  Kb,N = b.shape
  assert Ka == Kb
  
  out = torch.zeros(size=(M,N), dtype=a.dtype, device=DEVICE)
  matmul_kernel(a, b, out)
  return out








def test(size:tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
  a = torch.randn(size, device=device)
  b = torch.randn(size, device=device).T.contiguous()

  ref = torch.matmul(a, b)
  tri = triton_matmul(a, b)
  torch.testing.assert_close(ref, tri, atol=atol, rtol=rtol)
  print("PASS")




  