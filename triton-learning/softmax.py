import triton
import torch
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def torch_softmax(x):
  return torch.nn.functional.softmax(x,dim=-1)

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_rows, n_cols, BLOCK_SIZE:tl.constexpr, num_stage:tl.constexpr):
  pid = tl.program_id(axis=0)
  num_programs = tl.num_programs(axis=0)
  for row_idx in tl.range(pid, n_rows, num_programs, num_stages=num_stage):

    offset =  tl.arange(0, BLOCK_SIZE)
    x_offset = x_ptr + row_idx * n_cols + offset
    mask = offset < n_cols
    x_data = tl.load(x_offset, mask=mask, other=float("-inf"))
    # y_data = tl.load(offset, mask=mask)
    xmax = tl.max(x_data, axis=-1)
    x_data = x_data - xmax
    x_data = tl.exp(x_data)
    dominator = tl.sum(x_data, axis=-1)
    x_data = x_data / dominator
    y_offset = y_ptr + row_idx * n_cols + offset
    tl.store(y_offset, x_data, mask=mask)


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
TOTAL_REGS_PER_SM = properties["max_num_regs"] #每个SM有多少register
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
WARP_SIZE = properties["warpSize"]



def triton_softmax(x):
  out = torch.empty_like(x)
  n_rows, n_cols = out.shape
  # 每次处理一行, 利用multi-stage串行处理多行
  BLOCK_SIZE = triton.next_power_of_2(n_cols)
  num_warp = 4
  if BLOCK_SIZE >= 2048:
    num_warp = 8
  if BLOCK_SIZE >= 4096:
    num_warp = 16
  num_stage = 4
  kernel = softmax_kernel.warmup(x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warp, num_stage=num_stage, grid=(1,))
  kernel._init_handles()
  n_regs = kernel.n_regs # 记录每个thread需要使用多少register
  sram_per_program = kernel.metadata.shared # 由于只启动了一个program, 因此读取整个kernel的shared, 就能知道这个program消耗了多少sram
  
  # 计算整个program需要多少register:
  regs_per_program = num_warp * WARP_SIZE * n_regs

  # 分别计算, 每个SM在sram和register限制下, 分别能容纳多少program:
  regs_cond = TOTAL_REGS_PER_SM // regs_per_program
  sram_cond = TOTAL_SRAM_PER_SM // sram_per_program

  num_programs_per_sm = min(regs_cond, sram_cond)

  # 计算整个GPU一共可以启动多少program:
  total_programs = NUM_SM * num_programs_per_sm
  # 实际上最多用到n_rows这么多, 因为一个program处理一行. 因此我们可以再取一次min
  grid = (min(total_programs, n_rows),1,1)
  kernel[grid](x, out, n_rows, n_cols)
  return out

def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    """
    Here is where we test the warpper function and kernel that we wrote 
    above to ensure all our values are correct, using pytorch as the 
    correct answer to compare against

    we'll use an irregular number of rows & cols to verify that our padding mechanism works
    """
    # create input data
    torch.manual_seed(0)
    assert type(size) is tuple and len(size) == 2
    x = torch.randn(size[0], size[1], device=DEVICE)
    # run kernel & pytorch reference implementation
    z_ref = torch_softmax(x)
    z_tri = triton_softmax(x)
        # notice our implementation doesn't give a choice for what axis to softmax along.
        # this is a common theme of custom GPU kernels; because pytorch has to write code that
        #  is more general, it is slower than it could be
    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

if __name__ == "__main__":
    # always run unit-tests
    test_softmax_kernel(size=(1823, 781))

    # Only run benchmark if explicitly requested
    # import sys
    # if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
    #     benchmark.run(save_path='.', print_data=False)


