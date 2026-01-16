from triton import language as tl
import triton
import torch

DEVICE = torch.device("cuda",0)

def add_kernel(**kwargs):
    pass

def triton_add(x:torch.Tensor,y:torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid_fn = lambda meta: 
    add_kernel(x, y,
               out,
               n_elements,
               BLOCK_SIZE = 1024)
