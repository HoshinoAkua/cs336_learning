from triton import language as tl
import triton
import torch

DEVICE = torch.device("cuda",0)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offset < n_elements

    x_load = tl.load(x_ptr+offset, mask=mask, other=None)
    y_load = tl.load(y_ptr+offset, mask=mask, other=None)

    out = x_load + y_load
    tl.store(out_ptr+offset, out, mask = mask)

def triton_add(x:torch.Tensor,y:torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid_fn = lambda meta: (triton.cdiv(meta["n_elements"],meta["BLOCK_SIZE"]),)
    add_kernel[grid_fn](x, y,
               out,
               n_elements,
               BLOCK_SIZE = 1024)
    return out


def test_add_kernel(size, atol=1e-3,rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)

    z_tri = triton_add(x,y)
    z_ref = torch.add(x,y)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("triton 和 torch 的精度误差通过")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different values of x_names to benchmark
        x_log = True, # makes x-axis logarithmic
        line_arg='provider', # title of the legend 
        line_vals=['triton', 'torch'], # designators of the different entries in the legend
        line_names=['Triton', 'Torch'], # names to visibly go in the legend
        styles=[('blue', '-'), ('green', '-')], # triton will be blue; pytorch will be green
        ylabel='GB/s', # label name for y-axis
        plot_name='vector-add-performance', # also used as file name for saving plot
        args={}, # we'll see how this is used in a later tutorial; need it even if it's empty
    )
)
def benchmark(size, provider):
    # creating our input data
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    # each benchmark runs multiple times and quantiles tells matplotlib what confidence intervals to plot
    quantiles = [0.5, 0.05, 0.95]
    # defining which function this benchmark instance runs
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_add(x, y), quantiles=quantiles)
    # turning the raw millisecond measurement into meaninful units
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        # 3 = number of memory operations (2 reads + 1 write)
        # x.numel() = number of elements
        # x.element_size() = bytes per element (4 for float32, 2 for float16)
        # 1e-9 converts bytes to GB
        # 1e-3 converts milliseconds to seconds
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # always run unit-tests
    test_add_kernel(size=98432)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)