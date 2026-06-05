import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    blockidx = tl.program_id(axis=0)
    blockstart = blockidx * BLOCK_SIZE
    offsets = tl.arange(0,BLOCK_SIZE) + blockstart
    mask = offsets < n
    sqrt2 = 0.7071067811865475
    x = tl.load(x_ptr + offsets, mask=mask)
    output = 0.5 * x * (1.0 + tl.math.erf(x * sqrt2))
    tl.store(out_ptr + offsets, output, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gelu_kernel: out = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)