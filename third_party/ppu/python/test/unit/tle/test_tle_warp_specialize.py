"""PPU runtime tests of TLE warp specialization.

* ``tle.gpu.warp_specialize`` with named-barrier handover (``alloc_barriers`` +
  ``barrier_wait`` / ``barrier_arrive`` without a phase): lowered to the PPU hardware
  named barriers, one per barrier slot, with the partitions running concurrently.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_ppu

tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires the PPU backend")


@triton.jit
def _wait2(bars, i):
    if i == 0:
        tle.gpu.barrier_wait(bars[0])
    else:
        tle.gpu.barrier_wait(bars[1])


@triton.jit
def _arrive2(bars, i):
    if i == 0:
        tle.gpu.barrier_arrive(bars[0])
    else:
        tle.gpu.barrier_arrive(bars[1])


@triton.jit
def _ring_consumer(ring, full, empty, y, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        s = t % 2
        _wait2(full, s)
        v = tl.load(tle.gpu.local_ptr(ring.slot(s)))
        _arrive2(empty, s)
        tl.store(y + t * BLOCK + tl.arange(0, BLOCK), v * 3.0)


@triton.jit
def _ring_producer(ring, full, empty, x, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        s = t % 2
        if t >= 2:
            _wait2(empty, s)
        v = tl.load(x + t * BLOCK + tl.arange(0, BLOCK))
        tl.store(tle.gpu.local_ptr(ring.slot(s)), v)
        _arrive2(full, s)


@triton.jit
def _ring_kernel(x, y, n_tiles, BLOCK: tl.constexpr, THREADS: tl.constexpr):
    ring = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    full = tle.gpu.alloc_barriers(2, arrive_count=THREADS)
    empty = tle.gpu.alloc_barriers(2, arrive_count=THREADS)
    tle.gpu.warp_specialize(
        [(_ring_consumer, (ring, full, empty, y, n_tiles, BLOCK)),
         (_ring_producer, (ring, full, empty, x, n_tiles, BLOCK))],
        [4],
        [128],
    )


@pytest.mark.parametrize("n_tiles", [1, 2, 3, 64])
def test_warp_specialize_named_barrier_ring(n_tiles):
    block = 128
    x = torch.randn(n_tiles * block, device="cuda")
    y = torch.zeros_like(x)
    ck = _ring_kernel[(4, )](x, y, n_tiles, BLOCK=block, THREADS=256, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, 3 * x)
    assert ck.metadata.num_warps == 8
    assert "ttg.warp_specialize" in ck.asm["ttgir"]
    # one hardware named barrier per barrier slot, waited for and arrived at
    llir = ck.asm["llir"]
    assert "__ppukernel_barrier_sync_cnt_void" in llir
    assert "__ppukernel_barrier_arrive_void" in llir


@triton.jit
def _idle():
    pass


@triton.jit
def _async_load_partition(x, y, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        src = tl.make_block_ptr(x, (n_tiles * 64, BLOCK), (BLOCK, 1), (t * 64, 0), (64, BLOCK), (1, 0))
        dst = tl.make_block_ptr(y, (n_tiles * 64, BLOCK), (BLOCK, 1), (t * 64, 0), (64, BLOCK), (1, 0))
        tile = tle.load(src, boundary_check=(0, 1), is_async=True)
        tl.store(dst, (tile * 2).to(y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _async_load_kernel(x, y, n_tiles, BLOCK: tl.constexpr):
    tle.gpu.warp_specialize([(_idle, ()), (_async_load_partition, (x, y, n_tiles, BLOCK))], [4], [128])


def test_async_load_in_worker_partition():
    """The async copy addresses shared memory by the warp's index inside its warp group."""
    n_tiles, block = 4, 64
    x = torch.randn(n_tiles * 64, block, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros_like(x)
    _async_load_kernel[(1, )](x, y, n_tiles, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, 2 * x)
