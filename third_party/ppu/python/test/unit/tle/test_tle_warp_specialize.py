"""PPU runtime tests of TLE warp specialization and shared-memory views.

* ``tle.gpu.warp_specialize`` with named-barrier handover (``alloc_barriers`` +
  ``barrier_wait`` / ``barrier_arrive`` without a phase): lowered to the PPU hardware
  named barriers, one per barrier slot, with the partitions running concurrently.
* ``buffered_tensor.subslice`` as a dot operand of a swizzled ring slot.
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


@triton.constexpr_function
def _ring_layout(vec, max_phase):
    return tle.gpu.swizzled_shared_layout(vec, 1, max_phase, [2, 1, 0], [1] * 3, [1] * 3, [1] * 3, [2, 1, 0])


@triton.jit
def _subslice_ring_kernel(a, b, out, n_blk, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out = a @ sum_k b[k]: the tiles of b go through a three-slot swizzled ring; the dot reads the two column halves
    of a slot as subslices."""
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    ring = tle.gpu.alloc([3, BK, BN], dtype=tl.bfloat16, layout=_ring_layout(8, 8), scope=tle.gpu.smem)
    a_t = tl.load(a + offs_m[:, None] * BK + offs_k[None, :])
    b_ptr = b + offs_k[:, None] * BN + offs_n[None, :]
    acc0 = tl.zeros([BM, BN // 2], dtype=tl.float32)
    acc1 = tl.zeros([BM, BN // 2], dtype=tl.float32)
    for g in range(n_blk):
        slot = ring.slot(g % 3)
        tle.gpu.copy(b_ptr + g * BK * BN, slot, [BK, BN])
        acc0 = tl.dot(a_t, tl.load(tle.gpu.local_ptr(slot.subslice(0, BN // 2, 1))), acc0)
        acc1 = tl.dot(a_t, tl.load(tle.gpu.local_ptr(slot.subslice(BN // 2, BN // 2, 1))), acc1)
    o_ptr = out + offs_m[:, None] * BN + tl.arange(0, BN // 2)[None, :]
    tl.store(o_ptr, acc0)
    tl.store(o_ptr + BN // 2, acc1)


@pytest.mark.parametrize("n_blk", [1, 2, 7])
def test_subslice_dot_operands_of_ring_slot(n_blk):
    bm, bk, bn = 64, 64, 128
    a = torch.randn(bm, bk, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n_blk, bk, bn, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    _subslice_ring_kernel[(1, )](a, b, out, n_blk, BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    ref = sum(a.float() @ b[k].float() for k in range(n_blk))
    torch.testing.assert_close(out, ref, atol=1e-2 * n_blk, rtol=1e-2)
