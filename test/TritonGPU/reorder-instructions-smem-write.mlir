// REQUIRES: flagtree-backend-ppu
// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions | FileCheck %s

// A load of mutable shared memory is not sunk into a loop past a write to
// that memory: the loop would read the rewritten data.
// CHECK-LABEL: no_sink_load_past_write
//       CHECK: ttg.local_load
//       CHECK: scf.for
//       CHECK:   ttg.local_store
//       CHECK:   ttg.local_load
//   CHECK-NOT:   ttg.local_load
//       CHECK:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_load_past_write(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      ttg.local_store %B, %S : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %BD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Without a write in between, the load is still sunk into the loop.
// CHECK-LABEL: sink_load_into_loop
//   CHECK-NOT: ttg.local_load
//       CHECK: scf.for
//       CHECK:   ttg.local_load
//       CHECK:   ttg.local_load
//       CHECK:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_load_into_loop(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
      %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
