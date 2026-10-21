// Each tile op expands into the machine operations that compute it. A minitpu.matmul stages the zero tile and the
// resident activations, then per column tile streams the weight chunks double buffered, each k-tile loading a weight
// tile, pushing the eight token groups and popping eight results into the accumulators, and stores and drains the
// column tile. A minitpu.add streams its operands chunk by chunk, double buffered, and a minitpu.layernorm gathers
// one token at a time. Every expansion then allocates, lowers, packs and schedules.

// RUN: minitpu-opt %s -split-input-file -pass-pipeline='builtin.module(minitpu.kernel(minitpu-tile-to-vector))' | FileCheck %s
// RUN: minitpu-opt %s -split-input-file -pass-pipeline='builtin.module(minitpu.kernel(minitpu-tile-to-vector,minitpu-allocate-vmem,minitpu-allocate-registers),convert-minitpu-to-isa,minitpuisa.program(minitpuisa-pack,minitpuisa-schedule))' | FileCheck %s --check-prefix=PROGRAM

// CHECK-LABEL: minitpu.kernel @gemm
// CHECK:       minitpu.alloc : !minitpu.vmem<16>
// CHECK:       minitpu.alloc : !minitpu.vmem<512>
// CHECK:       minitpu.vmemld {{.*}} rows 16, slot 0
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 1
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK-COUNT-4: minitpu.vld
// CHECK:           minitpu.vmatload
// CHECK-COUNT-8:   minitpu.vmatpush
// CHECK-COUNT-8:   minitpu.vmatpop
// CHECK:           minitpu.vadd
// CHECK:         minitpu.vmemst {{.*}} rows 32
// CHECK-NOT:   minitpu.matmul
// CHECK-NOT:   minitpu.vgelu
// PROGRAM-LABEL: minitpuisa.program @gemm
// PROGRAM:       minitpuisa.vmatload
// PROGRAM:       minitpuisa.halt
minitpu.kernel @gemm {
  %activations = minitpu.arg 0
  %weights = minitpu.arg 1
  %drain = minitpu.arg 2
  minitpu.matmul %activations, %weights -> %drain tokens 32 depth 256 columns 32
}

// -----

// An activating product passes the resident activations through the GELU in place once, ahead of every column tile.

// CHECK-LABEL: minitpu.kernel @gemm_gelu
// CHECK:       scf.for
// CHECK:         minitpu.vld
// CHECK-NEXT:    minitpu.vgelu
// CHECK-NEXT:    minitpu.vst
// CHECK:       scf.for
// CHECK-NOT:   minitpu.vgelu
minitpu.kernel @gemm_gelu {
  %activations = minitpu.arg 0
  %weights = minitpu.arg 1
  %drain = minitpu.arg 2
  minitpu.matmul %activations, %weights -> %drain tokens 32 depth 256 columns 32 activate
}

// -----

// A product with a bias stages the bias row in place of the zero tile, and each column tile's accumulators load that
// tile's bias register, so the bias needs no launch of its own.

// CHECK-LABEL: minitpu.kernel @gemm_bias
// CHECK:       minitpu.alloc : !minitpu.vmem<8>
// CHECK:       minitpu.vmemld {{.*}} rows 8, slot 0
// CHECK:       scf.for %[[COLUMN:arg[0-9]+]] =
// CHECK:         %[[SCALED:.*]] = arith.muli %[[COLUMN]], %{{.*}}
// CHECK:         %[[ROW:[0-9]+]] = arith.addi %[[SCALED]], %{{.*}}
// CHECK-COUNT-8: minitpu.vld %{{.*}}[%[[ROW]]] : !minitpu.vmem<8>
// CHECK-NOT:   minitpu.vmem<16>
// PROGRAM-LABEL: minitpuisa.program @gemm_bias
// PROGRAM:       minitpuisa.smov_arg {{.*}}, arg 3
// PROGRAM:       minitpuisa.halt
minitpu.kernel @gemm_bias {
  %activations = minitpu.arg 0
  %weights = minitpu.arg 1
  %drain = minitpu.arg 2
  %bias = minitpu.arg 3
  minitpu.matmul %activations, %weights -> %drain bias %bias tokens 32 depth 256 columns 32
}

// -----

// A contraction ending part way through a chunk loads only the rows of its last chunk and walks only its k-tiles.

// CHECK-LABEL: minitpu.kernel @gemm_short_chunk
// CHECK:       minitpu.alloc : !minitpu.vmem<288>
// CHECK:       scf.for
// CHECK:         minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:         minitpu.vmemld {{.*}} rows 16, slot 1
// CHECK:         %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %[[EIGHT]]
// CHECK:         %[[ONE:.*]] = arith.constant 1 : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %[[ONE]]
// PROGRAM-LABEL: minitpuisa.program @gemm_short_chunk
// PROGRAM:       minitpuisa.halt
minitpu.kernel @gemm_short_chunk {
  %activations = minitpu.arg 0
  %weights = minitpu.arg 1
  %drain = minitpu.arg 2
  minitpu.matmul %activations, %weights -> %drain tokens 32 depth 144 columns 32
}

// -----

// A contraction deeper than one bank holds splits into segments of whole chunks, each staging its activations and
// walking every column tile. A later segment reads its pointers afresh and starts each column tile from the partial
// product the segment before drained, read back on slot 1 beside the first weight chunk.

// CHECK-LABEL: minitpu.kernel @gemm_split
// CHECK:       minitpu.alloc : !minitpu.vmem<4096>
// CHECK:       scf.for
// CHECK-NOT:     rows 32, slot 1
// CHECK:         minitpu.vmemst {{.*}} rows 32
// CHECK:       minitpu.arg 0
// CHECK:       minitpu.arg 1
// CHECK:       minitpu.arg 2
// CHECK:       scf.for
// CHECK:         minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:         minitpu.vmemld {{.*}} rows 32, slot 1
// CHECK-NEXT:    minitpu.flush_slot 2
// CHECK-COUNT-8: minitpu.vld %{{.*}}[%{{.*}}] : !minitpu.vmem<32>
// CHECK:         minitpu.vmemst {{.*}} rows 32
// CHECK-NOT:   scf.for
// PROGRAM-LABEL: minitpuisa.program @gemm_split
// PROGRAM:       minitpuisa.halt
minitpu.kernel @gemm_split {
  %activations = minitpu.arg 0
  %weights = minitpu.arg 1
  %drain = minitpu.arg 2
  minitpu.matmul %activations, %weights -> %drain tokens 32 depth 4096 columns 16
}

// -----

// A streamed sum loads both operands' first chunks on the two slots, peels its first chunk, runs the middle chunks two
// per iteration, and peels its last, each chunk prefetching the next before its additions and draining on slot 0.

// CHECK-LABEL: minitpu.kernel @add
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 1
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:       minitpu.flush_slot 3
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 1
// CHECK-COUNT-32: minitpu.vadd
// CHECK:       minitpu.flush_slot 1
// CHECK:       minitpu.vmemst {{.*}} rows 128, slot 0
// CHECK:       scf.for
// CHECK-COUNT-2: minitpu.vmemst {{.*}} rows 128, slot 0
// CHECK:       scf.yield
// CHECK:       minitpu.vmemst {{.*}} rows 128, slot 0
// CHECK:       minitpu.flush_slot 1
// CHECK-NOT:   minitpu.add
// PROGRAM-LABEL: minitpuisa.program @add
// PROGRAM:       minitpuisa.lbegin 3
// PROGRAM:       minitpuisa.halt
minitpu.kernel @add {
  %left = minitpu.arg 0
  %right = minitpu.arg 1
  %output = minitpu.arg 2
  minitpu.add %left, %right -> %output tokens 32 features 512
}

// -----

// A broadcast sum stages the whole bias row first and reads it once per feature tile, draining on slot 1. Two chunks
// leave no middle, so no loop is built.

// CHECK-LABEL: minitpu.kernel @add_bias
// CHECK:       minitpu.alloc : !minitpu.vmem<32>
// CHECK:       minitpu.vmemld {{.*}} rows 32, slot 0
// CHECK:       minitpu.flush_slot 1
// CHECK:       minitpu.flush_slot 1
// CHECK-COUNT-36: minitpu.vld
// CHECK:       minitpu.vmemst {{.*}} rows 128, slot 1
// CHECK:       minitpu.flush_slot 3
// CHECK-NOT:   scf.for
// CHECK:       minitpu.vmemst {{.*}} rows 128, slot 1
// CHECK:       minitpu.flush_slot 2
// PROGRAM-LABEL: minitpuisa.program @add_bias
// PROGRAM-NOT:   minitpuisa.lbegin
// PROGRAM:       minitpuisa.halt
minitpu.kernel @add_bias {
  %left = minitpu.arg 0
  %bias = minitpu.arg 1
  %output = minitpu.arg 2
  minitpu.add %left, %bias -> %output tokens 32 features 128 broadcast
}

// -----

// A LayerNorm stages its constants and keeps the reciprocal width and epsilon in registers, then gathers each token
// with a stride of one block of tokens, sums its registers as a tree before each vredsum, scales and shifts it, and
// drains it on slot 1. The first and last tokens are peeled around a loop of token pairs.

// CHECK-LABEL: minitpu.kernel @layernorm
// CHECK:       minitpu.alloc : !minitpu.vmem<24>
// CHECK:       minitpu.vmemld {{.*}} rows 24, slot 0
// CHECK:       minitpu.flush_slot 1
// CHECK:       minitpu.vld
// CHECK:       minitpu.vld
// CHECK:       arith.constant 1024 : i32
// CHECK:       minitpu.vmemld {{.*}} rows 8, slot 0
// CHECK:       minitpu.vadd
// CHECK-NEXT:  minitpu.vredsum
// CHECK:       minitpu.vrsqrt
// CHECK:       minitpu.vmemst {{.*}} rows 8, slot 1
// CHECK:       scf.for
// CHECK-COUNT-2: minitpu.vmemst {{.*}} rows 8, slot 1
// CHECK:       scf.yield
// CHECK:       minitpu.vmemst {{.*}} rows 8, slot 1
// CHECK:       minitpu.flush_slot 2
// CHECK-NOT:   minitpu.layernorm
// PROGRAM-LABEL: minitpuisa.program @layernorm
// PROGRAM:       minitpuisa.lbegin 15
// PROGRAM:       minitpuisa.vrsqrt
// PROGRAM:       minitpuisa.halt
minitpu.kernel @layernorm {
  %rows = minitpu.arg 0
  %constants = minitpu.arg 1
  %output = minitpu.arg 2
  minitpu.layernorm %rows, %constants -> %output tokens 32 features 128 epsilon 1.0e-05
}

// -----

// A copy alternates two chunk buffers, loading each chunk on slot 0 while the previous drains on slot 1, its steady
// chunks in a loop over pairs and its partial last chunk loaded at its own size.
// CHECK-LABEL: minitpu.kernel @copy
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:       minitpu.flush_slot 1
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:       minitpu.vmemst {{.*}} rows 128, slot 1
// CHECK:       scf.for
// CHECK-COUNT-2: minitpu.flush_slot 3
// CHECK:       scf.yield
// CHECK:       minitpu.vmemld {{.*}} rows 64, slot 0
// CHECK:       minitpu.vmemst {{.*}} rows 128, slot 1
// CHECK:       minitpu.flush_slot 3
// CHECK:       minitpu.vmemst {{.*}} rows 64, slot 1
// CHECK:       minitpu.flush_slot 2
// CHECK-NOT:   minitpu.copy
// PROGRAM-LABEL: minitpuisa.program @copy
// PROGRAM:       minitpuisa.lbegin 1
// PROGRAM:       minitpuisa.halt
minitpu.kernel @copy {
  %source = minitpu.arg 0
  %destination = minitpu.arg 2
  minitpu.copy %source -> %destination rows 576 stride 64
}

// -----

// A decode attention walks the heads, staging each head's queries and transposing them, then per cached key token
// block stages the keys and values and attends both halves: the transposed scores from the keys pushed against the
// transposed queries, the masked maximum and sum folded through VMEM, and the transposed values pushed against the
// probabilities held stationary. The output is normalised, transposed back and drained over the queries.
// CHECK-LABEL: minitpu.kernel @decode_attention
// CHECK:       scf.for
// CHECK:       minitpu.vtxin
// CHECK:       scf.for
// CHECK-COUNT-2: minitpu.vmemld {{.*}} rows 32
// CHECK:       minitpu.vmatload
// CHECK:       minitpu.vmatpush
// CHECK:       minitpu.vmatpop
// CHECK:       minitpu.vst {{.*}} mask 1
// CHECK:       minitpu.vexp
// CHECK:       minitpu.vrecip
// CHECK:       minitpu.vmemst {{.*}} rows 16
// CHECK-NOT:   minitpu.attention
// PROGRAM-LABEL: minitpuisa.program @decode_attention
// PROGRAM:       minitpuisa.lbegin 2
// PROGRAM:       minitpuisa.lbegin 1
// PROGRAM:       minitpuisa.vtxout
// PROGRAM:       minitpuisa.halt
minitpu.kernel @decode_attention {
  %constants = minitpu.arg 0
  %cache = minitpu.arg 1
  %queries = minitpu.arg 2
  minitpu.attention %constants, %queries cache %cache heads 2 head 16 keys 32
}
