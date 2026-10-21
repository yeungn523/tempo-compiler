// Each tile op expands into the machine operations that compute it. A tempo.matmul stages the zero tile and the
// resident activations, then per column tile streams the weight chunks double buffered, each k-tile loading a weight
// tile, pushing the eight token groups and popping eight results into the accumulators, and stores and drains the
// column tile. A tempo.add streams its operands chunk by chunk, double buffered, and a tempo.layernorm gathers
// one token at a time. Every expansion then allocates, lowers, packs and schedules.

// RUN: tempo-opt %s -split-input-file -pass-pipeline='builtin.module(tempo.kernel(tempo-tile-to-vector))' | FileCheck %s
// RUN: tempo-opt %s -split-input-file -pass-pipeline='builtin.module(tempo.kernel(tempo-tile-to-vector,tempo-allocate-vmem,tempo-allocate-registers),convert-tempo-to-isa,tempoisa.program(tempoisa-pack,tempoisa-schedule))' | FileCheck %s --check-prefix=PROGRAM

// CHECK-LABEL: tempo.kernel @gemm
// CHECK:       tempo.alloc : !tempo.vmem<16>
// CHECK:       tempo.alloc : !tempo.vmem<512>
// CHECK:       tempo.vmemld {{.*}} rows 16, slot 0
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 0
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 1
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK-COUNT-4: tempo.vld
// CHECK:           tempo.vmatload
// CHECK-COUNT-8:   tempo.vmatpush
// CHECK-COUNT-8:   tempo.vmatpop
// CHECK:           tempo.vadd
// CHECK:         tempo.vmemst {{.*}} rows 32
// CHECK-NOT:   tempo.matmul
// CHECK-NOT:   tempo.vgelu
// PROGRAM-LABEL: tempoisa.program @gemm
// PROGRAM:       tempoisa.vmatload
// PROGRAM:       tempoisa.halt
tempo.kernel @gemm {
  %activations = tempo.arg 0
  %weights = tempo.arg 1
  %drain = tempo.arg 2
  tempo.matmul %activations, %weights -> %drain tokens 32 depth 256 columns 32
}

// -----

// An activating product passes the resident activations through the GELU in place once, ahead of every column tile.

// CHECK-LABEL: tempo.kernel @gemm_gelu
// CHECK:       scf.for
// CHECK:         tempo.vld
// CHECK-NEXT:    tempo.vgelu
// CHECK-NEXT:    tempo.vst
// CHECK:       scf.for
// CHECK-NOT:   tempo.vgelu
tempo.kernel @gemm_gelu {
  %activations = tempo.arg 0
  %weights = tempo.arg 1
  %drain = tempo.arg 2
  tempo.matmul %activations, %weights -> %drain tokens 32 depth 256 columns 32 activate
}

// -----

// A product with a bias stages the bias row in place of the zero tile, and each column tile's accumulators load that
// tile's bias register, so the bias needs no launch of its own.

// CHECK-LABEL: tempo.kernel @gemm_bias
// CHECK:       tempo.alloc : !tempo.vmem<8>
// CHECK:       tempo.vmemld {{.*}} rows 8, slot 0
// CHECK:       scf.for %[[COLUMN:arg[0-9]+]] =
// CHECK:         %[[SCALED:.*]] = arith.muli %[[COLUMN]], %{{.*}}
// CHECK:         %[[ROW:[0-9]+]] = arith.addi %[[SCALED]], %{{.*}}
// CHECK-COUNT-8: tempo.vld %{{.*}}[%[[ROW]]] : !tempo.vmem<8>
// CHECK-NOT:   tempo.vmem<16>
// PROGRAM-LABEL: tempoisa.program @gemm_bias
// PROGRAM:       tempoisa.smov_arg {{.*}}, arg 3
// PROGRAM:       tempoisa.halt
tempo.kernel @gemm_bias {
  %activations = tempo.arg 0
  %weights = tempo.arg 1
  %drain = tempo.arg 2
  %bias = tempo.arg 3
  tempo.matmul %activations, %weights -> %drain bias %bias tokens 32 depth 256 columns 32
}

// -----

// A contraction ending part way through a chunk loads only the rows of its last chunk and walks only its k-tiles.

// CHECK-LABEL: tempo.kernel @gemm_short_chunk
// CHECK:       tempo.alloc : !tempo.vmem<288>
// CHECK:       scf.for
// CHECK:         tempo.vmemld {{.*}} rows 128, slot 0
// CHECK:         tempo.vmemld {{.*}} rows 16, slot 1
// CHECK:         %[[EIGHT:.*]] = arith.constant 8 : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %[[EIGHT]]
// CHECK:         %[[ONE:.*]] = arith.constant 1 : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %[[ONE]]
// PROGRAM-LABEL: tempoisa.program @gemm_short_chunk
// PROGRAM:       tempoisa.halt
tempo.kernel @gemm_short_chunk {
  %activations = tempo.arg 0
  %weights = tempo.arg 1
  %drain = tempo.arg 2
  tempo.matmul %activations, %weights -> %drain tokens 32 depth 144 columns 32
}

// -----

// A contraction deeper than one bank holds splits into segments of whole chunks, each staging its activations and
// walking every column tile. A later segment reads its pointers afresh and starts each column tile from the partial
// product the segment before drained, read back on slot 1 beside the first weight chunk.

// CHECK-LABEL: tempo.kernel @gemm_split
// CHECK:       tempo.alloc : !tempo.vmem<4096>
// CHECK:       scf.for
// CHECK-NOT:     rows 32, slot 1
// CHECK:         tempo.vmemst {{.*}} rows 32
// CHECK:       tempo.arg 0
// CHECK:       tempo.arg 1
// CHECK:       tempo.arg 2
// CHECK:       scf.for
// CHECK:         tempo.vmemld {{.*}} rows 128, slot 0
// CHECK:         tempo.vmemld {{.*}} rows 32, slot 1
// CHECK-NEXT:    tempo.flush_slot 2
// CHECK-COUNT-8: tempo.vld %{{.*}}[%{{.*}}] : !tempo.vmem<32>
// CHECK:         tempo.vmemst {{.*}} rows 32
// CHECK-NOT:   scf.for
// PROGRAM-LABEL: tempoisa.program @gemm_split
// PROGRAM:       tempoisa.halt
tempo.kernel @gemm_split {
  %activations = tempo.arg 0
  %weights = tempo.arg 1
  %drain = tempo.arg 2
  tempo.matmul %activations, %weights -> %drain tokens 32 depth 4096 columns 16
}

// -----

// A streamed sum loads both operands' first chunks on the two slots, peels its first chunk, runs the middle chunks two
// per iteration, and peels its last, each chunk prefetching the next before its additions and draining on slot 0.

// CHECK-LABEL: tempo.kernel @add
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 1
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 0
// CHECK:       tempo.flush_slot 3
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 0
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 1
// CHECK-COUNT-32: tempo.vadd
// CHECK:       tempo.flush_slot 1
// CHECK:       tempo.vmemst {{.*}} rows 128, slot 0
// CHECK:       scf.for
// CHECK-COUNT-2: tempo.vmemst {{.*}} rows 128, slot 0
// CHECK:       scf.yield
// CHECK:       tempo.vmemst {{.*}} rows 128, slot 0
// CHECK:       tempo.flush_slot 1
// CHECK-NOT:   tempo.add
// PROGRAM-LABEL: tempoisa.program @add
// PROGRAM:       tempoisa.lbegin 3
// PROGRAM:       tempoisa.halt
tempo.kernel @add {
  %left = tempo.arg 0
  %right = tempo.arg 1
  %output = tempo.arg 2
  tempo.add %left, %right -> %output tokens 32 features 512
}

// -----

// A broadcast sum stages the whole bias row first and reads it once per feature tile, draining on slot 1. Two chunks
// leave no middle, so no loop is built.

// CHECK-LABEL: tempo.kernel @add_bias
// CHECK:       tempo.alloc : !tempo.vmem<32>
// CHECK:       tempo.vmemld {{.*}} rows 32, slot 0
// CHECK:       tempo.flush_slot 1
// CHECK:       tempo.flush_slot 1
// CHECK-COUNT-36: tempo.vld
// CHECK:       tempo.vmemst {{.*}} rows 128, slot 1
// CHECK:       tempo.flush_slot 3
// CHECK-NOT:   scf.for
// CHECK:       tempo.vmemst {{.*}} rows 128, slot 1
// CHECK:       tempo.flush_slot 2
// PROGRAM-LABEL: tempoisa.program @add_bias
// PROGRAM-NOT:   tempoisa.lbegin
// PROGRAM:       tempoisa.halt
tempo.kernel @add_bias {
  %left = tempo.arg 0
  %bias = tempo.arg 1
  %output = tempo.arg 2
  tempo.add %left, %bias -> %output tokens 32 features 128 broadcast
}

// -----

// A LayerNorm stages its constants and keeps the reciprocal width and epsilon in registers, then gathers each token
// with a stride of one block of tokens, sums its registers as a tree before each vredsum, scales and shifts it, and
// drains it on slot 1. The first and last tokens are peeled around a loop of token pairs.

// CHECK-LABEL: tempo.kernel @layernorm
// CHECK:       tempo.alloc : !tempo.vmem<24>
// CHECK:       tempo.vmemld {{.*}} rows 24, slot 0
// CHECK:       tempo.flush_slot 1
// CHECK:       tempo.vld
// CHECK:       tempo.vld
// CHECK:       arith.constant 1024 : i32
// CHECK:       tempo.vmemld {{.*}} rows 8, slot 0
// CHECK:       tempo.vadd
// CHECK-NEXT:  tempo.vredsum
// CHECK:       tempo.vrsqrt
// CHECK:       tempo.vmemst {{.*}} rows 8, slot 1
// CHECK:       scf.for
// CHECK-COUNT-2: tempo.vmemst {{.*}} rows 8, slot 1
// CHECK:       scf.yield
// CHECK:       tempo.vmemst {{.*}} rows 8, slot 1
// CHECK:       tempo.flush_slot 2
// CHECK-NOT:   tempo.layernorm
// PROGRAM-LABEL: tempoisa.program @layernorm
// PROGRAM:       tempoisa.lbegin 15
// PROGRAM:       tempoisa.vrsqrt
// PROGRAM:       tempoisa.halt
tempo.kernel @layernorm {
  %rows = tempo.arg 0
  %constants = tempo.arg 1
  %output = tempo.arg 2
  tempo.layernorm %rows, %constants -> %output tokens 32 features 128 epsilon 1.0e-05
}

// -----

// A copy alternates two chunk buffers, loading each chunk on slot 0 while the previous drains on slot 1, its steady
// chunks in a loop over pairs and its partial last chunk loaded at its own size.
// CHECK-LABEL: tempo.kernel @copy
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 0
// CHECK:       tempo.flush_slot 1
// CHECK:       tempo.vmemld {{.*}} rows 128, slot 0
// CHECK:       tempo.vmemst {{.*}} rows 128, slot 1
// CHECK:       scf.for
// CHECK-COUNT-2: tempo.flush_slot 3
// CHECK:       scf.yield
// CHECK:       tempo.vmemld {{.*}} rows 64, slot 0
// CHECK:       tempo.vmemst {{.*}} rows 128, slot 1
// CHECK:       tempo.flush_slot 3
// CHECK:       tempo.vmemst {{.*}} rows 64, slot 1
// CHECK:       tempo.flush_slot 2
// CHECK-NOT:   tempo.copy
// PROGRAM-LABEL: tempoisa.program @copy
// PROGRAM:       tempoisa.lbegin 1
// PROGRAM:       tempoisa.halt
tempo.kernel @copy {
  %source = tempo.arg 0
  %destination = tempo.arg 2
  tempo.copy %source -> %destination rows 576 stride 64
}

// -----

// A decode attention walks the heads, staging each head's queries and transposing them, then per cached key token
// block stages the keys and values and attends both halves: the transposed scores from the keys pushed against the
// transposed queries, the masked maximum and sum folded through VMEM, and the transposed values pushed against the
// probabilities held stationary. The output is normalised, transposed back and drained over the queries.
// CHECK-LABEL: tempo.kernel @decode_attention
// CHECK:       scf.for
// CHECK:       tempo.vtxin
// CHECK:       scf.for
// CHECK-COUNT-2: tempo.vmemld {{.*}} rows 32
// CHECK:       tempo.vmatload
// CHECK:       tempo.vmatpush
// CHECK:       tempo.vmatpop
// CHECK:       tempo.vst {{.*}} mask 1
// CHECK:       tempo.vexp
// CHECK:       tempo.vrecip
// CHECK:       tempo.vmemst {{.*}} rows 16
// CHECK-NOT:   tempo.attention
// PROGRAM-LABEL: tempoisa.program @decode_attention
// PROGRAM:       tempoisa.lbegin 2
// PROGRAM:       tempoisa.lbegin 1
// PROGRAM:       tempoisa.vtxout
// PROGRAM:       tempoisa.halt
tempo.kernel @decode_attention {
  %constants = tempo.arg 0
  %cache = tempo.arg 1
  %queries = tempo.arg 2
  tempo.attention %constants, %queries cache %cache heads 2 head 16 keys 32
}
