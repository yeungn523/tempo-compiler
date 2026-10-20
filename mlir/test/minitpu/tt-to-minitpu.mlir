// Every GPT-2 small kernel a tile op covers converts from Triton into the minitpu kernel computing one of its token
// blocks, its launch arguments in the order the Triton kernel takes its pointers. Attention converts into no tile op
// yet and is refused.

// RUN: minitpu-opt %S/../Inputs/gemm_768x2304.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=GEMM
// RUN: minitpu-opt %S/../Inputs/gemm_3072x768_gelu.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=GELU
// RUN: minitpu-opt %S/../Inputs/gemm_3072x768_gelu_bias.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=BIAS
// RUN: minitpu-opt %S/../Inputs/add_768.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=ADD
// RUN: minitpu-opt %S/../Inputs/add_bias_768.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=ADDBIAS
// RUN: minitpu-opt %S/../Inputs/layernorm_768.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=NORM
// RUN: not minitpu-opt %S/../Inputs/attention_64.ttir --convert-tt-to-minitpu 2>&1 | FileCheck %s --check-prefix=ATTENTION

// GEMM-LABEL: minitpu.kernel @gpt2_matmul {
// GEMM-NEXT:    %[[A:.*]] = minitpu.arg 0
// GEMM-NEXT:    %[[W:.*]] = minitpu.arg 1
// GEMM-NEXT:    %[[D:.*]] = minitpu.arg 2
// GEMM-NEXT:    minitpu.matmul %[[A]], %[[W]] -> %[[D]] tokens 32 depth 768 columns 2304{{$}}
// GEMM-NEXT:  }
// GEMM-NOT:   tt.

// GELU-LABEL: minitpu.kernel @gpt2_matmul_gelu {
// GELU:         minitpu.matmul %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 depth 3072 columns 768 activate{{$}}

// BIAS-LABEL: minitpu.kernel @gpt2_matmul_bias {
// BIAS:         %[[B:.*]] = minitpu.arg 3
// BIAS-NEXT:    minitpu.matmul %{{.*}}, %{{.*}} -> %{{.*}} bias %[[B]] tokens 32 depth 3072 columns 768 activate{{$}}

// ADD-LABEL: minitpu.kernel @gpt2_add {
// ADD:         minitpu.add %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 features 768{{$}}

// ADDBIAS-LABEL: minitpu.kernel @gpt2_add_bias {
// ADDBIAS:         minitpu.add %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 features 768 broadcast{{$}}

// NORM-LABEL: minitpu.kernel @gpt2_layernorm {
// NORM:         minitpu.layernorm %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 features 768 epsilon 9.99999974E-6{{$}}

// ATTENTION: error: takes 2 dot products. A product converts into minitpu.matmul when one dot product accumulates its result.
