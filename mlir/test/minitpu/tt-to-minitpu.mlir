// Every GPT-2 small kernel a tile op covers converts from Triton into the minitpu kernel computing one of its token
// blocks, its launch arguments in the order the Triton kernel takes its pointers. Attention, written for one head,
// converts into the image walking the heads the pass is given, causal over the queries tensor's own keys or masked
// to a length over a cache.

// RUN: minitpu-opt %S/../Inputs/gemm_768x2304.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=GEMM
// RUN: minitpu-opt %S/../Inputs/gemm_3072x768_gelu.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=GELU
// RUN: minitpu-opt %S/../Inputs/gemm_3072x768_gelu_bias.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=BIAS
// RUN: minitpu-opt %S/../Inputs/add_768.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=ADD
// RUN: minitpu-opt %S/../Inputs/add_bias_768.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=ADDBIAS
// RUN: minitpu-opt %S/../Inputs/layernorm_768.ttir --convert-tt-to-minitpu | FileCheck %s --check-prefix=NORM
// RUN: minitpu-opt %S/../Inputs/attention_64.ttir --convert-tt-to-minitpu=attention-heads=12 | FileCheck %s --check-prefix=ATTENTION
// RUN: minitpu-opt %S/../Inputs/decode_attention_256.ttir --convert-tt-to-minitpu=attention-heads=12 | FileCheck %s --check-prefix=DECODE

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

// ATTENTION-LABEL: minitpu.kernel @gpt2_attention {
// ATTENTION-NEXT:    %[[C:.*]] = minitpu.arg 0
// ATTENTION-NEXT:    %[[Q:.*]] = minitpu.arg 2
// ATTENTION-NEXT:    minitpu.attention %[[C]], %[[Q]] heads 12 head 64 keys 64 causal{{$}}

// DECODE-LABEL: minitpu.kernel @gpt2_decode_attention {
// DECODE-NEXT:    %[[C:.*]] = minitpu.arg 0
// DECODE-NEXT:    %[[K:.*]] = minitpu.arg 1
// DECODE-NEXT:    %[[Q:.*]] = minitpu.arg 2
// DECODE-NEXT:    minitpu.attention %[[C]], %[[Q]] cache %[[K]] heads 12 head 64 keys 256{{$}}
