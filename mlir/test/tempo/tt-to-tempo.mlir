// Every GPT-2 small kernel a tile op covers converts from Triton into the tempo kernel computing one of its token
// blocks, its launch arguments in the order the Triton kernel takes its pointers. Attention, written for one head,
// converts into the image walking the heads the pass is given, causal over the queries tensor's own keys or masked
// to a length over a cache.

// RUN: tempo-opt %S/../Inputs/gemm_768x2304.ttir --convert-tt-to-tempo | FileCheck %s --check-prefix=GEMM
// RUN: tempo-opt %S/../Inputs/gemm_3072x768_gelu.ttir --convert-tt-to-tempo | FileCheck %s --check-prefix=GELU
// RUN: tempo-opt %S/../Inputs/gemm_3072x768_gelu_bias.ttir --convert-tt-to-tempo | FileCheck %s --check-prefix=BIAS
// RUN: tempo-opt %S/../Inputs/add_768.ttir --convert-tt-to-tempo | FileCheck %s --check-prefix=ADD
// RUN: tempo-opt %S/../Inputs/add_bias_768.ttir --convert-tt-to-tempo | FileCheck %s --check-prefix=ADDBIAS
// RUN: tempo-opt %S/../Inputs/layernorm_768.ttir --convert-tt-to-tempo | FileCheck %s --check-prefix=NORM
// RUN: tempo-opt %S/../Inputs/attention_64.ttir --convert-tt-to-tempo=attention-heads=12 | FileCheck %s --check-prefix=ATTENTION
// RUN: tempo-opt %S/../Inputs/decode_attention_256.ttir --convert-tt-to-tempo=attention-heads=12 | FileCheck %s --check-prefix=DECODE

// GEMM-LABEL: tempo.kernel @gpt2_matmul {
// GEMM-NEXT:    %[[A:.*]] = tempo.arg 0
// GEMM-NEXT:    %[[W:.*]] = tempo.arg 1
// GEMM-NEXT:    %[[D:.*]] = tempo.arg 2
// GEMM-NEXT:    tempo.matmul %[[A]], %[[W]] -> %[[D]] tokens 32 depth 768 columns 2304{{$}}
// GEMM-NEXT:  }
// GEMM-NOT:   tt.

// GELU-LABEL: tempo.kernel @gpt2_matmul_gelu {
// GELU:         tempo.matmul %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 depth 3072 columns 768 activate{{$}}

// BIAS-LABEL: tempo.kernel @gpt2_matmul_bias {
// BIAS:         %[[B:.*]] = tempo.arg 3
// BIAS-NEXT:    tempo.matmul %{{.*}}, %{{.*}} -> %{{.*}} bias %[[B]] tokens 32 depth 3072 columns 768 activate{{$}}

// ADD-LABEL: tempo.kernel @gpt2_add {
// ADD:         tempo.add %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 features 768{{$}}

// ADDBIAS-LABEL: tempo.kernel @gpt2_add_bias {
// ADDBIAS:         tempo.add %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 features 768 broadcast{{$}}

// NORM-LABEL: tempo.kernel @gpt2_layernorm {
// NORM:         tempo.layernorm %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 features 768 epsilon 9.99999974E-6{{$}}

// ATTENTION-LABEL: tempo.kernel @gpt2_attention {
// ATTENTION-NEXT:    %[[C:.*]] = tempo.arg 0
// ATTENTION-NEXT:    %[[Q:.*]] = tempo.arg 2
// ATTENTION-NEXT:    tempo.attention %[[C]], %[[Q]] heads 12 head 64 keys 64 causal{{$}}

// DECODE-LABEL: tempo.kernel @gpt2_decode_attention {
// DECODE-NEXT:    %[[C:.*]] = tempo.arg 0
// DECODE-NEXT:    %[[K:.*]] = tempo.arg 1
// DECODE-NEXT:    %[[Q:.*]] = tempo.arg 2
// DECODE-NEXT:    tempo.attention %[[C]], %[[Q]] cache %[[K]] heads 12 head 64 keys 256{{$}}
