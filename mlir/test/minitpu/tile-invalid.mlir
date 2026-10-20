// A tile op whose shape the lowering does not cover is refused with its own diagnostic.

// RUN: minitpu-opt %s -split-input-file -verify-diagnostics

minitpu.kernel @wrong_tokens {
  %a = minitpu.arg 0
  %w = minitpu.arg 1
  %d = minitpu.arg 2
  // expected-error @below {{is unable to cover 16 tokens. One launch covers 32}}
  minitpu.matmul %a, %w -> %d tokens 16 depth 256 columns 32
}

// -----

minitpu.kernel @partial_chunk {
  %a = minitpu.arg 0
  %w = minitpu.arg 1
  %d = minitpu.arg 2
  // expected-error @below {{The depth must be a whole number of 128-row weight chunks}}
  minitpu.matmul %a, %w -> %d tokens 32 depth 200 columns 32
}

// -----

minitpu.kernel @past_one_bank {
  %a = minitpu.arg 0
  %w = minitpu.arg 1
  %d = minitpu.arg 2
  // expected-error @below {{is unable to stage a depth of 4096}}
  minitpu.matmul %a, %w -> %d tokens 32 depth 4096 columns 32
}

// -----

minitpu.kernel @add_wrong_tokens {
  %l = minitpu.arg 0
  %r = minitpu.arg 1
  %o = minitpu.arg 2
  // expected-error @below {{is unable to cover 64 tokens. One launch covers 32}}
  minitpu.add %l, %r -> %o tokens 64 features 768
}

// -----

minitpu.kernel @add_odd_chunks {
  %l = minitpu.arg 0
  %r = minitpu.arg 1
  %o = minitpu.arg 2
  // expected-error @below {{The width must be a whole number of 128, an even number of 128-row chunks}}
  minitpu.add %l, %r -> %o tokens 32 features 192 broadcast
}

// -----

minitpu.kernel @add_past_one_bank {
  %l = minitpu.arg 0
  %r = minitpu.arg 1
  %o = minitpu.arg 2
  // expected-error @below {{is unable to stage 32768 features}}
  minitpu.add %l, %r -> %o tokens 32 features 32768 broadcast
}

// -----

minitpu.kernel @layernorm_partial_register {
  %r = minitpu.arg 0
  %c = minitpu.arg 1
  %o = minitpu.arg 2
  // expected-error @below {{The width must be a whole number of 64, and at most 2048}}
  minitpu.layernorm %r, %c -> %o tokens 32 features 96
}

// -----

minitpu.kernel @layernorm_past_one_descriptor {
  %r = minitpu.arg 0
  %c = minitpu.arg 1
  %o = minitpu.arg 2
  // expected-error @below {{is unable to cover 4096 features}}
  minitpu.layernorm %r, %c -> %o tokens 32 features 4096
}
