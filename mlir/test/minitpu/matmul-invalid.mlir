// A product whose shape the lowering does not cover is refused with its own diagnostic.

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
