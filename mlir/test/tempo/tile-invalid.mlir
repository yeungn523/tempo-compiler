// A tile op whose shape the lowering does not cover is refused with its own diagnostic.

// RUN: tempo-opt %s -split-input-file -verify-diagnostics

tempo.kernel @wrong_tokens {
  %a = tempo.arg 0
  %w = tempo.arg 1
  %d = tempo.arg 2
  // expected-error @below {{is unable to cover 16 tokens. One launch covers 32}}
  tempo.matmul %a, %w -> %d tokens 16 depth 256 columns 32
}

// -----

tempo.kernel @partial_tile {
  %a = tempo.arg 0
  %w = tempo.arg 1
  %d = tempo.arg 2
  // expected-error @below {{Both must be a whole number of 16-wide tiles}}
  tempo.matmul %a, %w -> %d tokens 32 depth 200 columns 32
}

// -----

tempo.kernel @bias_past_one_bank {
  %a = tempo.arg 0
  %w = tempo.arg 1
  %d = tempo.arg 2
  %b = tempo.arg 3
  // expected-error @below {{is unable to stage 32768 columns. Its bias row and weight chunks take 8480 VMEM rows}}
  tempo.matmul %a, %w -> %d bias %b tokens 32 depth 256 columns 32768
}

// -----

tempo.kernel @add_wrong_tokens {
  %l = tempo.arg 0
  %r = tempo.arg 1
  %o = tempo.arg 2
  // expected-error @below {{is unable to cover 64 tokens. One launch covers 32}}
  tempo.add %l, %r -> %o tokens 64 features 768
}

// -----

tempo.kernel @add_odd_chunks {
  %l = tempo.arg 0
  %r = tempo.arg 1
  %o = tempo.arg 2
  // expected-error @below {{The width must be a whole number of 128, an even number of 128-row chunks}}
  tempo.add %l, %r -> %o tokens 32 features 192 broadcast
}

// -----

tempo.kernel @add_past_one_bank {
  %l = tempo.arg 0
  %r = tempo.arg 1
  %o = tempo.arg 2
  // expected-error @below {{is unable to stage 32768 features}}
  tempo.add %l, %r -> %o tokens 32 features 32768 broadcast
}

// -----

tempo.kernel @layernorm_partial_register {
  %r = tempo.arg 0
  %c = tempo.arg 1
  %o = tempo.arg 2
  // expected-error @below {{The width must be a whole number of 64, and at most 2048}}
  tempo.layernorm %r, %c -> %o tokens 32 features 96 epsilon 1.0e-05
}

// -----

tempo.kernel @layernorm_past_one_descriptor {
  %r = tempo.arg 0
  %c = tempo.arg 1
  %o = tempo.arg 2
  // expected-error @below {{is unable to cover 4096 features}}
  tempo.layernorm %r, %c -> %o tokens 32 features 4096 epsilon 1.0e-05
}

// -----

tempo.kernel @copy_partial_row {
  %s = tempo.arg 0
  %d = tempo.arg 2
  // expected-error @below {{is unable to step 48 bytes between rows}}
  tempo.copy %s -> %d rows 8 stride 48
}

// -----

tempo.kernel @attention_keys_twice {
  %c = tempo.arg 0
  %k = tempo.arg 1
  %q = tempo.arg 2
  // expected-error @below {{is unable to tell where its keys lie}}
  tempo.attention %c, %q cache %k heads 12 head 64 keys 64 causal
}

// -----

tempo.kernel @attention_partial_tile {
  %c = tempo.arg 0
  %q = tempo.arg 2
  // expected-error @below {{is unable to cover a head of 24 over 64 keys}}
  tempo.attention %c, %q heads 12 head 24 keys 64 causal
}

// -----

tempo.kernel @attention_past_one_bank {
  %c = tempo.arg 0
  %k = tempo.arg 1
  %q = tempo.arg 2
  // expected-error @below {{is unable to stage 8192 keys}}
  tempo.attention %c, %q cache %k heads 12 head 64 keys 8192
}
