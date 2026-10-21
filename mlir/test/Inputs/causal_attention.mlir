// The causal attention tempo/manifest.mlir exports a constants block from.

tempo.kernel @causal {
  %c = tempo.arg 0
  %q = tempo.arg 2
  tempo.attention %c, %q heads 12 head 64 keys 64 causal
}
