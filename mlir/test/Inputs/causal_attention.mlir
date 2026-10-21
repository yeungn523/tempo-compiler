// The causal attention minitpu/manifest.mlir exports a constants block from.

minitpu.kernel @causal {
  %c = minitpu.arg 0
  %q = minitpu.arg 2
  minitpu.attention %c, %q heads 12 head 64 keys 64 causal
}
