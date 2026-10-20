// A Triton kernel the conversion does not recognise, or one addressing an operand other than as the host packs it,
// is refused with a diagnostic naming what differs.

// RUN: minitpu-opt %s -split-input-file -verify-diagnostics --convert-tt-to-minitpu | FileCheck %s

// A sum laid out as the host packs it converts.
// CHECK-LABEL: minitpu.kernel @sum
// CHECK: minitpu.add %{{.*}}, %{{.*}} -> %{{.*}} tokens 32 features 256{{$}}

module {
tt.func public @sum(%l: !tt.ptr<bf16>, %r: !tt.ptr<bf16>, %o: !tt.ptr<bf16>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %stride = arith.constant dense<256> : tensor<32x1xi32>
  %wide = arith.constant dense<512> : tensor<32x1xi32>
  %mask = arith.constant dense<true> : tensor<32x256xi1>
  scf.for %b = %c0 to %c64 step %c32 : i32 {
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %first = tt.splat %b : i32 -> tensor<32xi32>
    %m = arith.addi %first, %rows : tensor<32xi32>
    scf.for %f = %c0 to %c256 step %c256 : i32 {
      %columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %start = tt.splat %f : i32 -> tensor<256xi32>
      %n = arith.addi %start, %columns : tensor<256xi32>
      %row = tt.expand_dims %m {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %scaled = arith.muli %row, %stride : tensor<32x1xi32>
      %spread = arith.muli %row, %wide : tensor<32x1xi32>
      %column = tt.expand_dims %n {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
      %rowb = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x256xi32>
      %spreadb = tt.broadcast %spread : tensor<32x1xi32> -> tensor<32x256xi32>
      %columnb = tt.broadcast %column : tensor<1x256xi32> -> tensor<32x256xi32>
      %offsets = arith.addi %rowb, %columnb : tensor<32x256xi32>
      %far = arith.addi %spreadb, %columnb : tensor<32x256xi32>
      %lbase = tt.splat %l : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %laddr = tt.addptr %lbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %rbase = tt.splat %r : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %raddr = tt.addptr %rbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %obase = tt.splat %o : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %oaddr = tt.addptr %obase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %left = tt.load %laddr : tensor<32x256x!tt.ptr<bf16>>
      %right = tt.load %raddr : tensor<32x256x!tt.ptr<bf16>>
      %sum = arith.addf %left, %right : tensor<32x256xbf16>
      tt.store %oaddr, %sum : tensor<32x256x!tt.ptr<bf16>>
    }
  }
  tt.return
}
}


// -----

// A loop over token blocks must have constant bounds.

module {
tt.func public @unbounded(%l: !tt.ptr<bf16>, %r: !tt.ptr<bf16>, %o: !tt.ptr<bf16>, %count: i32) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %stride = arith.constant dense<256> : tensor<32x1xi32>
  %wide = arith.constant dense<512> : tensor<32x1xi32>
  %mask = arith.constant dense<true> : tensor<32x256xi1>
  // expected-error @+1 {{is a loop the conversion is unable to read}}
  scf.for %b = %c0 to %count step %c32 : i32 {
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %first = tt.splat %b : i32 -> tensor<32xi32>
    %m = arith.addi %first, %rows : tensor<32xi32>
    scf.for %f = %c0 to %c256 step %c256 : i32 {
      %columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %start = tt.splat %f : i32 -> tensor<256xi32>
      %n = arith.addi %start, %columns : tensor<256xi32>
      %row = tt.expand_dims %m {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %scaled = arith.muli %row, %stride : tensor<32x1xi32>
      %spread = arith.muli %row, %wide : tensor<32x1xi32>
      %column = tt.expand_dims %n {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
      %rowb = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x256xi32>
      %spreadb = tt.broadcast %spread : tensor<32x1xi32> -> tensor<32x256xi32>
      %columnb = tt.broadcast %column : tensor<1x256xi32> -> tensor<32x256xi32>
      %offsets = arith.addi %rowb, %columnb : tensor<32x256xi32>
      %far = arith.addi %spreadb, %columnb : tensor<32x256xi32>
      %lbase = tt.splat %l : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %laddr = tt.addptr %lbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %rbase = tt.splat %r : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %raddr = tt.addptr %rbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %obase = tt.splat %o : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %oaddr = tt.addptr %obase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %left = tt.load %laddr : tensor<32x256x!tt.ptr<bf16>>
      %right = tt.load %raddr : tensor<32x256x!tt.ptr<bf16>>
      %sum = arith.addf %left, %right : tensor<32x256xbf16>
      tt.store %oaddr, %sum : tensor<32x256x!tt.ptr<bf16>>
    }
  }
  tt.return
}
}


// -----

// An operand read with rows other than one width apart is no dense row-major tensor.

module {
tt.func public @strided(%l: !tt.ptr<bf16>, %r: !tt.ptr<bf16>, %o: !tt.ptr<bf16>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %stride = arith.constant dense<256> : tensor<32x1xi32>
  %wide = arith.constant dense<512> : tensor<32x1xi32>
  %mask = arith.constant dense<true> : tensor<32x256xi1>
  scf.for %b = %c0 to %c64 step %c32 : i32 {
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %first = tt.splat %b : i32 -> tensor<32xi32>
    %m = arith.addi %first, %rows : tensor<32xi32>
    scf.for %f = %c0 to %c256 step %c256 : i32 {
      %columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %start = tt.splat %f : i32 -> tensor<256xi32>
      %n = arith.addi %start, %columns : tensor<256xi32>
      %row = tt.expand_dims %m {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %scaled = arith.muli %row, %stride : tensor<32x1xi32>
      %spread = arith.muli %row, %wide : tensor<32x1xi32>
      %column = tt.expand_dims %n {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
      %rowb = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x256xi32>
      %spreadb = tt.broadcast %spread : tensor<32x1xi32> -> tensor<32x256xi32>
      %columnb = tt.broadcast %column : tensor<1x256xi32> -> tensor<32x256xi32>
      %offsets = arith.addi %rowb, %columnb : tensor<32x256xi32>
      %far = arith.addi %spreadb, %columnb : tensor<32x256xi32>
      %lbase = tt.splat %l : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %laddr = tt.addptr %lbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %rbase = tt.splat %r : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %raddr = tt.addptr %rbase, %far : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %obase = tt.splat %o : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %oaddr = tt.addptr %obase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %left = tt.load %laddr : tensor<32x256x!tt.ptr<bf16>>
      // expected-error @+2 {{addresses the right operand in a layout the host does not pack it in}}
      // expected-note @+1 {{expected element strides [256, 1] and found [512, 1]}}
      %right = tt.load %raddr : tensor<32x256x!tt.ptr<bf16>>
      %sum = arith.addf %left, %right : tensor<32x256xbf16>
      tt.store %oaddr, %sum : tensor<32x256x!tt.ptr<bf16>>
    }
  }
  tt.return
}
}


// -----

// A masked load is no operand the host stages whole.

module {
tt.func public @masked(%l: !tt.ptr<bf16>, %r: !tt.ptr<bf16>, %o: !tt.ptr<bf16>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %stride = arith.constant dense<256> : tensor<32x1xi32>
  %wide = arith.constant dense<512> : tensor<32x1xi32>
  %mask = arith.constant dense<true> : tensor<32x256xi1>
  scf.for %b = %c0 to %c64 step %c32 : i32 {
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %first = tt.splat %b : i32 -> tensor<32xi32>
    %m = arith.addi %first, %rows : tensor<32xi32>
    scf.for %f = %c0 to %c256 step %c256 : i32 {
      %columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %start = tt.splat %f : i32 -> tensor<256xi32>
      %n = arith.addi %start, %columns : tensor<256xi32>
      %row = tt.expand_dims %m {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %scaled = arith.muli %row, %stride : tensor<32x1xi32>
      %spread = arith.muli %row, %wide : tensor<32x1xi32>
      %column = tt.expand_dims %n {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
      %rowb = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x256xi32>
      %spreadb = tt.broadcast %spread : tensor<32x1xi32> -> tensor<32x256xi32>
      %columnb = tt.broadcast %column : tensor<1x256xi32> -> tensor<32x256xi32>
      %offsets = arith.addi %rowb, %columnb : tensor<32x256xi32>
      %far = arith.addi %spreadb, %columnb : tensor<32x256xi32>
      %lbase = tt.splat %l : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %laddr = tt.addptr %lbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %rbase = tt.splat %r : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %raddr = tt.addptr %rbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %obase = tt.splat %o : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %oaddr = tt.addptr %obase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %left = tt.load %laddr : tensor<32x256x!tt.ptr<bf16>>
      %right = tt.load %raddr, %mask : tensor<32x256x!tt.ptr<bf16>>
      %sum = arith.addf %left, %right : tensor<32x256xbf16>
      // expected-error @+1 {{is unable to be read as a sum}}
      tt.store %oaddr, %sum : tensor<32x256x!tt.ptr<bf16>>
    }
  }
  tt.return
}
}


// -----

// A kernel converts into one tile op, which writes one result.

module {
// expected-error @+1 {{stores 2 times}}
tt.func public @twice(%l: !tt.ptr<bf16>, %r: !tt.ptr<bf16>, %o: !tt.ptr<bf16>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %stride = arith.constant dense<256> : tensor<32x1xi32>
  %wide = arith.constant dense<512> : tensor<32x1xi32>
  %mask = arith.constant dense<true> : tensor<32x256xi1>
  scf.for %b = %c0 to %c64 step %c32 : i32 {
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %first = tt.splat %b : i32 -> tensor<32xi32>
    %m = arith.addi %first, %rows : tensor<32xi32>
    scf.for %f = %c0 to %c256 step %c256 : i32 {
      %columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %start = tt.splat %f : i32 -> tensor<256xi32>
      %n = arith.addi %start, %columns : tensor<256xi32>
      %row = tt.expand_dims %m {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %scaled = arith.muli %row, %stride : tensor<32x1xi32>
      %spread = arith.muli %row, %wide : tensor<32x1xi32>
      %column = tt.expand_dims %n {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
      %rowb = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x256xi32>
      %spreadb = tt.broadcast %spread : tensor<32x1xi32> -> tensor<32x256xi32>
      %columnb = tt.broadcast %column : tensor<1x256xi32> -> tensor<32x256xi32>
      %offsets = arith.addi %rowb, %columnb : tensor<32x256xi32>
      %far = arith.addi %spreadb, %columnb : tensor<32x256xi32>
      %lbase = tt.splat %l : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %laddr = tt.addptr %lbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %rbase = tt.splat %r : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %raddr = tt.addptr %rbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %obase = tt.splat %o : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %oaddr = tt.addptr %obase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %left = tt.load %laddr : tensor<32x256x!tt.ptr<bf16>>
      %right = tt.load %raddr : tensor<32x256x!tt.ptr<bf16>>
      %sum = arith.addf %left, %right : tensor<32x256xbf16>
      tt.store %oaddr, %sum : tensor<32x256x!tt.ptr<bf16>>
      tt.store %oaddr, %left : tensor<32x256x!tt.ptr<bf16>>
    }
  }
  tt.return
}
}


// -----

// The launch passes the pointers in the order the kernel takes them, which must be the operands' order.

module {
// expected-error @+1 {{must take exactly its left, right, result pointers, in that order}}
tt.func public @reordered(%r: !tt.ptr<bf16>, %l: !tt.ptr<bf16>, %o: !tt.ptr<bf16>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %stride = arith.constant dense<256> : tensor<32x1xi32>
  %wide = arith.constant dense<512> : tensor<32x1xi32>
  %mask = arith.constant dense<true> : tensor<32x256xi1>
  scf.for %b = %c0 to %c64 step %c32 : i32 {
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %first = tt.splat %b : i32 -> tensor<32xi32>
    %m = arith.addi %first, %rows : tensor<32xi32>
    scf.for %f = %c0 to %c256 step %c256 : i32 {
      %columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
      %start = tt.splat %f : i32 -> tensor<256xi32>
      %n = arith.addi %start, %columns : tensor<256xi32>
      %row = tt.expand_dims %m {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %scaled = arith.muli %row, %stride : tensor<32x1xi32>
      %spread = arith.muli %row, %wide : tensor<32x1xi32>
      %column = tt.expand_dims %n {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
      %rowb = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x256xi32>
      %spreadb = tt.broadcast %spread : tensor<32x1xi32> -> tensor<32x256xi32>
      %columnb = tt.broadcast %column : tensor<1x256xi32> -> tensor<32x256xi32>
      %offsets = arith.addi %rowb, %columnb : tensor<32x256xi32>
      %far = arith.addi %spreadb, %columnb : tensor<32x256xi32>
      %lbase = tt.splat %l : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %laddr = tt.addptr %lbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %rbase = tt.splat %r : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %raddr = tt.addptr %rbase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %obase = tt.splat %o : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>>
      %oaddr = tt.addptr %obase, %offsets : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi32>
      %left = tt.load %laddr : tensor<32x256x!tt.ptr<bf16>>
      %right = tt.load %raddr : tensor<32x256x!tt.ptr<bf16>>
      %sum = arith.addf %left, %right : tensor<32x256xbf16>
      tt.store %oaddr, %sum : tensor<32x256x!tt.ptr<bf16>>
    }
  }
  tt.return
}
}
