# mlir/

The compiler's MLIR pipeline: Triton's tt dialect in, MiniTPU images out, as one C++ tool (`minitpu-opt`) built
against the LLVM Triton 3.8.0 pins. The Python compiler under `codegen/` stays the byte-for-byte oracle for each stage
until that stage replaces it.

## Building

The build needs two things the repository does not carry: Triton 3.8.0's source, for its tt dialect, and the LLVM
build that release pins (`cmake/llvm-info.json`, LLVM `5f07f818`), which Triton publishes prebuilt.

```sh
git clone --depth 1 --branch v3.8.0 https://github.com/triton-lang/triton.git ~/toolchains/triton-3.8.0
mkdir -p ~/.triton/llvm && cd ~/.triton/llvm
curl -LO https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-5f07f818-ubuntu-x64-1.tar.gz
tar xzf llvm-5f07f818-ubuntu-x64-1.tar.gz && rm llvm-5f07f818-ubuntu-x64-1.tar.gz

cd <repository>/mlir
LLVM=~/.triton/llvm/llvm-5f07f818-ubuntu-x64-1
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DMLIR_DIR=$LLVM/lib/cmake/mlir -DLLVM_DIR=$LLVM/lib/cmake/llvm -DLLVM_EXTERNAL_LIT=$(which lit)
ninja -C build check-minitpu
```

C++ follows the Sun Lab style, held by `.clang-format` and `.clang-tidy` in this directory.

`-DTRITON_SOURCE_DIR` names the Triton checkout when it is not `~/toolchains/triton-3.8.0`. `lit` comes from
`pip install lit`.

## Layout

| Path                          | Holds                                                                               |
|-------------------------------|-------------------------------------------------------------------------------------|
| `third_party/triton/`         | Triton's tt dialect, compiled from `TRITON_SOURCE_DIR` without the rest of Triton    |
| `include/minitpu/minitpuisa/` | The minitpuisa dialect: one op per instruction, its slots and opcodes read from `codegen/minitpuisa.td`, and the bundle bit layout (`encoding.h`) |
| `lib/minitpuisa/`             | The dialect's verifiers and the image import and export (`image.cpp`)                |
| `tools/minitpu-opt/`          | The `mlir-opt`-style driver every lit test runs                                     |
| `tools/minitpu-translate/`    | `--import-minitpu-image` reads a `.bin` into minitpuisa, `--export-minitpu-image` writes it back |
| `test/`                       | lit/FileCheck tests, one pass per file, including a byte-for-byte round trip of every board image |
| `test/Inputs/`                | Every GPT-2 small kernel's tt module, written by `tools/dump_gpt2_ttir.py`          |
