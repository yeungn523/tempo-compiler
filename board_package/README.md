# NPU Board Package

Programs the PL on a ZCU104, runs GPT-2 small's layers on it, and checks every result against a host reference.

___

## Usage

### Deploying

From the repository root on the host, with a session open on the board:

```bash
tools/board/prepare_board.sh <account> <board-id>  # exports GPT-2's weights, builds the images, deploys this package
```

The board erases its home directory whenever no session holds it, so rerun the script after every login.

### Running

On the board, from the deployed package:

```bash
make check                                                # verifies the package files and runs the host tests
sudo make minitpu                                         # sets the PL clock, programs the PL, provisions the devices
make kernels                                              # runs the board checks
make generate PROMPT="The capital of France is" TOKENS=8  # prints GPT-2 small's greedy continuation
```

`sudo make minitpu` is needed after every boot or PL replacement, and everything after it runs as a normal user.

***Warning!*** Program the PL only through this package's `sudo make minitpu`. Without the clock it sets, every check
still passes but every launch runs about 1.8 times too slow.

### Generating From Python

`make generate` wraps the same calls a script makes:

```python
import numpy as np

from gpt2_inference import generate, load_gpt2_weights
from tokenizer import Tokenizer

tokenizer = Tokenizer()
layers, shared = load_gpt2_weights()
prompt = np.asarray(tokenizer.encode("The capital of France is"), dtype=np.int64)
print(tokenizer.decode(generate(prompt, layers, shared, new_tokens=8)))
```

`prefill` and `decode_step` run the two halves of `generate` separately. Prompts are capped at 64 tokens, the longest
sequence a compiled attention image covers.

___

## Board Checks

`make kernels` runs ten checks, each against a host reference that rounds where the device rounds.

| Check                       | Shape                                                   | Passes when                 |
|-----------------------------|---------------------------------------------------------|-----------------------------|
| Halt                        | The one-bundle program, launched three times            | Every launch retires        |
| COPY round trip             | 48 KiB of random words through device memory            | Exact                       |
| Full GEMM                   | `[32,768] @ [768,3072]`, structured and varying         | Exact, and under 1% error   |
| Elementwise add             | `[32,768] + [32,768]`, and plus a bias row              | Exact                       |
| Decode attention, all heads | 12 heads, one query over 49 cached keys                 | Under 5% error              |
| LayerNorm                   | `[32,768]`                                              | Under 2% error              |
| KV cache                    | 64 prefill tokens and one decode token                  | Exact                       |
| Transformer block           | `[64,768]`, 12 causal heads                             | Under 5% error              |
| GPT-2 prefill               | A real prompt through 12 layers of GPT-2's weights      | Every layer under 5% error  |
| GPT-2 decode                | One greedy token through 12 layers                      | Every layer under 5% error  |

The GEMM check's varying operands differ along every axis the kernel indexes, which `assert_operands_discriminating`
enforces, so an image that confuses two tokens, contraction blocks, or output columns fails it.
