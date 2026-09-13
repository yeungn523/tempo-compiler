# Run these commands on the ZCU104 board.
SHELL := /bin/bash
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3 -u

.PHONY: help minitpu provision check chat kernels reproduce \
        identity batch throughput boundary trace

help:
	@echo "sudo make minitpu  - set the PL clock, program the PL, provision UIO/udmabuf"
	@echo "                     override the clock with PL_CLK_MHZ=<n> (default $(PL_CLK_MHZ))"
	@echo "make reproduce - every headline claim in the README, in order (~6 min)"
	@echo
	@echo "  make identity   - which bitstream is loaded            (seconds)"
	@echo "  make kernels    - 16/16 board checks, block 0.68 %     (~2 min)"
	@echo "  make batch      - B = 1..16 give bit-identical results (~10 s)"
	@echo "  make boundary   - the 2 activation copies a pass makes (~30 s)"
	@echo "  make throughput - warm body time, tokens/s, FLOP rates (~60 s)"
	@echo "  make chat       - TTFT and decode rate, 5-token prompt (~30 s)"
	@echo "  make trace      - the 18 launches one layer issues     (seconds)"
	@echo
	@echo "make check     - verify package files without hardware access"
	@echo
	@echo "make chat accepts the CLI's own flags through ARGS, e.g."
	@echo "  make chat ARGS='--prompt \"The capital of France is\" --temperature 0'"

# PL_CLK_MHZ is set here because it is NOT set by the bitstream. PL0_REF_CTRL is
# programmed by the FSBL inside BOOT.BIN -- dated May 2025, older than any
# bitstream here -- and fpga_manager does not touch it, so a board boots at
# 99.999 MHz however the design was closed. The clock must be set BEFORE
# programming: the fabric is not designed to survive a clock change underneath
# itself, and programming holds it in reset so it starts cleanly on the new one.
#
# 187 is the verified default: 16/16 board checks with accuracy identical to
# 100 MHz. set_pl_clock.sh refuses past 190 -- closure was 200 MHz at WNS
# +0.049 ns, which is 1 % margin.
PL_CLK_MHZ ?= 187

minitpu:
	@$(HERE)minitpu/scripts/quiesce_pl.sh
	@$(HERE)minitpu/scripts/set_pl_clock.sh $(PL_CLK_MHZ)
	@$(HERE)minitpu/scripts/program_pl.sh
	@$(HERE)minitpu/scripts/provision.sh

provision:
	@$(HERE)minitpu/scripts/provision.sh

check:
	@$(PYTHON) $(HERE)minitpu/scripts/check_package.py
	@$(PYTHON) -m unittest discover -s $(HERE)tests -q

# The chat CLI. ARGS passes the program's own flags through, so the target does
# not have to grow a mirror of them: make chat ARGS='--layers 1 --temperature 0'.
#
# WEIGHTS is checked here rather than left to the CLI because the failure is a
# deploy step, not a usage error: the checkpoint is 475 MB and is built on a host
# that has torch (tools/export_gpt2_weights.py), so it is gitignored and has to
# be copied over separately.
WEIGHTS ?= $(HERE)gpt2_weights.npz
ARGS ?=

chat:
	@test -f "$(WEIGHTS)" || { \
	    echo "No checkpoint at $(WEIGHTS)."; \
	    echo "Build one on a host that has torch, then copy it here:"; \
	    echo "  python3 tools/export_gpt2_weights.py --out build/gpt2_weights.npz"; \
	    echo "  scp build/gpt2_weights.npz <board>:$(WEIGHTS)"; \
	    exit 1; }
	@$(PYTHON) $(HERE)minichat.py --weights "$(WEIGHTS)" $(ARGS)

kernels:
	@$(PYTHON) $(HERE)gpt2_kernels.py

# ---------------------------------------------------------------------------
# Reproducing the README's headline claims. One target per claim, so the
# document and the commands cannot drift: if a number in the README is wrong,
# one of these prints a different one.
#
# The PL clock is a runtime setting, not a property of the bitstream, so the
# throughput percentages need to be told what it is. It is the same default the
# `minitpu` target programs.
CLOCK_MHZ ?= 187.5

# `chat` with no ARGS is the INTERACTIVE CLI, and an interactive chat inside a
# scripted run does not fail -- it sits on stdin holding the device open, which
# is one of the two ways to wedge a board (D6). So this passes a prompt.
REPRODUCE_PROMPT ?= --prompt "The capital of France is" --temperature 0 --max-new-tokens 2

reproduce: identity kernels batch boundary throughput
	@$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) chat ARGS='$(REPRODUCE_PROMPT)'
	@echo
	@echo "== every headline claim above, on $(shell hostname). Compare with README.md."

identity:
	@$(PYTHON) $(HERE)minitpu/scripts/read_build_id.py

batch:
	@$(PYTHON) $(HERE)check_batch_invariance.py --device

boundary:
	@$(PYTHON) $(HERE)trace_boundary.py 12

# Two shapes, because two claims come from here: the 32-row pass IS the warm
# twelve-layer body time, and 256 rows is where throughput plateaus. One shape per
# process is deliberate -- the host staging arena only grows -- so this is three
# processes: the header, then each shape.
throughput:
	@MINITPU_PL_CLOCK_MHZ=$(CLOCK_MHZ) $(PYTHON) $(HERE)bench_throughput.py
	@MINITPU_PL_CLOCK_MHZ=$(CLOCK_MHZ) $(PYTHON) $(HERE)bench_throughput.py 1 32
	@MINITPU_PL_CLOCK_MHZ=$(CLOCK_MHZ) $(PYTHON) $(HERE)bench_throughput.py 1 256

trace:
	@$(PYTHON) $(HERE)run_gpt2_forward.py --trace
