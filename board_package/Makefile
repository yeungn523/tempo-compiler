# Run these commands on the ZCU104 board.
SHELL := /bin/bash
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3 -u

.PHONY: help program provision check kernels identity

help:
	@echo "sudo make program     set the PL clock, program the PL, provision UIO/udmabuf"
	@echo "                      override the clock with PL_CLK_MHZ=<n> (default $(PL_CLK_MHZ))"
	@echo "sudo make provision   provision UIO/udmabuf on a PL that is already programmed"
	@echo "make check            verify package files and run the host tests, without hardware"
	@echo "make identity         which bitstream is loaded"
	@echo "make kernels          the board checks"

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

program:
	@$(HERE)minitpu/scripts/quiesce_pl.sh
	@$(HERE)minitpu/scripts/set_pl_clock.sh $(PL_CLK_MHZ)
	@$(HERE)minitpu/scripts/program_pl.sh
	@$(HERE)minitpu/scripts/provision.sh

provision:
	@$(HERE)minitpu/scripts/provision.sh

check:
	@$(PYTHON) $(HERE)minitpu/scripts/check_package.py
	@$(PYTHON) -m unittest discover -s $(HERE)tests -q

identity:
	@$(PYTHON) $(HERE)minitpu/scripts/read_build_id.py

kernels:
	@$(PYTHON) $(HERE)gpt2_kernels.py
