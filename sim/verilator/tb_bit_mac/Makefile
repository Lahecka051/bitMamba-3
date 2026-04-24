# Makefile for Verilator tb_bit_mac
#
# Usage:
#   make        # build
#   make run    # build + run
#   make clean

RTL_DIR    := ../../../rtl/cores/bitmamba3
RTL        := $(RTL_DIR)/bit_mac.v
TB_CPP     := tb_bit_mac.cpp
TOP        := bit_mac
OBJ_DIR    := obj_dir
EXE        := $(OBJ_DIR)/V$(TOP)
VEC        := ../vectors/bit_mac.bin

VERILATOR  ?= verilator
VFLAGS     := --cc --exe --build -Wall --timing --Mdir $(OBJ_DIR)

.PHONY: all run clean vectors

all: $(EXE)

$(EXE): $(RTL) $(TB_CPP)
	$(VERILATOR) $(VFLAGS) $(RTL) $(TB_CPP) --top-module $(TOP)

vectors: $(VEC)

$(VEC):
	cd ../../.. && python sim/verilator/gen_vectors_bit_mac.py --n 1000 --lanes 128

run: $(EXE) vectors
	$(EXE) $(VEC)

clean:
	rm -rf $(OBJ_DIR) $(VEC)
