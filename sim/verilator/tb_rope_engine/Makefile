RTL_DIR    := ../../../rtl/cores/bitmamba3
RTL        := $(RTL_DIR)/rope_engine.v
TB_CPP     := tb_rope_engine.cpp
TOP        := rope_engine
OBJ_DIR    := obj_dir
EXE        := $(OBJ_DIR)/V$(TOP)
VEC        := ../vectors/rope.bin
VERILATOR  ?= verilator
VFLAGS     := --cc --exe --build -Wall --timing --Mdir $(OBJ_DIR)

.PHONY: all run clean vectors

all: $(EXE)

$(EXE): $(RTL) $(TB_CPP)
	$(VERILATOR) $(VFLAGS) $(RTL) $(TB_CPP) --top-module $(TOP)

vectors: $(VEC)

$(VEC):
	cd ../../.. && python sim/verilator/gen_vectors_rope.py --n 1000

run: $(EXE) vectors
	$(EXE) $(VEC)

clean:
	rm -rf $(OBJ_DIR) $(VEC)
