// tb_bit_mac.cpp — Verilator testbench for bit_mac.v
//
// Consumes vectors produced by sim/verilator/gen_vectors_bit_mac.py
// Each vector: LANES bytes activations + LANES/4 bytes packed weights + INT32 expected.
// Drives DUT cycle by cycle, captures out_valid/out_acc, asserts bit-exact equality.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <verilated.h>
#include "Vbit_mac.h"

static constexpr int LANES = 128;
static constexpr int ACT_W = 8;
static constexpr int WEIGHT_CODE_W = 2;
static constexpr int PIPELINE_STAGES = 3;

static uint64_t sim_time = 0;

static void tick(Vbit_mac* dut) {
    dut->clk = 0;
    dut->eval();
    ++sim_time;
    dut->clk = 1;
    dut->eval();
    ++sim_time;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const char* vec_path = "sim/verilator/vectors/bit_mac.bin";
    if (argc > 1) vec_path = argv[1];

    std::ifstream in(vec_path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", vec_path);
        return 1;
    }

    Vbit_mac dut;

    // Reset
    dut.rst_n = 0;
    for (int i = 0; i < 4; ++i) tick(&dut);
    dut.rst_n = 1;

    size_t const act_bytes = LANES;
    size_t const w_bytes   = LANES / 4; // 2 bits each, 4 per byte
    size_t const exp_bytes = 4;

    std::vector<uint8_t> act(act_bytes);
    std::vector<uint8_t> w_packed(w_bytes);

    int ok = 0, bad = 0;

    struct Pending { int32_t expected; int cycles_left; };
    std::vector<Pending> pending;

    while (in.read(reinterpret_cast<char*>(act.data()), act_bytes)
           && in.read(reinterpret_cast<char*>(w_packed.data()), w_bytes)) {
        int32_t expected;
        in.read(reinterpret_cast<char*>(&expected), exp_bytes);

        // Pack activation into the wide signal (LSB lane 0)
        // Verilator flattens wide signals into uint32[] arrays.
        // Using direct byte copy is brittle; explicit per-lane assignment:
        for (int i = 0; i < LANES; ++i) {
            // set LSB-aligned 8 bits in in_act
            uint32_t* words = reinterpret_cast<uint32_t*>(&dut.in_act);
            int bit = i * ACT_W;
            int wi = bit / 32;
            int bo = bit % 32;
            uint32_t mask = 0xFFu << bo;
            uint32_t val = static_cast<uint32_t>(static_cast<uint8_t>(act[i])) << bo;
            words[wi] = (words[wi] & ~mask) | (val & mask);
        }
        // Pack 2-bit weight codes
        for (int i = 0; i < LANES; ++i) {
            int code = (w_packed[i / 4] >> ((i % 4) * 2)) & 0x3;
            uint32_t* words = reinterpret_cast<uint32_t*>(&dut.in_w_code);
            int bit = i * WEIGHT_CODE_W;
            int wi = bit / 32;
            int bo = bit % 32;
            uint32_t mask = 0x3u << bo;
            uint32_t val = static_cast<uint32_t>(code) << bo;
            words[wi] = (words[wi] & ~mask) | (val & mask);
        }

        dut.in_valid = 1;
        pending.push_back({expected, PIPELINE_STAGES});
        tick(&dut);
        dut.in_valid = 0;

        // Drain one cycle, check if any output ready
        if (dut.out_valid) {
            int32_t actual = static_cast<int32_t>(dut.out_acc);
            // Sign-extend 24-bit
            if (actual & 0x00800000) actual |= 0xFF000000;
            int32_t exp_front = pending.front().expected;
            pending.erase(pending.begin());
            if (actual == exp_front) ++ok; else { ++bad;
                std::fprintf(stderr, "MISMATCH: got %d expected %d\n", actual, exp_front);
            }
        }
    }

    // Drain pipeline
    for (int i = 0; i < PIPELINE_STAGES + 2; ++i) {
        tick(&dut);
        if (dut.out_valid && !pending.empty()) {
            int32_t actual = static_cast<int32_t>(dut.out_acc);
            if (actual & 0x00800000) actual |= 0xFF000000;
            int32_t exp_front = pending.front().expected;
            pending.erase(pending.begin());
            if (actual == exp_front) ++ok; else ++bad;
        }
    }

    std::printf("bit_mac TB: ok=%d bad=%d (pipeline_drain=%zu)\n", ok, bad, pending.size());
    return bad == 0 ? 0 : 1;
}
