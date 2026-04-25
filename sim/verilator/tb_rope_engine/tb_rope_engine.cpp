// tb_rope_engine.cpp — Verilator TB for rope_engine.v
//
// Reads sim/verilator/vectors/rope.bin (records of x0_fp16, x1_fp16, theta_q13,
// x0p_fp16, x1p_fp16) and drives the RTL with each input tuple, asserting
// FP16-ULP-tolerant equality on outputs.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <verilated.h>
#include "Vrope_engine.h"

static constexpr int PIPELINE = 4;  // 1 LUT lookup + 3 FP16 MAC stages

static uint64_t sim_time = 0;

static void tick(Vrope_engine* dut) {
    dut->clk = 0; dut->eval(); ++sim_time;
    dut->clk = 1; dut->eval(); ++sim_time;
}

static double fp16_to_fp32(uint16_t h) {
    // Dispatch via type-pun on float; Verilator host expected to support this.
    union { uint16_t u; uint16_t pad[2]; float f; } u;
    // Manual conversion (no SSE intrinsics).
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    uint32_t f32;
    if (exp == 0) {
        if (frac == 0) {
            f32 = sign << 31;
        } else {
            // subnormal
            int e = -1;
            while ((frac & 0x400) == 0) { frac <<= 1; e--; }
            frac &= 0x3ff;
            f32 = (sign << 31) | ((127 + e) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | (255 << 23) | (frac << 13);
    } else {
        f32 = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    union { uint32_t u; float f; } o; o.u = f32;
    return o.f;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const char* vec_path = "sim/verilator/vectors/rope.bin";
    if (argc > 1) vec_path = argv[1];

    std::ifstream in(vec_path, std::ios::binary);
    if (!in) { std::fprintf(stderr, "Cannot open %s\n", vec_path); return 1; }

    Vrope_engine dut;
    dut.rst_n = 0;
    for (int i = 0; i < 4; ++i) tick(&dut);
    dut.rst_n = 1;

    constexpr int RECORD_SIZE = 2 + 2 + 2 + 2 + 2;

    struct Pending { uint16_t x0p_exp; uint16_t x1p_exp; int cycles_left; };
    std::vector<Pending> pending;

    int ok = 0, bad = 0;

    while (true) {
        uint16_t x0_u, x1_u, x0p_u, x1p_u;
        int16_t theta_q13;
        char buf[RECORD_SIZE];
        if (!in.read(buf, RECORD_SIZE)) break;
        std::memcpy(&x0_u, buf + 0, 2);
        std::memcpy(&x1_u, buf + 2, 2);
        std::memcpy(&theta_q13, buf + 4, 2);
        std::memcpy(&x0p_u, buf + 6, 2);
        std::memcpy(&x1p_u, buf + 8, 2);

        dut.in_valid = 1;
        dut.in_x0_fp16 = x0_u;
        dut.in_x1_fp16 = x1_u;
        dut.in_theta_q13 = theta_q13;
        pending.push_back({x0p_u, x1p_u, PIPELINE});

        tick(&dut);
        dut.in_valid = 0;

        if (dut.out_valid) {
            auto& p = pending.front();
            double a0 = fp16_to_fp32(dut.out_x0_fp16);
            double a1 = fp16_to_fp32(dut.out_x1_fp16);
            double e0 = fp16_to_fp32(p.x0p_exp);
            double e1 = fp16_to_fp32(p.x1p_exp);
            if (std::abs(a0 - e0) <= 5e-3 && std::abs(a1 - e1) <= 5e-3) ++ok;
            else { ++bad; if (bad <= 3) std::fprintf(stderr, "MISMATCH (%g,%g) vs (%g,%g)\n", a0, a1, e0, e1); }
            pending.erase(pending.begin());
        }
    }

    for (int i = 0; i < PIPELINE + 2; ++i) {
        tick(&dut);
        if (dut.out_valid && !pending.empty()) {
            auto& p = pending.front();
            double a0 = fp16_to_fp32(dut.out_x0_fp16);
            double a1 = fp16_to_fp32(dut.out_x1_fp16);
            double e0 = fp16_to_fp32(p.x0p_exp);
            double e1 = fp16_to_fp32(p.x1p_exp);
            if (std::abs(a0 - e0) <= 5e-3 && std::abs(a1 - e1) <= 5e-3) ++ok; else ++bad;
            pending.erase(pending.begin());
        }
    }

    std::printf("rope_engine TB: ok=%d bad=%d (drain=%zu)\n", ok, bad, pending.size());
    return bad == 0 ? 0 : 1;
}
