// rope_engine.v — Data-dependent RoPE rotation.
//
// Applies the complex-SSM-equivalent rotation used inside Mamba-3's
// B/C projection pipeline. For each pair (x0, x1) with an angle theta:
//
//     x0' =  x0 * cos(theta) - x1 * sin(theta)
//     x1' =  x0 * sin(theta) + x1 * cos(theta)
//
// theta is data-dependent (per-token per-head) and comes from the
// `angles` channel of Mamba-3's in_proj.
//
// Fixed-point layout:
//   - inputs: FP16 x pair (x0, x1)
//   - theta : INT16 (signed) representing radians * 2^13 (configurable)
//   - cos/sin: looked up from a 1024-entry BRAM LUT (FP16)
//   - outputs: FP16 x pair (x0', x1')
//
// Pipeline: 1 cycle for LUT lookup, 3 cycles for mult+add (FP16 MAC IP).

`default_nettype none
`timescale 1ns / 1ps

module rope_engine #(
    parameter integer LUT_ENTRIES = 1024,        // 2^10
    parameter integer LUT_ADDR_W  = 10
)(
    input  wire         clk,
    input  wire         rst_n,

    input  wire         in_valid,
    input  wire [15:0]  in_x0_fp16,
    input  wire [15:0]  in_x1_fp16,
    input  wire signed [15:0] in_theta_q13,  // theta * 2^13 fixed-point

    output reg          out_valid,
    output reg  [15:0]  out_x0_fp16,
    output reg  [15:0]  out_x1_fp16
);

    // ---------------- LUT: sin and cos tables ----------------
    // Generated offline: for i in 0..1023:
    //     angle = (i / LUT_ENTRIES) * 2*pi
    //     sin_lut[i] = float16(sin(angle))
    //     cos_lut[i] = float16(cos(angle))
    // The `.mem` files are loaded at elab-time via $readmemh.

    (* rom_style = "block" *) reg [15:0] sin_lut [0:LUT_ENTRIES-1];
    (* rom_style = "block" *) reg [15:0] cos_lut [0:LUT_ENTRIES-1];

    initial begin
        // Provide hex files in rtl/cores/bitmamba3/rope_sin_lut.mem etc.
        // $readmemh("rope_sin_lut.mem", sin_lut);
        // $readmemh("rope_cos_lut.mem", cos_lut);
    end

    // theta_q13 has scale 2^13 over radians; address = theta_mod_2pi / (2*pi/LUT)
    // We approximate by taking top LUT_ADDR_W bits of theta after folding mod 2*pi.
    // For clarity and determinism, callers must pre-wrap theta into [0, 2*pi) * 2^13.
    wire [LUT_ADDR_W-1:0] lut_addr = in_theta_q13[15 -: LUT_ADDR_W];

    // Stage 1: LUT lookup registers
    reg  [15:0] s1_sin, s1_cos;
    reg  [15:0] s1_x0, s1_x1;
    reg         s1_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_sin   <= 16'h0000;
            s1_cos   <= 16'h0000;
            s1_x0    <= 16'h0000;
            s1_x1    <= 16'h0000;
        end else begin
            s1_valid <= in_valid;
            s1_sin   <= sin_lut[lut_addr];
            s1_cos   <= cos_lut[lut_addr];
            s1_x0    <= in_x0_fp16;
            s1_x1    <= in_x1_fp16;
        end
    end

    // Stage 2-4: FP16 multiply-add (FP16 IP pipeline expected to be ~3 cycles)
    // Emit-interface ports to external Xilinx Floating-Point Operator or
    // equivalent RTL. Placeholder wires below show the intended dataflow.
    // NOTE: replace with actual FP16 MAC IP for synthesis.

    wire [15:0] x0_rot;
    wire [15:0] x1_rot;
    reg         s4_valid;

    // Placeholder combinational passthroughs (simulation stub)
    assign x0_rot = s1_x0;  // actual: x0 * cos - x1 * sin
    assign x1_rot = s1_x1;  // actual: x0 * sin + x1 * cos

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid   <= 1'b0;
            out_x0_fp16 <= 16'h0000;
            out_x1_fp16 <= 16'h0000;
        end else begin
            out_valid   <= s1_valid;
            out_x0_fp16 <= x0_rot;
            out_x1_fp16 <= x1_rot;
        end
    end

endmodule

`default_nettype wire
