// bit_mac.v — Ternary weight × INT8 activation MAC unit for BitMamba-3.
//
// Matches PyTorch BitLinear semantics (src/bitmamba3/bitlinear.py) bit-exact
// at the fixed-point level: weight ∈ {-1, 0, +1}, activation ∈ [-128, 127],
// partial sum held as signed INT24 for 128-wide dot product headroom
// (worst-case: 128 × 127 = 16,256 → fits in INT16, but INT24 leaves margin
// for wider future reductions).
//
// Weight encoding (2 bits per weight, packed 8 weights per byte):
//   00 -> 0  (skip add)
//   01 -> +1 (add activation)
//   10 -> -1 (subtract activation)
//   11 -> reserved (treat as 0)
//
// Interface: handshaked streaming, one 128-lane vector per cycle after
// pipeline fill. Three-stage pipeline: decode -> conditional add -> reduce.

`default_nettype none
`timescale 1ns / 1ps

module bit_mac #(
    parameter integer LANES       = 128,       // width of single MAC vector
    parameter integer ACT_W       = 8,         // activation bit width (INT8)
    parameter integer WEIGHT_CODE_W = 2,       // 2 bits per ternary weight
    parameter integer ACC_W       = 24         // accumulator bit width
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          in_valid,
    input  wire [LANES*ACT_W-1:0]        in_act,      // LANES x INT8
    input  wire [LANES*WEIGHT_CODE_W-1:0] in_w_code,  // LANES x 2-bit ternary
    output wire                          in_ready,    // always 1 (no backpressure)

    output reg                           out_valid,
    output reg  signed [ACC_W-1:0]       out_acc      // signed sum
);

    assign in_ready = 1'b1;

    // -------------------------------------------------------------
    // Stage 1: per-lane conditional add/sub based on weight code
    // -------------------------------------------------------------
    reg                            s1_valid;
    reg signed [ACT_W:0]           s1_addend [0:LANES-1];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            for (i = 0; i < LANES; i = i + 1) s1_addend[i] <= {(ACT_W+1){1'b0}};
        end else begin
            s1_valid <= in_valid;
            for (i = 0; i < LANES; i = i + 1) begin
                case (in_w_code[i*WEIGHT_CODE_W +: WEIGHT_CODE_W])
                    2'b00: s1_addend[i] <= {(ACT_W+1){1'b0}};
                    2'b01: s1_addend[i] <= {{1{in_act[i*ACT_W + ACT_W-1]}}, in_act[i*ACT_W +: ACT_W]}; // +act
                    2'b10: s1_addend[i] <= -{{1{in_act[i*ACT_W + ACT_W-1]}}, in_act[i*ACT_W +: ACT_W]}; // -act
                    default: s1_addend[i] <= {(ACT_W+1){1'b0}};
                endcase
            end
        end
    end

    // -------------------------------------------------------------
    // Stage 2: pairwise reductions (binary tree, combinational inside stage)
    // -------------------------------------------------------------
    // For LANES=128 we reduce in log2(128)=7 levels; fold into ~3 register stages
    // by grouping. Here for simplicity and to match prior GGUF Q4_0 MAC design
    // we do 2-level summation.

    localparam integer GROUP        = 16;
    localparam integer N_GROUPS     = LANES / GROUP;
    localparam integer STAGE2_W     = ACT_W + 1 + $clog2(GROUP);

    reg                             s2_valid;
    reg signed [STAGE2_W-1:0]       s2_group_sum [0:N_GROUPS-1];

    integer g, k;
    reg signed [STAGE2_W-1:0]       group_acc;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            for (g = 0; g < N_GROUPS; g = g + 1) s2_group_sum[g] <= {STAGE2_W{1'b0}};
        end else begin
            s2_valid <= s1_valid;
            for (g = 0; g < N_GROUPS; g = g + 1) begin
                group_acc = {STAGE2_W{1'b0}};
                for (k = 0; k < GROUP; k = k + 1) begin
                    group_acc = group_acc + s1_addend[g*GROUP + k];
                end
                s2_group_sum[g] <= group_acc;
            end
        end
    end

    // -------------------------------------------------------------
    // Stage 3: final reduction across groups
    // -------------------------------------------------------------
    reg signed [ACC_W-1:0]          s3_total;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_total  <= {ACC_W{1'b0}};
            out_valid <= 1'b0;
        end else begin
            out_valid <= s2_valid;
            begin : final_sum
                integer gg;
                reg signed [ACC_W-1:0] total;
                total = {ACC_W{1'b0}};
                for (gg = 0; gg < N_GROUPS; gg = gg + 1) begin
                    total = total + $signed(s2_group_sum[gg]);
                end
                s3_total <= total;
            end
        end
    end

    always @(*) out_acc = s3_total;

endmodule

`default_nettype wire
