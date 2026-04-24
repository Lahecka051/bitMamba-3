// mimo_matmul.v — MIMO projection tensor einsum in fixed-point.
//
// Applies mimo_x / mimo_z / mimo_o projections from the Mamba-3 MIMO path:
//     z_r  = einsum("bhp,rhp->brhp", z, mimo_z)
//     y    = einsum("brhp,rhp->bhp",  y, mimo_o)
//
// mimo_* parameters have shape (H, R, D_head). In the default Mamba-3 MIMO
// configuration they are kept in FP16/FP32 (not ternarized). A ternarization
// ablation can be added by swapping in a 2-bit encoded weight path.
//
// Interface: LANES-wide streaming, one token at a time.

`default_nettype none
`timescale 1ns / 1ps

module mimo_matmul #(
    parameter integer N_HEADS  = 8,
    parameter integer HEAD_DIM = 64,
    parameter integer MIMO_RANK = 4
)(
    input  wire                                       clk,
    input  wire                                       rst_n,

    input  wire                                       in_valid,
    input  wire [(N_HEADS*HEAD_DIM)*16-1:0]           in_bhp_fp16,
    input  wire [(N_HEADS*MIMO_RANK*HEAD_DIM)*16-1:0] in_weight_rhp_fp16,
    output wire                                       in_ready,

    output reg                                        out_valid,
    output reg  [(MIMO_RANK*N_HEADS*HEAD_DIM)*16-1:0] out_brhp_fp16
);

    assign in_ready = 1'b1;

    // Placeholder interface. The einsum reduces over the shared H and D_head
    // axes using FP16 MAC IP. For Zybo Z7-20, plan is:
    //   - 32 FP16 MAC units × 8 cycles = 256 op/cycle effective
    //   - Throughput: 1 token per ~N_HEADS*MIMO_RANK cycles = 32 cycles
    //
    // This stub establishes port shapes for integration with the top block.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid     <= 1'b0;
            out_brhp_fp16 <= {(MIMO_RANK*N_HEADS*HEAD_DIM*16){1'b0}};
        end else begin
            out_valid <= in_valid;
        end
    end

endmodule

`default_nettype wire
