// selective_scan_mimo.v — Mamba-3 MIMO SSM state update.
//
// Sequential recurrence per token t:
//     s_t = s_{t-1} * exp(A * dt_t) + dB_t * x_t       (vector form with rank-R)
//     y_t = C_t^T * s_t + D * x_t
//
// MIMO extension: B, C carry rank dimension R (R=4 default per Mamba-3 paper).
//     s_t  shape: (H, R, D_state, D_head) FP16
//     x_t  shape: (H, D_head)             INT8 after quant
//     B_t  shape: (R, H, D_state)         FP16
//     C_t  shape: (R, H, D_state)         FP16
//     A,D  shape: (H,)                    FP32
//     dt_t shape: (H,)                    FP32
//
// State storage: on-chip BRAM (per Zybo Z7-20 budget ≈ 32 KB for small head).
// One token per STATE_CYCLES cycles; pipelined across heads.
//
// Integration notes:
//   - This module is a structural skeleton. The actual FP16 multiply-add
//     fabric is expected to be built from Xilinx Floating-Point Operator IP
//     instances driven by the state machine below.
//   - For the paper's single-board Zybo demo, we may elect to fuse this with
//     `rope_engine` and `rmsnorm_int8` into a single dataflow block rather
//     than instantiating each separately.

`default_nettype none
`timescale 1ns / 1ps

module selective_scan_mimo #(
    parameter integer N_HEADS   = 8,
    parameter integer HEAD_DIM  = 64,
    parameter integer D_STATE   = 64,
    parameter integer MIMO_RANK = 4
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // Control
    input  wire                        start,
    input  wire                        clear_state,
    output reg                         busy,
    output reg                         token_done,

    // Per-token inputs (FP16 vectors, streamed one token at a time)
    input  wire [(N_HEADS*HEAD_DIM)*16-1:0]           token_x_fp16,
    input  wire [(MIMO_RANK*N_HEADS*D_STATE)*16-1:0]  token_B_fp16,
    input  wire [(MIMO_RANK*N_HEADS*D_STATE)*16-1:0]  token_C_fp16,
    input  wire [N_HEADS*32-1:0]                       token_dt_fp32,
    input  wire [N_HEADS*32-1:0]                       token_A_fp32,
    input  wire [N_HEADS*16-1:0]                       token_D_fp16,

    // Output (attention-equivalent output)
    output reg  [(N_HEADS*HEAD_DIM)*16-1:0]            token_y_fp16
);

    // -----------------------------------------------------------
    // State storage: (H, R, D_state, D_head) FP16 = 8 * 4 * 64 * 64 * 2 = 131,072 B = 128 KB
    // This is too large for Zybo Z7-20 BRAM (630 KB total, need room for others).
    // For single-board demo reduce H or chunk D_state onto DDR with tiled fetch.
    //
    // This file declares the interface and state-machine structure only. The
    // computational datapath is to be filled in with FP16 MAC IP instances.
    // -----------------------------------------------------------

    localparam [2:0] S_IDLE = 3'd0,
                     S_LOAD = 3'd1,
                     S_COMPUTE_EXPAND = 3'd2,
                     S_COMPUTE_Y = 3'd3,
                     S_STORE = 3'd4,
                     S_DONE = 3'd5;

    reg [2:0] state, next_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE: if (start) next_state = S_LOAD;
            S_LOAD: next_state = S_COMPUTE_EXPAND;
            S_COMPUTE_EXPAND: next_state = S_COMPUTE_Y;
            S_COMPUTE_Y: next_state = S_STORE;
            S_STORE: next_state = S_DONE;
            S_DONE:  next_state = S_IDLE;
            default: next_state = S_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy       <= 1'b0;
            token_done <= 1'b0;
            token_y_fp16 <= {(N_HEADS*HEAD_DIM*16){1'b0}};
        end else begin
            busy       <= (state != S_IDLE);
            token_done <= (state == S_DONE);
            // Datapath fills token_y_fp16 during S_COMPUTE_Y.
        end
    end

endmodule

`default_nettype wire
