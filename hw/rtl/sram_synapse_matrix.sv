// =============================================================================
// File: sram_synapse_matrix.sv
// Description: Dual-Port Synaptic Weight Crossbar Memory with on-chip Plasticity ALU.
// =============================================================================

include "defines.svh"

module sram_synapse_matrix #(
    parameter int NUM_PRE  = 16,
    parameter int NUM_POST = 16
)(
    input  logic                                clk,
    input  logic                                rst_n,

    // Port A: Read Interface (Spike Event Fan-Out)
    input  logic [(NUM_PRE)-1:0]          read_row,
    input  logic                                read_en,
    output q4_12_t                              read_weights [0:NUM_POST-1],
    output logic                                read_valid,

    // Port B: Write & Plasticity Interface (3-Factor R-STDP Update)
    input  logic [(NUM_PRE)-1:0]          write_row,
    input  logic                                write_en,
    input  q4_12_t                              write_weights [0:NUM_POST-1],

    // On-Chip 3-Factor Plasticity Engine Trigger
    input  logic                                plastic_update_en,
    input  logic [(NUM_PRE)-1:0]          plastic_pre_idx,
    input  q4_12_t                              dopamine_level
);

    // 16x16 SRAM Synaptic Storage Array in Q4.12 Fixed-Point
    q4_12_t weight_mem [0:NUM_PRE-1][0:NUM_POST-1];

    // Port A Read Operation (Single-cycle synchronous RAM read)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_valid <= 1'b0;
            for (int p = 0; p < NUM_POST; p++) begin
                read_weights[p] <= Q4_12_ZERO;
            end
        end else begin
            read_valid <= read_en;
            if (read_en) begin
                for (int p = 0; p < NUM_POST; p++) begin
                    read_weights[p] <= weight_mem[read_row][p];
                end
            end
        end
    end

    // Port B Write & Plasticity ALU Operation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize Naive Synaptic Weights
            for (int r = 0; r < NUM_PRE; r++) begin
                for (int c = 0; c < NUM_POST; c++) begin
                    weight_mem[r][c] <= Q4_12_ONE;
                end
            end
        end else if (write_en) begin
            for (int c = 0; c < NUM_POST; c++) begin
                weight_mem[write_row][c] <= write_weights[c];
            end
        end else if (plastic_update_en) begin
            // Hardware 3-Factor LTD: W_new = W_old - (dopamine >>> 1)
            for (int c = 0; c < NUM_POST; c++) begin
                q4_12_t delta_w;
                delta_w = (dopamine_level >>> 1); // eta = 0.5
                if (weight_mem[plastic_pre_idx][c] > delta_w) begin
                    weight_mem[plastic_pre_idx][c] <= weight_mem[plastic_pre_idx][c] - delta_w;
                end else begin
                    weight_mem[plastic_pre_idx][c] <= Q4_12_ZERO;
                end
            end
        end
    end

endmodule
