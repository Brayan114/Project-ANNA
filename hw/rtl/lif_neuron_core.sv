// =============================================================================
// File: lif_neuron_core.sv
// Description: Vectorized 16-channel Digital Leaky Integrate-and-Fire (LIF) NPU Core.
// =============================================================================

include "defines.svh"

module lif_neuron_core #(
    parameter int NEURONS_PER_CORE  = 16,
    parameter int LEAK_BITSHIFT     = 4,
    parameter int REFRACTORY_CYCLES = 2
)(
    input  logic                                clk,
    input  logic                                rst_n,

    // Step Timing Tick
    input  logic                                leak_tick,

    // Synaptic Driving Current Inputs (Q4.12 format)
    input  q4_12_t                              syn_current [0:NEURONS_PER_CORE-1],
    input  logic                                syn_valid,

    // Spike Event Outputs
    output logic [NEURONS_PER_CORE-1:0]         out_spikes,
    output q4_12_t                              v_mem_out   [0:NEURONS_PER_CORE-1]
);

    // Neuron State Storage Registers
    q4_12_t v_mem     [0:NEURONS_PER_CORE-1];
    logic [3:0] ref_timer [0:NEURONS_PER_CORE-1];

    // Combinational Next-State Signals
    q4_12_t v_leak    [0:NEURONS_PER_CORE-1];
    q4_12_t v_next    [0:NEURONS_PER_CORE-1];
    logic   spiked    [0:NEURONS_PER_CORE-1];

    genvar i;
    generate
        for (i = 0; i < NEURONS_PER_CORE; i++) begin : gen_neuron_lane

            // 1. Bit-Shift Exponential Leak Approximation: V_leak = V - (V >>> LEAK_BITSHIFT)
            assign v_leak[i] = v_mem[i] - (v_mem[i] >>> LEAK_BITSHIFT);

            // 2. Integration with incoming synaptic current
            always_comb begin
                if (leak_tick) begin
                    v_next[i] = v_leak[i] + (syn_valid ? syn_current[i] : Q4_12_ZERO);
                end else begin
                    v_next[i] = v_mem[i] + (syn_valid ? syn_current[i] : Q4_12_ZERO);
                end
            end

            // 3. Threshold Comparator
            assign spiked[i] = (v_next[i] >= Q4_12_V_TH) && (ref_timer[i] == '0);
            assign out_spikes[i] = spiked[i];
            assign v_mem_out[i]  = v_mem[i];

            // 4. Sequential State Register Updates
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    v_mem[i]     <= Q4_12_V_REST;
                    ref_timer[i] <= '0;
                end else begin
                    // Decrement refractory timer if active
                    if (leak_tick && (ref_timer[i] > '0)) begin
                        ref_timer[i] <= ref_timer[i] - 1'b1;
                    end

                    // Membrane voltage update and reset logic
                    if (spiked[i]) begin
                        v_mem[i]     <= Q4_12_V_RESET;
                        ref_timer[i] <= REFRACTORY_CYCLES[3:0];
                    end else if (ref_timer[i] == '0) begin
                        v_mem[i]     <= v_next[i];
                    end
                end
            end

        end
    endgenerate

endmodule
