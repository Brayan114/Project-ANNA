// =============================================================================
// File: ant_neuromorphic_top.sv
// Description: Top-level SoC integrating AER Router, LIF NPUs, and Synaptic SRAM.
// =============================================================================

include "defines.svh"

module ant_neuromorphic_top #(
    parameter int NUM_CORES        = 4,
    parameter int NEURONS_PER_CORE = 16
)(
    input  logic                                clk,
    input  logic                                rst_n,

    // Ingress AER Packet Interface (External sensors / host)
    input  aer_packet_t                         in_aer_packet,
    input  logic                                in_aer_valid,
    output logic                                in_aer_ready,

    // Master Biological Timestep Tick (1 kHz / 1 ms default)
    input  logic                                step_tick,

    // Egress Spiking Interface (Motor command / spike readout)
    output logic [NEURONS_PER_CORE-1:0]         core_spikes [0:NUM_CORES-1],
    output aer_packet_t                         egress_aer_packet,
    output logic                                egress_aer_valid
);

    // Internal Router Connections
    aer_packet_t routed_packet [0:NUM_CORES-1];
    logic        routed_valid  [0:NUM_CORES-1];
    logic        routed_ready  [0:NUM_CORES-1];

    // Instantiate Ingress AER Packet Router
    aer_router #(
        .FIFO_DEPTH(16),
        .NUM_CORES(NUM_CORES)
    ) u_router (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_packet  (in_aer_packet),
        .in_valid   (in_aer_valid),
        .in_ready   (in_aer_ready),
        .out_packet (routed_packet),
        .out_valid  (routed_valid),
        .out_ready  (routed_ready)
    );

    // Core Interconnect and Synaptic Crossbars
    genvar c;
    generate
        for (c = 0; c < NUM_CORES; c++) begin : gen_cores

            q4_12_t syn_drive [0:NEURONS_PER_CORE-1];
            logic   syn_valid;

            // Instantiate Synaptic Crossbar Memory for each Core
            sram_synapse_matrix #(
                .NUM_PRE(NEURONS_PER_CORE),
                .NUM_POST(NEURONS_PER_CORE)
            ) u_sram (
                .clk               (clk),
                .rst_n             (rst_n),
                .read_row          (routed_packet[c].neuron_id[(NEURONS_PER_CORE)-1:0]),
                .read_en           (routed_valid[c]),
                .read_weights      (syn_drive),
                .read_valid        (syn_valid),
                .write_row         ('0),
                .write_en          (1'b0),
                .write_weights     ('{default: Q4_12_ZERO}),
                .plastic_update_en (1'b0),
                .plastic_pre_idx   ('0),
                .dopamine_level    (Q4_12_ZERO)
            );

            assign routed_ready[c] = 1'b1;

            // Instantiate Vectorized LIF Neuron Core
            lif_neuron_core #(
                .NEURONS_PER_CORE(NEURONS_PER_CORE),
                .LEAK_BITSHIFT(4),
                .REFRACTORY_CYCLES(2)
            ) u_npu (
                .clk         (clk),
                .rst_n       (rst_n),
                .leak_tick   (step_tick),
                .syn_current (syn_drive),
                .syn_valid   (syn_valid),
                .out_spikes  (core_spikes[c]),
                .v_mem_out   ()
            );

        end
    endgenerate

    // Egress Packet Generation (Broadcast Core 0 Spikes for Readout)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            egress_aer_valid <= 1'b0;
            egress_aer_packet <= '0;
        end else if (|core_spikes[0]) begin
            egress_aer_valid <= 1'b1;
            egress_aer_packet.core_id   <= 4'd0;
            egress_aer_packet.neuron_id <= 10'd0;
            egress_aer_packet.timestamp <= 10'd0;
            egress_aer_packet.spike_val <= 1'b1;
        end else begin
            egress_aer_valid <= 1'b0;
        end
    end

endmodule
