// =============================================================================
// File: defines.svh
// Description: SystemVerilog definitions for Ant Neuromorphic SNN Processor.
// Standards: IEEE 1800-2017 Synthesizable SystemVerilog
// =============================================================================

ifndef ANT_NEUROMORPHIC_DEFINES_SVH
define ANT_NEUROMORPHIC_DEFINES_SVH

// 16-bit Signed Fixed-Point Arithmetic (Q4.12 Format)
// 1 sign bit, 3 integer bits, 12 fractional bits
// Range: [-8.0, +7.999755859375], Resolution: 1/4096 = 0.000244140625
typedef logic signed [15:0] q4_12_t;

// Fixed-Point Mathematical Constants
localparam q4_12_t Q4_12_ZERO    = 16'sh0000;         //  0.0
localparam q4_12_t Q4_12_ONE     = 16'sh1000;         // +1.0 (4096)
localparam q4_12_t Q4_12_HALF    = 16'sh0800;         // +0.5 (2048)
localparam q4_12_t Q4_12_V_REST  = 16'sh0000;         //  0.0 (Normalized resting potential)
localparam q4_12_t Q4_12_V_RESET = -16'sh0400;        // -0.25 (Reset potential)
localparam q4_12_t Q4_12_V_TH    = 16'sh1000;         // +1.0 (Spiking threshold)

// Core Hardware Sizing Constants
localparam int NUM_CORES         = 4;                 // 0: PB Compass, 1: FB Integrator, 2: MB KC, 3: MBON/Steering
localparam int NEURONS_PER_CORE  = 16;                // 16 vectorized neurons per execution lane
localparam int FIFO_DEPTH        = 16;                // Circular event FIFO depth
localparam int LEAK_BITSHIFT     = 4;                 // Leak decay: V <= V - (V >>> 4) => 15/16 decay (0.9375)
localparam int REFRACTORY_CYCLES = 2;                 // Refractory timer clock cycles

// Address-Event Representation (AER) Spike Packet (25 bits packed)
typedef struct packed {
    logic [3:0]  core_id;      // Destination Core ID [0..15]
    logic [9:0]  neuron_id;    // Destination Neuron Index [0..1023]
    logic [9:0]  timestamp;    // Event timestamp (millisecond counter)
    logic        spike_val;    // Binary event trigger (1 = spike)
} aer_packet_t;

endif // ANT_NEUROMORPHIC_DEFINES_SVH
