// =============================================================================
// File: aer_router.sv
// Description: Address-Event Representation (AER) Spike Packet Router with FIFO queue.
// =============================================================================

include "defines.svh"

module aer_router #(
    parameter int FIFO_DEPTH = 16,
    parameter int NUM_CORES  = 4
)(
    input  logic              clk,
    input  logic              rst_n,

    // Ingress AER Packet Interface
    input  aer_packet_t       in_packet,
    input  logic              in_valid,
    output logic              in_ready,

    // Egress Core Routing Interfaces
    output aer_packet_t       out_packet [0:NUM_CORES-1],
    output logic              out_valid  [0:NUM_CORES-1],
    input  logic              out_ready  [0:NUM_CORES-1]
);

    // Circular FIFO Queue Storage
    aer_packet_t fifo_mem [0:FIFO_DEPTH-1];
    logic [(FIFO_DEPTH):0] wr_ptr;
    logic [(FIFO_DEPTH):0] rd_ptr;
    logic [(FIFO_DEPTH):0] count;

    wire fifo_full  = (count == FIFO_DEPTH);
    wire fifo_empty = (count == 0);

    assign in_ready = !fifo_full;

    // FIFO Write Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (in_valid && in_ready) begin
            fifo_mem[wr_ptr[(FIFO_DEPTH)-1:0]] <= in_packet;
            wr_ptr <= wr_ptr + 1'b1;
        end
    end

    // Routing and Dispatch Logic
    aer_packet_t current_packet;
    assign current_packet = fifo_mem[rd_ptr[(FIFO_DEPTH)-1:0]];

    logic [3:0] target_core;
    assign target_core = current_packet.core_id;

    always_comb begin
        for (int i = 0; i < NUM_CORES; i++) begin
            out_packet[i] = current_packet;
            out_valid[i]  = (!fifo_empty && (target_core == i[3:0]));
        end
    end

    // FIFO Read Logic
    wire dispatch_accepted = (!fifo_empty && (target_core < NUM_CORES) && out_ready[target_core]);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            case ({in_valid && in_ready, dispatch_accepted})
                2'b10: count <= count + 1'b1;
                2'b01: begin
                    rd_ptr <= rd_ptr + 1'b1;
                    count  <= count - 1'b1;
                end
                2'b11: begin
                    rd_ptr <= rd_ptr + 1'b1;
                end
                default: ;
            endcase
        end
    end

endmodule
