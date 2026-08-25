"""
Cycle-Accurate Digital Hardware Emulator for Ant Neuromorphic SNN Processor.
Replicates bit-exact SystemVerilog RTL registers, Q4.12 fixed-point integer math, and AER routing.
"""
import numpy as np
from typing import List, Dict, Any, Tuple


class Q4_12:
    """16-bit Signed Q4.12 Fixed-Point Converter and ALU Helpers."""
    SCALE = 4096.0  # 2^12
    MIN_VAL = -32768
    MAX_VAL = 32767

    @staticmethod
    def float_to_fixed(val: float) -> int:
        clamped = max(-8.0, min(7.999755859375, float(val)))
        return int(np.clip(int(np.round(clamped * Q4_12.SCALE)), Q4_12.MIN_VAL, Q4_12.MAX_VAL))

    @staticmethod
    def fixed_to_float(raw: int) -> float:
        return float(np.int16(raw)) / Q4_12.SCALE


class AntNeuromorphicChipEmulator:
    """
    Cycle-Accurate Register-Transfer Level (RTL) Hardware Emulator.
    """

    def __init__(
        self,
        num_cores: int = 4,
        neurons_per_core: int = 16,
        leak_bitshift: int = 4,
        refractory_cycles: int = 2,
    ):
        self.num_cores = num_cores
        self.neurons_per_core = neurons_per_core
        self.leak_bitshift = leak_bitshift
        self.refractory_cycles = refractory_cycles

        # Fixed-point thresholds in raw integer Q4.12
        self.v_rest = Q4_12.float_to_fixed(0.0)
        self.v_reset = Q4_12.float_to_fixed(-0.25)
        self.v_thresh = Q4_12.float_to_fixed(1.0)

        # Hardware Register Arrays (int16 signed)
        self.v_mem = np.full((num_cores, neurons_per_core), self.v_rest, dtype=np.int16)
        self.ref_timer = np.zeros((num_cores, neurons_per_core), dtype=np.uint8)
        self.out_spikes = np.zeros((num_cores, neurons_per_core), dtype=np.uint8)

        # 16x16 SRAM Synaptic Matrices per Core: Shape (num_cores, 16, 16)
        self.sram_weights = np.full((num_cores, neurons_per_core, neurons_per_core), Q4_12.float_to_fixed(1.0), dtype=np.int16)

        # AER Circular FIFO
        self.aer_fifo: List[Tuple[int, int, int]] = []  # (core_id, neuron_id, timestamp)
        self.fifo_depth = 16

        # Cycle and operation counters
        self.clock_cycles = 0
        self.total_synops = 0

    def reset(self) -> None:
        self.v_mem.fill(self.v_rest)
        self.ref_timer.fill(0)
        self.out_spikes.fill(0)
        self.sram_weights.fill(Q4_12.float_to_fixed(1.0))
        self.aer_fifo.clear()
        self.clock_cycles = 0
        self.total_synops = 0

    def write_sram_row(self, core_id: int, row_idx: int, weights_float: np.ndarray) -> None:
        """Write a row of synaptic weights into on-chip SRAM."""
        for c in range(self.neurons_per_core):
            self.sram_weights[core_id, row_idx, c] = Q4_12.float_to_fixed(weights_float[c])

    def inject_aer_packet(self, core_id: int, neuron_id: int, timestamp: int = 0) -> bool:
        """Inject an incoming AER spike packet into the router FIFO."""
        if len(self.aer_fifo) < self.fifo_depth:
            self.aer_fifo.append((core_id, neuron_id, timestamp))
            return True
        return False

    def step_clock(self, leak_tick: bool = True) -> np.ndarray:
        """
        Execute one hardware clock cycle across all NPU cores.
        """
        self.clock_cycles += 1
        self.out_spikes.fill(0)

        # 1. AER Router Dispatch
        syn_drive = np.zeros((self.num_cores, self.neurons_per_core), dtype=np.int32)
        if len(self.aer_fifo) > 0:
            target_core, pre_neuron, _ = self.aer_fifo.pop(0)
            if target_core < self.num_cores:
                # Read row from SRAM
                row_weights = self.sram_weights[target_core, pre_neuron % self.neurons_per_core]
                syn_drive[target_core] += row_weights
                self.total_synops += self.neurons_per_core

        # 2. Vectorized LIF Core Execution
        for c in range(self.num_cores):
            for i in range(self.neurons_per_core):
                curr_v = int(self.v_mem[c, i])

                # Bit-shift leak: V_leak = V - (V >> 4)
                if leak_tick:
                    v_leak = curr_v - (curr_v >> self.leak_bitshift)
                    v_next = v_leak + syn_drive[c, i]
                    if self.ref_timer[c, i] > 0:
                        self.ref_timer[c, i] -= 1
                else:
                    v_next = curr_v + syn_drive[c, i]

                # Threshold comparator
                if v_next >= self.v_thresh and self.ref_timer[c, i] == 0:
                    self.out_spikes[c, i] = 1
                    self.v_mem[c, i] = np.int16(self.v_reset)
                    self.ref_timer[c, i] = self.refractory_cycles
                elif self.ref_timer[c, i] == 0:
                    self.v_mem[c, i] = np.int16(np.clip(v_next, Q4_12.MIN_VAL, Q4_12.MAX_VAL))

        return self.out_spikes.copy()

    def get_membrane_potentials(self, core_id: int) -> np.ndarray:
        """Return floating-point membrane potentials for diagnostics."""
        return np.array([Q4_12.fixed_to_float(raw) for raw in self.v_mem[core_id]])
