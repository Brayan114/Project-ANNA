import numpy as np
import pytest
from hw.sim.chip_emulator import AntNeuromorphicChipEmulator, Q4_12

def test_q4_12_precision():
    # Test float to fixed-point conversion
    assert Q4_12.float_to_fixed(0.0) == 0
    assert Q4_12.float_to_fixed(1.0) == 4096
    assert Q4_12.float_to_fixed(-0.25) == -1024
    
    # Test roundtrip accuracy within quantization resolution (1/4096 = 0.000244)
    val = 0.625
    assert np.isclose(Q4_12.fixed_to_float(Q4_12.float_to_fixed(val)), val, atol=1e-3)

def test_hardware_leak_and_threshold_firing():
    chip = AntNeuromorphicChipEmulator(num_cores=4, neurons_per_core=16)
    chip.reset()
    
    # 1. Inject strong current into Core 0 Neuron 0
    chip.v_mem[0, 0] = Q4_12.float_to_fixed(0.9)
    
    # Inject AER packet targeting Core 0, Neuron 0
    chip.inject_aer_packet(core_id=0, neuron_id=0)
    
    # Step clock: should trigger spike and reset
    spikes = chip.step_clock(leak_tick=True)
    assert spikes[0, 0] == 1
    assert chip.v_mem[0, 0] == chip.v_reset
    assert chip.ref_timer[0, 0] == 2

def test_aer_router_and_sram_broadcast():
    chip = AntNeuromorphicChipEmulator(num_cores=4, neurons_per_core=16)
    chip.reset()
    
    # Write custom row weights to SRAM Core 1 Row 3
    custom_weights = np.full(16, 0.1)
    chip.write_sram_row(core_id=1, row_idx=3, weights_float=custom_weights)
    
    # Inject spike packet to Core 1 Row 3
    chip.inject_aer_packet(core_id=1, neuron_id=3)
    
    # Clock tick: Core 1 neurons should integrate the 0.1 weight
    chip.step_clock(leak_tick=False)
    v_pot = chip.get_membrane_potentials(core_id=1)
    
    assert np.allclose(v_pot, 0.1, atol=1e-3)
    assert chip.total_synops == 16
