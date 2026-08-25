"""
Benchmark: Neuromorphic Spiking Efficiency vs. Dense Recurrent Baselines (LSTM / RNN).
Profiles Synaptic Operations (SynOps), Floating Point Operations (FLOPs), and Energy consumption.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from src.central_complex.cx_agent import CentralComplexAgent, AgentMode
from env.arena import AntArena2D


def benchmark_efficiency(n_steps: int = 2000):
    print(f'Running Neuromorphic Efficiency Benchmark ({n_steps} timesteps)...')

    agent = CentralComplexAgent(n_columns=16, dt=1.0, default_speed=1.0)
    arena = AntArena2D(nest_pos=(0.0, 0.0), dt=0.05)

    agent.reset(initial_heading=0.0)
    arena.reset(initial_heading=0.0)

    t0 = time.perf_counter()
    total_compass_spikes = 0
    total_integrator_spikes = 0

    for step in range(n_steps):
        speed_cmd, ang_cmd, diag = agent.step(
            celestial_heading=arena.heading,
            celestial_confidence=1.0,
            forward_speed_sensor=1.0,
            angular_velocity_sensor=0.05,
            foraging_turn_bias=0.02,
            dt_phys=arena.dt,
        )
        arena.step(speed_cmd, ang_cmd)
        total_compass_spikes += diag['compass_spikes']
        total_integrator_spikes += diag['integrator_spikes']

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    compass_synops = total_compass_spikes * 16
    integrator_synops = total_integrator_spikes * 16
    total_synops = compass_synops + integrator_synops

    lstm_flops_per_step = 17408
    total_lstm_flops = lstm_flops_per_step * n_steps

    snn_energy_uJ = (total_synops * 0.5e-12) * 1e6
    lstm_energy_uJ = (total_lstm_flops * 5.0e-12) * 1e6

    print('')
    print('================ COMPUTATIONAL EFFICIENCY BENCHMARK ================')
    print(f'Total Timesteps Simulated:      {n_steps}')
    print(f'Wall-Clock Execution Time:      {elapsed_ms:.2f} ms ({elapsed_ms/n_steps:.3f} ms/step)')
    print(f'Total SNN Spikes Emitted:       {total_compass_spikes + total_integrator_spikes}')
    print(f'Spike Sparsity Rate:            {100.0 * (1.0 - (total_compass_spikes + total_integrator_spikes)/(32 * n_steps)):.2f}%')
    print(f'Total Neuromorphic SynOps:      {total_synops:,} SynOps')
    print(f'Dense LSTM Baseline FLOPs:      {total_lstm_flops:,} FLOPs')
    print(f'Computational Operation Savings:{100.0 * (1.0 - total_synops / total_lstm_flops):.2f}%')
    print(f'Estimated SNN Energy (0.5pJ):   {snn_energy_uJ:.4f} microJoules')
    print(f'Estimated LSTM Energy (5.0pJ):  {lstm_energy_uJ:.4f} microJoules')
    print(f'Energy Efficiency Advantage:    {lstm_energy_uJ / (snn_energy_uJ + 1e-9):.1f}x lower energy')
    print('====================================================================')
    print('')


if __name__ == '__main__':
    benchmark_efficiency()
