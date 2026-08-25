"""
Experiment 1: Desert Ant Foraging and Closed-Loop Neuromorphic Path Integration.
Validates zero-drift vector homing and generates publication-grade multi-panel figures.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.central_complex.cx_agent import CentralComplexAgent, AgentMode
from env.arena import AntArena2D
from env.sensors import CelestialCompassSensor, OpticFlowSensor


def run_experiment(seed: int = 42, outbound_steps: int = 200, homing_steps: int = 400):
    np.random.seed(seed)
    os.makedirs('figures', exist_ok=True)

    # Initialize environment and sensors
    arena = AntArena2D(nest_pos=(0.0, 0.0), dt=0.05)
    compass_sensor = CelestialCompassSensor(noise_std=0.02)
    flow_sensor = OpticFlowSensor(speed_noise_std=0.01, gyro_noise_std=0.01)

    # Initialize Neuromorphic Central Complex Spiking Agent
    agent = CentralComplexAgent(n_columns=16, dt=1.0, steer_gain=4.5, default_speed=1.2)

    initial_heading = np.random.uniform(0, 2 * np.pi)
    agent.reset(initial_heading=initial_heading)
    arena.reset(initial_heading=initial_heading)

    history = {
        'traj': [],
        'headings': [],
        'compass_bumps': [],
        'integrator_states': [],
        'compass_spikes': [],
        'integrator_spikes': [],
        'modes': [],
    }

    print(f'=== Starting Outbound Foraging Phase ({outbound_steps} steps) ===')
    turn_noise = 0.0

    for step in range(outbound_steps):
        turn_noise = 0.85 * turn_noise + np.random.normal(0.0, 0.4)

        meas_heading, conf = compass_sensor.read(arena.heading)
        meas_speed, meas_gyro = flow_sensor.read(agent.default_speed, turn_noise)

        speed_cmd, ang_cmd, diag = agent.step(
            celestial_heading=meas_heading,
            celestial_confidence=conf,
            forward_speed_sensor=meas_speed,
            angular_velocity_sensor=meas_gyro,
            foraging_turn_bias=turn_noise,
            dt_phys=arena.dt,
        )

        arena.step(speed_cmd, ang_cmd)

        history['traj'].append(np.copy(arena.pos))
        history['headings'].append(arena.heading)
        history['compass_bumps'].append(np.copy(agent.compass.activity_trace))
        history['integrator_states'].append(np.copy(agent.integrator.accumulated_memory))
        history['compass_spikes'].append(np.copy(agent.compass.neurons.spikes))
        history['integrator_spikes'].append(np.copy(agent.integrator.neurons.spikes))
        history['modes'].append(1)

    food_pos = np.copy(arena.pos)
    food_dist = float(np.linalg.norm(food_pos))
    print(f'Food discovered at: {food_pos}, Distance from nest: {food_dist:.2f} m')

    print(f'=== Starting Closed-Loop Homing Phase ({homing_steps} steps) ===')
    agent.set_mode(AgentMode.HOMING)
    homing_start_idx = len(arena.trajectory)

    for step in range(homing_steps):
        meas_heading, conf = compass_sensor.read(arena.heading)
        meas_speed, meas_gyro = flow_sensor.read(agent.default_speed, 0.0)

        speed_cmd, ang_cmd, diag = agent.step(
            celestial_heading=meas_heading,
            celestial_confidence=conf,
            forward_speed_sensor=meas_speed,
            angular_velocity_sensor=ang_cmd,
            dt_phys=arena.dt,
        )

        arena.step(speed_cmd, ang_cmd)

        history['traj'].append(np.copy(arena.pos))
        history['headings'].append(arena.heading)
        history['compass_bumps'].append(np.copy(agent.compass.activity_trace))
        history['integrator_states'].append(np.copy(agent.integrator.accumulated_memory))
        history['compass_spikes'].append(np.copy(agent.compass.neurons.spikes))
        history['integrator_spikes'].append(np.copy(agent.integrator.neurons.spikes))
        history['modes'].append(2)

        dist_nest = np.linalg.norm(arena.pos - arena.nest_pos)
        if dist_nest < 0.15:
            print(f'>>> Nest reached successfully at homing step {step}! Final dist: {dist_nest:.3f} m')
            break

    # Calculate metrics
    metrics = arena.get_path_metrics(homing_start_idx)
    total_spikes_compass = np.sum(history['compass_spikes'])
    total_spikes_integrator = np.sum(history['integrator_spikes'])
    total_steps = len(history['traj'])
    total_neurons = 16 + 16
    sparsity = 1.0 - ((total_spikes_compass + total_spikes_integrator) / (total_neurons * total_steps))

    print('')
    print('================ EXPERIMENTAL RESULTS ================')
    print(f'Homing Error (eps_home):    {metrics["homing_error"]:.3f} m')
    print(f'Path Tortuosity Index (tau):{metrics["tortuosity"]:.3f} (Ideal = 1.000)')
    print(f'Ideal Homing Distance:      {metrics["ideal_path_length"]:.2f} m')
    print(f'Actual Homing Distance:     {metrics["homing_path_length"]:.2f} m')
    print(f'Population Spike Sparsity:  {sparsity * 100:.2f}% (Active Neurons < {(1 - sparsity)*100:.2f}%)')
    print(f'Total Compass Spikes:       {int(total_spikes_compass)}')
    print(f'Total Integrator Spikes:    {int(total_spikes_integrator)}')
    print('======================================================')
    print('')

    # Plotting
    traj = np.array(history['traj'])
    compass_bumps = np.array(history['compass_bumps'])
    integrator_states = np.array(history['integrator_states'])
    compass_spikes = np.array(history['compass_spikes'])

    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    # Panel A: 2D Spatial Arena Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traj[:outbound_steps, 0], traj[:outbound_steps, 1], color='#E67E22', lw=2.0, label='Outbound Search (Foraging)')
    ax1.plot(traj[outbound_steps:, 0], traj[outbound_steps:, 1], color='#2980B9', lw=2.5, label='Inbound Path Integration (Homing)')
    ax1.plot(0, 0, marker='*', markersize=16, color='#27AE60', markeredgecolor='black', label='Nest (Home Origin)')
    ax1.plot(food_pos[0], food_pos[1], marker='o', markersize=10, color='#C0392B', markeredgecolor='black', label='Food Source')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) 2D Ant Closed-Loop Foraging & Homing Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', frameon=True, fontsize=9)
    ax1.set_aspect('equal')

    # Panel B: Protocerebral Bridge Compass Bump Activity
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(compass_bumps.T, aspect='auto', cmap='plasma', origin='lower', extent=[0, total_steps, 0, 360])
    ax2.axvline(x=outbound_steps, color='white', linestyle='--', lw=2.0, label='Food Reached / Homing Start')
    ax2.set_xlabel('Simulation Timestep', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Preferred Azimuth Heading (deg)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Protocerebral Bridge (PB) Ring Attractor Bump Activity', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Neuron Activation')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)

    # Panel C: Fan-Shaped Body CPU4 Vector Memory
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(integrator_states.T, aspect='auto', cmap='viridis', origin='lower', extent=[0, total_steps, 0, 16])
    ax3.axvline(x=outbound_steps, color='red', linestyle='--', lw=2.0)
    ax3.set_xlabel('Simulation Timestep', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CPU4 Column Index', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Fan-Shaped Body (FB) CPU4 Home Vector Memory', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Accumulated Memory')

    # Panel D: Neuromorphic Spike Raster
    ax4 = fig.add_subplot(gs[1, 1])
    spike_times, spike_neurons = np.where(compass_spikes > 0)
    ax4.scatter(spike_times, spike_neurons, s=3, color='#8E44AD', alpha=0.8)
    ax4.axvline(x=outbound_steps, color='red', linestyle='--', lw=2.0)
    ax4.set_xlabel('Simulation Timestep', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Compass Neuron Index (0-15)', fontsize=11, fontweight='bold')
    ax4.set_title(f'(D) Neuromorphic Spike Raster (Sparsity: {sparsity*100:.1f}%)', fontsize=12, fontweight='bold')
    ax4.set_ylim(-0.5, 15.5)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plot_path = 'figures/exp1_path_integration.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'Publication figure saved to: {plot_path}')


if __name__ == '__main__':
    run_experiment()
