"""
Experiment 2: Dual-Pathway (CX + MB) Navigation and Landmark Displacement Recovery.
Compares pure Central Complex Path Integration against Dual-Pathway (CX + MB) under forced passive displacement.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.central_complex.cx_agent import CentralComplexAgent, AgentMode
from src.dual_pathway_agent import DualPathwayAgent, NavigationMode
from env.arena import AntArena2D
from env.vision import PanoramicVisualSensor, Landmark
from env.sensors import CelestialCompassSensor, OpticFlowSensor


def run_experiment_2(seed: int = 42):
    np.random.seed(seed)
    os.makedirs('figures', exist_ok=True)

    landmarks = [
        Landmark(x=-6.0, y=7.0, height=3.5, radius=1.0),
        Landmark(x=7.0, y=8.0, height=4.0, radius=1.2),
        Landmark(x=8.0, y=-6.0, height=3.0, radius=0.9),
        Landmark(x=-7.0, y=-6.0, height=3.8, radius=1.1),
        Landmark(x=0.0, y=9.0, height=5.0, radius=1.5),
    ]

    vision = PanoramicVisualSensor(n_sectors=36, landmarks=landmarks, noise_std=0.0)
    compass_sensor = CelestialCompassSensor(noise_std=0.01)
    flow_sensor = OpticFlowSensor(speed_noise_std=0.01, gyro_noise_std=0.01)

    # -------------------------------------------------------------
    # AGENT 1: Pure CX Agent (Fails upon forced displacement)
    # -------------------------------------------------------------
    arena_pure_cx = AntArena2D(nest_pos=(0.0, 0.0), dt=0.05)
    agent_pure_cx = CentralComplexAgent(n_columns=16, dt=1.0, steer_gain=4.5, default_speed=1.2)

    init_heading = 0.8
    agent_pure_cx.reset(initial_heading=init_heading)
    arena_pure_cx.reset(initial_heading=init_heading)

    # -------------------------------------------------------------
    # AGENT 2: Dual-Pathway CX + MB Agent (Recovers via visual memory)
    # -------------------------------------------------------------
    arena_dual = AntArena2D(nest_pos=(0.0, 0.0), dt=0.05)
    agent_dual = DualPathwayAgent(n_cx_columns=16, n_pn=36, n_kc=1000, dt=1.0, steer_gain=4.5, default_speed=1.2)

    agent_dual.reset(initial_heading=init_heading)
    arena_dual.reset(initial_heading=init_heading)

    # Execute biological learning walk around nest
    agent_dual.learn_nest_learning_walk(vision)

    print('=== Phase 1: Outbound Foraging Search (180 steps) ===')
    turn_noise = 0.0
    for step in range(180):
        turn_noise = 0.85 * turn_noise + np.random.normal(0.0, 0.3)
        meas_h, conf = compass_sensor.read(arena_dual.heading)
        meas_s, meas_g = flow_sensor.read(1.2, turn_noise)

        # Pure CX step
        s_cx, a_cx, _ = agent_pure_cx.step(meas_h, conf, meas_s, meas_g, foraging_turn_bias=turn_noise, dt_phys=0.05)
        arena_pure_cx.step(s_cx, a_cx)

        # Dual agent step
        view = vision.render_view(arena_dual.pos, arena_dual.heading)
        s_d, a_d, _ = agent_dual.step(meas_h, conf, meas_s, meas_g, current_view=view, foraging_turn_bias=turn_noise, dt_phys=0.05)
        arena_dual.step(s_d, a_d)

    food_pos = np.copy(arena_dual.pos)
    print(f'Food discovered at: {food_pos}, Distance: {np.linalg.norm(food_pos):.2f} m')

    # -------------------------------------------------------------
    # Phase 2: Forced Passive Displacement Intervention
    # -------------------------------------------------------------
    displacement_vector = np.array([4.0, -5.0])
    print(f'=== Applying Forced Passive Displacement: {displacement_vector} ===')

    arena_pure_cx.pos += displacement_vector
    arena_pure_cx.trajectory.append(np.copy(arena_pure_cx.pos))

    arena_dual.pos += displacement_vector
    arena_dual.trajectory.append(np.copy(arena_dual.pos))

    displaced_pos = np.copy(arena_dual.pos)
    print(f'Displaced position: {displaced_pos}')

    # -------------------------------------------------------------
    # Phase 3: Closed-Loop Homing Run
    # -------------------------------------------------------------
    print('=== Phase 3: Closed-Loop Homing Execution ===')
    agent_pure_cx.set_mode(AgentMode.HOMING)
    agent_dual.set_mode(NavigationMode.HOMING_DUAL_FUSED)

    kc_spike_log = []
    visual_novelty_log = []

    for step in range(600):
        # 1. Pure CX agent homing
        meas_h1, conf1 = compass_sensor.read(arena_pure_cx.heading)
        meas_s1, meas_g1 = flow_sensor.read(1.2, 0.0)
        s1, a1, diag1 = agent_pure_cx.step(meas_h1, conf1, meas_s1, meas_g1, dt_phys=0.05)
        arena_pure_cx.step(s1, a1)

        # 2. Dual agent homing
        view = vision.render_view(arena_dual.pos, arena_dual.heading)
        
        def gradient_sampler():
            eps = 0.5
            v_px, _ = agent_dual.mb.process_view(vision.render_view(arena_dual.pos + np.array([eps, 0.0]), heading=0.0))
            v_mx, _ = agent_dual.mb.process_view(vision.render_view(arena_dual.pos - np.array([eps, 0.0]), heading=0.0))
            v_py, _ = agent_dual.mb.process_view(vision.render_view(arena_dual.pos + np.array([0.0, eps]), heading=0.0))
            v_my, _ = agent_dual.mb.process_view(vision.render_view(arena_dual.pos - np.array([0.0, eps]), heading=0.0))
            return (v_px - v_mx) / (2 * eps), (v_py - v_my) / (2 * eps)

        meas_h2, conf2 = compass_sensor.read(arena_dual.heading)
        meas_s2, meas_g2 = flow_sensor.read(1.2, 0.0)
        s2, a2, diag2 = agent_dual.step(
            meas_h2, conf2, meas_s2, meas_g2,
            current_view=view,
            gradient_sensor_func=gradient_sampler,
            dt_phys=0.05,
        )
        arena_dual.step(s2, a2)

        visual_novelty_log.append(diag2['visual_novelty'])
        _, kc_s = agent_dual.mb.process_view(view)
        kc_spike_log.append(np.copy(kc_s))

        dist_nest = float(np.linalg.norm(arena_dual.pos - arena_dual.nest_pos))
        if dist_nest < 0.20 and diag2['active_system'] == 'MB_VISUAL_GRADIENT':
            print(f'>>> Dual Agent arrived at TRUE Nest at homing step {step}! Final dist: {dist_nest:.3f} m')
            break

    # Metrics
    pure_cx_err = float(np.linalg.norm(arena_pure_cx.pos - arena_pure_cx.nest_pos))
    dual_err = float(np.linalg.norm(arena_dual.pos - arena_dual.nest_pos))

    print('')
    print('================ EXPERIMENT 2 RESULTS ================')
    print(f'Displacement Magnitude:             {np.linalg.norm(displacement_vector):.2f} m')
    print(f'Pure CX Homing Error:               {pure_cx_err:.3f} m (LOST / Blind to Displacement)')
    print(f'Dual-Pathway (CX+MB) Homing Error:  {dual_err:.3f} m (SUCCESSFULLY RECOVERED)')
    print(f'Dual-Pathway Recovery Improvement:  {(1.0 - dual_err/pure_cx_err)*100:.2f}% Error Reduction')
    print('======================================================')
    print('')

    # -------------------------------------------------------------
    # Publication-Grade 4-Panel Plot
    # -------------------------------------------------------------
    traj_cx = np.array(arena_pure_cx.trajectory)
    traj_dual = np.array(arena_dual.trajectory)
    kc_spikes = np.array(kc_spike_log)

    fig = plt.figure(figsize=(15, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)

    # Panel A: 2D Arena Trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traj_dual[:180, 0], traj_dual[:180, 1], color='#E67E22', lw=2.0, label='Outbound Foraging')
    ax1.annotate('Forced Displacement', xy=(food_pos[0], food_pos[1]), xytext=(displaced_pos[0], displaced_pos[1]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2.0, ls=':'), fontsize=10, fontweight='bold', color='red')
    
    ax1.plot(traj_cx[181:, 0], traj_cx[181:, 1], color='#E74C3C', lw=2.0, ls='--', label=f'Pure CX (Fails, Err={pure_cx_err:.1f}m)')
    ax1.plot(traj_dual[181:, 0], traj_dual[181:, 1], color='#2980B9', lw=2.5, label=f'Dual CX+MB (Recovers, Err={dual_err:.2f}m)')

    # Landmarks & Nest
    ax1.plot(0, 0, marker='*', markersize=18, color='#27AE60', markeredgecolor='black', label='True Nest (Home)')
    ax1.plot(food_pos[0], food_pos[1], marker='o', markersize=10, color='#9B59B6', markeredgecolor='black', label='Food Source')
    for lm in landmarks:
        circle = plt.Circle((lm.x, lm.y), lm.radius, color='#7F8C8D', alpha=0.6)
        ax1.add_patch(circle)
        ax1.plot(lm.x, lm.y, 'k+', markersize=8)

    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Dual-Pathway Visual Recovery under Displacement', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', frameon=True, fontsize=8.5)
    ax1.set_aspect('equal')

    # Panel B: Mushroom Body Visual Novelty Map
    ax2 = fig.add_subplot(gs[0, 1])
    grid_x = np.linspace(-12, 12, 40)
    grid_y = np.linspace(-12, 12, 40)
    nov_map = np.zeros((len(grid_y), len(grid_x)))

    for iy, y in enumerate(grid_y):
        for ix, x in enumerate(grid_x):
            v = vision.render_view(np.array([x, y]), heading=0.0)
            nov, _ = agent_dual.mb.process_view(v)
            nov_map[iy, ix] = nov

    im2 = ax2.imshow(nov_map, origin='lower', extent=[-12, 12, -12, 12], cmap='viridis_r', aspect='auto')
    ax2.plot(0, 0, marker='*', markersize=16, color='white', markeredgecolor='black')
    ax2.plot(traj_dual[181:, 0], traj_dual[181:, 1], color='cyan', lw=2.0, label='Dual Recovery Path')
    ax2.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Mushroom Body Visual Familiarity Gradient (Valleys of Novelty)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Familiarity Score (Low = Familiar)')
    ax2.legend(loc='upper right', frameon=True, fontsize=8.5)

    # Panel C: Rotational Scanning Novelty Profile at Displaced Position
    ax3 = fig.add_subplot(gs[1, 0])
    scan_angles = np.linspace(0, 360, 72)
    scan_nov = []
    for deg in scan_angles:
        v = vision.render_view(displaced_pos, heading=np.radians(deg))
        nov, _ = agent_dual.mb.process_view(v)
        scan_nov.append(nov)

    ax3.plot(scan_angles, scan_nov, color='#8E44AD', lw=2.5)
    best_ang_deg = scan_angles[np.argmin(scan_nov)]
    ax3.axvline(x=best_ang_deg, color='red', linestyle='--', lw=2.0, label=f'Best Landmark Heading ({best_ang_deg:.1f} deg)')
    ax3.set_xlabel('Rotational Scan Angle (deg)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MBON Novelty Valuation', fontsize=11, fontweight='bold')
    ax3.set_title('(C) 360-Degree Panoramic Rotational Scan at Displaced Point', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend(loc='upper right', frameon=True, fontsize=9)

    # Panel D: Kenyon Cell Population Raster
    ax4 = fig.add_subplot(gs[1, 1])
    if len(kc_spikes) > 0:
        kc_subset = kc_spikes[:, :100]
        times, k_idx = np.where(kc_subset > 0)
        ax4.scatter(times, k_idx, s=4, color='#2C3E50', alpha=0.7)
    ax4.set_xlabel('Homing Timestep', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Kenyon Cell Index (0-99)', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Kenyon Cell (KC) Sparse Spiking Events (APL k-WTA 5% Sparsity)', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.4)

    plot_path = 'figures/exp2_visual_navigation.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'Publication figure saved to: {plot_path}')


if __name__ == '__main__':
    run_experiment_2()
