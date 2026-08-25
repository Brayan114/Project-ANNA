import numpy as np
import pytest
from src.central_complex.cx_agent import CentralComplexAgent, AgentMode
from env.arena import AntArena2D

def test_closed_loop_homing():
    agent = CentralComplexAgent(n_columns=16, dt=1.0, steer_gain=4.5, default_speed=1.2)
    arena = AntArena2D(nest_pos=(0.0, 0.0), dt=0.05)
    
    agent.reset(initial_heading=0.5)
    arena.reset(initial_heading=0.5)
    
    # 1. Outbound foraging walk (150 steps)
    for _ in range(150):
        speed_cmd, ang_cmd, diag = agent.step(
            celestial_heading=arena.heading,
            celestial_confidence=1.0,
            forward_speed_sensor=1.2,
            angular_velocity_sensor=0.0,
            foraging_turn_bias=0.0,
            dt_phys=arena.dt,
        )
        arena.step(speed_cmd, ang_cmd)
        
    outbound_pos = np.copy(arena.pos)
    dist_out = np.linalg.norm(outbound_pos)
    assert dist_out > 5.0
    
    # 2. Switch to HOMING mode
    agent.set_mode(AgentMode.HOMING)
    homing_start_idx = len(arena.trajectory)
    
    # Run closed-loop homing
    for _ in range(400):
        speed_cmd, ang_cmd, diag = agent.step(
            celestial_heading=arena.heading,
            celestial_confidence=1.0,
            forward_speed_sensor=speed_cmd,
            angular_velocity_sensor=ang_cmd,
            dt_phys=arena.dt,
        )
        arena.step(speed_cmd, ang_cmd)
        if np.linalg.norm(arena.pos - arena.nest_pos) < 0.5:
            break
            
    metrics = arena.get_path_metrics(homing_start_idx)
    assert metrics['homing_error'] < 0.5
    assert metrics['tortuosity'] < 1.4
