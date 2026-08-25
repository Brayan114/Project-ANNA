import numpy as np
import pytest
from src.dual_pathway_agent import DualPathwayAgent, NavigationMode
from env.arena import AntArena2D
from env.vision import PanoramicVisualSensor

def test_dual_pathway_navigation():
    agent = DualPathwayAgent(n_cx_columns=16, n_pn=36, n_kc=1000, default_speed=1.2)
    arena = AntArena2D(nest_pos=(0.0, 0.0), dt=0.05)
    vision = PanoramicVisualSensor(n_sectors=36)
    
    agent.reset(initial_heading=0.0)
    arena.reset(initial_heading=0.0)
    
    # Train nest learning walk
    agent.learn_nest_learning_walk(vision)
    
    # 1. Outbound foraging walk
    for _ in range(120):
        view = vision.render_view(arena.pos, arena.heading)
        speed_cmd, ang_cmd, diag = agent.step(
            celestial_heading=arena.heading,
            celestial_confidence=1.0,
            forward_speed_sensor=1.2,
            angular_velocity_sensor=0.0,
            current_view=view,
            foraging_turn_bias=0.0,
            dt_phys=arena.dt,
        )
        arena.step(speed_cmd, ang_cmd)
        
    assert np.linalg.norm(arena.pos) > 4.0
    
    # 2. Homing return
    agent.set_mode(NavigationMode.HOMING_DUAL_FUSED)
    start_idx = len(arena.trajectory)
    
    for _ in range(400):
        view = vision.render_view(arena.pos, arena.heading)
        def gradient_sampler():
            eps = 0.5
            v_px, _ = agent.mb.process_view(vision.render_view(arena.pos + np.array([eps, 0.0]), heading=0.0))
            v_mx, _ = agent.mb.process_view(vision.render_view(arena.pos - np.array([eps, 0.0]), heading=0.0))
            v_py, _ = agent.mb.process_view(vision.render_view(arena.pos + np.array([0.0, eps]), heading=0.0))
            v_my, _ = agent.mb.process_view(vision.render_view(arena.pos - np.array([0.0, eps]), heading=0.0))
            return (v_px - v_mx) / (2 * eps), (v_py - v_my) / (2 * eps)
            
        speed_cmd, ang_cmd, diag = agent.step(
            celestial_heading=arena.heading,
            celestial_confidence=1.0,
            forward_speed_sensor=speed_cmd,
            angular_velocity_sensor=ang_cmd,
            current_view=view,
            gradient_sensor_func=gradient_sampler,
            dt_phys=arena.dt,
        )
        arena.step(speed_cmd, ang_cmd)
        if np.linalg.norm(arena.pos - arena.nest_pos) < 0.5:
            break
            
    metrics = arena.get_path_metrics(start_idx)
    assert metrics['homing_error'] < 0.5
