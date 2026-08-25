"""
Unified Central Complex (CX) Spiking Brain Agent.
Coordinates compass ring attractor, vector path integrator, and pontine steering circuit.
"""
import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple

from src.central_complex.compass import CentralComplexCompass
from src.central_complex.integrator import FanShapedBodyIntegrator
from src.central_complex.steering import CentralComplexSteering


class AgentMode(Enum):
    OUTBOUND_FORAGING = 1
    HOMING = 2


class CentralComplexAgent:
    """
    Autonomous Insect-Inspired Neuromorphic Spiking Agent.
    """

    def __init__(
        self,
        n_columns: int = 16,
        dt: float = 1.0,
        steer_gain: float = 4.0,
        default_speed: float = 1.0,
    ):
        self.n_columns = n_columns
        self.dt = dt
        self.default_speed = default_speed

        self.compass = CentralComplexCompass(n_columns=n_columns, dt=dt)
        self.integrator = FanShapedBodyIntegrator(n_columns=n_columns, dt=dt)
        self.steering = CentralComplexSteering(n_columns=n_columns, dt=dt, steer_gain=steer_gain)

        self.mode = AgentMode.OUTBOUND_FORAGING
        self.step_count = 0
        self.locked_home_angle: float = 0.0
        self.locked_home_dist: float = 0.0
        self.distance_traveled_homing: float = 0.0

    def reset(self, initial_heading: float = 0.0) -> None:
        """Reset all neural subsystems and start at nest origin."""
        self.compass.reset(initial_heading=initial_heading)
        self.integrator.reset()
        self.steering.reset()
        self.mode = AgentMode.OUTBOUND_FORAGING
        self.step_count = 0
        self.locked_home_angle = 0.0
        self.locked_home_dist = 0.0
        self.distance_traveled_homing = 0.0

    def set_mode(self, mode: AgentMode) -> None:
        """Switch agent state (e.g. from OUTBOUND_FORAGING to HOMING)."""
        self.mode = mode
        if mode == AgentMode.HOMING:
            dist, angle = self.integrator.decode_home_vector()
            self.locked_home_dist = dist
            self.locked_home_angle = angle
            self.distance_traveled_homing = 0.0

    def step(
        self,
        celestial_heading: float,
        celestial_confidence: float,
        forward_speed_sensor: float,
        angular_velocity_sensor: float,
        foraging_turn_bias: float = 0.0,
        dt_phys: float = 0.05,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Process sensory input and output motor action (speed, angular_velocity).
        """
        self.step_count += 1

        # 1. Update PB Heading Compass
        compass_spikes = self.compass.step(
            celestial_heading=celestial_heading,
            celestial_confidence=celestial_confidence,
            angular_velocity=angular_velocity_sensor,
        )

        decoded_heading = self.compass.decode_heading()

        # 2. Update Path Integrator
        if self.mode == AgentMode.OUTBOUND_FORAGING:
            integrator_spikes = self.integrator.step(
                compass_activity=self.compass.activity_trace,
                forward_speed=forward_speed_sensor,
                dt_phys=dt_phys,
            )
            home_dist, home_angle = self.integrator.decode_home_vector()
        else:
            integrator_spikes = np.zeros(self.n_columns)
            home_dist = max(0.0, self.locked_home_dist - self.distance_traveled_homing)
            home_angle = self.locked_home_angle

        # 3. Compute Steering & Speed
        if self.mode == AgentMode.HOMING:
            if home_dist < 0.1:
                motor_speed = 0.0
                motor_ang_vel = 0.0
            else:
                heading_err = float(np.arctan2(np.sin(home_angle - decoded_heading), np.cos(home_angle - decoded_heading)))
                motor_ang_vel = float(np.clip(4.0 * heading_err, -4.0, 4.0))
                speed_factor = max(0.0, float(np.cos(heading_err)))**2
                motor_speed = self.default_speed * speed_factor
                self.distance_traveled_homing += motor_speed * dt_phys
        else:
            motor_ang_vel = foraging_turn_bias
            motor_speed = self.default_speed

        diagnostics = {
            "step": self.step_count,
            "mode": self.mode.name,
            "decoded_heading": decoded_heading,
            "home_vector_distance": home_dist,
            "home_vector_angle": home_angle,
            "compass_spikes": int(np.sum(compass_spikes)),
            "integrator_spikes": int(np.sum(integrator_spikes)),
            "total_spikes": int(np.sum(compass_spikes) + np.sum(integrator_spikes)),
        }

        return motor_speed, motor_ang_vel, diagnostics
