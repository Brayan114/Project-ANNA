"""
Dual-Pathway Ant-Inspired Neuromorphic Navigation Agent (CX + MB).
Integrates Central Complex (Vector Path Integration) and Mushroom Body (Visual Snapshot Memory).
"""
import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Callable, Optional

from src.central_complex.compass import CentralComplexCompass
from src.central_complex.integrator import FanShapedBodyIntegrator
from src.central_complex.steering import CentralComplexSteering
from src.mushroom_body.mb_network import MushroomBodyNetwork


class NavigationMode(Enum):
    OUTBOUND_FORAGING = 1
    HOMING_CX_DOMINANT = 2
    HOMING_MB_VISUAL = 3
    HOMING_DUAL_FUSED = 4


class DualPathwayAgent:
    """
    Unified Central Complex (CX) and Mushroom Body (MB) Spiking Agent.
    """

    def __init__(
        self,
        n_cx_columns: int = 16,
        n_pn: int = 36,
        n_kc: int = 1000,
        dt: float = 1.0,
        steer_gain: float = 4.5,
        default_speed: float = 1.2,
    ):
        self.dt = dt
        self.default_speed = default_speed

        # Central Complex (CX)
        self.compass = CentralComplexCompass(n_columns=n_cx_columns, dt=dt)
        self.integrator = FanShapedBodyIntegrator(n_columns=n_cx_columns, dt=dt)
        self.steering = CentralComplexSteering(n_columns=n_cx_columns, dt=dt, steer_gain=steer_gain)

        # Mushroom Body (MB)
        self.mb = MushroomBodyNetwork(n_pn=n_pn, n_kc=n_kc, dt=dt)

        self.mode = NavigationMode.OUTBOUND_FORAGING
        self.step_count = 0
        self.locked_home_dist = 0.0
        self.locked_home_angle = 0.0
        self.distance_traveled_homing = 0.0

    def reset(self, initial_heading: float = 0.0) -> None:
        self.compass.reset(initial_heading=initial_heading)
        self.integrator.reset()
        self.steering.reset()
        self.mb.reset()
        self.mode = NavigationMode.OUTBOUND_FORAGING
        self.step_count = 0
        self.locked_home_dist = 0.0
        self.locked_home_angle = 0.0
        self.distance_traveled_homing = 0.0

    def learn_nest_learning_walk(self, vision_sensor) -> None:
        """
        Execute biological learning walk around nest:
        Memorizes panoramic views at expanding radii around the nest entrance.
        """
        for r in [0.0, 0.8, 1.6, 2.4, 3.2, 4.0]:
            n_samples = 1 if r == 0.0 else 12
            for theta in np.linspace(0, 2 * np.pi, n_samples, endpoint=False):
                p = np.array([r * np.cos(theta), r * np.sin(theta)])
                v = vision_sensor.render_view(p, heading=0.0)
                self.mb.train_snapshot(v, reward=1.0)

    def set_mode(self, mode: NavigationMode) -> None:
        self.mode = mode
        if mode in [NavigationMode.HOMING_CX_DOMINANT, NavigationMode.HOMING_DUAL_FUSED, NavigationMode.HOMING_MB_VISUAL]:
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
        current_view: np.ndarray,
        gradient_sensor_func: Optional[Callable[[], Tuple[float, float]]] = None,
        foraging_turn_bias: float = 0.0,
        dt_phys: float = 0.05,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Advance dual-pathway brain.
        """
        self.step_count += 1

        # 1. Update PB Compass
        compass_spikes = self.compass.step(
            celestial_heading=celestial_heading,
            celestial_confidence=celestial_confidence,
            angular_velocity=angular_velocity_sensor,
        )
        decoded_heading = self.compass.decode_heading()

        # 2. Update FB Path Integrator
        if self.mode == NavigationMode.OUTBOUND_FORAGING:
            integrator_spikes = self.integrator.step(
                compass_activity=self.compass.activity_trace,
                forward_speed=forward_speed_sensor,
                dt_phys=dt_phys,
            )
            home_dist, home_angle = self.integrator.decode_home_vector()
        else:
            integrator_spikes = np.zeros(self.compass.n_columns)
            home_dist = max(0.0, self.locked_home_dist - self.distance_traveled_homing)
            home_angle = self.locked_home_angle

        # 3. Mushroom Body Visual Novelty Evaluation
        current_novelty, kc_spikes = self.mb.process_view(current_view)

        # 4. Dual-Pathway Motor Arbitration
        if self.mode == NavigationMode.OUTBOUND_FORAGING:
            motor_ang_vel = foraging_turn_bias
            motor_speed = self.default_speed
            active_system = "EXPLORATION"

        else:
            # Homing Mode
            if home_dist > 0.2:
                # Phase 1: CX Vector guidance while unwinding home vector
                target_angle = home_angle
                active_system = "CX_VECTOR"
                heading_err = float(np.arctan2(np.sin(target_angle - decoded_heading), np.cos(target_angle - decoded_heading)))
                motor_ang_vel = float(np.clip(4.0 * heading_err, -4.0, 4.0))
                speed_factor = max(0.0, float(np.cos(heading_err)))**2
                motor_speed = self.default_speed * speed_factor
                self.distance_traveled_homing += motor_speed * dt_phys
            else:
                # Phase 2: Home vector unwound -> Visual Landmark Gradient takes over
                if gradient_sensor_func is not None and self.mb.stored_snapshots_count > 0:
                    gx, gy = gradient_sensor_func()
                    g_norm = float(np.sqrt(gx**2 + gy**2))
                    if g_norm > 1e-4:
                        target_angle = float(np.mod(np.arctan2(-gy, -gx), 2 * np.pi))
                        active_system = "MB_VISUAL_GRADIENT"
                        heading_err = float(np.arctan2(np.sin(target_angle - decoded_heading), np.cos(target_angle - decoded_heading)))
                        motor_ang_vel = float(np.clip(4.0 * heading_err, -4.0, 4.0))
                        speed_factor = max(0.0, float(np.cos(heading_err)))**2
                        motor_speed = self.default_speed * speed_factor
                    else:
                        motor_speed = 0.0
                        motor_ang_vel = 0.0
                        active_system = "NEST_ARRIVED"
                else:
                    motor_speed = 0.0
                    motor_ang_vel = 0.0
                    active_system = "NEST_ARRIVED"

        diagnostics = {
            "step": self.step_count,
            "mode": self.mode.name,
            "active_system": active_system,
            "decoded_heading": decoded_heading,
            "home_vector_distance": home_dist,
            "home_vector_angle": home_angle,
            "visual_novelty": current_novelty,
            "kc_active_count": int(np.sum(kc_spikes)),
            "kc_sparsity": float(np.mean(kc_spikes)),
            "total_spikes": int(np.sum(compass_spikes) + np.sum(kc_spikes)),
        }

        return motor_speed, motor_ang_vel, diagnostics
