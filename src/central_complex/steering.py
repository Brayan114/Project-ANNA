"""
Central Complex CPU1 / Pontine Steering Circuit.
Computes motor turning command to align current heading with the accumulated home vector.
"""
import numpy as np
from src.neuro.lif import LIFNeuronGroup


class CentralComplexSteering:
    """
    CPU1 / Lateral Accessory Lobe (LAL) Spiking Steering Comparator.
    """

    def __init__(
        self,
        n_columns: int = 16,
        dt: float = 1.0,
        steer_gain: float = 4.0,
    ):
        self.n_columns = n_columns
        self.dt = dt
        self.steer_gain = steer_gain
        self.preferred_angles = np.linspace(0, 2 * np.pi, n_columns, endpoint=False)

        # Spiking premotor neurons (Left Turn vs Right Turn)
        self.turn_neurons = LIFNeuronGroup(
            n_neurons=2,  # 0: Turn Left, 1: Turn Right
            tau_m=10.0,
            v_rest=-70.0,
            v_reset=-75.0,
            v_th=-55.0,
            dt=dt,
        )

    def reset(self) -> None:
        self.turn_neurons.reset()

    def compute_steering(
        self,
        current_heading: float,
        target_heading: float,
        target_distance: float,
    ) -> float:
        """
        Compute turning torque (rad/s) to align current heading with target home vector.
        
        Args:
            current_heading: Heading angle in radians [0, 2pi) from compass.
            target_heading: Target home angle in radians [0, 2pi) from integrator.
            target_distance: Estimated remaining distance to home.
            
        Returns:
            angular_velocity: Motor turning setpoint (rad/s).
        """
        if target_distance < 0.1:
            # Reached home nest - small systematic search wiggle
            return 0.0

        # Angular error in [-pi, pi]
        heading_error = target_heading - current_heading
        heading_error = float(np.arctan2(np.sin(heading_error), np.cos(heading_error)))

        # Drive Left/Right turn motor neurons
        current_left = max(0.0, heading_error * 30.0)
        current_right = max(0.0, -heading_error * 30.0)

        spikes = self.turn_neurons.step(np.array([current_left, current_right]))

        # Smooth proportional motor setpoint bounded by steer_gain
        ang_cmd = self.steer_gain * np.clip(heading_error, -1.0, 1.0)
        return float(ang_cmd)
