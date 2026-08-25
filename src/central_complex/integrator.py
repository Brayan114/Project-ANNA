"""
Fan-Shaped Body (FB) Path Integrator Neurons (CPU4 / PF-Col).
Biophysically grounded vector accumulator implementing cosine-projected velocity integration.
"""
import numpy as np
from src.neuro.lif import LIFNeuronGroup


class FanShapedBodyIntegrator:
    """
    Fan-Shaped Body CPU4 / PF-Col Columnar Vector Integrator.
    """

    def __init__(
        self,
        n_columns: int = 16,
        dt: float = 1.0,
        tau_acc: float = 500000.0,
    ):
        self.n_columns = n_columns
        self.dt = dt
        self.preferred_angles = np.linspace(0, 2 * np.pi, n_columns, endpoint=False)
        self.tau_acc = tau_acc

        # Cosine projection matrix from compass columns to FB accumulator columns
        diff = self.preferred_angles[:, None] - self.preferred_angles[None, :]
        self.W_proj = np.cos(diff)

        # Exact norm scale for normalized Gaussian bump integration
        diff_zero = np.arctan2(np.sin(self.preferred_angles), np.cos(self.preferred_angles))
        bump_zero = np.exp(-0.5 * (diff_zero / 0.5)**2)
        bump_zero = bump_zero / np.max(bump_zero)
        self.norm_scale = float(np.sum((self.W_proj @ bump_zero) * np.cos(self.preferred_angles)))

        # Accumulated vector memory per column
        self.accumulated_memory = np.zeros(n_columns, dtype=np.float64)

        self.neurons = LIFNeuronGroup(
            n_neurons=n_columns,
            tau_m=20.0,
            v_rest=-70.0,
            v_reset=-75.0,
            v_th=-50.0,
            dt=dt,
        )

    def reset(self) -> None:
        self.accumulated_memory.fill(0.0)
        self.neurons.reset()

    def step(self, compass_activity: np.ndarray, forward_speed: float, dt_phys: float = 0.05) -> np.ndarray:
        """
        Advance integrator by one physical time step.
        """
        decay = np.exp(-self.dt / self.tau_acc) if self.tau_acc > 0 else 1.0

        act = np.asarray(compass_activity, dtype=np.float64)
        if np.max(act) > 1e-6:
            act = act / np.max(act)

        projected_drive = self.W_proj @ act
        step_dist = forward_speed * dt_phys

        gain = step_dist / (self.norm_scale + 1e-9)
        self.accumulated_memory = self.accumulated_memory * decay + gain * projected_drive

        driving_current = np.maximum(0.0, self.accumulated_memory) * 5.0 + 2.0
        spikes = self.neurons.step(driving_current)
        return spikes

    def decode_home_vector(self) -> tuple:
        """
        Decode the estimated 2D home vector (dx, dy) pointing back to the nest.
        Returns:
            (distance, homing_angle_rad)
        """
        vx = np.sum(self.accumulated_memory * np.cos(self.preferred_angles))
        vy = np.sum(self.accumulated_memory * np.sin(self.preferred_angles))

        home_vx = -vx
        home_vy = -vy

        distance = float(np.sqrt(home_vx**2 + home_vy**2))
        homing_angle = float(np.mod(np.arctan2(home_vy, home_vx), 2 * np.pi))
        return distance, homing_angle
