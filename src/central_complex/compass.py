"""
Protocerebral Bridge (PB) / Ellipsoid Body (EB) Spiking Ring Attractor Compass.
Maintains continuous representation of agent heading theta in [0, 2pi).
"""
import numpy as np
from src.neuro.lif import LIFNeuronGroup
from src.neuro.synapses import create_ring_attractor_matrix, create_shift_matrix


class CentralComplexCompass:
    """
    Protocerebral Bridge (PB) 16-column Heading Compass Network.
    """

    def __init__(
        self,
        n_columns: int = 16,
        dt: float = 1.0,
        tau_m: float = 20.0,
        w_exc: float = 4.0,
        w_inh: float = 2.0,
        w_shift: float = 3.0,
    ):
        self.n_columns = n_columns
        self.dt = dt
        self.preferred_angles = np.linspace(0, 2 * np.pi, n_columns, endpoint=False)

        # Core compass neuron population
        self.neurons = LIFNeuronGroup(
            n_neurons=n_columns,
            tau_m=tau_m,
            v_rest=-70.0,
            v_reset=-75.0,
            v_th=-50.0,
            r_m=1.0,
            t_ref=1.0,
            dt=dt,
        )

        # Recurrent ring attractor and shift matrices
        self.W_ring = create_ring_attractor_matrix(n_columns, w_exc=w_exc, w_inh=w_inh)
        self.W_shift_cw = create_shift_matrix(n_columns, direction=1, weight=w_shift)
        self.W_shift_ccw = create_shift_matrix(n_columns, direction=-1, weight=w_shift)

        # Filtered rate activity
        self.activity_trace = np.zeros(n_columns, dtype=np.float64)
        self.trace_decay = np.exp(-dt / 15.0)

    def reset(self, initial_heading: float = 0.0) -> None:
        """Reset compass and initialize bump at specified heading."""
        self.neurons.reset()
        diff = self.preferred_angles - initial_heading
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        bump = np.exp(-0.5 * (diff / 0.5)**2)
        self.activity_trace = np.copy(bump)
        init_v = -70.0 + 25.0 * bump
        self.neurons.reset(v_init=init_v)

    def step(
        self,
        celestial_heading: float = None,
        celestial_confidence: float = 0.0,
        angular_velocity: float = 0.0,
    ) -> np.ndarray:
        """
        Advance compass by one step.
        """
        # 1. Recurrent attractor dynamics
        recurrent_input = self.W_ring @ self.activity_trace

        # 2. Shift driven by angular velocity
        # Scale by angular velocity (rad/s) and time step
        shift_gain = angular_velocity * 10.0
        shift_input = np.zeros(self.n_columns, dtype=np.float64)
        if shift_gain > 0:
            shift_input = shift_gain * (self.W_shift_cw @ self.activity_trace)
        elif shift_gain < 0:
            shift_input = (-shift_gain) * (self.W_shift_ccw @ self.activity_trace)

        # 3. Direct celestial polarization sensory drive (DRA)
        celestial_input = np.zeros(self.n_columns, dtype=np.float64)
        if celestial_heading is not None and celestial_confidence > 0.0:
            diff = self.preferred_angles - celestial_heading
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            celestial_input = (celestial_confidence * 12.0) * np.maximum(0.0, np.cos(diff))

        total_current = recurrent_input + shift_input + celestial_input + 5.0
        spikes = self.neurons.step(total_current)

        # Smooth activity trace update
        # If external cue is available, lock onto it; otherwise maintain ring attractor bump
        if celestial_heading is not None and celestial_confidence > 0.5:
            diff = self.preferred_angles - celestial_heading
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            cue_bump = np.exp(-0.5 * (diff / 0.5)**2)
            self.activity_trace = 0.8 * cue_bump + 0.2 * spikes
        else:
            self.activity_trace = self.activity_trace * self.trace_decay + (1.0 - self.trace_decay) * (spikes * 2.0 + np.maximum(0.0, (self.neurons.v - self.neurons.v_rest) / 20.0))
            if np.max(self.activity_trace) > 1e-4:
                self.activity_trace = self.activity_trace / np.max(self.activity_trace)

        return spikes

    def decode_heading(self) -> float:
        """Decode heading estimate in [0, 2pi) via population vector decoding."""
        weights = self.activity_trace
        sin_sum = np.sum(weights * np.sin(self.preferred_angles))
        cos_sum = np.sum(weights * np.cos(self.preferred_angles))
        heading = np.arctan2(sin_sum, cos_sum)
        return float(np.mod(heading, 2 * np.pi))
