"""
Vectorized Leaky Integrate-and-Fire (LIF) Neuron Model.
Supports biological continuous dynamics and Q4.12 fixed-point quantization for neuromorphic hardware parity.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any


class LIFNeuronGroup:
    """
    Vectorized Leaky Integrate-and-Fire (LIF) neuron population.
    
    Differential Equation:
        tau_m * dV/dt = -(V - V_rest) + R_m * I(t)
    
    Discrete-Time Update (Euler):
        V[t+dt] = V[t] + (dt / tau_m) * (-(V[t] - V_rest) + R_m * I[t])
        If V[t+dt] >= V_th:
            Spike = 1
            V[t+dt] = V_reset
            Refractory_Timer = t_ref / dt
    """

    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 20.0,       # Membrane time constant (ms)
        v_rest: float = -70.0,      # Resting potential (mV)
        v_reset: float = -75.0,     # Reset potential (mV)
        v_th: float = -50.0,        # Spiking threshold (mV)
        r_m: float = 1.0,           # Membrane resistance (MOhm)
        t_ref: float = 2.0,         # Refractory period (ms)
        dt: float = 1.0,            # Simulation step (ms)
        fixed_point: bool = False,  # Emulate Q4.12 fixed-point arithmetic
    ):
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.r_m = r_m
        self.t_ref = t_ref
        self.dt = dt
        self.fixed_point = fixed_point

        # Precompute leak decay factor
        self.decay = np.exp(-self.dt / self.tau_m) if self.tau_m > 0 else 0.0
        self.ref_steps = int(np.round(self.t_ref / self.dt))

        # State vectors
        self.v = np.full(self.n_neurons, self.v_rest, dtype=np.float64)
        self.refractory_timer = np.zeros(self.n_neurons, dtype=np.int32)
        self.spikes = np.zeros(self.n_neurons, dtype=np.float64)
        self.total_spikes = 0
        self.total_timesteps = 0

    def reset(self, v_init: Optional[np.ndarray] = None) -> None:
        """Reset neuron membrane potentials and refractory timers."""
        if v_init is not None:
            self.v = np.copy(v_init).astype(np.float64)
        else:
            self.v.fill(self.v_rest)
        self.refractory_timer.fill(0)
        self.spikes.fill(0.0)
        self.total_spikes = 0
        self.total_timesteps = 0

    def step(self, current_input: np.ndarray) -> np.ndarray:
        """
        Advance the neuron population by one simulation step dt.
        
        Args:
            current_input: Synaptic / external driving current vector of shape (n_neurons,)
            
        Returns:
            spikes: Binary array (0 or 1) of shape (n_neurons,) indicating spiking neurons
        """
        current_input = np.asarray(current_input, dtype=np.float64)
        assert current_input.shape[0] == self.n_neurons, f"Expected input shape ({self.n_neurons},), got {current_input.shape}"

        self.total_timesteps += 1

        # Decrement refractory timers for active refractory neurons
        in_refractory = self.refractory_timer > 0
        self.refractory_timer[in_refractory] -= 1

        # Non-refractory neurons update membrane potential
        non_ref = ~in_refractory

        # Leaky integration: V[t+1] = V_rest + (V[t] - V_rest) * decay + R_m * I * (1 - decay)
        # Or standard Euler: V[t+1] = V[t] + (dt / tau_m) * (-(V[t] - V_rest) + R_m * I)
        leak_drive = (self.v_rest - self.v[non_ref]) + self.r_m * current_input[non_ref]
        self.v[non_ref] += (self.dt / self.tau_m) * leak_drive

        if self.fixed_point:
            # Emulate Q4.12 fixed-point truncation
            scale = 4096.0  # 2^12
            self.v[non_ref] = np.round(self.v[non_ref] * scale) / scale

        # Threshold detection
        spiked = (self.v >= self.v_th) & non_ref
        self.spikes.fill(0.0)
        self.spikes[spiked] = 1.0

        # Reset spiked neurons and initiate refractory period
        self.v[spiked] = self.v_reset
        self.refractory_timer[spiked] = self.ref_steps

        self.total_spikes += int(np.sum(spiked))
        return self.spikes.copy()

    @property
    def firing_rate(self) -> float:
        """Average firing rate in Hz across the population."""
        if self.total_timesteps == 0:
            return 0.0
        total_time_sec = (self.total_timesteps * self.dt) / 1000.0
        return self.total_spikes / (self.n_neurons * total_time_sec)
