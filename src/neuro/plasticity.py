"""
Synaptic Plasticity Engines: 3-Factor Reward-Modulated STDP (R-STDP) and Eligibility Traces.
"""
import numpy as np
from typing import Optional, Tuple


class RewardModulatedSTDP:
    """
    Three-Factor Synaptic Plasticity Engine (R-STDP / Dopaminergic Neuromodulation).
    
    Equations:
        Eligibility Trace:
            tau_e * dE_ij/dt = -E_ij + (S_post_i * S_pre_j - alpha_anti * S_pre_j)
            
        Weight Update:
            dW_ij/dt = eta * M(t) * E_ij
            
    For Mushroom Body Snapshot Learning:
        When dopamine M(t) > 0 (reward at nest/food), active KC synapses undergo LTD (Long-Term Depression),
        lowering the MBON response to familiar scenes (Anti-Hebbian familiarity encoding).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        eta: float = 0.05,           # Learning rate
        tau_e: float = 20.0,         # Eligibility trace decay (ms)
        w_init: float = 1.0,         # Initial synaptic weight
        w_min: float = 0.0,          # Minimum weight bound
        w_max: float = 1.0,          # Maximum weight bound
        dt: float = 1.0,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.eta = eta
        self.tau_e = tau_e
        self.w_min = w_min
        self.w_max = w_max
        self.dt = dt

        self.e_decay = np.exp(-dt / tau_e) if tau_e > 0 else 0.0

        # Synaptic weight matrix: Shape (n_post, n_pre)
        self.W = np.full((n_post, n_pre), w_init, dtype=np.float64)
        # Eligibility trace matrix: Shape (n_post, n_pre)
        self.E = np.zeros((n_post, n_pre), dtype=np.float64)

    def reset(self, w_init: Optional[float] = None) -> None:
        """Reset eligibility traces and optionally re-initialize weights."""
        self.E.fill(0.0)
        if w_init is not None:
            self.W.fill(w_init)

    def update_traces(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> np.ndarray:
        """
        Update eligibility traces based on pre- and post-synaptic activity.
        """
        pre = np.asarray(pre_spikes, dtype=np.float64)
        post = np.asarray(post_spikes, dtype=np.float64)

        # Outer product of post and pre spikes
        hebbian_activity = np.outer(post, pre)

        # Update trace: E[t+1] = E[t] * decay + activity
        self.E = self.E * self.e_decay + hebbian_activity
        return self.E

    def apply_modulation(self, dopamine_signal: float, pre_spikes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply 3-factor weight update driven by dopamine concentration M(t).
        
        If pre_spikes is provided directly, applies single-shot Anti-Hebbian LTD
        proportional to dopamine and active pre-synaptic cells:
            dW = -eta * dopamine * pre_spikes
        """
        if dopamine_signal == 0.0:
            return self.W

        if pre_spikes is not None:
            # Direct single-shot snapshot learning at goal
            pre = np.asarray(pre_spikes, dtype=np.float64)
            delta_W = -self.eta * dopamine_signal * pre[None, :]
        else:
            # Trace-based update
            delta_W = -self.eta * dopamine_signal * self.E

        self.W += delta_W
        self.W = np.clip(self.W, self.w_min, self.w_max)
        return self.W
