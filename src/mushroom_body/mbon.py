"""
Mushroom Body Output Neuron (MBON) for Novelty & Familiarity Valuation.
"""
import numpy as np
from src.neuro.lif import LIFNeuronGroup


class MushroomBodyOutputNeuron:
    """
    Mushroom Body Output Neuron (MBON).
    Computes visual novelty: high output for novel views, low output for familiar views.
    """

    def __init__(
        self,
        n_kc: int = 1000,
        dt: float = 1.0,
    ):
        self.n_kc = n_kc
        self.dt = dt

        self.neuron = LIFNeuronGroup(
            n_neurons=1,
            tau_m=20.0,
            v_rest=-70.0,
            v_reset=-75.0,
            v_th=-50.0,
            dt=dt,
        )

    def reset(self) -> None:
        self.neuron.reset()

    def compute_novelty(self, kc_spikes: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute visual novelty score from Kenyon cell spikes and synaptic weights.
        
        Args:
            kc_spikes: Binary array of shape (n_kc,)
            weights: Weight array of shape (1, n_kc) or (n_kc,)
            
        Returns:
            novelty_score: Float >= 0.0 (lower means more familiar).
        """
        w = np.asarray(weights).flatten()
        kc = np.asarray(kc_spikes).flatten()

        # Novelty is proportional to total un-depressed synaptic activation
        novelty = float(np.sum(w * kc))
        return novelty
