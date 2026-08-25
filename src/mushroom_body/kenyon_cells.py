"""
Kenyon Cell (KC) Population with Anterior Paired Lateral (APL) Feedback Inhibition.
Implements high-dimensional sparse expansion of sensory inputs with biological k-WTA dynamics.
"""
import numpy as np
from src.neuro.lif import LIFNeuronGroup


class KenyonCellPopulation:
    """
    Kenyon Cell (KC) Sparse Expansion Layer.
    
    Neuroanatomy:
        - N_PN sensory inputs (Projection Neurons) from visual/antennal lobes.
        - High-dimensional expansion to N_KC cells (e.g. 1,000 neurons).
        - Random sparse projection (each KC connects to k_conn random PNs).
        - APL (Anterior Paired Lateral) inhibitory interneuron enforcing k-WTA population sparsity <= 5%.
    """

    def __init__(
        self,
        n_pn: int = 36,               # Number of Projection Neurons (input dimensions)
        n_kc: int = 1000,             # Number of Kenyon Cells
        k_conn: int = 5,              # Number of PN connections per KC
        target_sparsity: float = 0.05,# Target fraction of active KCs (e.g. 5%)
        dt: float = 1.0,
    ):
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.k_conn = k_conn
        self.target_sparsity = target_sparsity
        self.k_top = max(1, int(n_kc * target_sparsity))
        self.dt = dt

        # Random sparse binary projection matrix W_PN_to_KC of shape (n_kc, n_pn)
        np.random.seed(42)
        self.W_pn_kc = np.zeros((n_kc, n_pn), dtype=np.float64)
        for i in range(n_kc):
            conn_indices = np.random.choice(n_pn, size=k_conn, replace=False)
            self.W_pn_kc[i, conn_indices] = 1.0 / np.sqrt(k_conn)

        self.spikes = np.zeros(n_kc, dtype=np.float64)
        self.drive = np.zeros(n_kc, dtype=np.float64)

    def reset(self) -> None:
        self.spikes.fill(0.0)
        self.drive.fill(0.0)

    def step(self, pn_input: np.ndarray) -> np.ndarray:
        """
        Process projection neuron input and emit sparse Kenyon cell spikes.
        
        Args:
            pn_input: Array of shape (n_pn,) containing normalized visual features [0.0, 1.0].
            
        Returns:
            kc_spikes: Binary array of shape (n_kc,) indicating active Kenyon cells.
        """
        pn = np.asarray(pn_input, dtype=np.float64)

        # Feedforward excitatory drive from PNs
        self.drive = self.W_pn_kc @ pn

        # APL k-WTA inhibitory feedback: Select top k_top active cells
        threshold = np.partition(self.drive, -self.k_top)[-self.k_top]
        self.spikes.fill(0.0)
        self.spikes[self.drive >= threshold] = 1.0

        # Enforce exact top k_top count if ties occur
        if np.sum(self.spikes) > self.k_top:
            active_idx = np.where(self.spikes > 0)[0]
            sorted_idx = active_idx[np.argsort(-self.drive[active_idx])]
            self.spikes.fill(0.0)
            self.spikes[sorted_idx[:self.k_top]] = 1.0

        return self.spikes.copy()

    @property
    def current_sparsity(self) -> float:
        """Fraction of active Kenyon cells in the most recent step."""
        return float(np.mean(self.spikes))
