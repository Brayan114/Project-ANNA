"""
Unified Mushroom Body (MB) Spiking Network with Dopaminergic Learning (DAN).
"""
import numpy as np
from typing import Tuple, Dict, Any, Callable

from src.mushroom_body.kenyon_cells import KenyonCellPopulation
from src.mushroom_body.mbon import MushroomBodyOutputNeuron
from src.neuro.plasticity import RewardModulatedSTDP


class MushroomBodyNetwork:
    """
    Complete Mushroom Body Spiking System.
    
    Functions:
        1. High-dimensional sparse expansion via Kenyon Cells.
        2. Readout valuation via MBONs.
        3. Dopamine-modulated 3-factor Anti-Hebbian synaptic plasticity for one-shot snapshot learning.
        4. Panoramic rotational scanning to find familiar landmark alignments.
    """

    def __init__(
        self,
        n_pn: int = 36,
        n_kc: int = 1000,
        k_conn: int = 5,
        target_sparsity: float = 0.05,
        eta: float = 0.8,              # Plasticity learning rate
        dt: float = 1.0,
    ):
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.dt = dt

        self.kc = KenyonCellPopulation(
            n_pn=n_pn,
            n_kc=n_kc,
            k_conn=k_conn,
            target_sparsity=target_sparsity,
            dt=dt,
        )

        self.mbon = MushroomBodyOutputNeuron(n_kc=n_kc, dt=dt)

        self.plasticity = RewardModulatedSTDP(
            n_pre=n_kc,
            n_post=1,
            eta=eta,
            w_init=1.0,
            w_min=0.0,
            w_max=1.0,
            dt=dt,
        )

        self.stored_snapshots_count = 0

    def reset(self) -> None:
        self.kc.reset()
        self.mbon.reset()
        self.plasticity.reset(w_init=1.0)
        self.stored_snapshots_count = 0

    def process_view(self, pn_input: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Process a panoramic visual view and compute its novelty.
        
        Args:
            pn_input: Normalized visual feature array of shape (n_pn,)
            
        Returns:
            (novelty_score, kc_spikes)
        """
        kc_spikes = self.kc.step(pn_input)
        novelty = self.mbon.compute_novelty(kc_spikes, self.plasticity.W)
        return novelty, kc_spikes

    def train_snapshot(self, pn_input: np.ndarray, reward: float = 1.0) -> float:
        """
        One-shot snapshot learning: Dopamine release depresses active KC synapses.
        
        Args:
            pn_input: Visual features at the rewarded goal location.
            reward: Dopamine strength M(t) > 0.
            
        Returns:
            post_training_novelty: Novelty score after synaptic depression.
        """
        novelty_pre, kc_spikes = self.process_view(pn_input)
        self.plasticity.apply_modulation(dopamine_signal=reward, pre_spikes=kc_spikes)
        novelty_post, _ = self.process_view(pn_input)
        self.stored_snapshots_count += 1
        return novelty_post

    def scan_for_familiar_heading(
        self,
        view_sampler: Callable[[float], np.ndarray],
        n_scan_angles: int = 36,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Scan 360 degrees around current position to find the heading with minimal novelty (maximum familiarity).
        
        Args:
            view_sampler: Function taking heading_angle (rad) and returning PN feature vector (n_pn,).
            n_scan_angles: Number of azimuthal angles to evaluate.
            
        Returns:
            (best_heading_rad, min_novelty, novelty_profile)
        """
        test_angles = np.linspace(0, 2 * np.pi, n_scan_angles, endpoint=False)
        novelty_profile = np.zeros(n_scan_angles, dtype=np.float64)

        for i, angle in enumerate(test_angles):
            view = view_sampler(angle)
            nov, _ = self.process_view(view)
            novelty_profile[i] = nov

        min_idx = int(np.argmin(novelty_profile))
        best_heading = float(test_angles[min_idx])
        min_novelty = float(novelty_profile[min_idx])
        return best_heading, min_novelty, novelty_profile
