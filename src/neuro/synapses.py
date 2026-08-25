"""
Synaptic Connectivity Matrices and Topology Generators for SNNs.
"""
import numpy as np


def create_ring_attractor_matrix(
    n_neurons: int,
    w_exc: float = 2.5,
    w_inh: float = 1.2,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Generate recurrent ring attractor weight matrix with local excitation and global/lateral inhibition.
    
    Formula:
        W[i, j] = w_exc * cos(theta_i - theta_j) - w_inh
    or wrapped Gaussian:
        W[i, j] = w_exc * exp(-d(theta_i, theta_j)^2 / (2 * sigma^2)) - w_inh
    """
    angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
    # Pairwise angular difference wrapped to [-pi, pi]
    diff = angles[:, None] - angles[None, :]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    
    # Cosine tuning profile with global inhibition
    W = w_exc * np.cos(diff) - w_inh
    # Zero autapse (self-connection optional or tuned)
    return W


def create_shift_matrix(
    n_neurons: int,
    direction: int = 1,  # +1 for clockwise / right shift, -1 for counter-clockwise / left shift
    weight: float = 1.5,
) -> np.ndarray:
    """
    Generate an asymmetric directional shift connection matrix.
    When multiplied by heading spikes, excites adjacent columns in the specified direction.
    """
    W = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i in range(n_neurons):
        target = (i + direction) % n_neurons
        W[target, i] = weight
    return W
