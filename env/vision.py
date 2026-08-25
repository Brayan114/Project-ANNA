"""
Biomimetic 360-Degree Panoramic Horizon Visual Sensor.
Models the compound eye panoramic skyline view based on naturalistic landmark silhouettes.
"""
import numpy as np
from typing import List, Tuple, Dict, Any


class Landmark:
    """Visual landmark in the 2D environment (e.g. desert shrub, rock, burrow mound)."""
    def __init__(self, x: float, y: float, height: float = 2.0, radius: float = 0.5):
        self.x = x
        self.y = y
        self.height = height
        self.radius = radius


class PanoramicVisualSensor:
    """
    360-Degree Panoramic Compound Eye Visual Sensor.
    
    Generates a 36-dimensional projection neuron (PN) array representing the panoramic skyline
    contrast and horizon elevation profile relative to agent heading.
    """

    def __init__(
        self,
        n_sectors: int = 36,          # 10-degree resolution per compound eye sector
        landmarks: List[Landmark] = None,
        noise_std: float = 0.01,
    ):
        self.n_sectors = n_sectors
        self.noise_std = noise_std
        self.sector_angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

        # Default landmark constellation if none provided
        if landmarks is None:
            self.landmarks = [
                Landmark(x=-5.0, y=8.0, height=3.0, radius=1.0),
                Landmark(x=8.0, y=6.0, height=4.0, radius=1.2),
                Landmark(x=6.0, y=-7.0, height=2.5, radius=0.8),
                Landmark(x=-8.0, y=-5.0, height=3.5, radius=1.1),
                Landmark(x=0.0, y=10.0, height=5.0, radius=1.5),   # Nest beacon
            ]
        else:
            self.landmarks = landmarks

    def render_view(self, pos: np.ndarray, heading: float) -> np.ndarray:
        """
        Render the 360-degree panoramic skyline feature vector from the given position and heading.
        
        Args:
            pos: 2D agent coordinate [x, y]
            heading: Agent azimuthal heading in radians [0, 2pi)
            
        Returns:
            view: Array of shape (n_sectors,) normalized to [0.0, 1.0].
        """
        view = np.zeros(self.n_sectors, dtype=np.float64)

        for lm in self.landmarks:
            dx = lm.x - pos[0]
            dy = lm.y - pos[1]
            dist = float(np.sqrt(dx**2 + dy**2)) + 1e-4

            # Bearing angle in world coordinates
            world_bearing = np.arctan2(dy, dx)
            # Relative retinal bearing angle [0, 2pi)
            retinal_bearing = np.mod(world_bearing - heading, 2 * np.pi)

            # Apparent angular size and height
            angular_height = lm.height / dist
            angular_width = np.arctan2(lm.radius, dist)

            # Cast Gaussian profile onto eye sectors
            diff = self.sector_angles - retinal_bearing
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            profile = angular_height * np.exp(-0.5 * (diff / max(0.1, angular_width))**2)
            view += profile

        # Normalize and add subtle sensory photon noise
        if np.max(view) > 1e-5:
            view = view / np.max(view)

        if self.noise_std > 0:
            view = np.clip(view + np.random.normal(0.0, self.noise_std, size=self.n_sectors), 0.0, 1.0)

        return view
