import numpy as np
import pytest
from src.central_complex.compass import CentralComplexCompass

def test_compass_bump_initialization():
    compass = CentralComplexCompass(n_columns=16, dt=1.0)
    target_heading = np.pi / 2.0  # 90 degrees
    compass.reset(initial_heading=target_heading)
    decoded = compass.decode_heading()
    assert np.isclose(decoded, target_heading, atol=0.3)

def test_compass_bump_rotation():
    compass = CentralComplexCompass(n_columns=16, dt=1.0)
    compass.reset(initial_heading=0.0)
    
    # Rotate with angular velocity
    for _ in range(100):
        compass.step(celestial_heading=None, celestial_confidence=0.0, angular_velocity=0.5)
    
    decoded = compass.decode_heading()
    assert decoded > 0.05
