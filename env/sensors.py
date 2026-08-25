"""
Biomimetic Sensory Models: Dorsal Rim Area (DRA) Celestial Compass & Ventral Optic Flow.
"""
import numpy as np


class CelestialCompassSensor:
    """
    Dorsal Rim Area (DRA) Polarized Light Compass Sensor.
    Detects celestial E-vector polarization angle corresponding to the solar azimuth.
    """

    def __init__(self, noise_std: float = 0.05, drop_probability: float = 0.0):
        self.noise_std = noise_std
        self.drop_probability = drop_probability

    def read(self, true_heading: float) -> tuple:
        """
        Returns:
            (measured_heading, confidence)
        """
        if np.random.rand() < self.drop_probability:
            # Occluded / overcast sky
            return 0.0, 0.0

        # Inject sensory noise
        noise = np.random.normal(0.0, self.noise_std)
        measured_heading = float(np.mod(true_heading + noise, 2 * np.pi))
        confidence = 1.0
        return measured_heading, confidence


class OpticFlowSensor:
    """
    Ventral Optic Flow and Proprioceptive Stride Integration Sensor.
    Measures linear translation speed and rotational angular velocity.
    """

    def __init__(self, speed_noise_std: float = 0.02, gyro_noise_std: float = 0.02):
        self.speed_noise_std = speed_noise_std
        self.gyro_noise_std = gyro_noise_std

    def read(self, true_speed: float, true_angular_vel: float) -> tuple:
        """
        Returns:
            (measured_speed, measured_angular_vel)
        """
        speed_noise = np.random.normal(0.0, self.speed_noise_std)
        gyro_noise = np.random.normal(0.0, self.gyro_noise_std)

        measured_speed = max(0.0, true_speed + speed_noise)
        measured_angular_vel = true_angular_vel + gyro_noise
        return measured_speed, measured_angular_vel
