"""
2D Continuous Desert Ant Foraging and Navigation Arena.
"""
import numpy as np
from typing import Tuple, Dict, Any, List


class AntArena2D:
    """
    Continuous 2D Desert Foraging Arena.
    Simulates kinematics, wind drift, nest location, and sensory feedback.
    """

    def __init__(
        self,
        nest_pos: Tuple[float, float] = (0.0, 0.0),
        dt: float = 0.05,  # 50ms physical timestep
        wind_drift: Tuple[float, float] = (0.0, 0.0),
    ):
        self.nest_pos = np.array(nest_pos, dtype=np.float64)
        self.dt = dt
        self.wind_drift = np.array(wind_drift, dtype=np.float64)

        # Agent state: [x, y, heading_theta]
        self.pos = np.copy(self.nest_pos)
        self.heading = 0.0  # Radians [0, 2pi)

        # History tracking
        self.trajectory: List[Tuple[float, float]] = []
        self.headings: List[float] = []

    def reset(self, initial_heading: float = 0.0) -> Tuple[np.ndarray, float]:
        """Reset agent to nest location."""
        self.pos = np.copy(self.nest_pos)
        self.heading = float(np.mod(initial_heading, 2 * np.pi))
        self.trajectory = [tuple(self.pos)]
        self.headings = [self.heading]
        return np.copy(self.pos), self.heading

    def step(self, motor_speed: float, motor_ang_vel: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Advance physical state.
        
        Args:
            motor_speed: Commanded forward velocity (units/s)
            motor_ang_vel: Commanded turning rate (rad/s)
            
        Returns:
            (pos, heading, info)
        """
        # Update orientation
        self.heading = float(np.mod(self.heading + motor_ang_vel * self.dt, 2 * np.pi))

        # Kinematic translation + environmental wind drift
        dx = (motor_speed * np.cos(self.heading) + self.wind_drift[0]) * self.dt
        dy = (motor_speed * np.sin(self.heading) + self.wind_drift[1]) * self.dt

        self.pos[0] += dx
        self.pos[1] += dy

        self.trajectory.append(tuple(self.pos))
        self.headings.append(self.heading)

        dist_to_nest = float(np.linalg.norm(self.pos - self.nest_pos))

        info = {
            "pos": np.copy(self.pos),
            "heading": self.heading,
            "dist_to_nest": dist_to_nest,
        }
        return np.copy(self.pos), self.heading, info

    def get_path_metrics(self, homing_start_idx: int) -> Dict[str, float]:
        """
        Calculate homing accuracy and path tortuosity index.
        """
        traj = np.array(self.trajectory)
        if len(traj) <= homing_start_idx:
            return {"homing_error": 0.0, "tortuosity": 1.0}

        homing_path = traj[homing_start_idx:]
        start_pt = homing_path[0]
        end_pt = homing_path[-1]

        # Euclidean straight-line distance from food to nest
        euclidean_dist = np.linalg.norm(start_pt - self.nest_pos)
        # Actual distance traveled during homing
        segment_diffs = np.diff(homing_path, axis=0)
        actual_dist = np.sum(np.linalg.norm(segment_diffs, axis=1))

        # Final homing error
        homing_error = float(np.linalg.norm(end_pt - self.nest_pos))
        tortuosity = float(actual_dist / (euclidean_dist + 1e-6))

        return {
            "homing_error": homing_error,
            "tortuosity": tortuosity,
            "homing_path_length": float(actual_dist),
            "ideal_path_length": float(euclidean_dist),
        }
