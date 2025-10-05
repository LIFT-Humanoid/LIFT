from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp
import jax
import dill
from pathlib import Path


class g1Utils:
    """Utility functions for the Go1."""

    """
    Properties
    """

    # ===== Values read directly from the provided G1-12DoF MJCF =====

    # Position gains (kp): default g1 joints kp=75, ankle_pitch kp=20, ankle_roll kp=2
    # Order per leg: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    KP = jp.array([
        40.17923847137318, 99.09842777666113, 40.17923847137318, 99.09842777666113, 28.50124619574858,  28.50124619574858,   # left leg
        40.17923847137318, 99.09842777666113, 40.17923847137318, 99.09842777666113, 28.50124619574858,  28.50124619574858,   # right leg
    ])

    # Damping (kd): default g1 joints damping=2, ankle_pitch damping=1, ankle_roll damping=0.2
    KD = jp.array([
        2.5578897650279457,  6.3088018534966395,  2.5578897650279457,  6.3088018534966395, 1.814445686584846, 1.814445686584846,    # left leg
        2.5578897650279457,  6.3088018534966395,  2.5578897650279457,  6.3088018534966395, 1.814445686584846, 1.814445686584846,    # right leg
    ])

    # Standing pose from custom numeric "init_qpos" (after the 7-DoF free base):
    # repeated 6D per leg: [-0.1, 0, 0, 0.3, -0.2, 0]
    STANDING_JOINT_ANGLES_L = jp.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0])
    STANDING_JOINT_ANGLES_F = jp.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0])

    ALL_STANDING_JOINT_ANGLES = jp.concatenate([
        STANDING_JOINT_ANGLES_L,
        STANDING_JOINT_ANGLES_F,
    ])

    # Joint angle limits from each joint's <range="lower upper">
    LOWER_JOINT_LIMITS = jp.array([
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,   # left leg
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,   # right leg
    ])

    UPPER_JOINT_LIMITS = jp.array([
        2.8798,  2.9671,  2.7576,  2.8798,    0.5236,   0.2618,   # left leg
        2.8798,  2.9671,  2.7576,  2.8798,    0.5236,   0.2618,   # right leg
    ])

    # Motor torque limits from actuatorfrcrange (absolute values):
    # hip_pitch/hip_yaw: ±88, hip_roll/knee: ±139, ankle_pitch/ankle_roll: ±50
    MOTOR_TORQUE_LIMIT = jp.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2)

    # ===== Not present in this MJCF (set here only if you still need placeholders) =====
    # No velocity limits are defined in the XML; enforce in controller if required.
    MOTOR_VEL_LIMIT = jp.array([
                                32, 20, 32, 20, 37, 37,
                                32, 20, 32, 20, 37, 37
                                ])
    ALL_VEL_LIMIT = None

    # Whole-state limits (including free base) are not defined in the XML.
    UPPER_ALL_POS_LIMIT = None
    LOWER_ALL_POS_LIMIT = None



    """constant: the velocity limit for the motors"""


    # TODO
    CACHE_PATH = None#epath.resource_path('brax') / 'robots/go1/.cache'

    @staticmethod
    def get_system(used_cached: bool = False) -> System:
        """Returns the system for the Go1."""

        if used_cached:
            sys = g1Utils._load_cached_system(approx_system=False)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/g1/g1_locomotion.xml'
            
            sys = mjcf.load(path)

        return sys


    @staticmethod
    def get_approx_system(used_cached: bool = False) -> System:
        """Returns the approximate system for the Go1."""

        if used_cached:
            sys = g1Utils._load_cached_system(approx_system=True)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/g1/g1_locomotion.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def _cache_system(approx_system: bool) -> System:
        """Cache the system for the Go1 to avoid reloading the xml file."""
        sys = g1Utils.get_system()
        Path(g1Utils.CACHE_PATH).mkdir(parents=True, exist_ok=True)
        with open(g1Utils._cache_path(approx_system), 'wb') as f:
            dill.dump(sys, f)
        return sys

    @staticmethod
    def _load_cached_system(approx_system: bool) -> System:
        """Load the cached system for the Go1."""
        try:
            with open(g1Utils._cache_path(approx_system), 'rb') as f:
                sys = dill.load(f)
        except FileNotFoundError:
            sys = g1Utils._cache_system(approx_system)
        return sys

    @staticmethod
    def _cache_path(approx_system: bool) -> epath.Path:
        """Get the path to the cached system for the Go1."""
        if approx_system:
            path = g1Utils.CACHE_PATH / 'g1_locomotion.pkl'
        else:
            path = g1Utils.CACHE_PATH / 'g1_locomotion.pkl'
        return path





