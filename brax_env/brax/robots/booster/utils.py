from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp
import jax
import dill
from pathlib import Path


class BoosterUtils:
    """Utility functions for the Go1."""

    """
    Properties
    """

    KP = jp.array([50.0, 50.0, 50.0, 50.0, 30.0, 30.0, 
                  50.0, 50.0, 50.0, 50.0, 30.0, 30.0,])

    KD = jp.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 
                  3.0, 3.0, 3.0, 3.0, 1.0, 1.0,])
                  
    STANDING_JOINT_ANGLES_L = jp.array([-0.2, 0.0, 0.0 , 0.4, -0.25, 0.0])
    STANDING_JOINT_ANGLES_F = jp.array([-0.2, 0.0, 0.0 , 0.4, -0.25, 0.0])

    ALL_STANDING_JOINT_ANGLES = jp.concatenate([
        STANDING_JOINT_ANGLES_L,
        STANDING_JOINT_ANGLES_F,
    ])


    LOWER_JOINT_LIMITS = jp.array([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, 
                                   -1.8, -1.57, -1.0, 0.0, -0.87, -0.44])
    """constant: the lower joint angle limits for a leg"""

    UPPER_JOINT_LIMITS = jp.array([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 
                                   1.57, 0.3, 1.0, 2.34, 0.35, 0.44])
    """constant: the upper joint angle limits for a leg"""

    MOTOR_TORQUE_LIMIT = jp.tile(jp.array([45.0, 45.0, 30.0, 65.0, 24.0, 15.0]), 2)
    """constant: the torque limit for the motors"""

    MOTOR_VEL_LIMIT = jp.array([
                                12.5, 10.9, 10.9, 11.7, 18.8, 12.4,
                                12.5, 10.9, 10.9, 11.7, 18.8, 12.4
                                ])
    
    ALL_VEL_LIMIT = jp.array([
                            2.0, 2.0, 2.0,
                            1.0, 1.0, 1.0,
                            12.5, 10.9, 10.9, 11.7, 18.8, 12.4,
                            12.5, 10.9, 10.9, 11.7, 18.8, 12.4
                            ])
    
    UPPER_ALL_POS_LIMIT = jp.array([
                                100.0, 100.0, 0.8,
                                1.0, 1.0, 1.0, 1.0,
                                   1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 
                                   1.57, 0.3, 1.0, 2.34, 0.35, 0.44
                                ])
    LOWER_ALL_POS_LIMIT = jp.array([
                                -100.0, -100.0, 0.0,
                                -1.0, -1.0, -1.0, -1.0,
                                -1.8, -0.3, -1.0, 0.0, -0.87, -0.44, 
                                -1.8, -1.57, -1.0, 0.0, -0.87, -0.44
                                ])
    """constant: the velocity limit for the motors"""


    # TODO
    CACHE_PATH = None#epath.resource_path('brax') / 'robots/go1/.cache'

    @staticmethod
    def get_system(used_cached: bool = False) -> System:
        """Returns the system for the Go1."""

        if used_cached:
            sys = BoosterUtils._load_cached_system(approx_system=False)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/booster/T1_locomotion.xml'
            
            sys = mjcf.load(path)

        return sys


    @staticmethod
    def get_approx_system(used_cached: bool = False) -> System:
        """Returns the approximate system for the Go1."""

        if used_cached:
            sys = BoosterUtils._load_cached_system(approx_system=True)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/booster/T1_locomotion.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def _cache_system(approx_system: bool) -> System:
        """Cache the system for the Go1 to avoid reloading the xml file."""
        sys = BoosterUtils.get_system()
        Path(BoosterUtils.CACHE_PATH).mkdir(parents=True, exist_ok=True)
        with open(BoosterUtils._cache_path(approx_system), 'wb') as f:
            dill.dump(sys, f)
        return sys

    @staticmethod
    def _load_cached_system(approx_system: bool) -> System:
        """Load the cached system for the Go1."""
        try:
            with open(BoosterUtils._cache_path(approx_system), 'rb') as f:
                sys = dill.load(f)
        except FileNotFoundError:
            sys = BoosterUtils._cache_system(approx_system)
        return sys

    @staticmethod
    def _cache_path(approx_system: bool) -> epath.Path:
        """Get the path to the cached system for the Go1."""
        if approx_system:
            path = BoosterUtils.CACHE_PATH / 'T1_locomotion.pkl'
        else:
            path = BoosterUtils.CACHE_PATH / 'T1_locomotion.pkl'
        return path





