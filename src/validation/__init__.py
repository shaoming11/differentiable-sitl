"""Validation module for parameter identification."""

from .rollout import (
    hold_out_validation,
    split_train_test,
    compare_trajectories
)

from .metrics import (
    quaternion_to_euler,
    compute_attitude_rmse,
    compute_velocity_rmse,
    compute_angular_velocity_rmse,
    summarize_validation_metrics
)

__all__ = [
    'hold_out_validation',
    'split_train_test',
    'compare_trajectories',
    'quaternion_to_euler',
    'compute_attitude_rmse',
    'compute_velocity_rmse',
    'compute_angular_velocity_rmse',
    'summarize_validation_metrics'
]
