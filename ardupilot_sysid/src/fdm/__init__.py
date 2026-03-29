"""Differentiable Flight Dynamics Models implemented in JAX."""

from .frame_configs import (
    FRAME_CONFIGS,
    get_frame_config,
    validate_frame_config,
    QUAD_X,
    HEXA_X,
    QUAD_PLUS,
    OCTO_X,
)

from .motor_model import (
    normalize_pwm,
    denormalize_pwm,
    pwm_to_angular_velocity,
    angular_velocity_to_thrust,
    angular_velocity_to_torque,
    pwm_to_thrust_torque,
    estimate_hover_pwm,
)

from .multicopter_jax import (
    fdm_step,
    rollout,
    loss_fn,
    flatten_params,
    unflatten_params,
    quat_to_rotation,
    quat_integrate,
    get_default_state_weights,
    quaternion_distance,
    validate_state,
)

__all__ = [
    # Frame configs
    'FRAME_CONFIGS',
    'get_frame_config',
    'validate_frame_config',
    'QUAD_X',
    'HEXA_X',
    'QUAD_PLUS',
    'OCTO_X',
    # Motor model
    'normalize_pwm',
    'denormalize_pwm',
    'pwm_to_angular_velocity',
    'angular_velocity_to_thrust',
    'angular_velocity_to_torque',
    'pwm_to_thrust_torque',
    'estimate_hover_pwm',
    # FDM core
    'fdm_step',
    'rollout',
    'loss_fn',
    'flatten_params',
    'unflatten_params',
    'quat_to_rotation',
    'quat_integrate',
    'get_default_state_weights',
    'quaternion_distance',
    'validate_state',
]
