"""
Differentiable multicopter flight dynamics model in JAX.

Implements pure JAX multicopter physics with exact gradients via automatic
differentiation for system identification.

State representation:
    [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    - q: quaternion (world-to-body rotation)
    - v: linear velocity in world frame (m/s)
    - w: angular velocity in body frame (rad/s)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from functools import partial

from .motor_model import pwm_to_angular_velocity


def quat_to_rotation(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix (world-to-body).

    Uses the standard quaternion-to-rotation-matrix formula.

    Args:
        q: (4,) array [qw, qx, qy, qz] - unit quaternion

    Returns:
        (3, 3) rotation matrix R such that v_body = R @ v_world

    Example:
        >>> q = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        >>> R = quat_to_rotation(q)
        >>> # R is identity matrix
    """
    qw, qx, qy, qz = q

    # Precompute repeated terms
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qwqx, qwqy, qwqz = qw * qx, qw * qy, qw * qz
    qxqy, qxqz, qyqz = qx * qy, qx * qz, qy * qz

    # Rotation matrix (world-to-body)
    return jnp.array([
        [1 - 2*(qy2 + qz2),     2*(qxqy - qwqz),     2*(qxqz + qwqy)],
        [    2*(qxqy + qwqz), 1 - 2*(qx2 + qz2),     2*(qyqz - qwqx)],
        [    2*(qxqz - qwqy),     2*(qyqz + qwqx), 1 - 2*(qx2 + qy2)]
    ])


def quat_inverse(q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        q: (4,) array [qw, qx, qy, qz]

    Returns:
        (4,) array [qw, -qx, -qy, -qz]
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_integrate(
    q: jnp.ndarray,
    omega: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Integrate quaternion with angular velocity using Euler method.

    Implements quaternion kinematics:
        q̇ = 0.5 * q ⊗ [0, ωx, ωy, ωz]

    Args:
        q: (4,) array [qw, qx, qy, qz] - current quaternion
        omega: (3,) array [wx, wy, wz] - angular velocity in body frame (rad/s)
        dt: timestep in seconds

    Returns:
        (4,) array - updated quaternion (normalized)

    Note:
        Uses simple Euler integration. For higher accuracy, consider RK4.
    """
    qw, qx, qy, qz = q
    wx, wy, wz = omega

    # Quaternion derivative: q̇ = 0.5 * q ⊗ [0, ω]
    # Using quaternion multiplication formula
    q_dot = 0.5 * jnp.array([
        -qx*wx - qy*wy - qz*wz,
         qw*wx + qy*wz - qz*wy,
         qw*wy - qx*wz + qz*wx,
         qw*wz + qx*wy - qy*wx
    ])

    # Euler step
    q_next = q + q_dot * dt

    # Normalize to maintain unit quaternion constraint
    return q_next / jnp.linalg.norm(q_next)


def compute_motor_forces(
    pwm_normalized: jnp.ndarray,
    params: Dict[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute thrust forces and angular velocities for all motors.

    Args:
        pwm_normalized: (N,) array of normalized PWM values [0-1]
        params: dict containing:
            - pwm_to_omega_poly: (3,) array [a₀, a₁, a₂]
            - kT: scalar thrust coefficient

    Returns:
        Tuple of:
        - thrusts: (N,) array of thrust forces in Newtons
        - omega_motors: (N,) array of angular velocities in rad/s
    """
    # PWM → angular velocity
    omega_motors = pwm_to_angular_velocity(
        pwm_normalized,
        params['pwm_to_omega_poly']
    )

    # Angular velocity → thrust: F = kT * ω²
    thrusts = params['kT'] * omega_motors ** 2

    return thrusts, omega_motors


def compute_total_torque(
    thrusts: jnp.ndarray,
    omega_motors: jnp.ndarray,
    omega_body: jnp.ndarray,
    params: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """
    Compute total torque on the body in body frame.

    Total torque consists of:
    1. Spatial torques: τ_spatial = r × F (moment arm from motor positions)
    2. Reaction torques: τ_reaction = ±kQ * ω² (motor drag torque)
    3. Rotational drag: τ_drag = -c_drag * ω * |ω|

    Args:
        thrusts: (N,) array of motor thrusts in Newtons
        omega_motors: (N,) array of motor angular velocities (rad/s)
        omega_body: (3,) array of body angular velocity (rad/s)
        params: dict containing:
            - motor_positions: (N, 3) array of motor positions (m)
            - motor_directions: (N,) array (+1=CW, -1=CCW)
            - kQ: scalar torque coefficient
            - c_drag: scalar rotational drag coefficient

    Returns:
        (3,) array of total torque in body frame (N·m)
    """
    motor_positions = params['motor_positions']
    motor_directions = params['motor_directions']

    # 1. Spatial torques: τ = r × F
    # All motors thrust along +z in body frame
    thrust_vectors = jnp.stack([
        jnp.zeros_like(thrusts),
        jnp.zeros_like(thrusts),
        thrusts
    ], axis=1)  # (N, 3)

    spatial_torques = jnp.cross(motor_positions, thrust_vectors)  # (N, 3)
    total_spatial = jnp.sum(spatial_torques, axis=0)  # (3,)

    # 2. Reaction torques: ±kQ * ω² along z-axis
    # CW motors (+1): positive reaction torque (opposes motor spin)
    # CCW motors (-1): negative reaction torque
    reaction_torques_z = motor_directions * params['kQ'] * omega_motors ** 2
    total_reaction = jnp.array([0.0, 0.0, jnp.sum(reaction_torques_z)])

    # 3. Rotational drag: -c_drag * ω * |ω|
    # Component-wise drag proportional to velocity squared
    drag_torque = -params['c_drag'] * omega_body * jnp.abs(omega_body)

    # Total torque
    return total_spatial + total_reaction + drag_torque


def compute_angular_acceleration(
    torque: jnp.ndarray,
    omega: jnp.ndarray,
    inertia: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute angular acceleration from Euler's rotation equation.

    Implements:
        I · α = τ - ω × (I · ω)

    where the cross term accounts for gyroscopic effects.

    Args:
        torque: (3,) array of total torque in body frame (N·m)
        omega: (3,) array of angular velocity in body frame (rad/s)
        inertia: (3,) array [Ixx, Iyy, Izz] - diagonal inertia tensor (kg·m²)

    Returns:
        (3,) array of angular acceleration in body frame (rad/s²)
    """
    # Gyroscopic term: ω × (I · ω)
    I_omega = inertia * omega  # Element-wise for diagonal inertia
    gyroscopic = jnp.cross(omega, I_omega)

    # Euler's equation: I · α = τ - gyroscopic
    # For diagonal inertia: α = (τ - gyroscopic) / I
    alpha = (torque - gyroscopic) / inertia

    return alpha


@partial(jax.jit, static_argnames=['dt'])
def fdm_step(
    state: jnp.ndarray,
    pwm: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    dt: float
) -> jnp.ndarray:
    """
    One timestep of multicopter flight dynamics model.

    Integrates the equations of motion:
    1. Translational dynamics: F = ma (with gravity and thrust)
    2. Rotational dynamics: I·α = τ (with gyroscopic coupling)
    3. Quaternion kinematics: q̇ = 0.5 * q ⊗ [0, ω]

    Args:
        state: (10,) array [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            - q: quaternion (world-to-body)
            - v: linear velocity in world frame (m/s)
            - ω: angular velocity in body frame (rad/s)
        pwm: (N,) array of normalized PWM values [0-1] per motor
        params: dict with keys:
            - mass: scalar (kg)
            - kT: scalar thrust coefficient (N/(rad/s)²)
            - kQ: scalar torque coefficient (N·m/(rad/s)²)
            - inertia: (3,) array [Ixx, Iyy, Izz] (kg·m²)
            - c_drag: scalar rotational drag coefficient
            - pwm_to_omega_poly: (3,) array [a₀, a₁, a₂]
            - motor_positions: (N, 3) array (meters)
            - motor_directions: (N,) array (+1=CW, -1=CCW)
        dt: timestep (seconds)

    Returns:
        (10,) array - next state

    Example:
        >>> state = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # At rest
        >>> pwm = jnp.ones(4) * 0.5  # 50% throttle
        >>> params = {...}  # Parameter dict
        >>> next_state = fdm_step(state, pwm, params, dt=0.01)
    """
    # Unpack state
    q = state[0:4]      # Quaternion
    v = state[4:7]      # Linear velocity (world frame)
    omega = state[7:10] # Angular velocity (body frame)

    # === THRUST CALCULATION ===
    thrusts, omega_motors = compute_motor_forces(pwm, params)

    # Total thrust in body frame (all motors thrust along +z_body)
    F_body_total = jnp.array([0.0, 0.0, jnp.sum(thrusts)])

    # === TRANSLATIONAL DYNAMICS ===
    # Rotate thrust to world frame (R^T = R^-1 for rotation matrices)
    R = quat_to_rotation(q)
    F_world = R.T @ F_body_total  # R.T transforms body→world

    # Add gravity (world frame): F_gravity = -mg * ẑ
    F_total_world = F_world - jnp.array([0.0, 0.0, params['mass'] * 9.81])

    # Acceleration: a = F / m
    accel = F_total_world / params['mass']

    # === ROTATIONAL DYNAMICS ===
    # Compute total torque
    torque_total = compute_total_torque(
        thrusts, omega_motors, omega, params
    )

    # Angular acceleration from Euler's equation
    alpha = compute_angular_acceleration(
        torque_total, omega, params['inertia']
    )

    # === INTEGRATION (Euler method) ===
    q_next = quat_integrate(q, omega, dt)
    v_next = v + accel * dt
    omega_next = omega + alpha * dt

    # Assemble next state
    return jnp.concatenate([q_next, v_next, omega_next])


@partial(jax.jit, static_argnames=['dt'])
def rollout(
    state_init: jnp.ndarray,
    pwm_sequence: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    dt: float
) -> jnp.ndarray:
    """
    Roll out FDM for multiple timesteps using jax.lax.scan.

    This function is efficiently compiled and differentiated by JAX.

    Args:
        state_init: (10,) initial state
        pwm_sequence: (T, N) array of PWM commands over time
        params: parameter dict (see fdm_step documentation)
        dt: timestep in seconds

    Returns:
        (T, 10) array of states at each timestep

    Example:
        >>> state_init = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> pwm_seq = jnp.ones((1000, 4)) * 0.5
        >>> trajectory = rollout(state_init, pwm_seq, params, dt=0.01)
        >>> # trajectory.shape = (1000, 10)
    """
    def step_fn(carry, pwm_t):
        """Single step function for jax.lax.scan."""
        state = carry
        next_state = fdm_step(state, pwm_t, params, dt)
        return next_state, next_state

    # Use scan for efficient sequential computation
    _, trajectory = jax.lax.scan(step_fn, state_init, pwm_sequence)

    return trajectory


def flatten_params(params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, Dict]:
    """
    Flatten parameter dict to 1D array for optimization.

    Args:
        params: Dict of parameter arrays

    Returns:
        Tuple of:
        - flat_params: (n_params,) 1D array
        - template: Template dict for unflattening
    """
    flat_list = []
    template = {}

    # Define parameter order (must be consistent)
    param_keys = ['mass', 'kT', 'kQ', 'inertia', 'c_drag', 'pwm_to_omega_poly']

    for key in param_keys:
        if key in params:
            value = params[key]
            flat_value = jnp.atleast_1d(value).flatten()
            flat_list.append(flat_value)
            template[key] = {
                'shape': value.shape if hasattr(value, 'shape') else (),
                'size': flat_value.size
            }

    return jnp.concatenate(flat_list), template


def unflatten_params(
    params_flat: jnp.ndarray,
    template: Dict,
    params_fixed: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """
    Unflatten 1D parameter array back to dict.

    Args:
        params_flat: (n_params,) flattened parameter vector
        template: Template from flatten_params
        params_fixed: Dict of fixed parameters (motor geometry, etc.)

    Returns:
        Complete parameter dict ready for fdm_step
    """
    params = dict(params_fixed)  # Start with fixed parameters
    offset = 0

    param_keys = ['mass', 'kT', 'kQ', 'inertia', 'c_drag', 'pwm_to_omega_poly']

    for key in param_keys:
        if key in template:
            size = template[key]['size']
            shape = template[key]['shape']

            # Use integer offsets (computed from template)
            param_slice = params_flat[offset:offset + size]
            params[key] = param_slice.reshape(shape) if shape else param_slice[0]

            offset += size

    return params


def loss_fn(
    params_flat: jnp.ndarray,
    template: Dict,
    params_fixed: Dict[str, jnp.ndarray],
    state_trajectory_target: jnp.ndarray,
    pwm_sequence: jnp.ndarray,
    weights: jnp.ndarray,
    dt: float
) -> float:
    """
    Loss function for optimization: weighted MSE between predicted and target states.

    Note: Not JIT-compiled by default due to dict template. Call jax.jit manually if needed.

    Args:
        params_flat: (n_params,) flattened parameter vector to optimize
        template: Template for unflattening parameters
        params_fixed: Dict of fixed parameters (geometry, etc.)
        state_trajectory_target: (T, 10) target state trajectory
        pwm_sequence: (T-1, N) PWM sequence
        weights: (10,) per-state-dimension weights
        dt: timestep

    Returns:
        Scalar loss (weighted MSE)

    Example:
        >>> loss = loss_fn(params_flat, template, params_fixed,
        ...                target_traj, pwm_seq, weights, dt=0.01)
        >>> # Use jax.grad to compute gradients
        >>> grad_fn = jax.grad(loss_fn, argnums=0)
        >>> gradients = grad_fn(params_flat, template, params_fixed, ...)
    """
    # Unflatten parameters (outside JIT)
    params = unflatten_params(params_flat, template, params_fixed)

    # Run FDM rollout (this part is JIT-compiled)
    predicted = rollout(state_trajectory_target[0], pwm_sequence, params, dt)

    # Compute weighted residuals
    residuals = predicted - state_trajectory_target[1:]
    weighted_residuals = residuals * weights  # Broadcasting

    # Mean squared error
    return jnp.mean(weighted_residuals ** 2)


def get_default_state_weights() -> jnp.ndarray:
    """
    Get default weights for state dimensions in loss function.

    Weights should reflect:
    - Relative importance of each state dimension
    - Measurement accuracy (higher weight for more accurate measurements)
    - Magnitude scaling (prevent large values from dominating)

    Returns:
        (10,) array of weights for [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    """
    return jnp.array([
        1.0, 1.0, 1.0, 1.0,  # Quaternion (already normalized)
        1.0, 1.0, 1.0,        # Velocity (m/s)
        10.0, 10.0, 10.0      # Angular velocity (rad/s) - higher priority
    ])


def quaternion_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> float:
    """
    Compute angular distance between two quaternions.

    Uses the formula: d = 1 - |q1 · q2|

    Args:
        q1: (4,) quaternion
        q2: (4,) quaternion

    Returns:
        Scalar distance in [0, 1], where 0 = identical orientation
    """
    dot_product = jnp.abs(jnp.dot(q1, q2))
    return 1.0 - jnp.clip(dot_product, 0.0, 1.0)


def validate_state(state: jnp.ndarray) -> bool:
    """
    Validate that state vector is physically reasonable.

    Checks for:
    - NaN or Inf values
    - Quaternion normalization
    - Reasonable velocity/angular velocity magnitudes

    Args:
        state: (10,) state vector

    Returns:
        True if valid, False otherwise
    """
    # Check for NaN/Inf
    if not jnp.all(jnp.isfinite(state)):
        return False

    # Check quaternion norm (should be close to 1)
    q_norm = jnp.linalg.norm(state[0:4])
    if jnp.abs(q_norm - 1.0) > 1e-3:
        return False

    # Check velocities are reasonable (< 100 m/s)
    v_mag = jnp.linalg.norm(state[4:7])
    if v_mag > 100.0:
        return False

    # Check angular velocities are reasonable (< 50 rad/s ~ 477 deg/s)
    omega_mag = jnp.linalg.norm(state[7:10])
    if omega_mag > 50.0:
        return False

    return True
