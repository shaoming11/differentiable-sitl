"""
State space models for Unscented Kalman Filter.

This module defines the state transition and observation models for the
13-dimensional state vector used in the RTS smoother:

State vector x = [qw, qx, qy, qz,  # Quaternion (4)
                  vx, vy, vz,      # Velocity in world frame (3)
                  wx, wy, wz,      # Angular velocity in body frame (3)
                  px, py, pz]      # Position in world frame (3)
"""

import jax.numpy as jnp
from typing import Tuple


def state_transition_model(
    x: jnp.ndarray,
    dt: float,
    process_noise_cov: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Predict next state using constant angular velocity model.

    Dynamics:
    - Quaternion: integrate with angular velocity
    - Velocity: assume constant (no dynamics model yet)
    - Angular velocity: assume constant (will be corrected by measurements)
    - Position: integrate velocity

    Args:
        x: (13,) current state [qw,qx,qy,qz, vx,vy,vz, wx,wy,wz, px,py,pz]
        dt: timestep in seconds
        process_noise_cov: (13, 13) process noise covariance Q (unused in deterministic prediction)

    Returns:
        x_next: (13,) predicted state
    """
    # Extract state components
    q = x[0:4]      # Quaternion
    v = x[4:7]      # Velocity (world frame)
    omega = x[7:10] # Angular velocity (body frame)
    p = x[10:13]    # Position (world frame)

    # Quaternion integration using first-order Euler method
    # q_dot = 0.5 * Omega(omega) * q, where Omega is the quaternion matrix
    qw, qx, qy, qz = q
    wx, wy, wz = omega

    q_dot = 0.5 * jnp.array([
        -qx*wx - qy*wy - qz*wz,  # dqw/dt
         qw*wx + qy*wz - qz*wy,  # dqx/dt
         qw*wy - qx*wz + qz*wx,  # dqy/dt
         qw*wz + qx*wy - qy*wx   # dqz/dt
    ])

    q_next = q + q_dot * dt

    # Normalize quaternion to maintain unit norm constraint
    q_next = q_next / jnp.linalg.norm(q_next)

    # Velocity: assume constant (no force model or aerodynamic effects yet)
    v_next = v

    # Angular velocity: assume constant (piecewise constant between measurements)
    omega_next = omega

    # Position: integrate velocity (simple Euler integration)
    p_next = p + v * dt

    # Concatenate updated state
    x_next = jnp.concatenate([q_next, v_next, omega_next, p_next])

    return x_next


def imu_observation_model(x: jnp.ndarray) -> jnp.ndarray:
    """
    IMU observation model: measures angular velocity and (simplified) acceleration.

    The IMU measures:
    - Angular velocity in body frame (directly from state)
    - Acceleration in body frame (simplified - returns zero for now)

    Note: In a full implementation, acceleration would be computed from
    the derivative of velocity, rotated to body frame, plus gravity compensation.
    For the initial implementation, we focus on angular velocity which is
    directly observable in the state.

    Args:
        x: (13,) state vector

    Returns:
        z: (6,) observation [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    """
    # Extract angular velocity from state (body frame)
    omega = x[7:10]

    # For now, return zero acceleration (no dynamics model)
    # In future: compute expected acceleration from state derivatives
    # accounting for gravity and rotation
    acc = jnp.zeros(3)

    return jnp.concatenate([acc, omega])


def gps_observation_model(x: jnp.ndarray) -> jnp.ndarray:
    """
    GPS observation model: measures velocity and position.

    GPS provides:
    - Velocity in NED (North-East-Down) frame
    - Position in geodetic coordinates (lat, lng, alt)

    For simplicity, we assume the world frame is aligned with NED,
    so velocities and positions are directly observable.

    Args:
        x: (13,) state vector

    Returns:
        z: (6,) observation [vel_n, vel_e, vel_d, lat, lng, alt]
    """
    v = x[4:7]      # Velocity (world/NED frame)
    p = x[10:13]    # Position (world/NED frame)

    # In a full implementation, position would need conversion from
    # local NED to geodetic (lat/lng/alt), but for local operation
    # we can treat these as approximately linear
    return jnp.concatenate([v, p])


def baro_observation_model(x: jnp.ndarray) -> jnp.ndarray:
    """
    Barometer observation model: measures altitude.

    The barometer provides altitude (Down component in NED frame).

    Args:
        x: (13,) state vector

    Returns:
        z: (1,) observation [altitude]
    """
    # Altitude is the negative of the Down component (z position)
    # In NED frame: positive down, so altitude = -z
    # For simplicity, we'll use the z component directly
    return x[12:13]  # z position


def quaternion_to_euler(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: (4,) quaternion [qw, qx, qy, qz]

    Returns:
        euler: (3,) Euler angles [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    # Clamp to avoid numerical issues at poles
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.array([roll, pitch, yaw])


def euler_to_quaternion(euler: jnp.ndarray) -> jnp.ndarray:
    """
    Convert Euler angles to quaternion.

    Args:
        euler: (3,) Euler angles [roll, pitch, yaw] in radians

    Returns:
        q: (4,) quaternion [qw, qx, qy, qz]
    """
    roll, pitch, yaw = euler

    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return jnp.array([qw, qx, qy, qz])


def rotate_vector_body_to_world(v_body: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a vector from body frame to world frame using quaternion.

    Args:
        v_body: (3,) vector in body frame
        q: (4,) quaternion [qw, qx, qy, qz]

    Returns:
        v_world: (3,) vector in world frame
    """
    qw, qx, qy, qz = q
    vx, vy, vz = v_body

    # Quaternion rotation: v' = q * v * q^*
    # Using the formula: v' = v + 2 * qv x (qv x v + qw * v)
    # where qv = [qx, qy, qz]

    # First cross product: qv x v
    t0 = qy * vz - qz * vy
    t1 = qz * vx - qx * vz
    t2 = qx * vy - qy * vx

    # Second term: qv x v + qw * v
    t0 += qw * vx
    t1 += qw * vy
    t2 += qw * vz

    # Second cross product: qv x (qv x v + qw * v)
    u0 = qy * t2 - qz * t1
    u1 = qz * t0 - qx * t2
    u2 = qx * t1 - qy * t0

    # Final result: v + 2 * u
    v_world = v_body + 2.0 * jnp.array([u0, u1, u2])

    return v_world


def rotate_vector_world_to_body(v_world: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a vector from world frame to body frame using quaternion.

    Args:
        v_world: (3,) vector in world frame
        q: (4,) quaternion [qw, qx, qy, qz]

    Returns:
        v_body: (3,) vector in body frame
    """
    # Conjugate quaternion for inverse rotation
    q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])
    return rotate_vector_body_to_world(v_world, q_conj)
