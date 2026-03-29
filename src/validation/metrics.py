import numpy as np
from typing import Dict


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: (4,) or (N, 4) quaternion(s) [qw, qx, qy, qz]

    Returns:
        (3,) or (N, 3) Euler angles [roll, pitch, yaw] in radians
    """
    if q.ndim == 1:
        qw, qx, qy, qz = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
    else:
        # Vectorized for multiple quaternions
        return np.array([quaternion_to_euler(q[i]) for i in range(len(q))])


def compute_attitude_rmse(
    predicted: np.ndarray,
    actual: np.ndarray
) -> Dict[str, float]:
    """
    Compute RMSE for attitude (roll, pitch, yaw) from quaternions.

    Args:
        predicted: (T, 10) predicted states (quaternion in first 4 dims)
        actual: (T, 10) actual states

    Returns:
        Dict with keys 'roll_deg', 'pitch_deg', 'yaw_deg' (RMSE in degrees)
    """
    # Extract quaternions
    q_pred = predicted[:, 0:4]
    q_actual = actual[:, 0:4]

    # Convert to Euler angles
    euler_pred = quaternion_to_euler(q_pred)  # (T, 3)
    euler_actual = quaternion_to_euler(q_actual)

    # Compute RMSE for each axis
    roll_error = euler_pred[:, 0] - euler_actual[:, 0]
    pitch_error = euler_pred[:, 1] - euler_actual[:, 1]

    # Handle yaw wrap-around (-π to π)
    yaw_error = euler_pred[:, 2] - euler_actual[:, 2]
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # Wrap to [-π, π]

    # RMSE in degrees
    rmse_roll = float(np.sqrt(np.mean(roll_error**2)) * 180 / np.pi)
    rmse_pitch = float(np.sqrt(np.mean(pitch_error**2)) * 180 / np.pi)
    rmse_yaw = float(np.sqrt(np.mean(yaw_error**2)) * 180 / np.pi)

    return {
        'roll_deg': rmse_roll,
        'pitch_deg': rmse_pitch,
        'yaw_deg': rmse_yaw
    }


def compute_velocity_rmse(
    predicted: np.ndarray,
    actual: np.ndarray
) -> Dict[str, float]:
    """
    Compute RMSE for linear velocities.

    Args:
        predicted: (T, 10) predicted states (velocity in dims 4-7)
        actual: (T, 10) actual states

    Returns:
        Dict with keys 'vx', 'vy', 'vz', 'v_total' (RMSE in m/s)
    """
    v_pred = predicted[:, 4:7]
    v_actual = actual[:, 4:7]

    rmse_vx = float(np.sqrt(np.mean((v_pred[:, 0] - v_actual[:, 0])**2)))
    rmse_vy = float(np.sqrt(np.mean((v_pred[:, 1] - v_actual[:, 1])**2)))
    rmse_vz = float(np.sqrt(np.mean((v_pred[:, 2] - v_actual[:, 2])**2)))

    # Total velocity magnitude error
    v_mag_pred = np.linalg.norm(v_pred, axis=1)
    v_mag_actual = np.linalg.norm(v_actual, axis=1)
    rmse_v_total = float(np.sqrt(np.mean((v_mag_pred - v_mag_actual)**2)))

    return {
        'vx': rmse_vx,
        'vy': rmse_vy,
        'vz': rmse_vz,
        'v_total': rmse_v_total
    }


def compute_angular_velocity_rmse(
    predicted: np.ndarray,
    actual: np.ndarray
) -> Dict[str, float]:
    """
    Compute RMSE for angular velocities.

    Args:
        predicted: (T, 10) predicted states (omega in dims 7-10)
        actual: (T, 10) actual states

    Returns:
        Dict with keys 'wx', 'wy', 'wz' (RMSE in rad/s)
    """
    omega_pred = predicted[:, 7:10]
    omega_actual = actual[:, 7:10]

    rmse_wx = float(np.sqrt(np.mean((omega_pred[:, 0] - omega_actual[:, 0])**2)))
    rmse_wy = float(np.sqrt(np.mean((omega_pred[:, 1] - omega_actual[:, 1])**2)))
    rmse_wz = float(np.sqrt(np.mean((omega_pred[:, 2] - omega_actual[:, 2])**2)))

    return {
        'wx': rmse_wx,
        'wy': rmse_wy,
        'wz': rmse_wz
    }


def summarize_validation_metrics(validation_result: Dict) -> Dict:
    """
    Compute comprehensive validation metrics.

    Args:
        validation_result: Output from hold_out_validation()

    Returns:
        Dict with all validation metrics organized by category
    """
    predicted = validation_result['predicted']
    actual = validation_result['actual']

    metrics = {
        'attitude': compute_attitude_rmse(predicted, actual),
        'velocity': compute_velocity_rmse(predicted, actual),
        'angular_velocity': compute_angular_velocity_rmse(predicted, actual),
        'duration_s': float(validation_result['timestamps'][-1]),
        'n_samples': len(predicted)
    }

    return metrics
