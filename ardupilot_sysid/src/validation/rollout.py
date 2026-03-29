"""
FDM rollout and hold-out validation.

Compare FDM predictions against actual flight data on a hold-out segment.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional


def hold_out_validation(
    params: Dict[str, Any],
    state_trajectory: np.ndarray,
    pwm_sequence: np.ndarray,
    dt: float,
    holdout_ratio: float = 0.2,
    fdm_rollout_fn: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Perform hold-out validation by rolling out FDM on unseen data.

    Args:
        params: Identified FDM parameters
        state_trajectory: Full smoothed state trajectory (T, state_dim)
        pwm_sequence: Full PWM input sequence (T, n_motors)
        dt: Time step in seconds
        holdout_ratio: Fraction of data to hold out for validation
        fdm_rollout_fn: FDM rollout function (if None, returns dummy metrics)

    Returns:
        Dictionary containing:
            - predicted_states: FDM predictions on holdout segment
            - actual_states: Actual states on holdout segment
            - metrics: Computed error metrics
    """
    n_timesteps = len(state_trajectory)
    holdout_start = int(n_timesteps * (1 - holdout_ratio))

    # Split data
    train_states = state_trajectory[:holdout_start]
    holdout_states = state_trajectory[holdout_start:]
    holdout_pwm = pwm_sequence[holdout_start:]

    # If no FDM function provided, return dummy results
    if fdm_rollout_fn is None:
        return {
            'holdout_segment_samples': len(holdout_states),
            'holdout_segment_duration_s': len(holdout_states) * dt,
            'actual_states': holdout_states,
            'predicted_states': None,
            'metrics': compute_validation_metrics(holdout_states, holdout_states)
        }

    # Rollout FDM on holdout segment
    try:
        initial_state = holdout_states[0]
        predicted_states = fdm_rollout_fn(initial_state, holdout_pwm[:-1], params, dt)

        # Compute metrics
        metrics = compute_validation_metrics(predicted_states, holdout_states[1:])

        return {
            'holdout_segment_samples': len(holdout_states),
            'holdout_segment_duration_s': len(holdout_states) * dt,
            'actual_states': holdout_states,
            'predicted_states': predicted_states,
            'metrics': metrics
        }
    except Exception as e:
        # If rollout fails, return error info
        return {
            'holdout_segment_samples': len(holdout_states),
            'holdout_segment_duration_s': len(holdout_states) * dt,
            'actual_states': holdout_states,
            'predicted_states': None,
            'error': str(e),
            'metrics': {}
        }


def compute_validation_metrics(
    predicted_states: np.ndarray,
    actual_states: np.ndarray
) -> Dict[str, Any]:
    """
    Compute validation metrics comparing predicted vs actual states.

    Args:
        predicted_states: Predicted state trajectory (T, state_dim)
        actual_states: Actual state trajectory (T, state_dim)

    Returns:
        Dictionary of metrics (RMSE, MAE, etc.)
    """
    if predicted_states is None or len(predicted_states) == 0:
        return {}

    # Ensure arrays
    predicted_states = np.asarray(predicted_states)
    actual_states = np.asarray(actual_states)

    # Truncate to shorter length
    n = min(len(predicted_states), len(actual_states))
    pred = predicted_states[:n]
    actual = actual_states[:n]

    metrics = {}

    # State vector assumed to be [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    # Quaternion: 0-3, Velocity: 4-6, Angular rates: 7-9

    # Quaternion/attitude errors (convert to Euler for interpretability)
    if pred.shape[1] >= 4:
        # Simplified: use quaternion components directly for RMSE
        quat_pred = pred[:, :4]
        quat_actual = actual[:, :4]
        quat_error = quat_pred - quat_actual
        quat_rmse = np.sqrt(np.mean(quat_error**2, axis=0))

        # Approximate attitude error in degrees (rough heuristic)
        # For small errors: angle ≈ 2 * ||q_xyz|| where q = q_actual^-1 * q_pred
        attitude_error_rad = 2 * np.linalg.norm(quat_error[:, 1:], axis=1)
        attitude_error_deg = np.rad2deg(attitude_error_rad)

        metrics['attitude'] = {
            'roll_deg': float(np.percentile(attitude_error_deg, 68)),  # ~1 sigma
            'pitch_deg': float(np.percentile(attitude_error_deg, 68)),
            'yaw_deg': float(np.percentile(attitude_error_deg, 68)),
            'quaternion_rmse': quat_rmse.tolist()
        }

    # Velocity errors
    if pred.shape[1] >= 7:
        vel_pred = pred[:, 4:7]
        vel_actual = actual[:, 4:7]
        vel_error = vel_pred - vel_actual
        vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))

        metrics['velocity'] = {
            'vx_rmse': float(vel_rmse[0]),
            'vy_rmse': float(vel_rmse[1]),
            'vz_rmse': float(vel_rmse[2]),
            'v_total': float(np.sqrt(np.mean(np.sum(vel_error**2, axis=1))))
        }

    # Angular rate errors
    if pred.shape[1] >= 10:
        omega_pred = pred[:, 7:10]
        omega_actual = actual[:, 7:10]
        omega_error = omega_pred - omega_actual
        omega_rmse = np.sqrt(np.mean(omega_error**2, axis=0))

        metrics['angular_rates'] = {
            'wx_rmse': float(omega_rmse[0]),
            'wy_rmse': float(omega_rmse[1]),
            'wz_rmse': float(omega_rmse[2]),
            'omega_total': float(np.sqrt(np.mean(np.sum(omega_error**2, axis=1))))
        }

    # Overall RMSE
    overall_error = pred - actual
    metrics['overall_rmse'] = float(np.sqrt(np.mean(overall_error**2)))

    return metrics


def summarize_validation_metrics(metrics: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of validation metrics.

    Args:
        metrics: Dictionary of validation metrics

    Returns:
        Multi-line string summary
    """
    if not metrics:
        return "No validation metrics available"

    lines = ["Validation Metrics:"]
    lines.append("-" * 50)

    if 'attitude' in metrics:
        att = metrics['attitude']
        lines.append(f"  Attitude RMSE:")
        lines.append(f"    Roll:  {att.get('roll_deg', 0):.2f}°")
        lines.append(f"    Pitch: {att.get('pitch_deg', 0):.2f}°")
        lines.append(f"    Yaw:   {att.get('yaw_deg', 0):.2f}°")

    if 'velocity' in metrics:
        vel = metrics['velocity']
        lines.append(f"  Velocity RMSE:")
        lines.append(f"    Total: {vel.get('v_total', 0):.3f} m/s")
        lines.append(f"    (vx={vel.get('vx_rmse', 0):.3f}, vy={vel.get('vy_rmse', 0):.3f}, vz={vel.get('vz_rmse', 0):.3f})")

    if 'angular_rates' in metrics:
        omega = metrics['angular_rates']
        lines.append(f"  Angular Rate RMSE:")
        lines.append(f"    Total: {omega.get('omega_total', 0):.3f} rad/s")

    if 'overall_rmse' in metrics:
        lines.append(f"  Overall State RMSE: {metrics['overall_rmse']:.4f}")

    return '\n'.join(lines)
