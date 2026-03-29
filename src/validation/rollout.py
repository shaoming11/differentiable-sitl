import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List

from ardupilot_sysid.src.fdm.multicopter_jax import rollout


def hold_out_validation(
    params: Dict,
    state_trajectory_holdout: jnp.ndarray,
    pwm_sequence_holdout: jnp.ndarray,
    dt: float
) -> Dict:
    """
    Validate identified parameters on hold-out data.

    Roll out FDM with identified parameters on hold-out PWM sequence
    and compare to actual (smoothed) states.

    Args:
        params: Identified FDM parameters
        state_trajectory_holdout: (T, 10) smoothed states from RTS (ground truth)
        pwm_sequence_holdout: (T-1, N) PWM commands for hold-out segment
        dt: Timestep

    Returns:
        Dict with keys:
            - 'predicted': (T, 10) predicted state trajectory
            - 'actual': (T, 10) actual state trajectory
            - 'residuals': (T, 10) prediction errors
            - 'timestamps': (T,) time vector

    Example:
        >>> result = hold_out_validation(params_optimal, states_holdout, pwm_holdout, dt)
        >>> print(f"Mean prediction error: {np.mean(np.abs(result['residuals'])):.4f}")
    """
    # Initial state from hold-out segment
    state_init = state_trajectory_holdout[0]

    # Roll out FDM
    predicted = rollout(state_init, pwm_sequence_holdout, params, dt)

    # Actual trajectory (skip first state since rollout produces T-1 states)
    actual = state_trajectory_holdout[1:]

    # Compute residuals
    residuals = predicted - actual

    # Generate timestamps
    T = len(predicted)
    timestamps = np.arange(T) * dt

    return {
        'predicted': np.array(predicted),
        'actual': np.array(actual),
        'residuals': np.array(residuals),
        'timestamps': timestamps
    }


def split_train_test(
    segments: List[Tuple[float, float]],
    test_ratio: float = 0.2
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Split segments into training and test sets.

    Strategy: Use last test_ratio of segments for testing (chronologically).

    Args:
        segments: List of (start_time, end_time) tuples
        test_ratio: Fraction for testing (default 0.2 = 20%)

    Returns:
        (train_segments, test_segments)
    """
    n_test = max(1, int(len(segments) * test_ratio))

    train_segments = segments[:-n_test]
    test_segments = segments[-n_test:]

    return train_segments, test_segments


def compare_trajectories(
    predicted: np.ndarray,
    actual: np.ndarray,
    state_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute comparison metrics between predicted and actual trajectories.

    Args:
        predicted: (T, n_states) predicted states
        actual: (T, n_states) actual states
        state_names: List of state dimension names

    Returns:
        Dict of metrics per state dimension
    """
    if state_names is None:
        state_names = ['qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']

    metrics = {}

    for i, name in enumerate(state_names):
        pred_i = predicted[:, i]
        actual_i = actual[:, i]

        # RMSE
        rmse = float(np.sqrt(np.mean((pred_i - actual_i)**2)))

        # MAE
        mae = float(np.mean(np.abs(pred_i - actual_i)))

        # Correlation
        if np.std(pred_i) > 1e-10 and np.std(actual_i) > 1e-10:
            corr = float(np.corrcoef(pred_i, actual_i)[0, 1])
        else:
            corr = 1.0 if mae < 1e-10 else 0.0

        metrics[name] = {
            'rmse': rmse,
            'mae': mae,
            'correlation': corr
        }

    return metrics
