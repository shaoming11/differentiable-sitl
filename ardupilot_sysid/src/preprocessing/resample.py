"""
Resampling to uniform time grid.

This module provides functions to resample sensor data streams from their
native (potentially irregular) time grids to a common uniform time grid.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Dict, Optional, Tuple

from ..parser.message_types import TIMESTAMP_COL


def resample_to_uniform_grid(
    dataframes: Dict[str, pd.DataFrame],
    target_rate_hz: float = 400.0,
    time_range: Optional[Tuple[float, float]] = None,
    method: str = 'linear'
) -> Dict[str, pd.DataFrame]:
    """
    Resample all sensor streams to a uniform time grid.

    This function creates a common time base for all sensors, making subsequent
    processing (filtering, differentiation, etc.) much easier. The target rate
    should be chosen based on the highest-rate sensor (typically IMU at 400 Hz).

    Args:
        dataframes: dict of DataFrames (keys: 'imu', 'gps', 'rcout', 'baro', 'ekf', etc.)
                   Each DataFrame must have a 'timestamp' column.
        target_rate_hz: Target sample rate in Hz (default 400 Hz)
        time_range: (start_time, end_time) in seconds, or None for full overlap range
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        dict of resampled DataFrames on uniform grid

    Raises:
        ValueError: If dataframes is empty, DataFrames missing timestamp column,
                   or invalid interpolation method

    Example:
        >>> dataframes = {'imu': imu_df, 'gps': gps_df, 'rcout': rcout_df}
        >>> resampled = resample_to_uniform_grid(dataframes, target_rate_hz=400)
        >>> imu_uniform = resampled['imu']
    """
    if not dataframes:
        raise ValueError("dataframes dictionary is empty")

    # Validate all DataFrames have timestamp column
    for name, df in dataframes.items():
        if df.empty:
            raise ValueError(f"DataFrame '{name}' is empty")
        if TIMESTAMP_COL not in df.columns:
            raise ValueError(f"DataFrame '{name}' missing '{TIMESTAMP_COL}' column")

    # Validate interpolation method
    valid_methods = ['linear', 'cubic', 'nearest']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

    # Determine time range
    if time_range is None:
        # Use the overlap region (latest start, earliest end)
        t_start = max(df[TIMESTAMP_COL].iloc[0] for df in dataframes.values())
        t_end = min(df[TIMESTAMP_COL].iloc[-1] for df in dataframes.values())
    else:
        t_start, t_end = time_range

    if t_start >= t_end:
        raise ValueError(f"Invalid time range: start ({t_start:.3f}) >= end ({t_end:.3f})")

    # Create uniform time grid
    dt = 1.0 / target_rate_hz
    t_uniform = np.arange(t_start, t_end, dt)

    if len(t_uniform) == 0:
        raise ValueError(f"Time grid is empty. Check time range [{t_start:.3f}, {t_end:.3f}] "
                        f"with dt={dt:.6f}")

    # Resample each DataFrame
    resampled = {}
    for key, df in dataframes.items():
        try:
            resampled[key] = _interpolate_dataframe(df, t_uniform, method=method)
        except Exception as e:
            raise ValueError(f"Failed to resample DataFrame '{key}': {e}") from e

    return resampled


def _interpolate_dataframe(
    df: pd.DataFrame,
    t_new: np.ndarray,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Interpolate DataFrame to new time grid.

    Args:
        df: Input DataFrame with 'timestamp' column
        t_new: New time grid (1D array)
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        Resampled DataFrame with same columns as input

    Notes:
        - For 'cubic', requires at least 4 points
        - Extrapolation is NOT performed - values outside range are NaN
        - This is safer than extrapolation which can produce unrealistic values
    """
    # Create output DataFrame with new timestamps
    df_new = pd.DataFrame({TIMESTAMP_COL: t_new})

    # Get original timestamps
    t_old = df[TIMESTAMP_COL].values

    # Interpolate each column (except timestamp)
    for col in df.columns:
        if col == TIMESTAMP_COL:
            continue

        values_old = df[col].values

        # Check for NaNs in input data
        valid_mask = ~np.isnan(values_old)
        if not np.any(valid_mask):
            # All NaNs - just fill with NaN
            df_new[col] = np.nan
            continue

        # Only use valid points for interpolation
        t_valid = t_old[valid_mask]
        values_valid = values_old[valid_mask]

        # Check if we have enough points for the method
        if method == 'cubic' and len(t_valid) < 4:
            # Fall back to linear for sparse data
            interp_method = 'linear'
        else:
            interp_method = method

        try:
            # Create interpolator
            if interp_method == 'nearest':
                interp_func = interpolate.interp1d(
                    t_valid,
                    values_valid,
                    kind='nearest',
                    bounds_error=False,
                    fill_value=np.nan
                )
            elif interp_method == 'cubic':
                interp_func = interpolate.interp1d(
                    t_valid,
                    values_valid,
                    kind='cubic',
                    bounds_error=False,
                    fill_value=np.nan
                )
            else:  # linear
                interp_func = interpolate.interp1d(
                    t_valid,
                    values_valid,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

            # Interpolate to new time grid
            df_new[col] = interp_func(t_new)

        except Exception as e:
            raise ValueError(f"Interpolation failed for column '{col}': {e}") from e

    return df_new


def resample_single_stream(
    df: pd.DataFrame,
    target_rate_hz: float,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Resample a single DataFrame to a uniform time grid.

    Convenience function for resampling a single stream without the overhead
    of the multi-stream interface.

    Args:
        df: Input DataFrame with 'timestamp' column
        target_rate_hz: Target sample rate in Hz
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        Resampled DataFrame on uniform grid

    Example:
        >>> imu_uniform = resample_single_stream(imu_df, target_rate_hz=400)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"DataFrame missing '{TIMESTAMP_COL}' column")

    # Get time range from input data
    t_start = df[TIMESTAMP_COL].iloc[0]
    t_end = df[TIMESTAMP_COL].iloc[-1]

    # Create uniform time grid
    dt = 1.0 / target_rate_hz
    t_uniform = np.arange(t_start, t_end, dt)

    # Interpolate
    return _interpolate_dataframe(df, t_uniform, method=method)


def compute_resampling_stats(
    original_df: pd.DataFrame,
    resampled_df: pd.DataFrame
) -> Dict:
    """
    Compute statistics about the resampling operation.

    Args:
        original_df: Original DataFrame before resampling
        resampled_df: Resampled DataFrame

    Returns:
        Dictionary with resampling statistics

    Example:
        >>> stats = compute_resampling_stats(imu_df, imu_resampled)
        >>> print(f"Upsampling factor: {stats['rate_ratio']:.2f}x")
    """
    if original_df.empty or resampled_df.empty:
        return {
            'original_samples': len(original_df),
            'resampled_samples': len(resampled_df),
            'original_rate_hz': 0.0,
            'resampled_rate_hz': 0.0,
            'rate_ratio': 0.0,
            'time_span_s': 0.0,
        }

    # Compute original sample rate
    dt_orig = np.median(np.diff(original_df[TIMESTAMP_COL].values))
    rate_orig = 1.0 / dt_orig if dt_orig > 0 else 0.0

    # Compute resampled sample rate
    dt_resamp = np.median(np.diff(resampled_df[TIMESTAMP_COL].values))
    rate_resamp = 1.0 / dt_resamp if dt_resamp > 0 else 0.0

    # Time span
    t_span = resampled_df[TIMESTAMP_COL].iloc[-1] - resampled_df[TIMESTAMP_COL].iloc[0]

    return {
        'original_samples': len(original_df),
        'resampled_samples': len(resampled_df),
        'original_rate_hz': float(rate_orig),
        'resampled_rate_hz': float(rate_resamp),
        'rate_ratio': float(rate_resamp / rate_orig) if rate_orig > 0 else 0.0,
        'time_span_s': float(t_span),
    }


def print_resampling_report(original_dfs: Dict[str, pd.DataFrame],
                            resampled_dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Print a human-readable report of resampling results.

    Args:
        original_dfs: Dictionary of original DataFrames
        resampled_dfs: Dictionary of resampled DataFrames
    """
    print("\n" + "="*60)
    print("RESAMPLING REPORT")
    print("="*60)

    for name in original_dfs.keys():
        if name not in resampled_dfs:
            continue

        stats = compute_resampling_stats(original_dfs[name], resampled_dfs[name])

        print(f"\n{name.upper()}:")
        print(f"  Original: {stats['original_samples']} samples @ {stats['original_rate_hz']:.1f} Hz")
        print(f"  Resampled: {stats['resampled_samples']} samples @ {stats['resampled_rate_hz']:.1f} Hz")
        print(f"  Rate ratio: {stats['rate_ratio']:.2f}x")

        if stats['rate_ratio'] > 1.0:
            print(f"  → Upsampling (interpolation)")
        elif stats['rate_ratio'] < 1.0:
            print(f"  → Downsampling")
        else:
            print(f"  → No change in rate")

    # Overall stats
    total_orig = sum(len(df) for df in original_dfs.values())
    total_resamp = sum(len(df) for df in resampled_dfs.values())

    print(f"\nTOTAL:")
    print(f"  Original: {total_orig:,} samples")
    print(f"  Resampled: {total_resamp:,} samples")

    # Time span (assume all are on same grid now)
    if resampled_dfs:
        first_df = next(iter(resampled_dfs.values()))
        t_span = first_df[TIMESTAMP_COL].iloc[-1] - first_df[TIMESTAMP_COL].iloc[0]
        print(f"  Time span: {t_span:.2f} s")

    print("="*60 + "\n")
