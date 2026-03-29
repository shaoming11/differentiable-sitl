"""
Timestamp alignment using cross-correlation.

This module aligns sensor streams by detecting hardware latencies through
cross-correlation of IMU and GPS velocity signals.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from typing import Tuple, Dict

from ..parser.message_types import TIMESTAMP_COL


def align_timestamps(
    imu_df: pd.DataFrame,
    gps_df: pd.DataFrame,
    rcout_df: pd.DataFrame,
    max_lag_ms: float = 500.0,
    highpass_cutoff: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Align sensor streams by cross-correlating IMU and GPS velocity.

    Strategy:
    1. Use IMU as reference clock (highest rate, lowest latency)
    2. Integrate IMU acceleration to get velocity estimate
    3. Cross-correlate with GPS velocity to find GPS latency
    4. Shift GPS timestamps to compensate
    5. Assume RCOUT latency is negligible (already accounted for in tau_motor)

    The cross-correlation approach works because:
    - GPS velocity is a direct measurement (but delayed)
    - Integrated IMU acceleration gives a velocity estimate (but drifts)
    - High-pass filtering removes DC offset/drift differences
    - The peak correlation lag reveals the hardware latency

    Args:
        imu_df: DataFrame with columns [timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        gps_df: DataFrame with columns [timestamp, vel_n, vel_e, vel_d, lat, lng, alt, ...]
        rcout_df: DataFrame with columns [timestamp, pwm_1, ..., pwm_N]
        max_lag_ms: Maximum lag to search in milliseconds (default 500ms)
        highpass_cutoff: High-pass filter cutoff frequency in Hz (default 0.5 Hz)

    Returns:
        (imu_df, gps_df_aligned, rcout_df, metadata)
        metadata: dict with keys ['gps_latency_ms', 'correlation_peak', 'lag_samples', 'quality']

    Raises:
        ValueError: If required columns are missing or DataFrames are empty
    """
    # Validate inputs
    _validate_dataframes(imu_df, gps_df, rcout_df)

    # Check if we have enough data for meaningful cross-correlation
    if len(imu_df) < 100:
        raise ValueError(f"IMU data too short for alignment: {len(imu_df)} samples (need at least 100)")

    if len(gps_df) < 10:
        raise ValueError(f"GPS data too short for alignment: {len(gps_df)} samples (need at least 10)")

    # Estimate IMU sample rate
    imu_dt = np.median(np.diff(imu_df[TIMESTAMP_COL].values))
    imu_rate = 1.0 / imu_dt if imu_dt > 0 else 400.0

    # Convert max_lag from ms to samples
    max_lag_samples = int((max_lag_ms / 1000.0) * imu_rate)

    # Resample GPS velocity to IMU grid for cross-correlation
    gps_vel_n_resampled = np.interp(
        imu_df[TIMESTAMP_COL].values,
        gps_df[TIMESTAMP_COL].values,
        gps_df['vel_n'].values
    )

    gps_vel_e_resampled = np.interp(
        imu_df[TIMESTAMP_COL].values,
        gps_df[TIMESTAMP_COL].values,
        gps_df['vel_e'].values
    )

    # Integrate IMU acceleration to get velocity estimate
    # Note: This assumes body frame is approximately aligned with NED
    # For more accuracy, should rotate using attitude, but for cross-correlation
    # the rough alignment is sufficient
    imu_vel_n_integrated = cumulative_trapezoid(
        imu_df['acc_x'].values,
        imu_df[TIMESTAMP_COL].values,
        initial=0
    )

    imu_vel_e_integrated = cumulative_trapezoid(
        imu_df['acc_y'].values,
        imu_df[TIMESTAMP_COL].values,
        initial=0
    )

    # High-pass filter both signals to remove DC offset/drift
    # This is crucial because:
    # - GPS and integrated IMU have different absolute velocities
    # - Integration causes drift in IMU velocity
    # - We only care about dynamic correlation, not DC level
    try:
        sos = signal.butter(4, highpass_cutoff, 'highpass', fs=imu_rate, output='sos')

        # Filter north velocity
        imu_vel_n_filtered = signal.sosfilt(sos, imu_vel_n_integrated)
        gps_vel_n_filtered = signal.sosfilt(sos, gps_vel_n_resampled)

        # Filter east velocity
        imu_vel_e_filtered = signal.sosfilt(sos, imu_vel_e_integrated)
        gps_vel_e_filtered = signal.sosfilt(sos, gps_vel_e_resampled)

    except ValueError as e:
        raise ValueError(f"Filter design failed (sample rate: {imu_rate:.1f} Hz, "
                        f"cutoff: {highpass_cutoff} Hz). Try a lower cutoff frequency.") from e

    # Cross-correlate both axes and average for more robust estimate
    # Use 'same' mode and restrict to max_lag
    correlation_n = signal.correlate(gps_vel_n_filtered, imu_vel_n_filtered, mode='same')
    correlation_e = signal.correlate(gps_vel_e_filtered, imu_vel_e_filtered, mode='same')

    # Average correlations (gives more stable peak)
    correlation = (correlation_n + correlation_e) / 2.0

    # Create lag array centered at zero
    lags = signal.correlation_lags(len(gps_vel_n_filtered), len(imu_vel_n_filtered), mode='same')

    # Restrict search to reasonable lag range
    valid_mask = np.abs(lags) <= max_lag_samples
    correlation_restricted = correlation[valid_mask]
    lags_restricted = lags[valid_mask]

    # Find peak (positive lag means GPS is delayed relative to IMU)
    peak_idx = np.argmax(np.abs(correlation_restricted))
    lag_samples = lags_restricted[peak_idx]
    lag_seconds = lag_samples / imu_rate

    # Quality metrics
    correlation_peak = float(correlation_restricted[peak_idx])
    correlation_std = float(np.std(correlation_restricted))
    quality = correlation_peak / correlation_std if correlation_std > 0 else 0.0

    # Shift GPS timestamps backward to compensate for latency
    # If lag is positive, GPS was delayed, so subtract the lag
    gps_df_aligned = gps_df.copy()
    gps_df_aligned[TIMESTAMP_COL] = gps_df_aligned[TIMESTAMP_COL] - lag_seconds

    metadata = {
        'gps_latency_ms': float(lag_seconds * 1000),
        'correlation_peak': correlation_peak,
        'lag_samples': int(lag_samples),
        'quality': quality,
        'imu_sample_rate_hz': float(imu_rate),
        'search_range_ms': float(max_lag_ms),
    }

    return imu_df, gps_df_aligned, rcout_df, metadata


def check_timestamp_jitter(df: pd.DataFrame, name: str = "data") -> Dict:
    """
    Analyze timestamp jitter (variation in sample intervals).

    Timestamp jitter can indicate:
    - CPU scheduling delays
    - Buffer overflow/underflow
    - Hardware timing issues

    Typical jitter should be < 1ms for 400 Hz IMU.

    Args:
        df: DataFrame with 'timestamp' column
        name: Name of the data stream for reporting

    Returns:
        dict with keys ['name', 'mean_dt', 'std_dt', 'max_jitter_ms', 'sample_rate_hz']

    Raises:
        ValueError: If DataFrame is empty or missing timestamp column
    """
    if df.empty:
        raise ValueError(f"DataFrame '{name}' is empty")

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"DataFrame '{name}' missing '{TIMESTAMP_COL}' column")

    if len(df) < 2:
        raise ValueError(f"DataFrame '{name}' needs at least 2 samples to compute jitter")

    timestamps = df[TIMESTAMP_COL].values
    dt = np.diff(timestamps)

    # Remove outliers (e.g., data gaps) for statistics
    # Use median absolute deviation (MAD) for robust outlier detection
    median_dt = np.median(dt)
    mad = np.median(np.abs(dt - median_dt))

    # Keep samples within 5 MAD of median (robust to outliers)
    if mad > 0:
        valid_mask = np.abs(dt - median_dt) < 5 * mad
        dt_clean = dt[valid_mask]
    else:
        dt_clean = dt

    mean_dt = float(np.mean(dt_clean))
    std_dt = float(np.std(dt_clean))
    max_jitter_ms = float(np.max(np.abs(dt_clean - mean_dt)) * 1000)
    sample_rate_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0

    return {
        'name': name,
        'mean_dt': mean_dt,
        'std_dt': std_dt,
        'max_jitter_ms': max_jitter_ms,
        'sample_rate_hz': sample_rate_hz,
        'num_samples': len(df),
        'num_outliers': int(np.sum(~valid_mask)) if mad > 0 else 0,
    }


def _validate_dataframes(imu_df: pd.DataFrame, gps_df: pd.DataFrame, rcout_df: pd.DataFrame) -> None:
    """
    Validate that DataFrames have required columns and are not empty.

    Raises:
        ValueError: If validation fails
    """
    # Check IMU
    if imu_df.empty:
        raise ValueError("IMU DataFrame is empty")

    required_imu_cols = [TIMESTAMP_COL, 'acc_x', 'acc_y', 'acc_z']
    missing_imu = [col for col in required_imu_cols if col not in imu_df.columns]
    if missing_imu:
        raise ValueError(f"IMU DataFrame missing columns: {missing_imu}")

    # Check GPS
    if gps_df.empty:
        raise ValueError("GPS DataFrame is empty")

    required_gps_cols = [TIMESTAMP_COL, 'vel_n', 'vel_e']
    missing_gps = [col for col in required_gps_cols if col not in gps_df.columns]
    if missing_gps:
        raise ValueError(f"GPS DataFrame missing columns: {missing_gps}")

    # Check RCOUT
    if rcout_df.empty:
        raise ValueError("RCOUT DataFrame is empty")

    if TIMESTAMP_COL not in rcout_df.columns:
        raise ValueError(f"RCOUT DataFrame missing '{TIMESTAMP_COL}' column")


def print_alignment_report(metadata: Dict) -> None:
    """
    Print a human-readable report of timestamp alignment results.

    Args:
        metadata: Dictionary returned by align_timestamps()
    """
    print("\n" + "="*60)
    print("TIMESTAMP ALIGNMENT REPORT")
    print("="*60)
    print(f"IMU sample rate: {metadata['imu_sample_rate_hz']:.1f} Hz")
    print(f"GPS latency detected: {metadata['gps_latency_ms']:.1f} ms")
    print(f"Lag in samples: {metadata['lag_samples']}")
    print(f"Correlation peak: {metadata['correlation_peak']:.3e}")
    print(f"Alignment quality: {metadata['quality']:.2f}")
    print(f"Search range: ±{metadata['search_range_ms']:.0f} ms")

    # Quality assessment
    if metadata['quality'] > 10.0:
        quality_str = "EXCELLENT - High confidence in alignment"
    elif metadata['quality'] > 5.0:
        quality_str = "GOOD - Reliable alignment"
    elif metadata['quality'] > 2.0:
        quality_str = "FAIR - Moderate confidence"
    else:
        quality_str = "POOR - Low confidence, verify manually"

    print(f"Assessment: {quality_str}")

    # Latency assessment
    latency_ms = abs(metadata['gps_latency_ms'])
    if latency_ms < 50:
        latency_str = "Unusually low - verify GPS configuration"
    elif latency_ms < 250:
        latency_str = "Normal GPS latency"
    else:
        latency_str = "High latency - check GPS antenna/settings"

    print(f"Latency: {latency_str}")
    print("="*60 + "\n")
