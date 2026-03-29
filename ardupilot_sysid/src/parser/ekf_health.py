"""
EKF health monitoring and segment filtering.

This module provides functionality to identify time segments where the Extended Kalman Filter
is healthy (low innovation ratio) and filter flight data to include only these segments.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from .message_types import TIMESTAMP_COL, EKF_INNOVATION_COL


def filter_ekf_healthy_segments(
    ekf_df: pd.DataFrame,
    innovation_threshold: float = 1.0,
    min_segment_duration: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Find time ranges where EKF innovation ratio is below threshold.

    The EKF innovation ratio (sometimes called innovation consistency) should be < 1.0
    for healthy operation. Values > 1.0 indicate the filter is having trouble reconciling
    sensor measurements with its predictions, which can happen during:
    - GPS glitches or multipath
    - Compass interference
    - Vibration-induced accelerometer clipping
    - Rapid maneuvers exceeding the model assumptions

    This function identifies continuous segments where the innovation stays healthy.

    Args:
        ekf_df: DataFrame with 'timestamp' and 'innovation_ratio' columns
        innovation_threshold: Maximum allowed innovation ratio (default 1.0)
        min_segment_duration: Minimum duration in seconds for a segment to be included
                             (filters out brief healthy periods)

    Returns:
        List of (start_time, end_time) tuples in seconds, representing healthy segments
    """
    if ekf_df.empty:
        return []

    if TIMESTAMP_COL not in ekf_df.columns:
        raise ValueError(f"DataFrame must contain '{TIMESTAMP_COL}' column")

    if EKF_INNOVATION_COL not in ekf_df.columns:
        raise ValueError(f"DataFrame must contain '{EKF_INNOVATION_COL}' column")

    # Sort by timestamp to ensure chronological order
    ekf_df = ekf_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Boolean mask for healthy samples
    is_healthy = ekf_df[EKF_INNOVATION_COL] < innovation_threshold

    # Find transitions: healthy→unhealthy and unhealthy→healthy
    transitions = is_healthy.astype(int).diff()

    # Find segment boundaries
    # transitions == 1: transition from unhealthy (0) to healthy (1) → segment start
    # transitions == -1: transition from healthy (1) to unhealthy (0) → segment end
    segment_starts = ekf_df.index[transitions == 1].tolist()
    segment_ends = ekf_df.index[transitions == -1].tolist()

    # Handle edge cases: log starts or ends during healthy period
    if is_healthy.iloc[0]:
        segment_starts.insert(0, 0)
    if is_healthy.iloc[-1]:
        segment_ends.append(len(ekf_df) - 1)

    # Build segment list with timestamps
    segments = []
    for start_idx, end_idx in zip(segment_starts, segment_ends):
        start_time = ekf_df.loc[start_idx, TIMESTAMP_COL]
        end_time = ekf_df.loc[end_idx, TIMESTAMP_COL]
        duration = end_time - start_time

        # Only include segments longer than minimum duration
        if duration >= min_segment_duration:
            segments.append((float(start_time), float(end_time)))

    return segments


def apply_segment_filter(
    df: pd.DataFrame,
    segments: List[Tuple[float, float]]
) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows within EKF-healthy segments.

    Args:
        df: DataFrame with a 'timestamp' column
        segments: List of (start_time, end_time) tuples defining valid segments

    Returns:
        Filtered DataFrame containing only data within the specified segments
    """
    if df.empty or not segments:
        return pd.DataFrame(columns=df.columns)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"DataFrame must contain '{TIMESTAMP_COL}' column")

    # Build boolean mask for all segments
    mask = pd.Series(False, index=df.index)

    for start_time, end_time in segments:
        segment_mask = (df[TIMESTAMP_COL] >= start_time) & (df[TIMESTAMP_COL] <= end_time)
        mask |= segment_mask

    return df[mask].copy()


def compute_segment_statistics(
    ekf_df: pd.DataFrame,
    segments: List[Tuple[float, float]]
) -> dict:
    """
    Compute statistics about EKF health segments.

    Args:
        ekf_df: DataFrame with EKF data
        segments: List of healthy segments

    Returns:
        Dictionary with segment statistics including:
        - num_segments: Number of healthy segments found
        - total_duration: Total duration of healthy segments (seconds)
        - mean_duration: Average segment duration (seconds)
        - min_duration: Shortest segment duration (seconds)
        - max_duration: Longest segment duration (seconds)
        - coverage: Fraction of total log time that is healthy (0-1)
    """
    if not segments:
        return {
            'num_segments': 0,
            'total_duration': 0.0,
            'mean_duration': 0.0,
            'min_duration': 0.0,
            'max_duration': 0.0,
            'coverage': 0.0,
        }

    durations = [end - start for start, end in segments]
    total_duration = sum(durations)

    # Compute total log duration
    log_duration = ekf_df[TIMESTAMP_COL].max() - ekf_df[TIMESTAMP_COL].min()

    return {
        'num_segments': len(segments),
        'total_duration': total_duration,
        'mean_duration': np.mean(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        'coverage': total_duration / log_duration if log_duration > 0 else 0.0,
    }


def print_segment_report(
    ekf_df: pd.DataFrame,
    segments: List[Tuple[float, float]],
    innovation_threshold: float = 1.0
) -> None:
    """
    Print a human-readable report of EKF health segments.

    Args:
        ekf_df: DataFrame with EKF data
        segments: List of healthy segments
        innovation_threshold: Threshold used for filtering
    """
    stats = compute_segment_statistics(ekf_df, segments)

    print("\n" + "="*60)
    print("EKF HEALTH SEGMENT REPORT")
    print("="*60)
    print(f"Innovation threshold: {innovation_threshold:.2f}")
    print(f"Number of healthy segments: {stats['num_segments']}")
    print(f"Total healthy duration: {stats['total_duration']:.1f} s")
    print(f"Coverage: {stats['coverage']*100:.1f}% of log")

    if stats['num_segments'] > 0:
        print(f"Average segment duration: {stats['mean_duration']:.1f} s")
        print(f"Shortest segment: {stats['min_duration']:.1f} s")
        print(f"Longest segment: {stats['max_duration']:.1f} s")

        print("\nSegment details:")
        for i, (start, end) in enumerate(segments):
            duration = end - start
            print(f"  [{i+1}] {start:.1f} - {end:.1f} s  (duration: {duration:.1f} s)")

    if stats['coverage'] < 0.5:
        print("\n⚠ WARNING: Less than 50% of the log has healthy EKF data.")
        print("  Consider using a different flight log with better GPS/compass conditions.")

    print("="*60 + "\n")
