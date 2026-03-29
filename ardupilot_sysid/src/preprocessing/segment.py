"""
Data segmentation by EKF health.

This module provides functions to segment flight data based on EKF (Extended Kalman Filter)
health metrics. Only segments with good EKF health should be used for system identification.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

from ..parser.message_types import TIMESTAMP_COL


def segment_by_ekf_health(
    ekf_df: pd.DataFrame,
    innovation_threshold: float = 1.0,
    min_segment_duration_s: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Find time ranges where EKF is healthy (innovation ratio < threshold).

    The EKF innovation ratio (sometimes called innovation consistency) indicates
    how well the filter is tracking reality:
    - < 0.5: Excellent - filter predictions match measurements very well
    - 0.5-1.0: Good - acceptable tracking
    - > 1.0: Poor - filter is struggling, measurements disagree with predictions

    Unhealthy EKF can occur due to:
    - GPS glitches or multipath interference
    - Compass interference (magnetic disturbances)
    - Vibration-induced accelerometer clipping
    - Rapid maneuvers exceeding model assumptions
    - Sensor failures or dropouts

    Args:
        ekf_df: DataFrame with columns [timestamp, innovation_ratio (SV), ...]
        innovation_threshold: Max allowed innovation ratio (default 1.0)
        min_segment_duration_s: Minimum segment duration to keep (seconds)

    Returns:
        List of (start_time, end_time) tuples representing healthy segments

    Example:
        >>> segments = segment_by_ekf_health(ekf_df, innovation_threshold=0.8)
        >>> print(f"Found {len(segments)} healthy segments")
    """
    if ekf_df.empty:
        return []

    if TIMESTAMP_COL not in ekf_df.columns:
        raise ValueError(f"EKF DataFrame must contain '{TIMESTAMP_COL}' column")

    # Check for innovation ratio column (try different possible names)
    innovation_col = None
    possible_cols = ['innovation_ratio', 'SV', 'innov_ratio']

    for col in possible_cols:
        if col in ekf_df.columns:
            innovation_col = col
            break

    if innovation_col is None:
        # If no EKF health data, return full time range as single segment
        # This allows processing to continue even without EKF data
        return [(ekf_df[TIMESTAMP_COL].iloc[0], ekf_df[TIMESTAMP_COL].iloc[-1])]

    # Sort by timestamp to ensure chronological order
    ekf_df = ekf_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Find healthy samples (innovation below threshold)
    healthy = ekf_df[innovation_col].values < innovation_threshold

    # Handle case where all samples are unhealthy
    if not np.any(healthy):
        return []

    # Handle case where all samples are healthy
    if np.all(healthy):
        t_start = ekf_df[TIMESTAMP_COL].iloc[0]
        t_end = ekf_df[TIMESTAMP_COL].iloc[-1]
        duration = t_end - t_start

        if duration >= min_segment_duration_s:
            return [(float(t_start), float(t_end))]
        else:
            return []

    # Find segment boundaries (where healthy changes)
    # diff converts boolean to int: False=0, True=1
    # 0→1 (transition to healthy) gives diff=+1
    # 1→0 (transition to unhealthy) gives diff=-1
    changes = np.diff(healthy.astype(int))
    segment_starts = np.where(changes == 1)[0] + 1  # +1 because diff shifts indices
    segment_ends = np.where(changes == -1)[0] + 1

    # Handle edge cases: log starts or ends during healthy period
    if healthy[0]:
        segment_starts = np.concatenate([[0], segment_starts])
    if healthy[-1]:
        segment_ends = np.concatenate([segment_ends, [len(healthy)]])

    # Convert to time ranges and filter by duration
    segments = []
    timestamps = ekf_df[TIMESTAMP_COL].values

    for start_idx, end_idx in zip(segment_starts, segment_ends):
        # Clamp indices to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(timestamps), end_idx)

        t_start = timestamps[start_idx]
        t_end = timestamps[end_idx - 1]
        duration = t_end - t_start

        # Only include segments longer than minimum duration
        if duration >= min_segment_duration_s:
            segments.append((float(t_start), float(t_end)))

    return segments


def apply_segments(
    df: pd.DataFrame,
    segments: List[Tuple[float, float]]
) -> List[pd.DataFrame]:
    """
    Split DataFrame into multiple DataFrames, one per segment.

    This is useful for processing each healthy segment independently,
    for example when fitting separate models to each segment.

    Args:
        df: DataFrame with a 'timestamp' column
        segments: List of (start_time, end_time) tuples defining valid segments

    Returns:
        List of DataFrames, each containing data for one segment

    Example:
        >>> segments = segment_by_ekf_health(ekf_df)
        >>> imu_segments = apply_segments(imu_df, segments)
        >>> print(f"Split into {len(imu_segments)} segments")
    """
    if df.empty:
        return []

    if not segments:
        return []

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"DataFrame must contain '{TIMESTAMP_COL}' column")

    result = []
    timestamps = df[TIMESTAMP_COL].values

    for t_start, t_end in segments:
        # Use boolean indexing for efficiency
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        segment_df = df[mask].copy()

        # Only include non-empty segments
        if len(segment_df) > 0:
            # Reset index for clean segment DataFrames
            segment_df = segment_df.reset_index(drop=True)
            result.append(segment_df)

    return result


def merge_close_segments(
    segments: List[Tuple[float, float]],
    max_gap_s: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Merge segments that are close together in time.

    Brief unhealthy periods (< max_gap_s) between healthy segments are often
    just transient glitches. Merging them produces longer, more useful segments
    for system identification.

    Args:
        segments: List of (start_time, end_time) tuples
        max_gap_s: Maximum gap in seconds to merge (default 2.0s)

    Returns:
        List of merged segments

    Example:
        >>> segments = [(0, 10), (12, 20), (25, 30)]
        >>> merged = merge_close_segments(segments, max_gap_s=3.0)
        >>> # Result: [(0, 20), (25, 30)]  (first two merged)
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])

    merged = [sorted_segments[0]]

    for current_start, current_end in sorted_segments[1:]:
        prev_start, prev_end = merged[-1]

        # Check if gap is small enough to merge
        gap = current_start - prev_end

        if gap <= max_gap_s:
            # Merge with previous segment
            merged[-1] = (prev_start, current_end)
        else:
            # Start new segment
            merged.append((current_start, current_end))

    return merged


def filter_segments_by_criteria(
    segments: List[Tuple[float, float]],
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    max_count: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Filter segments by duration and count criteria.

    Useful for selecting the best segments for system identification.
    For example, you might want only the longest segments, or segments
    within a specific duration range.

    Args:
        segments: List of (start_time, end_time) tuples
        min_duration_s: Minimum duration (None = no minimum)
        max_duration_s: Maximum duration (None = no maximum)
        max_count: Maximum number of segments to return (keeps longest)

    Returns:
        Filtered list of segments

    Example:
        >>> # Keep only the 3 longest segments above 10 seconds
        >>> best = filter_segments_by_criteria(segments, min_duration_s=10, max_count=3)
    """
    if not segments:
        return []

    filtered = segments.copy()

    # Filter by minimum duration
    if min_duration_s is not None:
        filtered = [(s, e) for s, e in filtered if (e - s) >= min_duration_s]

    # Filter by maximum duration
    if max_duration_s is not None:
        filtered = [(s, e) for s, e in filtered if (e - s) <= max_duration_s]

    # Limit count (keep longest segments)
    if max_count is not None and len(filtered) > max_count:
        # Sort by duration (longest first)
        filtered = sorted(filtered, key=lambda x: x[1] - x[0], reverse=True)
        filtered = filtered[:max_count]
        # Re-sort by start time
        filtered = sorted(filtered, key=lambda x: x[0])

    return filtered


def summarize_segments(segments: List[Tuple[float, float]]) -> Dict:
    """
    Compute summary statistics for segments.

    Args:
        segments: List of (start_time, end_time) tuples

    Returns:
        dict with keys ['n_segments', 'total_duration_s', 'mean_duration_s',
                       'min_duration_s', 'max_duration_s', 'std_duration_s']

    Example:
        >>> stats = summarize_segments(segments)
        >>> print(f"Found {stats['n_segments']} segments "
        ...       f"with mean duration {stats['mean_duration_s']:.1f}s")
    """
    if not segments:
        return {
            'n_segments': 0,
            'total_duration_s': 0.0,
            'mean_duration_s': 0.0,
            'min_duration_s': 0.0,
            'max_duration_s': 0.0,
            'std_duration_s': 0.0,
        }

    durations = [end - start for start, end in segments]

    return {
        'n_segments': len(segments),
        'total_duration_s': float(sum(durations)),
        'mean_duration_s': float(np.mean(durations)),
        'min_duration_s': float(min(durations)),
        'max_duration_s': float(max(durations)),
        'std_duration_s': float(np.std(durations)),
    }


def print_segment_report(segments: List[Tuple[float, float]],
                        ekf_df: Optional[pd.DataFrame] = None,
                        innovation_threshold: float = 1.0) -> None:
    """
    Print a human-readable report of segmentation results.

    Args:
        segments: List of healthy segments
        ekf_df: Optional EKF DataFrame for computing coverage statistics
        innovation_threshold: Threshold used for filtering
    """
    print("\n" + "="*60)
    print("EKF HEALTH SEGMENT REPORT")
    print("="*60)
    print(f"Innovation threshold: {innovation_threshold:.2f}")

    stats = summarize_segments(segments)

    print(f"Number of healthy segments: {stats['n_segments']}")
    print(f"Total healthy duration: {stats['total_duration_s']:.1f} s")

    if stats['n_segments'] > 0:
        print(f"Mean segment duration: {stats['mean_duration_s']:.1f} s")
        print(f"Shortest segment: {stats['min_duration_s']:.1f} s")
        print(f"Longest segment: {stats['max_duration_s']:.1f} s")
        print(f"Std deviation: {stats['std_duration_s']:.1f} s")

        # Compute coverage if EKF data provided
        if ekf_df is not None and not ekf_df.empty:
            log_duration = ekf_df[TIMESTAMP_COL].max() - ekf_df[TIMESTAMP_COL].min()
            coverage = stats['total_duration_s'] / log_duration if log_duration > 0 else 0.0
            print(f"Coverage: {coverage*100:.1f}% of log")

            if coverage < 0.3:
                print("\n⚠ WARNING: Less than 30% of the log has healthy EKF data.")
                print("  Consider using a different flight log with better GPS/compass conditions.")
            elif coverage < 0.5:
                print("\n⚠ NOTE: Only {:.1f}% of the log is usable.".format(coverage*100))
                print("  Results may be limited by small sample size.")

        print("\nSegment details:")
        for i, (start, end) in enumerate(segments):
            duration = end - start
            print(f"  [{i+1:2d}] {start:8.1f} - {end:8.1f} s  (duration: {duration:6.1f} s)")

    else:
        print("\nNo healthy segments found!")
        print("This usually indicates:")
        print("  - Poor GPS signal quality (multipath, low satellite count)")
        print("  - Compass interference (magnetic disturbances)")
        print("  - Excessive vibration")
        print("  - Aggressive flight exceeding EKF model assumptions")
        print("\nRecommendations:")
        print("  - Try a different flight log")
        print("  - Increase innovation_threshold (e.g., 1.5 or 2.0)")
        print("  - Check sensor calibration and mounting")

    print("="*60 + "\n")


def get_segment_indices(
    df: pd.DataFrame,
    segments: List[Tuple[float, float]]
) -> List[Tuple[int, int]]:
    """
    Convert time-based segments to index-based segments.

    Useful when you need to slice arrays by index rather than time.

    Args:
        df: DataFrame with 'timestamp' column
        segments: List of (start_time, end_time) tuples

    Returns:
        List of (start_index, end_index) tuples

    Example:
        >>> idx_segments = get_segment_indices(imu_df, segments)
        >>> for start_idx, end_idx in idx_segments:
        ...     segment_data = imu_df.iloc[start_idx:end_idx]
    """
    if df.empty or not segments:
        return []

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"DataFrame must contain '{TIMESTAMP_COL}' column")

    timestamps = df[TIMESTAMP_COL].values
    idx_segments = []

    for t_start, t_end in segments:
        # Find indices closest to segment times
        start_idx = np.searchsorted(timestamps, t_start, side='left')
        end_idx = np.searchsorted(timestamps, t_end, side='right')

        # Clamp to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(timestamps), end_idx)

        if start_idx < end_idx:
            idx_segments.append((int(start_idx), int(end_idx)))

    return idx_segments
