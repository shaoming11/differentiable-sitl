"""Data preprocessing: alignment, resampling, and segmentation."""

from .align import (
    align_timestamps,
    check_timestamp_jitter,
    print_alignment_report,
)

from .resample import (
    resample_to_uniform_grid,
    resample_single_stream,
    compute_resampling_stats,
    print_resampling_report,
)

from .segment import (
    segment_by_ekf_health,
    apply_segments,
    merge_close_segments,
    filter_segments_by_criteria,
    summarize_segments,
    print_segment_report,
    get_segment_indices,
)

__all__ = [
    # Alignment
    'align_timestamps',
    'check_timestamp_jitter',
    'print_alignment_report',
    # Resampling
    'resample_to_uniform_grid',
    'resample_single_stream',
    'compute_resampling_stats',
    'print_resampling_report',
    # Segmentation
    'segment_by_ekf_health',
    'apply_segments',
    'merge_close_segments',
    'filter_segments_by_criteria',
    'summarize_segments',
    'print_segment_report',
    'get_segment_indices',
]
