"""
ArduPilot DataFlash log parsing utilities.

This package provides tools for parsing ArduPilot .bin log files and extracting
sensor data into pandas DataFrames with normalized units and timestamps.

Modules:
    message_types: Constants and field mappings for ArduPilot message types
    dflog_reader: Main log parsing class (DFLogReader)
    ekf_health: EKF health monitoring and segment filtering

Example:
    >>> from ardupilot_sysid.src.parser import DFLogReader
    >>> reader = DFLogReader('flight.bin')
    >>> data = reader.parse()
    >>> imu_df = data['imu']
"""

from .dflog_reader import DFLogReader, print_log_summary
from .ekf_health import (
    filter_ekf_healthy_segments,
    apply_segment_filter,
    compute_segment_statistics,
    print_segment_report,
)
from .message_types import (
    # Message type constants
    IMU_MSG,
    RCOUT_MSG,
    ATT_MSG,
    GPS_MSG,
    BARO_MSG,
    EKF_MSG,
    PARM_MSG,
    # Column name constants
    TIMESTAMP_COL,
    GYRO_COLS,
    ACCEL_COLS,
    ATTITUDE_COLS,
    PWM_COLS,
    GPS_POSITION_COLS,
    GPS_VELOCITY_COLS,
    # Helper functions
    get_message_fields,
    get_normalized_columns,
    get_message_rate,
)

__all__ = [
    # Main classes
    'DFLogReader',
    # Log summary
    'print_log_summary',
    # EKF health filtering
    'filter_ekf_healthy_segments',
    'apply_segment_filter',
    'compute_segment_statistics',
    'print_segment_report',
    # Message types
    'IMU_MSG',
    'RCOUT_MSG',
    'ATT_MSG',
    'GPS_MSG',
    'BARO_MSG',
    'EKF_MSG',
    'PARM_MSG',
    # Column names
    'TIMESTAMP_COL',
    'GYRO_COLS',
    'ACCEL_COLS',
    'ATTITUDE_COLS',
    'PWM_COLS',
    'GPS_POSITION_COLS',
    'GPS_VELOCITY_COLS',
    # Helper functions
    'get_message_fields',
    'get_normalized_columns',
    'get_message_rate',
]
