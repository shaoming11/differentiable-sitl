"""
ArduPilot DataFlash message type constants and field mappings.

This module defines the message types and fields available in ArduPilot DataFlash logs,
along with normalized column names for standardized data processing.
"""

import numpy as np

# =============================================================================
# Message Type Names
# =============================================================================

IMU_MSG = 'IMU'
RCOUT_MSG = 'RCOUT'
ATT_MSG = 'ATT'
GPS_MSG = 'GPS'
BARO_MSG = 'BARO'
EKF_MSG = 'EKF1'  # EKF1 is the primary EKF message
PARM_MSG = 'PARM'

# =============================================================================
# Raw Field Mappings (as they appear in DataFlash logs)
# =============================================================================

IMU_FIELDS = ['TimeUS', 'GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']
RCOUT_FIELDS = ['TimeUS'] + [f'C{i}' for i in range(1, 15)]
ATT_FIELDS = ['TimeUS', 'Roll', 'Pitch', 'Yaw']
GPS_FIELDS = ['TimeUS', 'Lat', 'Lng', 'Alt', 'Spd', 'VelN', 'VelE', 'VelD']
BARO_FIELDS = ['TimeUS', 'Alt', 'Press', 'Temp']
EKF_FIELDS = ['TimeUS', 'SV']  # SV = innovation ratio (primary health metric)
PARM_FIELDS = ['Name', 'Value']

# =============================================================================
# Standard Column Names (normalized for processing)
# =============================================================================

# Time
TIMESTAMP_COL = 'timestamp'  # in seconds (converted from TimeUS)

# IMU columns
GYRO_COLS = ['gyr_x', 'gyr_y', 'gyr_z']  # rad/s (body frame)
ACCEL_COLS = ['acc_x', 'acc_y', 'acc_z']  # m/s^2 (body frame)

# PWM outputs (servo/motor channels)
PWM_COLS = [f'pwm_{i}' for i in range(1, 15)]  # microseconds (1000-2000)

# Attitude columns
ATTITUDE_COLS = ['roll', 'pitch', 'yaw']  # radians (Euler angles)

# GPS columns
GPS_POSITION_COLS = ['lat', 'lng', 'alt']  # degrees, degrees, meters
GPS_VELOCITY_COLS = ['vel_n', 'vel_e', 'vel_d']  # m/s (NED frame)
GPS_SPEED_COL = 'gps_speed'  # m/s (ground speed)

# Barometer columns
BARO_ALT_COL = 'baro_alt'  # meters (relative altitude)
BARO_PRESS_COL = 'baro_press'  # Pascals
BARO_TEMP_COL = 'baro_temp'  # degrees Celsius

# EKF health columns
EKF_INNOVATION_COL = 'innovation_ratio'  # dimensionless (should be < 1.0 for healthy)

# =============================================================================
# Message Rate Constants (typical values in Hz)
# =============================================================================

IMU_RATE_HZ = 400  # Standard IMU rate
RCOUT_RATE_HZ = 50  # Standard servo output rate
ATT_RATE_HZ = 400  # Attitude estimate rate
GPS_RATE_HZ = 10  # GPS update rate (can vary 5-10 Hz)
BARO_RATE_HZ = 10  # Barometer update rate
EKF_RATE_HZ = 10  # EKF update rate

# =============================================================================
# Unit Conversion Constants
# =============================================================================

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
US_TO_SEC = 1e-6  # Microseconds to seconds

# =============================================================================
# Field Mapping Dictionaries
# =============================================================================

# Maps message type to list of raw fields
MESSAGE_FIELDS = {
    IMU_MSG: IMU_FIELDS,
    RCOUT_MSG: RCOUT_FIELDS,
    ATT_MSG: ATT_FIELDS,
    GPS_MSG: GPS_FIELDS,
    BARO_MSG: BARO_FIELDS,
    EKF_MSG: EKF_FIELDS,
    PARM_MSG: PARM_FIELDS,
}

# Maps message type to normalized column names (after unit conversion)
NORMALIZED_COLUMNS = {
    IMU_MSG: [TIMESTAMP_COL] + GYRO_COLS + ACCEL_COLS,
    RCOUT_MSG: [TIMESTAMP_COL] + PWM_COLS,
    ATT_MSG: [TIMESTAMP_COL] + ATTITUDE_COLS,
    GPS_MSG: [TIMESTAMP_COL] + GPS_POSITION_COLS + [GPS_SPEED_COL] + GPS_VELOCITY_COLS,
    BARO_MSG: [TIMESTAMP_COL, BARO_ALT_COL, BARO_PRESS_COL],
    EKF_MSG: [TIMESTAMP_COL, EKF_INNOVATION_COL],
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_message_fields(msg_type: str) -> list[str]:
    """
    Get the list of raw fields for a given message type.

    Args:
        msg_type: Message type name (e.g., 'IMU', 'RCOUT')

    Returns:
        List of field names as they appear in the log

    Raises:
        ValueError: If message type is not recognized
    """
    if msg_type not in MESSAGE_FIELDS:
        raise ValueError(f"Unknown message type: {msg_type}. "
                        f"Known types: {list(MESSAGE_FIELDS.keys())}")
    return MESSAGE_FIELDS[msg_type]


def get_normalized_columns(msg_type: str) -> list[str]:
    """
    Get the list of normalized column names for a given message type.

    Args:
        msg_type: Message type name (e.g., 'IMU', 'RCOUT')

    Returns:
        List of normalized column names after unit conversion

    Raises:
        ValueError: If message type is not recognized
    """
    if msg_type not in NORMALIZED_COLUMNS:
        raise ValueError(f"Unknown message type: {msg_type}. "
                        f"Known types: {list(NORMALIZED_COLUMNS.keys())}")
    return NORMALIZED_COLUMNS[msg_type]


def get_message_rate(msg_type: str) -> float:
    """
    Get the typical update rate in Hz for a given message type.

    Args:
        msg_type: Message type name (e.g., 'IMU', 'RCOUT')

    Returns:
        Typical update rate in Hz
    """
    rates = {
        IMU_MSG: IMU_RATE_HZ,
        RCOUT_MSG: RCOUT_RATE_HZ,
        ATT_MSG: ATT_RATE_HZ,
        GPS_MSG: GPS_RATE_HZ,
        BARO_MSG: BARO_RATE_HZ,
        EKF_MSG: EKF_RATE_HZ,
    }
    return rates.get(msg_type, 1.0)  # Default to 1 Hz if unknown
