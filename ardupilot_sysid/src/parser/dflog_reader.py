"""
ArduPilot DataFlash log reader.

This module provides the DFLogReader class for parsing ArduPilot DataFlash .bin logs
and extracting sensor data into pandas DataFrames with normalized units and timestamps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from pymavlink import mavutil

from .message_types import (
    IMU_MSG, RCOUT_MSG, ATT_MSG, GPS_MSG, BARO_MSG, EKF_MSG, PARM_MSG,
    TIMESTAMP_COL, GYRO_COLS, ACCEL_COLS, PWM_COLS, ATTITUDE_COLS,
    GPS_POSITION_COLS, GPS_VELOCITY_COLS, GPS_SPEED_COL,
    BARO_ALT_COL, BARO_PRESS_COL, EKF_INNOVATION_COL,
    DEG_TO_RAD, US_TO_SEC,
)


class DFLogReader:
    """
    Reader for ArduPilot DataFlash binary logs.

    This class parses .bin log files using pymavlink and extracts relevant
    sensor messages into pandas DataFrames with standardized column names
    and SI units.
    """

    def __init__(self, log_path: str):
        """
        Initialize the log reader.

        Args:
            log_path: Path to the .bin log file

        Raises:
            FileNotFoundError: If log file does not exist
            ValueError: If log file cannot be opened
        """
        self.log_path = Path(log_path)

        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        if not self.log_path.suffix == '.bin':
            raise ValueError(f"Expected .bin file, got: {self.log_path.suffix}")

        self.mlog = None

    def parse(self) -> Dict[str, pd.DataFrame]:
        """
        Parse log file and return dictionary of DataFrames.

        Returns:
            Dictionary with keys:
                'imu': DataFrame [timestamp, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z]
                'rcout': DataFrame [timestamp, pwm_1, ..., pwm_14]
                'att': DataFrame [timestamp, roll, pitch, yaw] (radians)
                'gps': DataFrame [timestamp, lat, lng, alt, gps_speed, vel_n, vel_e, vel_d]
                'baro': DataFrame [timestamp, baro_alt, baro_press]
                'ekf': DataFrame [timestamp, innovation_ratio]
                'params': Dictionary {param_name: value}

        Raises:
            RuntimeError: If log parsing fails
        """
        try:
            # Open the log file
            self.mlog = mavutil.mavlink_connection(str(self.log_path))

            # Extract all message types
            result = {
                'imu': self._extract_imu(),
                'rcout': self._extract_rcout(),
                'att': self._extract_att(),
                'gps': self._extract_gps(),
                'baro': self._extract_baro(),
                'ekf': self._extract_ekf(),
                'params': self._extract_params(),
            }

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to parse log file: {e}") from e

        finally:
            # Close the log file
            if self.mlog is not None:
                self.mlog.close()

    def _extract_messages(
        self,
        msg_type: str,
        fields: List[str],
        max_messages: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract all messages of a given type into a DataFrame.

        Args:
            msg_type: Message type to extract (e.g., 'IMU', 'GPS')
            fields: List of field names to extract from the message
            max_messages: Optional limit on number of messages to extract

        Returns:
            DataFrame with columns for each field
        """
        data = {field: [] for field in fields}
        count = 0

        # Reset to start of log
        self.mlog.rewind()

        while True:
            msg = self.mlog.recv_match(type=msg_type)
            if msg is None:
                break

            # Extract all fields
            for field in fields:
                try:
                    value = getattr(msg, field)
                    data[field].append(value)
                except AttributeError:
                    # Field doesn't exist in this message instance
                    data[field].append(np.nan)

            count += 1
            if max_messages is not None and count >= max_messages:
                break

        # Convert to DataFrame
        df = pd.DataFrame(data)

        return df

    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert TimeUS (microseconds) to timestamp (seconds).

        The timestamp is normalized to start at 0.0 seconds.

        Args:
            df: DataFrame with 'TimeUS' column

        Returns:
            DataFrame with 'timestamp' column replacing 'TimeUS'
        """
        if df.empty or 'TimeUS' not in df.columns:
            return df

        # Convert to seconds
        timestamps = df['TimeUS'].values * US_TO_SEC

        # Normalize to start at 0
        timestamps = timestamps - timestamps[0]

        # Create new dataframe with timestamp column
        df = df.copy()
        df[TIMESTAMP_COL] = timestamps
        df = df.drop(columns=['TimeUS'])

        # Move timestamp to first column
        cols = [TIMESTAMP_COL] + [c for c in df.columns if c != TIMESTAMP_COL]
        df = df[cols]

        return df

    def _extract_imu(self) -> pd.DataFrame:
        """
        Extract IMU messages.

        Returns:
            DataFrame with columns: [timestamp, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z]
            Units: timestamp (s), gyro (rad/s), accel (m/s^2)
        """
        fields = ['TimeUS', 'GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']
        df = self._extract_messages(IMU_MSG, fields)

        if df.empty:
            return pd.DataFrame(columns=[TIMESTAMP_COL] + GYRO_COLS + ACCEL_COLS)

        # Normalize timestamps
        df = self._normalize_timestamps(df)

        # Rename columns to standard names
        # ArduPilot logs gyro in rad/s and accel in m/s^2 already (no conversion needed)
        df = df.rename(columns={
            'GyrX': 'gyr_x',
            'GyrY': 'gyr_y',
            'GyrZ': 'gyr_z',
            'AccX': 'acc_x',
            'AccY': 'acc_y',
            'AccZ': 'acc_z',
        })

        return df

    def _extract_rcout(self) -> pd.DataFrame:
        """
        Extract RCOUT (PWM output) messages.

        Returns:
            DataFrame with columns: [timestamp, pwm_1, ..., pwm_14]
            Units: timestamp (s), PWM (microseconds, typically 1000-2000)
        """
        fields = ['TimeUS'] + [f'C{i}' for i in range(1, 15)]
        df = self._extract_messages(RCOUT_MSG, fields)

        if df.empty:
            return pd.DataFrame(columns=[TIMESTAMP_COL] + PWM_COLS)

        # Normalize timestamps
        df = self._normalize_timestamps(df)

        # Rename columns to standard names
        rename_map = {f'C{i}': f'pwm_{i}' for i in range(1, 15)}
        df = df.rename(columns=rename_map)

        return df

    def _extract_att(self) -> pd.DataFrame:
        """
        Extract ATT (attitude) messages.

        Returns:
            DataFrame with columns: [timestamp, roll, pitch, yaw]
            Units: timestamp (s), angles (radians)
        """
        fields = ['TimeUS', 'Roll', 'Pitch', 'Yaw']
        df = self._extract_messages(ATT_MSG, fields)

        if df.empty:
            return pd.DataFrame(columns=[TIMESTAMP_COL] + ATTITUDE_COLS)

        # Normalize timestamps
        df = self._normalize_timestamps(df)

        # Convert degrees to radians
        df['Roll'] = df['Roll'] * DEG_TO_RAD
        df['Pitch'] = df['Pitch'] * DEG_TO_RAD
        df['Yaw'] = df['Yaw'] * DEG_TO_RAD

        # Rename columns to standard names
        df = df.rename(columns={
            'Roll': 'roll',
            'Pitch': 'pitch',
            'Yaw': 'yaw',
        })

        return df

    def _extract_gps(self) -> pd.DataFrame:
        """
        Extract GPS messages.

        Returns:
            DataFrame with columns: [timestamp, lat, lng, alt, gps_speed, vel_n, vel_e, vel_d]
            Units: timestamp (s), lat/lng (degrees), alt (m), velocities (m/s)
        """
        fields = ['TimeUS', 'Lat', 'Lng', 'Alt', 'Spd', 'VelN', 'VelE', 'VelD']
        df = self._extract_messages(GPS_MSG, fields)

        if df.empty:
            return pd.DataFrame(
                columns=[TIMESTAMP_COL] + GPS_POSITION_COLS + [GPS_SPEED_COL] + GPS_VELOCITY_COLS
            )

        # Normalize timestamps
        df = self._normalize_timestamps(df)

        # Rename columns to standard names
        # Note: GPS velocities are already in m/s in ArduPilot logs
        df = df.rename(columns={
            'Lat': 'lat',
            'Lng': 'lng',
            'Alt': 'alt',
            'Spd': 'gps_speed',
            'VelN': 'vel_n',
            'VelE': 'vel_e',
            'VelD': 'vel_d',
        })

        return df

    def _extract_baro(self) -> pd.DataFrame:
        """
        Extract BARO (barometer) messages.

        Returns:
            DataFrame with columns: [timestamp, baro_alt, baro_press]
            Units: timestamp (s), altitude (m), pressure (Pa)
        """
        fields = ['TimeUS', 'Alt', 'Press']
        df = self._extract_messages(BARO_MSG, fields)

        if df.empty:
            return pd.DataFrame(columns=[TIMESTAMP_COL, BARO_ALT_COL, BARO_PRESS_COL])

        # Normalize timestamps
        df = self._normalize_timestamps(df)

        # Rename columns to standard names
        df = df.rename(columns={
            'Alt': 'baro_alt',
            'Press': 'baro_press',
        })

        return df

    def _extract_ekf(self) -> pd.DataFrame:
        """
        Extract EKF health messages.

        Returns:
            DataFrame with columns: [timestamp, innovation_ratio]
            Units: timestamp (s), innovation_ratio (dimensionless)
        """
        fields = ['TimeUS', 'SV']
        df = self._extract_messages(EKF_MSG, fields)

        if df.empty:
            return pd.DataFrame(columns=[TIMESTAMP_COL, EKF_INNOVATION_COL])

        # Normalize timestamps
        df = self._normalize_timestamps(df)

        # Rename columns to standard names
        df = df.rename(columns={
            'SV': 'innovation_ratio',
        })

        return df

    def _extract_params(self) -> Dict[str, float]:
        """
        Extract parameter values from the log.

        Returns:
            Dictionary mapping parameter name to value
        """
        fields = ['Name', 'Value']
        df = self._extract_messages(PARM_MSG, fields)

        if df.empty:
            return {}

        # Convert to dictionary
        params = {}
        for _, row in df.iterrows():
            name = row['Name']
            value = row['Value']
            # Handle byte strings from pymavlink
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='ignore')
            params[name] = float(value)

        return params

    def get_log_summary(self, parsed_data: Dict) -> Dict:
        """
        Generate a summary of the parsed log data.

        Args:
            parsed_data: Dictionary returned by parse()

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'log_file': str(self.log_path),
        }

        # Count messages for each type
        for msg_type, df in parsed_data.items():
            if msg_type == 'params':
                summary[f'{msg_type}_count'] = len(df)
            elif isinstance(df, pd.DataFrame) and not df.empty:
                summary[f'{msg_type}_count'] = len(df)
                if TIMESTAMP_COL in df.columns:
                    duration = df[TIMESTAMP_COL].max() - df[TIMESTAMP_COL].min()
                    summary[f'{msg_type}_duration_s'] = float(duration)
            else:
                summary[f'{msg_type}_count'] = 0

        # Overall duration (use IMU as reference since it's the highest rate)
        if 'imu' in parsed_data and not parsed_data['imu'].empty:
            imu_df = parsed_data['imu']
            total_duration = imu_df[TIMESTAMP_COL].max() - imu_df[TIMESTAMP_COL].min()
            summary['total_duration_s'] = float(total_duration)
            summary['total_duration_min'] = float(total_duration / 60.0)

        return summary


def print_log_summary(summary: Dict) -> None:
    """
    Print a human-readable summary of the log data.

    Args:
        summary: Dictionary from get_log_summary()
    """
    print("\n" + "="*60)
    print("LOG PARSE SUMMARY")
    print("="*60)
    print(f"File: {summary.get('log_file', 'unknown')}")

    if 'total_duration_min' in summary:
        print(f"Duration: {summary['total_duration_min']:.1f} min ({summary['total_duration_s']:.1f} s)")

    print("\nMessage counts:")
    msg_types = ['imu', 'rcout', 'att', 'gps', 'baro', 'ekf', 'params']
    for msg_type in msg_types:
        key = f'{msg_type}_count'
        if key in summary:
            count = summary[key]
            print(f"  {msg_type.upper():8s}: {count:8d} messages", end='')
            duration_key = f'{msg_type}_duration_s'
            if duration_key in summary:
                duration = summary[duration_key]
                rate = count / duration if duration > 0 else 0
                print(f"  ({rate:.1f} Hz)")
            else:
                print()

    print("="*60 + "\n")
