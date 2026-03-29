#!/usr/bin/env python3
"""
Example usage of the ArduPilot DataFlash log parser.

This script demonstrates how to use the parser to extract and analyze
flight data from .bin log files.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ardupilot_sysid.src.parser.dflog_reader import DFLogReader, print_log_summary
from ardupilot_sysid.src.parser.ekf_health import (
    filter_ekf_healthy_segments,
    apply_segment_filter,
    print_segment_report,
)
from ardupilot_sysid.src.parser.message_types import (
    TIMESTAMP_COL,
    GYRO_COLS,
    ACCEL_COLS,
    ATTITUDE_COLS,
    PWM_COLS,
)


def demo_basic_parsing():
    """Demonstrate basic log parsing (with synthetic example)."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Log Parsing")
    print("="*70)

    # NOTE: This is a demonstration. In real use, you would do:
    #
    # reader = DFLogReader('/path/to/your/flight.bin')
    # data = reader.parse()
    #
    # The data dictionary contains:
    # - data['imu']: IMU sensor data (gyro + accel)
    # - data['rcout']: PWM outputs to motors/servos
    # - data['att']: Attitude estimates (roll, pitch, yaw)
    # - data['gps']: GPS position and velocity
    # - data['baro']: Barometer altitude and pressure
    # - data['ekf']: EKF health metrics
    # - data['params']: ArduPilot parameters from the log

    print("\nTo parse a real log file:")
    print("  reader = DFLogReader('/path/to/flight.bin')")
    print("  data = reader.parse()")
    print("\nExample data structure:")
    print("  data['imu']:")
    print("    - Columns: timestamp, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z")
    print("    - Units: timestamp (s), gyro (rad/s), accel (m/s^2)")
    print("    - Rate: ~400 Hz")
    print("\n  data['rcout']:")
    print("    - Columns: timestamp, pwm_1, pwm_2, ..., pwm_14")
    print("    - Units: timestamp (s), PWM (microseconds)")
    print("    - Rate: ~50 Hz")
    print("\n  data['att']:")
    print("    - Columns: timestamp, roll, pitch, yaw")
    print("    - Units: timestamp (s), angles (radians)")
    print("    - Rate: ~400 Hz")
    print("\n  data['gps']:")
    print("    - Columns: timestamp, lat, lng, alt, gps_speed, vel_n, vel_e, vel_d")
    print("    - Units: timestamp (s), lat/lng (deg), alt (m), velocities (m/s)")
    print("    - Rate: ~10 Hz")
    print("\n  data['ekf']:")
    print("    - Columns: timestamp, innovation_ratio")
    print("    - Units: timestamp (s), innovation_ratio (dimensionless)")
    print("    - Rate: ~10 Hz")
    print("\n  data['params']:")
    print("    - Dictionary: {param_name: value}")


def demo_ekf_filtering():
    """Demonstrate EKF health filtering with synthetic data."""
    print("\n" + "="*70)
    print("DEMO 2: EKF Health Filtering")
    print("="*70)

    # Create synthetic EKF data
    print("\nCreating synthetic EKF data with healthy and unhealthy periods...")
    timestamps = np.linspace(0, 120, 1200)  # 2 minutes at 10 Hz
    innovation_ratio = np.ones(1200) * 0.5  # Start healthy

    # Simulate GPS glitches at different times
    innovation_ratio[200:350] = 1.8   # Unhealthy from 20-35s
    innovation_ratio[600:750] = 2.5   # Unhealthy from 60-75s
    innovation_ratio[1000:1050] = 1.2 # Unhealthy from 100-105s

    ekf_df = pd.DataFrame({
        'timestamp': timestamps,
        'innovation_ratio': innovation_ratio,
    })

    print(f"Total log duration: {timestamps[-1]:.1f} seconds")
    print(f"Mean innovation ratio: {innovation_ratio.mean():.2f}")
    print(f"Max innovation ratio: {innovation_ratio.max():.2f}")

    # Find healthy segments
    print("\nFinding EKF-healthy segments...")
    segments = filter_ekf_healthy_segments(
        ekf_df,
        innovation_threshold=1.0,
        min_segment_duration=5.0
    )

    print_segment_report(ekf_df, segments, innovation_threshold=1.0)

    # Create synthetic IMU data and filter it
    print("Creating synthetic IMU data at 400 Hz...")
    imu_timestamps = np.linspace(0, 120, 48000)  # 400 Hz
    imu_df = pd.DataFrame({
        'timestamp': imu_timestamps,
        'gyr_x': np.random.randn(48000) * 0.1,
        'gyr_y': np.random.randn(48000) * 0.1,
        'gyr_z': np.random.randn(48000) * 0.1,
    })

    print(f"IMU data: {len(imu_df)} samples")

    # Apply segment filter
    print("\nFiltering IMU data to EKF-healthy segments...")
    filtered_imu = apply_segment_filter(imu_df, segments)

    print(f"Filtered IMU data: {len(filtered_imu)} samples")
    print(f"Reduction: {(1 - len(filtered_imu)/len(imu_df))*100:.1f}%")


def demo_data_access():
    """Demonstrate how to access specific data fields."""
    print("\n" + "="*70)
    print("DEMO 3: Accessing Parsed Data")
    print("="*70)

    # Create example DataFrames
    print("\nExample: Working with parsed IMU data")
    print("-" * 70)

    imu_df = pd.DataFrame({
        'timestamp': np.linspace(0, 10, 100),
        'gyr_x': np.random.randn(100) * 0.1,
        'gyr_y': np.random.randn(100) * 0.1,
        'gyr_z': np.random.randn(100) * 0.1,
        'acc_x': np.random.randn(100) * 0.5,
        'acc_y': np.random.randn(100) * 0.5,
        'acc_z': np.random.randn(100) * 0.5 + 9.81,
    })

    print(f"\nIMU DataFrame shape: {imu_df.shape}")
    print(f"Columns: {list(imu_df.columns)}")
    print(f"\nFirst few rows:")
    print(imu_df.head())

    print("\n\nAccessing specific columns:")
    print(f"  Gyro X values: imu_df['gyr_x']")
    print(f"  All gyro data: imu_df[{GYRO_COLS}]")
    print(f"  All accel data: imu_df[{ACCEL_COLS}]")

    print("\n\nComputing statistics:")
    print(f"  Mean gyro magnitude: {np.linalg.norm(imu_df[GYRO_COLS].values, axis=1).mean():.3f} rad/s")
    print(f"  Mean accel magnitude: {np.linalg.norm(imu_df[ACCEL_COLS].values, axis=1).mean():.3f} m/s^2")

    print("\n\nExample: Working with RCOUT (PWM) data")
    print("-" * 70)

    rcout_df = pd.DataFrame({
        'timestamp': np.linspace(0, 10, 50),
        'pwm_1': np.ones(50) * 1500,
        'pwm_2': np.ones(50) * 1500,
        'pwm_3': np.ones(50) * 1500,
        'pwm_4': np.ones(50) * 1500,
    })

    print(f"\nRCOUT DataFrame shape: {rcout_df.shape}")
    print(f"Columns: {list(rcout_df.columns)}")
    print(f"\nMotor 1 PWM range: {rcout_df['pwm_1'].min():.0f} - {rcout_df['pwm_1'].max():.0f} μs")


def demo_workflow():
    """Demonstrate a complete parsing workflow."""
    print("\n" + "="*70)
    print("DEMO 4: Complete Parsing Workflow")
    print("="*70)

    print("""
TYPICAL WORKFLOW:
-----------------

1. Parse the log file:
   ```python
   from ardupilot_sysid.src.parser.dflog_reader import DFLogReader

   reader = DFLogReader('flight_data.bin')
   data = reader.parse()
   summary = reader.get_log_summary(data)
   ```

2. Check the log summary:
   ```python
   from ardupilot_sysid.src.parser.dflog_reader import print_log_summary

   print_log_summary(summary)
   # Output:
   #   Duration: 8.3 min
   #   IMU: 199200 messages (400 Hz)
   #   GPS: 4980 messages (10 Hz)
   #   ...
   ```

3. Filter by EKF health:
   ```python
   from ardupilot_sysid.src.parser.ekf_health import (
       filter_ekf_healthy_segments,
       apply_segment_filter,
       print_segment_report
   )

   # Find healthy segments
   segments = filter_ekf_healthy_segments(
       data['ekf'],
       innovation_threshold=1.0,
       min_segment_duration=5.0
   )

   # Print report
   print_segment_report(data['ekf'], segments)

   # Filter all data streams
   imu_clean = apply_segment_filter(data['imu'], segments)
   rcout_clean = apply_segment_filter(data['rcout'], segments)
   att_clean = apply_segment_filter(data['att'], segments)
   ```

4. Use the cleaned data for analysis:
   ```python
   # Extract gyro data for parameter identification
   gyro_data = imu_clean[['timestamp', 'gyr_x', 'gyr_y', 'gyr_z']].values

   # Extract motor commands
   motor_pwm = rcout_clean[['timestamp', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4']].values

   # Ready for next stage: RTS smoothing and optimization
   ```

ADVANTAGES:
-----------
✓ All timestamps normalized to start at 0.0 seconds
✓ All units converted to SI (radians, m/s^2, etc.)
✓ EKF health filtering removes bad data
✓ Clean pandas DataFrames for easy manipulation
✓ Ready for subsequent processing stages
""")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("ArduPilot DataFlash Log Parser - Usage Examples")
    print("="*70)

    demo_basic_parsing()
    demo_ekf_filtering()
    demo_data_access()
    demo_workflow()

    print("\n" + "="*70)
    print("For more information, see:")
    print("  - ardupilot_sysid/src/parser/dflog_reader.py")
    print("  - ardupilot_sysid/src/parser/ekf_health.py")
    print("  - ardupilot_sysid/src/parser/message_types.py")
    print("  - tests/test_parser.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
