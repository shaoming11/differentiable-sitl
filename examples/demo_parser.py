#!/usr/bin/env python3
"""
Demo script for the parser module.

This script demonstrates how to use the DFLogReader, message types,
and EKF health filtering on a real ArduPilot log file.

Usage:
    python demo_parser.py /path/to/flight.bin
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ardupilot_sysid.src.parser.dflog_reader import DFLogReader, print_log_summary
from ardupilot_sysid.src.parser import ekf_health
from ardupilot_sysid.src.parser import message_types as mt


def demo_parser(log_path: str):
    """
    Demonstrate parser functionality on a real log file.

    Args:
        log_path: Path to ArduPilot .bin log file
    """
    print("="*80)
    print("ArduPilot Log Parser Demo")
    print("="*80)
    print(f"\nParsing log file: {log_path}\n")

    # Step 1: Parse the log file
    print("Step 1: Parsing log file...")
    reader = DFLogReader(log_path)

    try:
        data = reader.parse()
    except Exception as e:
        print(f"ERROR: Failed to parse log file: {e}")
        return

    # Step 2: Print summary
    print("\nStep 2: Log summary:")
    summary = reader.get_log_summary(data)
    print_log_summary(summary)

    # Step 3: Show sample data from each message type
    print("\nStep 3: Sample data from each message type:")
    print("-" * 80)

    for msg_type in ['imu', 'rcout', 'att', 'gps', 'baro', 'ekf']:
        df = data[msg_type]
        if not df.empty:
            print(f"\n{msg_type.upper()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  First 3 rows:")
            print(df.head(3).to_string(index=False))
        else:
            print(f"\n{msg_type.upper()}: No data found")

    # Step 4: Show parameters
    print("\n" + "-" * 80)
    print("PARAMETERS:")
    params = data['params']
    if params:
        print(f"  Total parameters: {len(params)}")
        print(f"  Sample parameters:")
        for i, (name, value) in enumerate(list(params.items())[:10]):
            print(f"    {name:20s} = {value}")
        if len(params) > 10:
            print(f"    ... and {len(params) - 10} more")
    else:
        print("  No parameters found")

    # Step 5: EKF health analysis
    print("\n" + "="*80)
    print("Step 4: EKF Health Analysis")
    print("="*80)

    ekf_df = data['ekf']
    if not ekf_df.empty:
        # Find healthy segments
        segments = ekf_health.filter_ekf_healthy_segments(
            ekf_df,
            innovation_threshold=1.0,
            min_segment_duration=5.0
        )

        # Print segment report
        ekf_health.print_segment_report(ekf_df, segments, innovation_threshold=1.0)

        # Step 6: Apply filtering to other data
        if segments:
            print("Step 5: Filtering data to EKF-healthy segments")
            print("-" * 80)

            for msg_type in ['imu', 'att', 'gps']:
                df = data[msg_type]
                if not df.empty:
                    filtered = ekf_health.apply_segment_filter(df, segments)
                    coverage = len(filtered) / len(df) * 100
                    print(f"  {msg_type.upper():6s}: {len(df):6d} → {len(filtered):6d} samples ({coverage:5.1f}% retained)")
    else:
        print("  No EKF data found in log")

    # Step 7: Data quality checks
    print("\n" + "="*80)
    print("Step 6: Data Quality Checks")
    print("="*80)

    checks = []

    # Check for sufficient data
    imu_df = data['imu']
    if not imu_df.empty:
        duration = imu_df[mt.TIMESTAMP_COL].max()
        if duration < 10:
            checks.append(f"⚠  WARNING: Short log duration ({duration:.1f}s). Recommend >30s for identification.")
        else:
            checks.append(f"✓  Log duration: {duration:.1f}s")

    # Check for GPS data
    gps_df = data['gps']
    if gps_df.empty:
        checks.append("⚠  WARNING: No GPS data found. Position-based identification not possible.")
    else:
        checks.append(f"✓  GPS data: {len(gps_df)} samples")

    # Check for RCOUT data
    rcout_df = data['rcout']
    if rcout_df.empty:
        checks.append("✗  ERROR: No RCOUT data found. Cannot identify motor parameters without PWM data.")
    else:
        checks.append(f"✓  RCOUT data: {len(rcout_df)} samples")

    # Check EKF health coverage
    if not ekf_df.empty and segments:
        stats = ekf_health.compute_segment_statistics(ekf_df, segments)
        coverage = stats['coverage'] * 100
        if coverage < 50:
            checks.append(f"⚠  WARNING: Low EKF health coverage ({coverage:.1f}%). Check GPS/compass.")
        else:
            checks.append(f"✓  EKF health coverage: {coverage:.1f}%")

    print("\n".join(checks))

    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python demo_parser.py <path_to_flight.bin>")
        print("\nThis demo requires a real ArduPilot .bin log file.")
        print("You can download sample logs from: https://logs.ardupilot.org/")
        sys.exit(1)

    log_path = sys.argv[1]

    if not Path(log_path).exists():
        print(f"ERROR: File not found: {log_path}")
        sys.exit(1)

    demo_parser(log_path)
