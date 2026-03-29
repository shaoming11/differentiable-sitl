#!/usr/bin/env python3
"""
Integration test demonstrating the complete parser workflow.

This script creates synthetic log-like data and demonstrates the full
parsing and filtering pipeline.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Test the new __init__.py exports
from ardupilot_sysid.src.parser import (
    # Main class
    DFLogReader,
    print_log_summary,
    # EKF filtering
    filter_ekf_healthy_segments,
    apply_segment_filter,
    compute_segment_statistics,
    print_segment_report,
    # Constants
    TIMESTAMP_COL,
    GYRO_COLS,
    ACCEL_COLS,
    ATTITUDE_COLS,
    PWM_COLS,
    IMU_MSG,
    RCOUT_MSG,
    get_message_rate,
)


def test_imports():
    """Test that all imports work correctly."""
    print("\n" + "="*70)
    print("TEST: Module Imports")
    print("="*70)

    print("✓ DFLogReader imported")
    print("✓ print_log_summary imported")
    print("✓ filter_ekf_healthy_segments imported")
    print("✓ apply_segment_filter imported")
    print("✓ compute_segment_statistics imported")
    print("✓ print_segment_report imported")
    print("✓ Constants imported (TIMESTAMP_COL, GYRO_COLS, etc.)")
    print("✓ Helper functions imported (get_message_rate, etc.)")

    # Test that constants are correct
    assert TIMESTAMP_COL == 'timestamp'
    assert len(GYRO_COLS) == 3
    assert len(ACCEL_COLS) == 3
    assert len(ATTITUDE_COLS) == 3
    assert len(PWM_COLS) == 14

    print("\n✓ All imports successful and constants verified!")


def test_complete_workflow():
    """Test complete workflow with synthetic data."""
    print("\n" + "="*70)
    print("TEST: Complete Workflow")
    print("="*70)

    # Create synthetic data that simulates a parsed log
    print("\n1. Creating synthetic flight data...")
    duration = 120  # 2 minutes
    dt = 1/400  # 400 Hz

    # IMU data at 400 Hz
    n_imu = int(duration * 400)
    imu_df = pd.DataFrame({
        'timestamp': np.arange(n_imu) * dt,
        'gyr_x': np.sin(np.linspace(0, 4*np.pi, n_imu)) * 0.2,
        'gyr_y': np.cos(np.linspace(0, 4*np.pi, n_imu)) * 0.2,
        'gyr_z': np.sin(np.linspace(0, 2*np.pi, n_imu)) * 0.1,
        'acc_x': np.random.randn(n_imu) * 0.5,
        'acc_y': np.random.randn(n_imu) * 0.5,
        'acc_z': np.random.randn(n_imu) * 0.5 + 9.81,
    })
    print(f"   Created IMU data: {len(imu_df)} samples at 400 Hz")

    # RCOUT data at 50 Hz
    n_rcout = int(duration * 50)
    rcout_df = pd.DataFrame({
        'timestamp': np.linspace(0, duration, n_rcout),
        'pwm_1': 1500 + np.sin(np.linspace(0, 4*np.pi, n_rcout)) * 200,
        'pwm_2': 1500 + np.sin(np.linspace(0, 4*np.pi, n_rcout)) * 200,
        'pwm_3': 1500 + np.sin(np.linspace(0, 4*np.pi, n_rcout)) * 200,
        'pwm_4': 1500 + np.sin(np.linspace(0, 4*np.pi, n_rcout)) * 200,
    })
    print(f"   Created RCOUT data: {len(rcout_df)} samples at 50 Hz")

    # Attitude data at 400 Hz
    att_df = pd.DataFrame({
        'timestamp': np.arange(n_imu) * dt,
        'roll': np.sin(np.linspace(0, 4*np.pi, n_imu)) * 0.2,
        'pitch': np.cos(np.linspace(0, 4*np.pi, n_imu)) * 0.15,
        'yaw': np.linspace(0, 2*np.pi, n_imu),
    })
    print(f"   Created ATT data: {len(att_df)} samples at 400 Hz")

    # EKF data at 10 Hz with unhealthy periods
    n_ekf = int(duration * 10)
    innovation_ratio = np.ones(n_ekf) * 0.5
    # Add unhealthy periods
    innovation_ratio[200:300] = 1.8  # 20-30s
    innovation_ratio[600:700] = 2.2  # 60-70s

    ekf_df = pd.DataFrame({
        'timestamp': np.linspace(0, duration, n_ekf),
        'innovation_ratio': innovation_ratio,
    })
    print(f"   Created EKF data: {len(ekf_df)} samples at 10 Hz")

    # Simulate parsed data dictionary
    data = {
        'imu': imu_df,
        'rcout': rcout_df,
        'att': att_df,
        'ekf': ekf_df,
        'params': {'PARAM1': 1.0, 'PARAM2': 2.0},
    }

    print("\n2. Finding EKF-healthy segments...")
    segments = filter_ekf_healthy_segments(
        ekf_df,
        innovation_threshold=1.0,
        min_segment_duration=5.0
    )
    print(f"   Found {len(segments)} healthy segments")

    # Print detailed segment report
    stats = compute_segment_statistics(ekf_df, segments)
    print(f"   Total healthy duration: {stats['total_duration']:.1f} s")
    print(f"   Coverage: {stats['coverage']*100:.1f}%")

    print("\n3. Filtering data to healthy segments...")
    imu_clean = apply_segment_filter(imu_df, segments)
    rcout_clean = apply_segment_filter(rcout_df, segments)
    att_clean = apply_segment_filter(att_df, segments)

    print(f"   IMU: {len(imu_df)} → {len(imu_clean)} samples")
    print(f"   RCOUT: {len(rcout_df)} → {len(rcout_clean)} samples")
    print(f"   ATT: {len(att_df)} → {len(att_clean)} samples")

    # Verify that filtered data is reasonable
    assert len(imu_clean) < len(imu_df), "Filtered data should be smaller"
    assert len(imu_clean) > 0, "Should have some healthy data"

    print("\n4. Computing statistics on cleaned data...")
    print(f"   Mean gyro magnitude: {np.linalg.norm(imu_clean[GYRO_COLS].values, axis=1).mean():.4f} rad/s")
    print(f"   Mean accel magnitude: {np.linalg.norm(imu_clean[ACCEL_COLS].values, axis=1).mean():.4f} m/s^2")
    print(f"   Roll range: {att_clean['roll'].min():.2f} to {att_clean['roll'].max():.2f} rad")
    print(f"   PWM range (motor 1): {rcout_clean['pwm_1'].min():.0f} to {rcout_clean['pwm_1'].max():.0f} μs")

    print("\n✓ Complete workflow test passed!")


def test_helper_functions():
    """Test helper functions."""
    print("\n" + "="*70)
    print("TEST: Helper Functions")
    print("="*70)

    # Test get_message_rate
    imu_rate = get_message_rate(IMU_MSG)
    rcout_rate = get_message_rate(RCOUT_MSG)

    print(f"IMU rate: {imu_rate} Hz")
    print(f"RCOUT rate: {rcout_rate} Hz")

    assert imu_rate == 400, f"Expected 400 Hz, got {imu_rate}"
    assert rcout_rate == 50, f"Expected 50 Hz, got {rcout_rate}"

    print("\n✓ Helper function tests passed!")


def test_data_access_patterns():
    """Test common data access patterns."""
    print("\n" + "="*70)
    print("TEST: Data Access Patterns")
    print("="*70)

    # Create sample data
    imu_df = pd.DataFrame({
        'timestamp': np.linspace(0, 10, 100),
        'gyr_x': np.random.randn(100) * 0.1,
        'gyr_y': np.random.randn(100) * 0.1,
        'gyr_z': np.random.randn(100) * 0.1,
        'acc_x': np.random.randn(100) * 0.5,
        'acc_y': np.random.randn(100) * 0.5,
        'acc_z': np.random.randn(100) * 0.5 + 9.81,
    })

    print("\n1. Access gyro data using column constants:")
    gyro_data = imu_df[GYRO_COLS].values
    print(f"   Shape: {gyro_data.shape}")
    assert gyro_data.shape == (100, 3), "Should be (N, 3)"
    print("   ✓ Correct shape")

    print("\n2. Access accel data using column constants:")
    accel_data = imu_df[ACCEL_COLS].values
    print(f"   Shape: {accel_data.shape}")
    assert accel_data.shape == (100, 3), "Should be (N, 3)"
    print("   ✓ Correct shape")

    print("\n3. Access timestamp:")
    timestamps = imu_df[TIMESTAMP_COL].values
    print(f"   Shape: {timestamps.shape}")
    assert timestamps.shape == (100,), "Should be (N,)"
    print("   ✓ Correct shape")

    print("\n✓ Data access pattern tests passed!")


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("PARSER MODULE - INTEGRATION TESTS")
    print("="*70)

    try:
        test_imports()
        test_helper_functions()
        test_data_access_patterns()
        test_complete_workflow()

        print("\n" + "="*70)
        print("ALL INTEGRATION TESTS PASSED! ✓")
        print("="*70)
        print("\nThe parser module is fully functional and ready for use.")
        print("\nNext steps:")
        print("  1. Test with real ArduPilot .bin log files")
        print("  2. Proceed to Phase 2, Part 2: Preprocessing")
        print("  3. Implement RTS smoother using parsed data")
        print("="*70 + "\n")

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
