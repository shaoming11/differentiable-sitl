"""
Example usage of the preprocessing module.

This script demonstrates the complete preprocessing pipeline:
1. Timestamp alignment via cross-correlation
2. Resampling to uniform time grid
3. Segmentation by EKF health
"""

import numpy as np
import pandas as pd

from ardupilot_sysid.src.preprocessing import (
    align_timestamps,
    check_timestamp_jitter,
    print_alignment_report,
    resample_to_uniform_grid,
    print_resampling_report,
    segment_by_ekf_health,
    apply_segments,
    print_segment_report,
)


def generate_example_data():
    """Generate example sensor data for demonstration."""
    print("Generating example sensor data...")

    # IMU: 400 Hz, 20 seconds
    t_imu = np.arange(0, 20.0, 1/400.0)
    n_imu = len(t_imu)

    imu_df = pd.DataFrame({
        'timestamp': t_imu,
        'acc_x': 0.5 * np.sin(2 * np.pi * 0.5 * t_imu) + 0.1 * np.random.randn(n_imu),
        'acc_y': 0.5 * np.cos(2 * np.pi * 0.5 * t_imu) + 0.1 * np.random.randn(n_imu),
        'acc_z': -9.81 + 0.2 * np.random.randn(n_imu),
        'gyr_x': 0.1 * np.random.randn(n_imu),
        'gyr_y': 0.1 * np.random.randn(n_imu),
        'gyr_z': 0.5 + 0.05 * np.random.randn(n_imu),
    })

    # GPS: 10 Hz, 20 seconds (with 150ms latency)
    t_gps = np.arange(0, 20.0, 0.1) + 0.150  # Add artificial latency
    n_gps = len(t_gps)

    gps_df = pd.DataFrame({
        'timestamp': t_gps,
        'vel_n': 3.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t_gps) + 0.1 * np.random.randn(n_gps),
        'vel_e': 2.0 + 0.5 * np.cos(2 * np.pi * 0.5 * t_gps) + 0.1 * np.random.randn(n_gps),
        'vel_d': 0.0 + 0.05 * np.random.randn(n_gps),
        'lat': 37.4 * np.ones(n_gps),
        'lng': -122.1 * np.ones(n_gps),
        'alt': 100.0 * np.ones(n_gps),
    })

    # RCOUT: 50 Hz, 20 seconds
    t_rcout = np.arange(0, 20.0, 0.02)
    n_rcout = len(t_rcout)

    rcout_data = {'timestamp': t_rcout}
    for i in range(1, 15):
        if i <= 4:
            rcout_data[f'pwm_{i}'] = 1500 + 100 * np.sin(2 * np.pi * 0.3 * t_rcout + i)
        else:
            rcout_data[f'pwm_{i}'] = 1000 * np.ones(n_rcout)

    rcout_df = pd.DataFrame(rcout_data)

    # EKF: 10 Hz, 20 seconds (with some unhealthy periods)
    t_ekf = np.arange(0, 20.0, 0.1)
    n_ekf = len(t_ekf)

    innovation_ratio = np.zeros(n_ekf)
    for i, t in enumerate(t_ekf):
        if t < 5.0:
            innovation_ratio[i] = 0.5 + 0.2 * np.random.randn()
        elif t < 7.0:
            innovation_ratio[i] = 1.5 + 0.3 * np.random.randn()  # Unhealthy
        elif t < 15.0:
            innovation_ratio[i] = 0.6 + 0.2 * np.random.randn()
        else:
            innovation_ratio[i] = 1.8 + 0.4 * np.random.randn()  # Unhealthy

    innovation_ratio = np.clip(innovation_ratio, 0.1, 5.0)

    ekf_df = pd.DataFrame({
        'timestamp': t_ekf,
        'innovation_ratio': innovation_ratio,
    })

    return imu_df, gps_df, rcout_df, ekf_df


def main():
    """Run the complete preprocessing pipeline."""
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE EXAMPLE")
    print("="*70 + "\n")

    # Step 1: Generate example data
    imu_df, gps_df, rcout_df, ekf_df = generate_example_data()

    print(f"Generated data:")
    print(f"  IMU: {len(imu_df)} samples @ ~400 Hz")
    print(f"  GPS: {len(gps_df)} samples @ ~10 Hz (with 150ms artificial latency)")
    print(f"  RCOUT: {len(rcout_df)} samples @ ~50 Hz")
    print(f"  EKF: {len(ekf_df)} samples @ ~10 Hz")

    # Step 2: Check timestamp jitter
    print("\n" + "-"*70)
    print("STEP 1: Timestamp Jitter Analysis")
    print("-"*70)

    imu_jitter = check_timestamp_jitter(imu_df, name='IMU')
    gps_jitter = check_timestamp_jitter(gps_df, name='GPS')
    rcout_jitter = check_timestamp_jitter(rcout_df, name='RCOUT')

    for jitter in [imu_jitter, gps_jitter, rcout_jitter]:
        print(f"\n{jitter['name']}:")
        print(f"  Sample rate: {jitter['sample_rate_hz']:.1f} Hz")
        print(f"  Mean dt: {jitter['mean_dt']*1000:.3f} ms")
        print(f"  Std dt: {jitter['std_dt']*1000:.3f} ms")
        print(f"  Max jitter: {jitter['max_jitter_ms']:.3f} ms")

    # Step 3: Align timestamps
    print("\n" + "-"*70)
    print("STEP 2: Timestamp Alignment")
    print("-"*70)

    imu_aligned, gps_aligned, rcout_aligned, align_metadata = align_timestamps(
        imu_df,
        gps_df,
        rcout_df
    )

    print_alignment_report(align_metadata)

    # Step 4: Resample to uniform grid
    print("-"*70)
    print("STEP 3: Resampling to Uniform Grid")
    print("-"*70)

    original_dfs = {
        'imu': imu_aligned,
        'gps': gps_aligned,
        'rcout': rcout_aligned,
        'ekf': ekf_df,
    }

    resampled_dfs = resample_to_uniform_grid(
        original_dfs,
        target_rate_hz=400.0
    )

    print_resampling_report(original_dfs, resampled_dfs)

    # Step 5: Segment by EKF health
    print("-"*70)
    print("STEP 4: EKF Health Segmentation")
    print("-"*70)

    segments = segment_by_ekf_health(
        resampled_dfs['ekf'],
        innovation_threshold=1.0,
        min_segment_duration_s=3.0
    )

    print_segment_report(segments, resampled_dfs['ekf'], innovation_threshold=1.0)

    # Step 6: Apply segments to all data
    print("-"*70)
    print("STEP 5: Apply Segments to All Data")
    print("-"*70)

    print("\nSplitting data into healthy segments...")

    segmented_data = {}
    for name, df in resampled_dfs.items():
        segmented_data[name] = apply_segments(df, segments)

    print(f"\nResulting segments:")
    for i, seg in enumerate(segments):
        t_start, t_end = seg
        duration = t_end - t_start

        imu_seg = segmented_data['imu'][i]
        gps_seg = segmented_data['gps'][i]
        rcout_seg = segmented_data['rcout'][i]

        print(f"\nSegment {i+1}: {t_start:.1f} - {t_end:.1f} s (duration: {duration:.1f} s)")
        print(f"  IMU: {len(imu_seg)} samples")
        print(f"  GPS: {len(gps_seg)} samples")
        print(f"  RCOUT: {len(rcout_seg)} samples")

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print("\nThe data is now ready for:")
    print("  - RTS smoothing (Phase 3)")
    print("  - System identification (Phases 4-6)")
    print("  - Validation (Phase 7)")
    print()


if __name__ == '__main__':
    main()
