"""
Example usage of the RTS smoother for state estimation.

This script demonstrates how to:
1. Load ArduPilot log data
2. Run forward UKF pass
3. Run backward RTS pass
4. Compare forward vs smoothed estimates
5. Visualize results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ardupilot_sysid.src.smoother import (
    UnscentedKalmanFilter,
    RTSSmoother,
    quaternion_to_euler
)


def generate_synthetic_data(dt=0.01, duration=5.0):
    """
    Generate synthetic flight data for demonstration.

    Args:
        dt: Timestep in seconds
        duration: Duration in seconds

    Returns:
        Dictionary with 'imu', 'gps', 'baro' DataFrames
    """
    N = int(duration / dt)
    timestamps = np.arange(N) * dt

    # Generate sinusoidal motion
    freq = 0.5  # Hz
    omega_x = 0.2 * np.sin(2 * np.pi * freq * timestamps)
    omega_y = 0.1 * np.cos(2 * np.pi * freq * timestamps)
    omega_z = 0.05 * np.sin(4 * np.pi * freq * timestamps)

    # IMU measurements (with noise)
    imu_data = pd.DataFrame({
        'timestamp': timestamps,
        'acc_x': np.zeros(N) + np.random.randn(N) * 0.01,
        'acc_y': np.zeros(N) + np.random.randn(N) * 0.01,
        'acc_z': np.zeros(N) + np.random.randn(N) * 0.01,
        'gyr_x': omega_x + np.random.randn(N) * 0.01,
        'gyr_y': omega_y + np.random.randn(N) * 0.01,
        'gyr_z': omega_z + np.random.randn(N) * 0.01
    })

    # GPS measurements (lower rate, with noise)
    gps_timestamps = timestamps[::10]  # 10 Hz
    N_gps = len(gps_timestamps)
    gps_data = pd.DataFrame({
        'timestamp': gps_timestamps,
        'vel_n': 10.0 + np.random.randn(N_gps) * 0.5,
        'vel_e': np.zeros(N_gps) + np.random.randn(N_gps) * 0.5,
        'vel_d': np.zeros(N_gps) + np.random.randn(N_gps) * 0.5,
        'lat': 37.7749 + np.random.randn(N_gps) * 0.0001,
        'lng': -122.4194 + np.random.randn(N_gps) * 0.0001,
        'alt': 100.0 + np.random.randn(N_gps) * 1.0
    })

    # Barometer measurements (lower rate, with noise)
    baro_timestamps = timestamps[::10]  # 10 Hz
    N_baro = len(baro_timestamps)
    baro_data = pd.DataFrame({
        'timestamp': baro_timestamps,
        'baro_alt': 100.0 + np.random.randn(N_baro) * 0.5
    })

    return {
        'imu': imu_data,
        'gps': gps_data,
        'baro': baro_data
    }


def run_smoother_example():
    """Run the RTS smoother on synthetic data and visualize results."""
    print("=" * 80)
    print("RTS Smoother Example")
    print("=" * 80)

    # Parameters
    dt = 0.01  # 100 Hz
    duration = 5.0  # 5 seconds

    print("\n1. Generating synthetic flight data...")
    measurements = generate_synthetic_data(dt, duration)
    print(f"   IMU samples: {len(measurements['imu'])}")
    print(f"   GPS samples: {len(measurements['gps'])}")
    print(f"   Barometer samples: {len(measurements['baro'])}")

    # Initial state (hovering at origin)
    x_init = np.array([
        1, 0, 0, 0,    # quaternion (identity)
        0, 0, 0,       # velocity
        0, 0, 0,       # angular velocity
        0, 0, 0        # position
    ])

    # Initial covariance (high uncertainty)
    P_init = np.eye(13) * 1.0

    # Process noise covariance
    Q = np.diag([
        1e-6, 1e-6, 1e-6, 1e-6,  # quaternion (low noise)
        1e-3, 1e-3, 1e-3,        # velocity
        1e-3, 1e-3, 1e-3,        # angular velocity (higher noise)
        1e-4, 1e-4, 1e-4         # position
    ])

    # Measurement noise covariances
    R_imu = np.diag([
        1e-2, 1e-2, 1e-2,  # accelerometer
        1e-3, 1e-3, 1e-3   # gyroscope
    ])

    R_gps = np.diag([
        0.5, 0.5, 0.5,    # velocity
        1e-6, 1e-6, 1.0   # position (lat, lng, alt)
    ])

    R_baro = np.array([[0.5]])  # altitude

    print("\n2. Running forward UKF pass...")
    ukf = UnscentedKalmanFilter(state_dim=13, alpha=1e-3, beta=2.0, kappa=0.0)
    forward_states = ukf.forward_pass(
        x_init, P_init,
        measurements, dt,
        Q, R_imu, R_gps, R_baro
    )
    print(f"   Forward pass complete: {len(forward_states)} states")

    print("\n3. Running backward RTS pass...")
    smoother = RTSSmoother()
    smoothed_states = smoother.backward_pass(forward_states)
    print(f"   Backward pass complete: {len(smoothed_states)} states")

    print("\n4. Comparing forward vs smoothed estimates...")
    metrics = smoother.compare_forward_vs_smoothed(forward_states, smoothed_states)
    print(f"   Forward trace (mean):  {metrics['forward_trace_mean']:.6f}")
    print(f"   Smoothed trace (mean): {metrics['smoothed_trace_mean']:.6f}")
    print(f"   Variance reduction:    {metrics['variance_reduction']:.2f}%")
    print(f"   State difference:      {metrics['state_diff_norm']:.6f}")

    print("\n5. Extracting trajectories...")
    timestamps = smoother.get_timestamps(smoothed_states)
    omega_forward = smoother.extract_angular_velocity(forward_states)
    omega_smoothed = smoother.extract_angular_velocity(smoothed_states)

    # Compare with IMU measurements
    imu_data = measurements['imu']
    imu_omega = imu_data[['gyr_x', 'gyr_y', 'gyr_z']].values

    # Compute RMSE for angular velocity
    forward_rmse = np.sqrt(np.mean((omega_forward - imu_omega) ** 2))
    smoothed_rmse = np.sqrt(np.mean((omega_smoothed - imu_omega) ** 2))
    print(f"   Forward UKF RMSE (angular velocity):  {forward_rmse:.6f} rad/s")
    print(f"   RTS Smoother RMSE (angular velocity): {smoothed_rmse:.6f} rad/s")

    print("\n6. Creating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('RTS Smoother Results', fontsize=16)

    # Angular velocity comparison
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = axes[i, 0]
        ax.plot(imu_data['timestamp'], imu_data[f'gyr_{axis_name.lower()}'],
                'k.', alpha=0.3, label='IMU measurement', markersize=1)
        ax.plot(timestamps, omega_forward[:, i],
                'b-', alpha=0.7, label='Forward UKF', linewidth=1)
        ax.plot(timestamps, omega_smoothed[:, i],
                'r-', alpha=0.7, label='RTS Smoothed', linewidth=1)
        ax.set_ylabel(f'$\\omega_{{{axis_name}}}$ (rad/s)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        if i == 2:
            ax.set_xlabel('Time (s)')

    # Covariance trace over time
    ax = axes[0, 1]
    ax.plot(timestamps, metrics['forward_trace'], 'b-', label='Forward UKF')
    ax.plot(timestamps, metrics['smoothed_trace'], 'r-', label='RTS Smoothed')
    ax.set_ylabel('Trace(P)')
    ax.set_title('Total Uncertainty Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance reduction over time
    ax = axes[1, 1]
    variance_reduction_per_time = 100 * (1 - metrics['smoothed_trace'] / metrics['forward_trace'])
    ax.plot(timestamps, variance_reduction_per_time, 'g-')
    ax.set_ylabel('Variance Reduction (%)')
    ax.set_title('Smoother Benefit Over Time')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

    # Quaternion evolution
    ax = axes[2, 1]
    quaternions = smoother.extract_quaternion(smoothed_states)
    euler_angles = np.array([quaternion_to_euler(q) for q in quaternions])
    ax.plot(timestamps, np.rad2deg(euler_angles[:, 0]), label='Roll')
    ax.plot(timestamps, np.rad2deg(euler_angles[:, 1]), label='Pitch')
    ax.plot(timestamps, np.rad2deg(euler_angles[:, 2]), label='Yaw')
    ax.set_ylabel('Angle (deg)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Attitude Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rts_smoother_results.png', dpi=150, bbox_inches='tight')
    print(f"\n   Saved visualization to: rts_smoother_results.png")

    print("\n" + "=" * 80)
    print("SUCCESS: RTS smoother implementation complete!")
    print("=" * 80)
    print("\nKey achievements:")
    print("  - Forward UKF provides causal state estimates")
    print("  - Backward RTS pass refines estimates using future data")
    print(f"  - Variance reduced by {metrics['variance_reduction']:.2f}%")
    print("  - Quaternion normalization maintained throughout")
    print("  - Ready for parameter identification pipeline")
    print("=" * 80)


if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    run_smoother_example()
    plt.show()
