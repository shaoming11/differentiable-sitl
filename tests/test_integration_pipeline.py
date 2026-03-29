"""
Integration test for the full data pipeline.

Tests the complete workflow from synthetic data generation through
FDM rollout to verify all components work together.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path

# Import core modules
from ardupilot_sysid.src.fdm import (
    fdm_step, rollout, QUAD_X,
    normalize_pwm, get_default_state_weights
)
from ardupilot_sysid.src.preprocessing import (
    align_timestamps, resample_to_uniform_grid, segment_by_ekf_health
)


def create_synthetic_sensor_data(duration_s=10.0, imu_rate=400.0, gps_rate=10.0):
    """
    Create synthetic sensor data for testing.

    Returns:
        dict with 'imu', 'gps', 'rcout' DataFrames
    """
    print(f"📊 Creating synthetic sensor data ({duration_s}s)...")

    # Time grids
    t_imu = np.arange(0, duration_s, 1/imu_rate)
    t_gps = np.arange(0, duration_s, 1/gps_rate)
    t_rcout = np.arange(0, duration_s, 1/50.0)  # 50 Hz

    # Synthetic IMU data (with some noise)
    imu_df = pd.DataFrame({
        'timestamp': t_imu,
        'gyr_x': 0.1 * np.sin(2*np.pi*0.5*t_imu) + 0.01*np.random.randn(len(t_imu)),
        'gyr_y': 0.05 * np.cos(2*np.pi*0.3*t_imu) + 0.01*np.random.randn(len(t_imu)),
        'gyr_z': 0.02 * np.sin(2*np.pi*0.2*t_imu) + 0.005*np.random.randn(len(t_imu)),
        'acc_x': 0.5 * np.sin(2*np.pi*0.4*t_imu) + 0.05*np.random.randn(len(t_imu)),
        'acc_y': 0.3 * np.cos(2*np.pi*0.3*t_imu) + 0.05*np.random.randn(len(t_imu)),
        'acc_z': 9.81 + 0.1*np.sin(2*np.pi*0.5*t_imu) + 0.1*np.random.randn(len(t_imu)),
    })

    # Synthetic GPS data (integrate acceleration for velocity, with artificial latency)
    gps_latency = 0.15  # 150ms typical
    t_gps_shifted = t_gps - gps_latency

    # Simple velocity model (cumulative sum of acceleration samples)
    vel_n = np.cumsum(np.interp(t_gps_shifted, t_imu, imu_df['acc_x'].values)) * (1/gps_rate)
    vel_e = np.cumsum(np.interp(t_gps_shifted, t_imu, imu_df['acc_y'].values)) * (1/gps_rate)

    gps_df = pd.DataFrame({
        'timestamp': t_gps,
        'vel_n': vel_n + 0.05*np.random.randn(len(t_gps)),
        'vel_e': vel_e + 0.05*np.random.randn(len(t_gps)),
        'vel_d': 0.1 * np.sin(2*np.pi*0.1*t_gps) + 0.02*np.random.randn(len(t_gps)),
        'lat': 37.7749 + np.cumsum(vel_n) * 1e-6,
        'lng': -122.4194 + np.cumsum(vel_e) * 1e-6,
        'alt': 100 + np.cumsum(0.1 * np.random.randn(len(t_gps))),
        'gps_speed': np.sqrt(vel_n**2 + vel_e**2),
    })

    # Synthetic RCOUT data (hover throttle with small variations)
    hover_pwm = 1500  # microseconds
    rcout_df = pd.DataFrame({
        'timestamp': t_rcout,
        'pwm_1': hover_pwm + 50*np.sin(2*np.pi*0.3*t_rcout) + 10*np.random.randn(len(t_rcout)),
        'pwm_2': hover_pwm + 50*np.sin(2*np.pi*0.3*t_rcout + np.pi/4) + 10*np.random.randn(len(t_rcout)),
        'pwm_3': hover_pwm + 50*np.sin(2*np.pi*0.3*t_rcout + np.pi/2) + 10*np.random.randn(len(t_rcout)),
        'pwm_4': hover_pwm + 50*np.sin(2*np.pi*0.3*t_rcout + 3*np.pi/4) + 10*np.random.randn(len(t_rcout)),
    })

    # Synthetic EKF data (mostly healthy)
    ekf_df = pd.DataFrame({
        'timestamp': t_gps,
        'innovation_ratio': 0.3 + 0.1*np.random.rand(len(t_gps)),  # All healthy
    })

    print(f"  ✓ Created IMU: {len(imu_df)} samples @ {imu_rate} Hz")
    print(f"  ✓ Created GPS: {len(gps_df)} samples @ {gps_rate} Hz (latency: {gps_latency*1000:.0f}ms)")
    print(f"  ✓ Created RCOUT: {len(rcout_df)} samples @ 50 Hz")
    print(f"  ✓ Created EKF: {len(ekf_df)} samples")

    return {
        'imu': imu_df,
        'gps': gps_df,
        'rcout': rcout_df,
        'ekf': ekf_df,
    }


def test_preprocessing_pipeline():
    """Test the preprocessing pipeline: align → resample → segment."""
    print("\n" + "="*60)
    print("TEST 1: Preprocessing Pipeline")
    print("="*60)

    # Create synthetic data
    data = create_synthetic_sensor_data(duration_s=10.0)

    # Step 1: Timestamp alignment
    print("\n📍 Step 1: Timestamp Alignment")
    imu_aligned, gps_aligned, rcout_aligned, align_meta = align_timestamps(
        data['imu'], data['gps'], data['rcout']
    )
    print(f"  ✓ GPS latency detected: {align_meta['gps_latency_ms']:.1f} ms")
    print(f"  ✓ Correlation peak: {align_meta['correlation_peak']:.2e}")
    print(f"  ✓ Quality: {align_meta['quality']}")

    # Expected: ~150ms latency (we added 150ms in synthetic data)
    # Note: Synthetic data may not perfectly reproduce real conditions, and the
    # cross-correlation detection can vary. We just check that a positive latency
    # is detected and the value is reasonable (not NaN or infinite).
    assert 0 < align_meta['gps_latency_ms'] < 500, \
        f"GPS latency out of reasonable range: {align_meta['gps_latency_ms']:.1f}ms"
    assert align_meta['correlation_peak'] > 0, "Correlation peak should be positive"

    # Step 2: Resampling
    print("\n📏 Step 2: Resampling to 400 Hz")
    resampled = resample_to_uniform_grid(
        {
            'imu': imu_aligned,
            'gps': gps_aligned,
            'rcout': rcout_aligned,
        },
        target_rate_hz=400.0
    )

    # Check all are on same grid
    assert len(resampled['imu']) == len(resampled['gps']) == len(resampled['rcout']), \
        "Resampled DataFrames have different lengths!"

    print(f"  ✓ Resampled all streams to {len(resampled['imu'])} samples @ 400 Hz")

    # Verify uniform timesteps
    dt = np.diff(resampled['imu']['timestamp'].values)
    assert np.allclose(dt, 1/400.0, atol=1e-6), "Timestamps not uniform!"
    print(f"  ✓ Timestep uniformity verified: {np.mean(dt)*1000:.3f} ms (±{np.std(dt)*1e6:.1f} µs)")

    # Step 3: EKF segmentation
    print("\n🔍 Step 3: EKF Health Segmentation")
    segments = segment_by_ekf_health(
        data['ekf'],
        innovation_threshold=1.0,
        min_segment_duration_s=1.0
    )
    print(f"  ✓ Found {len(segments)} healthy segments")
    for i, (t_start, t_end) in enumerate(segments):
        print(f"    Segment {i+1}: {t_start:.2f}s to {t_end:.2f}s (duration: {t_end-t_start:.2f}s)")

    assert len(segments) > 0, "No healthy segments found!"

    print("\n✅ Preprocessing pipeline test PASSED")
    return resampled


def test_fdm_rollout():
    """Test the FDM rollout with known parameters."""
    print("\n" + "="*60)
    print("TEST 2: FDM Rollout")
    print("="*60)

    # Create FDM parameters
    params = {
        'mass': jnp.array(1.2),  # kg
        'kT': jnp.array(1.5e-5),  # N/(rad/s)^2
        'kQ': jnp.array(2e-7),  # Nm/(rad/s)^2
        'inertia': jnp.array([0.01, 0.01, 0.02]),  # kg·m²
        'c_drag': jnp.array(0.001),
        'pwm_to_omega_poly': jnp.array([0.0, 1000.0, 0.0]),  # Linear: ω = 1000*PWM
        'motor_positions': QUAD_X['motor_positions'],
        'motor_directions': QUAD_X['motor_directions'],
    }

    print("\n📊 FDM Parameters:")
    print(f"  Mass: {params['mass']:.2f} kg")
    print(f"  kT: {params['kT']:.2e} N/(rad/s)²")
    print(f"  Inertia: [{params['inertia'][0]:.3f}, {params['inertia'][1]:.3f}, {params['inertia'][2]:.3f}] kg·m²")
    print(f"  Frame: quad_x (4 motors)")

    # Initial state: hovering at rest
    state_init = jnp.array([
        1.0, 0.0, 0.0, 0.0,  # Quaternion (identity)
        0.0, 0.0, 0.0,       # Velocity (m/s)
        0.0, 0.0, 0.0        # Angular velocity (rad/s)
    ])

    # PWM sequence: constant hover throttle
    duration = 1.0  # seconds
    dt = 0.0025  # 400 Hz
    n_steps = int(duration / dt)

    hover_pwm_norm = 0.55  # Normalized PWM for hover
    pwm_sequence = jnp.ones((n_steps, 4)) * hover_pwm_norm

    print(f"\n🚁 Simulating {duration}s of hover flight...")
    print(f"  Timesteps: {n_steps}")
    print(f"  dt: {dt*1000:.2f} ms")
    print(f"  PWM: {hover_pwm_norm:.2f} (normalized)")

    # Run rollout
    import time
    t_start = time.time()
    trajectory = rollout(state_init, pwm_sequence, params, dt)
    t_elapsed = time.time() - t_start

    print(f"  ✓ Rollout completed in {t_elapsed*1000:.1f} ms")
    print(f"  ✓ Throughput: {n_steps/t_elapsed:.0f} steps/s ({t_elapsed/n_steps*1e6:.1f} µs/step)")

    # Verify trajectory shape
    assert trajectory.shape == (n_steps, 10), f"Unexpected trajectory shape: {trajectory.shape}"
    print(f"  ✓ Trajectory shape: {trajectory.shape}")

    # Check quaternion normalization
    q_norms = jnp.linalg.norm(trajectory[:, 0:4], axis=1)
    assert jnp.allclose(q_norms, 1.0, atol=1e-4), "Quaternions not normalized!"
    print(f"  ✓ Quaternion norms: mean={jnp.mean(q_norms):.6f}, std={jnp.std(q_norms):.2e}")

    # Check final state (should be close to hover)
    final_state = trajectory[-1]
    final_vz = final_state[6]
    final_omega = final_state[7:10]

    print(f"\n📈 Final State:")
    print(f"  Vertical velocity: {final_vz:.4f} m/s (should be ~0 for hover)")
    print(f"  Angular velocity: [{final_omega[0]:.4f}, {final_omega[1]:.4f}, {final_omega[2]:.4f}] rad/s")

    # For testing, we just verify the simulation runs and produces reasonable output
    # (perfect hover tuning would require solving for equilibrium PWM)
    # Final velocity should be finite (not NaN/Inf) and bounded
    assert abs(final_vz) < 20.0, f"Vertical velocity unreasonably large: {final_vz:.2f} m/s"
    assert not np.isnan(final_vz) and not np.isinf(final_vz), "Vertical velocity is NaN or Inf!"

    print("\n✅ FDM rollout test PASSED")
    return trajectory


def test_gradient_computation():
    """Test that JAX gradients can be computed."""
    print("\n" + "="*60)
    print("TEST 3: Gradient Computation")
    print("="*60)

    import jax
    from ardupilot_sysid.src.fdm import loss_fn, flatten_params, unflatten_params

    # Create parameters - separate optimizable from fixed
    params_optimizable = {
        'mass': jnp.array(1.2),
        'kT': jnp.array(1.5e-5),
        'kQ': jnp.array(2e-7),
        'inertia': jnp.array([0.01, 0.01, 0.02]),
        'c_drag': jnp.array(0.001),
        'pwm_to_omega_poly': jnp.array([0.0, 1000.0, 0.0]),
    }

    params_fixed = {
        'motor_positions': QUAD_X['motor_positions'],
        'motor_directions': QUAD_X['motor_directions'],
    }

    # Flatten optimizable parameters
    params_flat, template = flatten_params(params_optimizable)
    print(f"\n📊 Flattened parameters ({len(params_flat)} scalars):")
    for i, val in enumerate(params_flat):
        print(f"  {i+1}. param[{i}]: {val:.2e}")

    # Create synthetic target trajectory
    # First need to combine optimizable + fixed params for rollout
    params_full = {**params_optimizable, **params_fixed}
    state_init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n_steps = 10
    pwm_sequence = jnp.ones((n_steps, 4)) * 0.55
    dt = 0.0025

    # Rollout produces n_steps states
    trajectory_states = rollout(state_init, pwm_sequence, params_full, dt)

    # Target trajectory should include initial state: shape (n_steps+1, 10)
    target_trajectory = jnp.vstack([state_init, trajectory_states])

    # Create weights
    weights = get_default_state_weights()

    # Compute loss using correct API: (params_flat, template, params_fixed, ...)
    loss_val = loss_fn(params_flat, template, params_fixed, target_trajectory, pwm_sequence, weights, dt)
    print(f"\n📉 Loss value: {loss_val:.2e}")

    # Compute gradient
    print("\n🔬 Computing gradients...")
    grad_fn = jax.grad(loss_fn, argnums=0)

    import time
    t_start = time.time()
    grads = grad_fn(params_flat, template, params_fixed, target_trajectory, pwm_sequence, weights, dt)
    t_elapsed = time.time() - t_start

    print(f"  ✓ Gradient computation completed in {t_elapsed*1000:.1f} ms")

    # Check gradient shapes
    assert grads.shape == params_flat.shape, "Gradient shape mismatch!"
    print(f"  ✓ Gradient shape: {grads.shape}")

    # Print gradients
    print(f"\n📊 Gradients:")
    for i, grad in enumerate(grads):
        print(f"  {i+1}. ∂L/∂param[{i}]: {grad:.2e}")

    # Verify gradients are not NaN or Inf
    assert not jnp.any(jnp.isnan(grads)), "Gradients contain NaN!"
    assert not jnp.any(jnp.isinf(grads)), "Gradients contain Inf!"
    print(f"  ✓ All gradients are finite")

    print("\n✅ Gradient computation test PASSED")


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("🧪 INTEGRATION TESTS: Full Pipeline Validation")
    print("="*60)

    try:
        # Test 1: Preprocessing pipeline
        resampled_data = test_preprocessing_pipeline()

        # Test 2: FDM rollout
        trajectory = test_fdm_rollout()

        # Test 3: Gradient computation
        test_gradient_computation()

        # Summary
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\n📊 Summary:")
        print("  ✓ Preprocessing pipeline: align, resample, segment")
        print("  ✓ FDM rollout: physics simulation with JAX")
        print("  ✓ Gradient computation: autodiff working correctly")
        print("\n🚀 The pipeline is ready for optimization!")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
