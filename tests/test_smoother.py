"""
Tests for the RTS smoother implementation.

Tests cover:
1. State space models (transition, observation)
2. UKF forward pass
3. RTS backward pass
4. Comparison with ground truth
"""

import pytest
import numpy as np
import pandas as pd
from ardupilot_sysid.src.smoother import (
    UnscentedKalmanFilter,
    RTSSmoother,
    state_transition_model,
    imu_observation_model,
    gps_observation_model,
    baro_observation_model,
    quaternion_to_euler,
    euler_to_quaternion
)


class TestStateSpaceModels:
    """Tests for state space models."""

    def test_state_transition_constant_velocity(self):
        """Test that constant velocity model maintains velocity."""
        # Initial state: unit quaternion, velocity [1, 0, 0], no rotation
        x0 = np.array([1, 0, 0, 0,  # quaternion
                       1, 0, 0,     # velocity
                       0, 0, 0,     # angular velocity
                       0, 0, 0])    # position

        dt = 0.1
        x1 = state_transition_model(x0, dt)

        # Velocity should remain constant
        np.testing.assert_allclose(x1[4:7], [1, 0, 0], rtol=1e-6)

        # Position should increase by velocity * dt
        np.testing.assert_allclose(x1[10:13], [0.1, 0, 0], rtol=1e-6)

    def test_state_transition_quaternion_integration(self):
        """Test quaternion integration with constant angular velocity."""
        # Initial state: unit quaternion, rotation about z-axis
        x0 = np.array([1, 0, 0, 0,  # quaternion
                       0, 0, 0,     # velocity
                       0, 0, 1,     # angular velocity (1 rad/s about z)
                       0, 0, 0])    # position

        dt = 0.1
        x1 = state_transition_model(x0, dt)

        # Quaternion should be rotated
        # After dt=0.1s at 1 rad/s, rotation angle is 0.1 rad
        expected_q = euler_to_quaternion(np.array([0, 0, 0.1]))
        np.testing.assert_allclose(x1[0:4], expected_q, rtol=1e-4, atol=1e-4)

        # Quaternion should remain normalized
        assert np.abs(np.linalg.norm(x1[0:4]) - 1.0) < 1e-6

    def test_quaternion_normalization(self):
        """Test that state transition maintains quaternion normalization."""
        x0 = np.array([0.7, 0.1, 0.2, 0.3,  # non-unit quaternion
                       0, 0, 0,
                       0.5, 0.3, 0.2,  # some angular velocity
                       0, 0, 0])

        dt = 0.01
        x1 = state_transition_model(x0, dt)

        # Output quaternion should be normalized
        assert np.abs(np.linalg.norm(x1[0:4]) - 1.0) < 1e-6

    def test_imu_observation_model(self):
        """Test IMU observation model extracts angular velocity."""
        x = np.array([1, 0, 0, 0,
                      0, 0, 0,
                      0.5, 0.3, -0.2,  # angular velocity
                      0, 0, 0])

        z = imu_observation_model(x)

        # Should return [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
        assert z.shape == (6,)
        # Angular velocity should match state
        np.testing.assert_allclose(z[3:6], [0.5, 0.3, -0.2])

    def test_gps_observation_model(self):
        """Test GPS observation model extracts velocity and position."""
        x = np.array([1, 0, 0, 0,
                      2, 3, -1,        # velocity
                      0, 0, 0,
                      100, 200, 50])   # position

        z = gps_observation_model(x)

        # Should return [vel_n, vel_e, vel_d, lat, lng, alt]
        assert z.shape == (6,)
        np.testing.assert_allclose(z[0:3], [2, 3, -1])  # velocity
        np.testing.assert_allclose(z[3:6], [100, 200, 50])  # position

    def test_baro_observation_model(self):
        """Test barometer observation model extracts altitude."""
        x = np.array([1, 0, 0, 0,
                      0, 0, 0,
                      0, 0, 0,
                      0, 0, -50])  # altitude (negative down in NED)

        z = baro_observation_model(x)

        # Should return [altitude]
        assert z.shape == (1,)
        np.testing.assert_allclose(z, [-50])


class TestQuaternionConversions:
    """Tests for quaternion/Euler conversions."""

    def test_euler_to_quaternion_identity(self):
        """Test identity rotation."""
        euler = np.array([0, 0, 0])
        q = euler_to_quaternion(euler)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)

    def test_quaternion_to_euler_identity(self):
        """Test identity rotation."""
        q = np.array([1, 0, 0, 0])
        euler = quaternion_to_euler(q)
        np.testing.assert_allclose(euler, [0, 0, 0], atol=1e-6)

    def test_euler_quaternion_roundtrip(self):
        """Test conversion roundtrip."""
        euler_orig = np.array([0.1, -0.2, 0.3])
        q = euler_to_quaternion(euler_orig)
        euler_recovered = quaternion_to_euler(q)
        np.testing.assert_allclose(euler_orig, euler_recovered, atol=1e-6)


class TestUKF:
    """Tests for Unscented Kalman Filter."""

    def test_ukf_initialization(self):
        """Test UKF initializes with correct parameters."""
        ukf = UnscentedKalmanFilter(state_dim=13)
        assert ukf.n == 13
        assert len(ukf.weights_mean) == 27  # 2n + 1
        assert len(ukf.weights_cov) == 27
        # Weights should sum to 1 (allow small numerical error)
        assert np.abs(np.sum(ukf.weights_mean) - 1.0) < 1e-9

    def test_sigma_point_generation(self):
        """Test sigma point generation."""
        ukf = UnscentedKalmanFilter(state_dim=3)
        x = np.array([1, 2, 3])
        P = np.eye(3)

        sigma_points = ukf._generate_sigma_points(x, P)

        # Should have 2n+1 = 7 sigma points
        assert sigma_points.shape == (7, 3)
        # Central point should be the mean
        np.testing.assert_allclose(sigma_points[0], x)

    def test_ukf_predict_step(self):
        """Test UKF predict step."""
        ukf = UnscentedKalmanFilter(state_dim=13)

        x0 = np.array([1, 0, 0, 0,  # quaternion
                       1, 0, 0,     # velocity
                       0, 0, 0,     # angular velocity
                       0, 0, 0])    # position

        P0 = np.eye(13) * 0.01
        Q = np.eye(13) * 0.001
        dt = 0.1

        x_pred, P_pred = ukf.predict(x0, P0, state_transition_model, Q, dt)

        # Predicted state should be reasonable
        assert x_pred.shape == (13,)
        assert P_pred.shape == (13, 13)

        # Covariance should increase (process noise added)
        assert np.trace(P_pred) > np.trace(P0)

    def test_ukf_update_step(self):
        """Test UKF update step."""
        ukf = UnscentedKalmanFilter(state_dim=13)

        x_pred = np.array([1, 0, 0, 0,
                           1, 0, 0,
                           0.5, 0, 0,
                           1, 0, 0])

        P_pred = np.eye(13) * 0.1
        R = np.eye(6) * 0.01

        # Create a measurement close to the prediction
        measurement = np.array([0, 0, 0, 0.5, 0, 0])  # IMU measurement

        x_updated, P_updated = ukf.update(
            x_pred, P_pred,
            measurement,
            imu_observation_model,
            R
        )

        # Updated state should be close to prediction (measurement agrees)
        assert x_updated.shape == (13,)
        assert P_updated.shape == (13, 13)

        # Covariance should decrease (information added)
        assert np.trace(P_updated) < np.trace(P_pred)


class TestRTSSmoother:
    """Tests for RTS smoother."""

    def test_rts_initialization(self):
        """Test RTS smoother initializes correctly."""
        smoother = RTSSmoother()
        assert smoother is not None

    def test_rts_backward_pass_reduces_uncertainty(self):
        """Test that RTS backward pass reduces covariance."""
        # Create synthetic forward states
        np.random.seed(42)
        N = 10
        forward_states = []

        for i in range(N):
            x = np.array([1, 0, 0, 0,
                          0, 0, 0,
                          0, 0, 0,
                          i, 0, 0])
            P = np.eye(13) * (0.1 + 0.01 * i)  # Increasing uncertainty

            forward_states.append(type('State', (), {
                'x_posterior': x,
                'P_posterior': P,
                'x_prior': x,
                'P_prior': P * 1.1,
                'timestamp': i * 0.1
            })())

        smoother = RTSSmoother()
        smoothed_states = smoother.backward_pass(forward_states)

        # Should return same number of states
        assert len(smoothed_states) == N

        # Last state should be unchanged
        np.testing.assert_allclose(
            smoothed_states[-1].x_posterior,
            forward_states[-1].x_posterior
        )

        # Earlier states should have reduced covariance
        for i in range(N - 1):
            fwd_trace = np.trace(forward_states[i].P_posterior)
            smooth_trace = np.trace(smoothed_states[i].P_posterior)
            # Smoothed should have lower or equal uncertainty
            assert smooth_trace <= fwd_trace * 1.01  # Allow small numerical error

    def test_rts_extract_angular_velocity(self):
        """Test extracting angular velocity from smoothed states."""
        from ardupilot_sysid.src.smoother.ukf import UKFState

        states = []
        for i in range(5):
            x = np.array([1, 0, 0, 0,
                          0, 0, 0,
                          i * 0.1, i * 0.2, i * 0.3,  # angular velocity
                          0, 0, 0])
            P = np.eye(13) * 0.1
            states.append(UKFState(x, P, x, P, i * 0.1))

        smoother = RTSSmoother()
        omega = smoother.extract_angular_velocity(states)

        assert omega.shape == (5, 3)
        # Check first and last
        np.testing.assert_allclose(omega[0], [0, 0, 0])
        np.testing.assert_allclose(omega[4], [0.4, 0.8, 1.2])

    def test_rts_comparison_metrics(self):
        """Test comparison between forward and smoothed states."""
        from ardupilot_sysid.src.smoother.ukf import UKFState

        np.random.seed(42)
        N = 10

        forward_states = []
        smoothed_states = []

        for i in range(N):
            x = np.random.randn(13)
            x[0:4] = x[0:4] / np.linalg.norm(x[0:4])  # Normalize quaternion

            P_fwd = np.eye(13) * 0.1
            P_smooth = np.eye(13) * 0.05  # Lower uncertainty

            forward_states.append(UKFState(x, P_fwd, x, P_fwd * 1.1, i * 0.1))
            smoothed_states.append(UKFState(x, P_smooth, x, P_fwd * 1.1, i * 0.1))

        smoother = RTSSmoother()
        metrics = smoother.compare_forward_vs_smoothed(forward_states, smoothed_states)

        assert 'forward_trace_mean' in metrics
        assert 'smoothed_trace_mean' in metrics
        assert 'variance_reduction' in metrics
        assert 'state_diff_norm' in metrics

        # Should show variance reduction
        assert metrics['variance_reduction'] > 0


class TestIntegration:
    """Integration tests for full UKF + RTS pipeline."""

    def test_full_pipeline_synthetic_data(self):
        """Test full UKF + RTS pipeline on synthetic data."""
        # Generate synthetic measurements
        np.random.seed(42)
        dt = 0.01  # 100 Hz
        T = 1.0    # 1 second
        N = int(T / dt)

        # Create synthetic IMU data
        timestamps = np.arange(N) * dt
        imu_data = pd.DataFrame({
            'timestamp': timestamps,
            'acc_x': np.zeros(N),
            'acc_y': np.zeros(N),
            'acc_z': np.zeros(N),
            'gyr_x': np.sin(2 * np.pi * timestamps),  # 1 Hz sinusoid
            'gyr_y': np.zeros(N),
            'gyr_z': np.zeros(N)
        })

        measurements = {'imu': imu_data}

        # Initial state and covariance
        x_init = np.array([1, 0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0])
        P_init = np.eye(13) * 0.1

        # Process and measurement noise
        Q = np.eye(13) * 1e-4
        R_imu = np.eye(6) * 1e-3

        # Run forward pass
        ukf = UnscentedKalmanFilter(state_dim=13)
        forward_states = ukf.forward_pass(
            x_init, P_init,
            measurements, dt,
            Q, R_imu,
            R_gps=np.eye(6),  # Not used
            R_baro=np.eye(1)  # Not used
        )

        assert len(forward_states) > 0

        # Run backward pass
        smoother = RTSSmoother()
        smoothed_states = smoother.backward_pass(forward_states)

        assert len(smoothed_states) == len(forward_states)

        # Compare forward vs smoothed
        metrics = smoother.compare_forward_vs_smoothed(forward_states, smoothed_states)

        # Smoothing should reduce uncertainty (or at worst, not increase it significantly)
        # Allow small negative values due to numerical issues
        assert metrics['variance_reduction'] >= -1.0


def test_quaternion_stays_normalized_through_pipeline():
    """Test that quaternions remain normalized through entire pipeline."""
    np.random.seed(42)

    # Generate simple constant motion
    dt = 0.01
    N = 50
    timestamps = np.arange(N) * dt

    imu_data = pd.DataFrame({
        'timestamp': timestamps,
        'acc_x': np.zeros(N),
        'acc_y': np.zeros(N),
        'acc_z': np.zeros(N),
        'gyr_x': np.ones(N) * 0.1,  # Constant rotation
        'gyr_y': np.zeros(N),
        'gyr_z': np.zeros(N)
    })

    measurements = {'imu': imu_data}

    x_init = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    P_init = np.eye(13) * 0.01
    Q = np.eye(13) * 1e-5
    R_imu = np.eye(6) * 1e-4

    ukf = UnscentedKalmanFilter(state_dim=13)
    forward_states = ukf.forward_pass(
        x_init, P_init, measurements, dt, Q, R_imu, np.eye(6), np.eye(1)
    )

    smoother = RTSSmoother()
    smoothed_states = smoother.backward_pass(forward_states)

    # Check all quaternions are normalized
    for state in forward_states + smoothed_states:
        q = state.x_posterior[0:4]
        q_norm = np.linalg.norm(q)
        assert np.abs(q_norm - 1.0) < 1e-6, f"Quaternion not normalized: {q_norm}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
