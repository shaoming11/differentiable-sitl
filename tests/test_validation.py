"""Tests for validation module."""

import numpy as np
import jax.numpy as jnp
import pytest
from src.validation.rollout import (
    hold_out_validation,
    split_train_test,
    compare_trajectories
)
from src.validation.metrics import (
    quaternion_to_euler,
    compute_attitude_rmse,
    compute_velocity_rmse,
    compute_angular_velocity_rmse,
    summarize_validation_metrics
)


class TestQuaternionConversion:
    """Test quaternion to Euler angle conversion."""

    def test_identity_quaternion(self):
        """Identity quaternion should give zero Euler angles."""
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        euler = quaternion_to_euler(q_identity)

        assert euler.shape == (3,)
        np.testing.assert_allclose(euler, [0.0, 0.0, 0.0], atol=1e-10)

    def test_90deg_roll(self):
        """Test 90° roll rotation."""
        # qw, qx, qy, qz for 90° around x-axis
        q = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0])
        euler = quaternion_to_euler(q)

        expected = np.array([np.pi/2, 0.0, 0.0])
        np.testing.assert_allclose(euler, expected, atol=1e-6)

    def test_90deg_pitch(self):
        """Test 90° pitch rotation."""
        q = np.array([np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0])
        euler = quaternion_to_euler(q)

        expected = np.array([0.0, np.pi/2, 0.0])
        np.testing.assert_allclose(euler, expected, atol=1e-6)

    def test_90deg_yaw(self):
        """Test 90° yaw rotation."""
        q = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        euler = quaternion_to_euler(q)

        expected = np.array([0.0, 0.0, np.pi/2])
        np.testing.assert_allclose(euler, expected, atol=1e-6)

    def test_batch_conversion(self):
        """Test batch conversion of multiple quaternions."""
        q_batch = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0],
            [np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0]
        ])

        euler_batch = quaternion_to_euler(q_batch)

        assert euler_batch.shape == (3, 3)
        np.testing.assert_allclose(euler_batch[0], [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(euler_batch[1], [np.pi/2, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(euler_batch[2], [0.0, np.pi/2, 0.0], atol=1e-6)


class TestAttitudeRMSE:
    """Test attitude RMSE computation."""

    def test_perfect_prediction(self):
        """Zero error when predicted equals actual."""
        states = np.random.randn(100, 10)
        # Normalize quaternions
        states[:, 0:4] /= np.linalg.norm(states[:, 0:4], axis=1, keepdims=True)

        rmse = compute_attitude_rmse(states, states)

        assert rmse['roll_deg'] < 1e-6
        assert rmse['pitch_deg'] < 1e-6
        assert rmse['yaw_deg'] < 1e-6

    def test_known_error(self):
        """Test with known attitude error."""
        T = 100

        # Actual: identity quaternions
        actual = np.zeros((T, 10))
        actual[:, 0] = 1.0  # qw = 1

        # Predicted: 10° roll error
        roll_error = np.deg2rad(10)
        q_error = np.array([np.cos(roll_error/2), np.sin(roll_error/2), 0.0, 0.0])
        predicted = actual.copy()
        predicted[:, 0:4] = q_error

        rmse = compute_attitude_rmse(predicted, actual)

        # Should be approximately 10° roll error
        assert 9.5 < rmse['roll_deg'] < 10.5
        assert rmse['pitch_deg'] < 0.5
        assert rmse['yaw_deg'] < 0.5

    def test_yaw_wraparound(self):
        """Test proper handling of yaw wraparound."""
        T = 100

        # Actual: 170° yaw
        actual = np.zeros((T, 10))
        yaw1 = np.deg2rad(170)
        actual[:, 0] = np.cos(yaw1/2)
        actual[:, 3] = np.sin(yaw1/2)

        # Predicted: -170° yaw (20° difference, but wraps around)
        predicted = np.zeros((T, 10))
        yaw2 = np.deg2rad(-170)
        predicted[:, 0] = np.cos(yaw2/2)
        predicted[:, 3] = np.sin(yaw2/2)

        rmse = compute_attitude_rmse(predicted, actual)

        # Should show ~20° error, not 340°
        assert rmse['yaw_deg'] < 25


class TestVelocityRMSE:
    """Test velocity RMSE computation."""

    def test_zero_error(self):
        """Test with identical velocities."""
        states = np.random.randn(100, 10)
        rmse = compute_velocity_rmse(states, states)

        assert rmse['vx'] < 1e-10
        assert rmse['vy'] < 1e-10
        assert rmse['vz'] < 1e-10
        assert rmse['v_total'] < 1e-10

    def test_known_error(self):
        """Test with known velocity error."""
        T = 100
        actual = np.zeros((T, 10))
        predicted = actual.copy()

        # Add 1 m/s error in vx
        predicted[:, 4] = 1.0

        rmse = compute_velocity_rmse(predicted, actual)

        assert abs(rmse['vx'] - 1.0) < 1e-10
        assert rmse['vy'] < 1e-10
        assert rmse['vz'] < 1e-10
        assert abs(rmse['v_total'] - 1.0) < 1e-10


class TestAngularVelocityRMSE:
    """Test angular velocity RMSE computation."""

    def test_zero_error(self):
        """Test with identical angular velocities."""
        states = np.random.randn(100, 10)
        rmse = compute_angular_velocity_rmse(states, states)

        assert rmse['wx'] < 1e-10
        assert rmse['wy'] < 1e-10
        assert rmse['wz'] < 1e-10

    def test_known_error(self):
        """Test with known angular velocity error."""
        T = 100
        actual = np.zeros((T, 10))
        predicted = actual.copy()

        # Add 0.5 rad/s error in wz
        predicted[:, 9] = 0.5

        rmse = compute_angular_velocity_rmse(predicted, actual)

        assert rmse['wx'] < 1e-10
        assert rmse['wy'] < 1e-10
        assert abs(rmse['wz'] - 0.5) < 1e-10


class TestSplitTrainTest:
    """Test train/test splitting."""

    def test_default_split(self):
        """Test 80/20 default split."""
        segments = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

        train, test = split_train_test(segments)

        assert len(train) == 4
        assert len(test) == 1
        assert train == segments[:-1]
        assert test == segments[-1:]

    def test_custom_ratio(self):
        """Test custom test ratio."""
        segments = [(i, i+1) for i in range(10)]

        train, test = split_train_test(segments, test_ratio=0.3)

        assert len(test) == 3
        assert len(train) == 7

    def test_minimum_test_size(self):
        """Ensure at least 1 test segment."""
        segments = [(0, 1), (1, 2)]

        train, test = split_train_test(segments, test_ratio=0.1)

        assert len(test) >= 1
        assert len(train) >= 1


class TestCompareTrajectories:
    """Test trajectory comparison metrics."""

    def test_perfect_match(self):
        """Test metrics for identical trajectories."""
        traj = np.random.randn(100, 10)

        metrics = compare_trajectories(traj, traj)

        for state_metrics in metrics.values():
            assert state_metrics['rmse'] < 1e-10
            assert state_metrics['mae'] < 1e-10
            assert abs(state_metrics['correlation'] - 1.0) < 1e-6

    def test_known_error(self):
        """Test metrics with known error."""
        actual = np.random.randn(100, 10)
        predicted = actual + 0.1  # Constant bias

        metrics = compare_trajectories(predicted, actual)

        # All states should have MAE ≈ 0.1
        for state_metrics in metrics.values():
            assert abs(state_metrics['mae'] - 0.1) < 0.01
            assert state_metrics['correlation'] > 0.99

    def test_custom_state_names(self):
        """Test with custom state names."""
        traj = np.random.randn(50, 3)
        state_names = ['x', 'y', 'z']

        metrics = compare_trajectories(traj, traj, state_names=state_names)

        assert set(metrics.keys()) == {'x', 'y', 'z'}

    def test_uncorrelated_trajectories(self):
        """Test correlation for uncorrelated data."""
        actual = np.random.randn(1000, 10)
        predicted = np.random.randn(1000, 10)

        metrics = compare_trajectories(predicted, actual)

        # Correlation should be near zero
        for state_metrics in metrics.values():
            assert abs(state_metrics['correlation']) < 0.1


class TestSummarizeValidationMetrics:
    """Test comprehensive validation metric summary."""

    def test_complete_summary(self):
        """Test that all metric categories are included."""
        T = 100

        # Create synthetic validation result
        predicted = np.random.randn(T, 10)
        predicted[:, 0:4] /= np.linalg.norm(predicted[:, 0:4], axis=1, keepdims=True)

        validation_result = {
            'predicted': predicted,
            'actual': predicted,  # Perfect match
            'residuals': np.zeros((T, 10)),
            'timestamps': np.arange(T) * 0.01
        }

        metrics = summarize_validation_metrics(validation_result)

        assert 'attitude' in metrics
        assert 'velocity' in metrics
        assert 'angular_velocity' in metrics
        assert 'duration_s' in metrics
        assert 'n_samples' in metrics

        assert metrics['duration_s'] == pytest.approx(0.99, abs=0.01)
        assert metrics['n_samples'] == T

    def test_realistic_errors(self):
        """Test with realistic prediction errors."""
        T = 200

        # Create actual trajectory
        actual = np.zeros((T, 10))
        actual[:, 0] = 1.0  # Identity quaternion

        # Create predicted with small errors
        predicted = actual.copy()
        predicted[:, 0:4] += np.random.randn(T, 4) * 0.01
        predicted[:, 0:4] /= np.linalg.norm(predicted[:, 0:4], axis=1, keepdims=True)
        predicted[:, 4:7] += np.random.randn(T, 3) * 0.05  # 5cm/s velocity noise
        predicted[:, 7:10] += np.random.randn(T, 3) * 0.1  # 0.1 rad/s angular velocity noise

        validation_result = {
            'predicted': predicted,
            'actual': actual,
            'residuals': predicted - actual,
            'timestamps': np.arange(T) * 0.01
        }

        metrics = summarize_validation_metrics(validation_result)

        # Check reasonable error magnitudes
        assert metrics['attitude']['roll_deg'] < 5
        assert metrics['attitude']['pitch_deg'] < 5
        assert metrics['attitude']['yaw_deg'] < 5
        assert metrics['velocity']['v_total'] < 0.2
        assert metrics['angular_velocity']['wx'] < 0.2
        assert metrics['angular_velocity']['wy'] < 0.2
        assert metrics['angular_velocity']['wz'] < 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
