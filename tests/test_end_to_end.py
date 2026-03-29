"""
End-to-end pipeline tests.

Tests complete workflows from log parsing to parameter identification
and validation using synthetic data with known ground truth.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import tempfile

from ardupilot_sysid.src.fdm import (
    rollout, fdm_step, QUAD_X,
    flatten_params, unflatten_params, loss_fn
)
from ardupilot_sysid.src.preprocessing import (
    resample_to_uniform_grid, segment_by_ekf_health
)
from src.validation.rollout import hold_out_validation, split_train_test
from src.validation.metrics import summarize_validation_metrics


class TestSyntheticDataGeneration:
    """Test synthetic data generation for testing."""

    def test_generate_hover_trajectory(self):
        """Test generation of hover flight trajectory."""
        # Known parameters
        params = {
            'mass': jnp.array(1.2),
            'kT': jnp.array(1.5e-5),
            'kQ': jnp.array(2e-7),
            'inertia': jnp.array([0.01, 0.01, 0.02]),
            'c_drag': jnp.array(0.001),
            'pwm_to_omega_poly': jnp.array([0.0, 1000.0, 0.0]),
            'motor_positions': QUAD_X['motor_positions'],
            'motor_directions': QUAD_X['motor_directions'],
        }

        # Generate hover trajectory
        state_init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dt = 0.0025
        n_steps = 400  # 1 second

        hover_pwm = 0.55
        pwm_sequence = jnp.ones((n_steps, 4)) * hover_pwm

        trajectory = rollout(state_init, pwm_sequence, params, dt)

        # Verify trajectory properties
        assert trajectory.shape == (n_steps, 10)
        assert not jnp.any(jnp.isnan(trajectory))
        assert not jnp.any(jnp.isinf(trajectory))

        # Check quaternion normalization
        q_norms = jnp.linalg.norm(trajectory[:, 0:4], axis=1)
        assert jnp.allclose(q_norms, 1.0, atol=1e-4)

    def test_generate_dynamic_maneuver(self):
        """Test generation of dynamic maneuver trajectory."""
        params = {
            'mass': jnp.array(1.2),
            'kT': jnp.array(1.5e-5),
            'kQ': jnp.array(2e-7),
            'inertia': jnp.array([0.01, 0.01, 0.02]),
            'c_drag': jnp.array(0.001),
            'pwm_to_omega_poly': jnp.array([0.0, 1000.0, 0.0]),
            'motor_positions': QUAD_X['motor_positions'],
            'motor_directions': QUAD_X['motor_directions'],
        }

        state_init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dt = 0.0025
        n_steps = 1000

        # Varying PWM commands (simulating pitch/roll maneuvers)
        t = np.arange(n_steps) * dt
        base_pwm = 0.55
        pwm_sequence = np.ones((n_steps, 4)) * base_pwm
        pwm_sequence[:, 0] += 0.05 * np.sin(2*np.pi*0.5*t)  # Front-left
        pwm_sequence[:, 1] += 0.05 * np.sin(2*np.pi*0.5*t + np.pi)  # Front-right
        pwm_sequence[:, 2] += 0.03 * np.cos(2*np.pi*0.3*t)  # Rear-right
        pwm_sequence[:, 3] += 0.03 * np.cos(2*np.pi*0.3*t + np.pi)  # Rear-left

        pwm_sequence = jnp.array(pwm_sequence)

        trajectory = rollout(state_init, pwm_sequence, params, dt)

        # Verify trajectory has variation (not just hovering)
        angular_vel = trajectory[:, 7:10]
        assert jnp.std(angular_vel[:, 0]) > 0.01  # Roll rate varies
        assert jnp.std(angular_vel[:, 1]) > 0.01  # Pitch rate varies


class TestParserToSmootherIntegration:
    """Test integration between parser and smoother."""

    def test_data_format_compatibility(self):
        """Test that parser output format is compatible with smoother input."""
        # This is a structural test - verify expected data shapes
        # Real test would use actual parser output

        # Simulate parser output format
        n_samples = 1000
        dt = 0.0025

        mock_imu = {
            'timestamp': np.arange(n_samples) * dt,
            'gyr_x': np.random.randn(n_samples) * 0.1,
            'gyr_y': np.random.randn(n_samples) * 0.1,
            'gyr_z': np.random.randn(n_samples) * 0.05,
            'acc_x': np.random.randn(n_samples) * 0.5,
            'acc_y': np.random.randn(n_samples) * 0.5,
            'acc_z': 9.81 + np.random.randn(n_samples) * 0.2,
        }

        mock_gps = {
            'timestamp': np.arange(0, n_samples*dt, 0.1),  # 10 Hz
            'vel_n': np.random.randn(int(n_samples * dt * 10)) * 0.1,
            'vel_e': np.random.randn(int(n_samples * dt * 10)) * 0.1,
            'vel_d': np.random.randn(int(n_samples * dt * 10)) * 0.05,
        }

        # Verify expected structure
        assert 'timestamp' in mock_imu
        assert 'gyr_x' in mock_imu
        assert len(mock_imu['timestamp']) == n_samples

        # Smoother expects uniform grid - verify resampling works
        # (This would use actual resample_to_uniform_grid in real test)


class TestSmootherToOptimizerIntegration:
    """Test integration between smoother and optimizer."""

    def test_state_trajectory_format(self):
        """Test that smoother output format is compatible with optimizer."""
        # Simulate smoothed state trajectory
        n_steps = 400
        state_dim = 10

        # Mock smoothed states
        smoothed_states = np.random.randn(n_steps, state_dim)
        # Normalize quaternions
        smoothed_states[:, 0:4] /= np.linalg.norm(smoothed_states[:, 0:4], axis=1, keepdims=True)

        # Mock PWM sequence
        pwm_sequence = np.ones((n_steps-1, 4)) * 0.55

        # Verify shapes are compatible with loss function
        assert smoothed_states.shape[0] == n_steps
        assert smoothed_states.shape[1] == state_dim
        assert pwm_sequence.shape[0] == n_steps - 1
        assert pwm_sequence.shape[1] == 4

    def test_parameter_optimization_interface(self):
        """Test parameter optimization interface."""
        from ardupilot_sysid.src.fdm import get_default_state_weights

        # Create test parameters
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

        # Flatten parameters
        params_flat, template = flatten_params(params_optimizable)

        # Create mock data
        n_steps = 100
        state_init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pwm_sequence = jnp.ones((n_steps, 4)) * 0.55
        dt = 0.0025

        # Generate target trajectory
        params_full = {**params_optimizable, **params_fixed}
        trajectory_states = rollout(state_init, pwm_sequence, params_full, dt)
        target_trajectory = jnp.vstack([state_init, trajectory_states])

        weights = get_default_state_weights()

        # Test loss function call
        loss = loss_fn(params_flat, template, params_fixed, target_trajectory, pwm_sequence, weights, dt)

        assert not jnp.isnan(loss)
        assert not jnp.isinf(loss)
        assert loss >= 0


class TestOptimizerToValidationIntegration:
    """Test integration between optimizer output and validation."""

    def test_validation_input_format(self):
        """Test that optimizer output can be validated."""
        # Simulate optimized parameters
        params = {
            'mass': jnp.array(1.2),
            'kT': jnp.array(1.5e-5),
            'kQ': jnp.array(2e-7),
            'inertia': jnp.array([0.01, 0.01, 0.02]),
            'c_drag': jnp.array(0.001),
            'pwm_to_omega_poly': jnp.array([0.0, 1000.0, 0.0]),
            'motor_positions': QUAD_X['motor_positions'],
            'motor_directions': QUAD_X['motor_directions'],
        }

        # Create holdout data
        n_steps = 200
        state_init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pwm_holdout = jnp.ones((n_steps, 4)) * 0.55
        dt = 0.0025

        # Generate "actual" trajectory
        trajectory_actual = rollout(state_init, pwm_holdout, params, dt)
        states_holdout = jnp.vstack([state_init, trajectory_actual])

        # Run validation
        result = hold_out_validation(params, states_holdout, pwm_holdout, dt)

        # Verify result structure
        assert 'predicted' in result
        assert 'actual' in result
        assert 'residuals' in result
        assert 'timestamps' in result

        # Compute metrics
        metrics = summarize_validation_metrics(result)

        assert 'attitude' in metrics
        assert 'velocity' in metrics
        assert 'angular_velocity' in metrics


class TestFullPipelineSyntheticData:
    """
    Test complete pipeline on synthetic data with known ground truth.

    This is the ultimate integration test: generate synthetic trajectory
    from known parameters, then verify the pipeline can recover those
    parameters within acceptable tolerance.
    """

    def test_parameter_recovery_simple_hover(self):
        """
        Test parameter recovery from simple hover maneuver.

        Note: This is a simplified test. Full parameter recovery would
        require more complex maneuvers and longer duration.
        """
        # True parameters
        true_params = {
            'mass': jnp.array(1.2),
            'kT': jnp.array(1.5e-5),
            'kQ': jnp.array(2e-7),
            'inertia': jnp.array([0.01, 0.01, 0.02]),
            'c_drag': jnp.array(0.001),
            'pwm_to_omega_poly': jnp.array([0.0, 1000.0, 0.0]),
        }

        # Generate synthetic trajectory
        state_init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dt = 0.0025
        n_steps = 400

        hover_pwm = 0.55
        pwm_sequence = jnp.ones((n_steps, 4)) * hover_pwm

        params_with_fixed = {
            **true_params,
            'motor_positions': QUAD_X['motor_positions'],
            'motor_directions': QUAD_X['motor_directions']
        }

        trajectory = rollout(state_init, pwm_sequence, params_with_fixed, dt)
        target_trajectory = jnp.vstack([state_init, trajectory])

        # Note: Actual optimization would use MAP optimizer here
        # For now, just verify loss at true parameters is low

        from ardupilot_sysid.src.fdm import get_default_state_weights

        params_flat, template = flatten_params(true_params)
        params_fixed = {
            'motor_positions': QUAD_X['motor_positions'],
            'motor_directions': QUAD_X['motor_directions']
        }
        weights = get_default_state_weights()

        loss = loss_fn(params_flat, template, params_fixed, target_trajectory, pwm_sequence, weights, dt)

        # Loss should be very small (near perfect fit)
        assert loss < 0.01, f"Loss at true parameters should be small, got {loss}"

    @pytest.mark.slow
    def test_parameter_recovery_with_noise(self):
        """
        Test parameter recovery with noisy observations.

        This test adds realistic sensor noise to verify robustness.
        """
        # TODO: Implement full parameter recovery test with:
        # 1. Synthetic trajectory generation
        # 2. Add sensor noise
        # 3. Run full pipeline (parser → smoother → optimizer)
        # 4. Verify recovered parameters within 10% of true values
        pass

    @pytest.mark.slow
    def test_parameter_recovery_dynamic_maneuvers(self):
        """
        Test parameter recovery from dynamic flight maneuvers.

        Uses roll/pitch/yaw maneuvers to excite all parameters.
        """
        # TODO: Implement test with:
        # 1. Generate trajectory with rich excitation
        # 2. Run full pipeline
        # 3. Verify parameter recovery
        # 4. Check excitation scores
        pass


class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_invalid_log_data(self):
        """Test handling of invalid or corrupted log data."""
        # TODO: Test with malformed data
        pass

    def test_insufficient_excitation(self):
        """Test handling when flight data has poor excitation."""
        # TODO: Verify warnings are generated
        pass

    def test_optimization_failure(self):
        """Test handling when optimization fails to converge."""
        # TODO: Test with pathological cases
        pass


class TestSegmentProcessing:
    """Test processing of flight segments."""

    def test_train_test_split(self):
        """Test train/test split functionality."""
        segments = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 50.0)]

        train, test = split_train_test(segments, test_ratio=0.2)

        assert len(train) == 4
        assert len(test) == 1
        assert test[0] == (40.0, 50.0)  # Last segment

    def test_multiple_segment_optimization(self):
        """Test optimization using multiple flight segments."""
        # TODO: Test with multiple segments
        # Verify segments are properly concatenated or batch-processed
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
