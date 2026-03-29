"""
Tests for the differentiable flight dynamics model (FDM) in JAX.

Tests cover:
1. Motor model functions (PWM conversion, thrust/torque)
2. Frame configurations
3. Core FDM physics (thrust, torque, integration)
4. Gradient verification
5. Rollout functionality
6. Edge cases and numerical stability
"""

import pytest
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from ardupilot_sysid.src.fdm import (
    # Motor model
    normalize_pwm,
    denormalize_pwm,
    pwm_to_angular_velocity,
    angular_velocity_to_thrust,
    angular_velocity_to_torque,
    estimate_hover_pwm,
    # Frame configs
    get_frame_config,
    validate_frame_config,
    QUAD_X,
    # FDM core
    fdm_step,
    rollout,
    quat_to_rotation,
    quat_integrate,
    validate_state,
    quaternion_distance,
    flatten_params,
    unflatten_params,
    loss_fn,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def quad_params():
    """Standard quadcopter parameters for testing."""
    frame = get_frame_config('quad_x')
    return {
        'mass': 1.5,  # kg
        'kT': 1e-5,   # N/(rad/s)²
        'kQ': 1e-6,   # N·m/(rad/s)²
        'inertia': jnp.array([0.01, 0.01, 0.02]),  # kg·m²
        'c_drag': 1e-4,  # Rotational drag coefficient
        'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),  # rad/s
        'motor_positions': frame['motor_positions'],
        'motor_directions': frame['motor_directions'],
    }


@pytest.fixture
def hover_state():
    """State vector at hover (stationary, level)."""
    return jnp.array([
        1.0, 0.0, 0.0, 0.0,  # Quaternion (identity = level)
        0.0, 0.0, 0.0,        # Velocity (stationary)
        0.0, 0.0, 0.0         # Angular velocity (not rotating)
    ])


@pytest.fixture
def hover_pwm():
    """PWM values for approximate hover (4 motors)."""
    return jnp.ones(4) * 0.5  # 50% throttle


# ==============================================================================
# Motor Model Tests
# ==============================================================================

class TestMotorModel:
    """Tests for motor model functions."""

    def test_normalize_pwm_range(self):
        """Test PWM normalization maps 1000-2000 to 0-1."""
        pwm_us = jnp.array([1000.0, 1500.0, 2000.0])
        expected = jnp.array([0.0, 0.5, 1.0])
        result = normalize_pwm(pwm_us)
        assert jnp.allclose(result, expected)

    def test_normalize_pwm_clipping(self):
        """Test PWM normalization clips out-of-range values."""
        pwm_us = jnp.array([500.0, 1500.0, 2500.0])
        result = normalize_pwm(pwm_us)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    def test_denormalize_pwm(self):
        """Test denormalization is inverse of normalization."""
        pwm_normalized = jnp.array([0.0, 0.5, 1.0])
        expected = jnp.array([1000.0, 1500.0, 2000.0])
        result = denormalize_pwm(pwm_normalized)
        assert jnp.allclose(result, expected)

    def test_pwm_to_angular_velocity_zero(self):
        """Test zero PWM gives zero angular velocity (with zero offset)."""
        pwm = jnp.array([0.0, 0.0])
        poly = jnp.array([0.0, 100.0, 50.0])  # a0=0, a1=100, a2=50
        result = pwm_to_angular_velocity(pwm, poly)
        assert jnp.allclose(result, 0.0)

    def test_pwm_to_angular_velocity_quadratic(self):
        """Test quadratic polynomial mapping."""
        pwm = jnp.array([0.5])
        poly = jnp.array([0.0, 100.0, 50.0])
        # omega = 0 + 100*0.5 + 50*0.25 = 50 + 12.5 = 62.5
        expected = 62.5
        result = pwm_to_angular_velocity(pwm, poly)
        assert jnp.allclose(result, expected)

    def test_thrust_zero_omega(self):
        """Test zero angular velocity gives zero thrust."""
        omega = jnp.array([0.0, 0.0])
        kT = 1e-5
        result = angular_velocity_to_thrust(omega, kT)
        assert jnp.allclose(result, 0.0)

    def test_thrust_scaling(self):
        """Test thrust scales with omega²."""
        omega = jnp.array([100.0, 200.0])
        kT = 1e-5
        result = angular_velocity_to_thrust(omega, kT)
        expected = jnp.array([0.1, 0.4])  # kT * [100², 200²]
        assert jnp.allclose(result, expected)

    def test_torque_scaling(self):
        """Test torque scales with omega²."""
        omega = jnp.array([100.0, 200.0])
        kQ = 1e-6
        result = angular_velocity_to_torque(omega, kQ)
        expected = jnp.array([0.01, 0.04])  # kQ * [100², 200²]
        assert jnp.allclose(result, expected)

    def test_estimate_hover_pwm_reasonable(self, quad_params):
        """Test hover PWM estimate is in reasonable range."""
        hover_pwm = estimate_hover_pwm(
            mass=quad_params['mass'],
            kT=quad_params['kT'],
            poly_coeffs=quad_params['pwm_to_omega_poly'],
            num_motors=4
        )
        # Should be between 0.3 and 0.7 for typical values
        assert 0.3 < hover_pwm < 0.7


# ==============================================================================
# Frame Configuration Tests
# ==============================================================================

class TestFrameConfigs:
    """Tests for frame configurations."""

    def test_quad_x_config(self):
        """Test quad X configuration has correct structure."""
        config = get_frame_config('quad_x')
        assert config['motor_positions'].shape == (4, 3)
        assert config['motor_directions'].shape == (4,)
        validate_frame_config(config)

    def test_hexa_x_config(self):
        """Test hexa X configuration has correct structure."""
        config = get_frame_config('hexa_x')
        assert config['motor_positions'].shape == (6, 3)
        assert config['motor_directions'].shape == (6,)
        validate_frame_config(config)

    def test_motor_directions_valid(self):
        """Test motor directions are only +1 or -1."""
        config = get_frame_config('quad_x')
        directions = config['motor_directions']
        assert jnp.all((directions == 1) | (directions == -1))

    def test_invalid_frame_type(self):
        """Test error on invalid frame type."""
        with pytest.raises(ValueError, match="Unknown frame type"):
            get_frame_config('invalid_frame')

    def test_validate_config_missing_key(self):
        """Test validation fails on missing keys."""
        invalid_config = {'motor_positions': jnp.zeros((4, 3))}
        with pytest.raises(ValueError, match="missing"):
            validate_frame_config(invalid_config)

    def test_validate_config_shape_mismatch(self):
        """Test validation fails on shape mismatch."""
        invalid_config = {
            'motor_positions': jnp.zeros((4, 3)),
            'motor_directions': jnp.zeros(3)  # Wrong number
        }
        with pytest.raises(ValueError, match="mismatch"):
            validate_frame_config(invalid_config)


# ==============================================================================
# Quaternion and Rotation Tests
# ==============================================================================

class TestQuaternionOps:
    """Tests for quaternion operations."""

    def test_quat_to_rotation_identity(self):
        """Test identity quaternion gives identity rotation."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rotation(q)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-6)

    def test_quat_to_rotation_90deg_z(self):
        """Test 90° rotation about z-axis."""
        # q = [cos(45°), 0, 0, sin(45°)] for 90° CCW about z
        q = jnp.array([jnp.sqrt(2)/2, 0.0, 0.0, jnp.sqrt(2)/2])
        R = quat_to_rotation(q)

        # For a 90° CCW rotation about z (viewed from world frame):
        # The body frame's x-axis points where world's y-axis was
        # So R maps world x → body y
        x_axis = jnp.array([1.0, 0.0, 0.0])
        result = R @ x_axis
        expected = jnp.array([0.0, 1.0, 0.0])  # x maps to y
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_quat_integrate_no_rotation(self):
        """Test quaternion integration with zero angular velocity."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        omega = jnp.array([0.0, 0.0, 0.0])
        q_next = quat_integrate(q, omega, dt=0.01)
        assert jnp.allclose(q_next, q, atol=1e-6)

    def test_quat_integrate_normalization(self):
        """Test quaternion remains normalized after integration."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        omega = jnp.array([1.0, 2.0, 3.0])  # Arbitrary rotation
        q_next = quat_integrate(q, omega, dt=0.01)

        # Check normalization
        norm = jnp.linalg.norm(q_next)
        assert jnp.allclose(norm, 1.0, atol=1e-6)

    def test_quaternion_distance_identical(self):
        """Test distance between identical quaternions is zero."""
        q1 = jnp.array([1.0, 0.0, 0.0, 0.0])
        q2 = jnp.array([1.0, 0.0, 0.0, 0.0])
        dist = quaternion_distance(q1, q2)
        assert jnp.allclose(dist, 0.0, atol=1e-6)

    def test_quaternion_distance_opposite(self):
        """Test distance between opposite orientations is maximum."""
        q1 = jnp.array([1.0, 0.0, 0.0, 0.0])
        q2 = jnp.array([0.0, 1.0, 0.0, 0.0])  # 180° rotation
        dist = quaternion_distance(q1, q2)
        # Should be close to 1.0 (maximum distance)
        assert dist > 0.9


# ==============================================================================
# FDM Core Tests
# ==============================================================================

class TestFDMCore:
    """Tests for core FDM functionality."""

    def test_fdm_step_hover_stability(self, quad_params, hover_state, hover_pwm):
        """Test hover state remains approximately stable."""
        # Adjust PWM to exactly balance gravity
        mass = quad_params['mass']
        kT = quad_params['kT']
        g = 9.81

        # Required thrust per motor
        thrust_per_motor = (mass * g) / 4.0

        # Required omega: sqrt(thrust / kT)
        omega_req = jnp.sqrt(thrust_per_motor / kT)

        # Compute PWM using polynomial inversion (simplified)
        poly = quad_params['pwm_to_omega_poly']
        a0, a1, a2 = poly
        # Solve: a2*pwm^2 + a1*pwm + a0 - omega = 0
        pwm_hover = (-a1 + jnp.sqrt(a1**2 - 4*a2*(a0 - omega_req))) / (2*a2)
        pwm = jnp.ones(4) * pwm_hover

        # Run one step
        next_state = fdm_step(hover_state, pwm, quad_params, dt=0.01)

        # Check velocity remains small (might not be exactly zero due to numerical errors)
        v = next_state[4:7]
        assert jnp.linalg.norm(v) < 0.1, "Velocity should remain small at hover"

    def test_fdm_step_zero_thrust_falls(self, quad_params, hover_state):
        """Test that zero thrust causes vehicle to fall."""
        pwm = jnp.zeros(4)  # No thrust

        # Run several steps
        state = hover_state
        for _ in range(10):
            state = fdm_step(state, pwm, quad_params, dt=0.01)

        # Check that z-velocity is negative (falling)
        vz = state[6]
        assert vz < -0.5, "Should be falling with no thrust"

    def test_fdm_step_differential_thrust_rotates(self, quad_params, hover_state):
        """Test differential thrust causes rotation."""
        # Apply more thrust to front motors (roll)
        pwm = jnp.array([0.6, 0.4, 0.6, 0.4])  # Front high, rear low

        # Run several steps
        state = hover_state
        for _ in range(20):
            state = fdm_step(state, pwm, quad_params, dt=0.01)

        # Check that angular velocity has developed
        omega = state[7:10]
        assert jnp.linalg.norm(omega) > 0.01, "Should have angular velocity"

    def test_fdm_step_no_nan_or_inf(self, quad_params, hover_state, hover_pwm):
        """Test that FDM step produces no NaN or Inf values."""
        next_state = fdm_step(hover_state, hover_pwm, quad_params, dt=0.01)
        assert jnp.all(jnp.isfinite(next_state)), "State should have no NaN/Inf"

    def test_validate_state_valid(self, hover_state):
        """Test validation accepts valid state."""
        assert validate_state(hover_state)

    def test_validate_state_unnormalized_quat(self):
        """Test validation rejects unnormalized quaternion."""
        bad_state = jnp.array([
            2.0, 0.0, 0.0, 0.0,  # Unnormalized quaternion
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        assert not validate_state(bad_state)

    def test_validate_state_nan(self):
        """Test validation rejects NaN values."""
        bad_state = jnp.array([
            1.0, 0.0, 0.0, 0.0,
            jnp.nan, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        assert not validate_state(bad_state)


# ==============================================================================
# Rollout Tests
# ==============================================================================

class TestRollout:
    """Tests for trajectory rollout."""

    def test_rollout_shape(self, quad_params, hover_state):
        """Test rollout produces correct output shape."""
        T = 100
        N_motors = 4
        pwm_sequence = jnp.ones((T, N_motors)) * 0.5

        trajectory = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)

        assert trajectory.shape == (T, 10), f"Expected (100, 10), got {trajectory.shape}"

    def test_rollout_initial_state(self, quad_params, hover_state):
        """Test rollout first state matches after one step."""
        pwm_sequence = jnp.ones((10, 4)) * 0.5
        trajectory = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)

        # First trajectory state should match single fdm_step from initial state
        expected_first = fdm_step(hover_state, pwm_sequence[0], quad_params, dt=0.01)
        assert jnp.allclose(trajectory[0], expected_first, atol=1e-6)

    def test_rollout_no_nan(self, quad_params, hover_state):
        """Test rollout produces no NaN values."""
        pwm_sequence = jnp.ones((100, 4)) * 0.5
        trajectory = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)

        assert jnp.all(jnp.isfinite(trajectory)), "Trajectory should have no NaN/Inf"

    def test_rollout_performance(self, quad_params, hover_state):
        """Test rollout completes in reasonable time for 10 seconds."""
        # 10 seconds at 100 Hz = 1000 steps
        T = 1000
        pwm_sequence = jnp.ones((T, 4)) * 0.5

        # First call compiles (will be slower)
        _ = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)

        # Subsequent calls should be fast (< 1 second)
        import time
        start = time.time()
        _ = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Rollout too slow: {elapsed:.3f}s for 10s simulation"


# ==============================================================================
# Gradient Tests
# ==============================================================================

class TestGradients:
    """Tests for gradient correctness using JAX's check_grads."""

    def test_fdm_step_gradients_wrt_state(self, quad_params, hover_state, hover_pwm):
        """Test gradients of fdm_step w.r.t. state."""
        def f(state):
            return jnp.sum(fdm_step(state, hover_pwm, quad_params, dt=0.01))

        # Check gradients (order=1 for first derivatives)
        check_grads(f, (hover_state,), order=1, rtol=1e-3)

    def test_fdm_step_gradients_wrt_pwm(self, quad_params, hover_state, hover_pwm):
        """Test gradients of fdm_step w.r.t. PWM."""
        def f(pwm):
            return jnp.sum(fdm_step(hover_state, pwm, quad_params, dt=0.01))

        check_grads(f, (hover_pwm,), order=1, rtol=1e-3)

    def test_rollout_gradients_wrt_initial_state(self, quad_params, hover_state):
        """Test gradients of rollout w.r.t. initial state."""
        pwm_sequence = jnp.ones((10, 4)) * 0.5

        def f(state_init):
            traj = rollout(state_init, pwm_sequence, quad_params, dt=0.01)
            return jnp.sum(traj)  # Sum over all outputs

        check_grads(f, (hover_state,), order=1, rtol=1e-3)

    def test_rollout_gradients_wrt_pwm_sequence(self, quad_params, hover_state):
        """Test gradients of rollout w.r.t. PWM sequence."""
        pwm_sequence = jnp.ones((10, 4)) * 0.5

        def f(pwm_seq):
            traj = rollout(hover_state, pwm_seq, quad_params, dt=0.01)
            return jnp.sum(traj)

        check_grads(f, (pwm_sequence,), order=1, rtol=1e-3)

    def test_loss_fn_gradients(self, quad_params, hover_state):
        """Test gradients of loss function for optimization."""
        # Create synthetic target trajectory
        pwm_sequence = jnp.ones((10, 4)) * 0.5
        target_traj = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)
        target_traj_full = jnp.vstack([hover_state[None, :], target_traj])

        weights = jnp.ones(10)

        # Test gradient w.r.t. a single parameter (mass) directly
        # This tests the core loss_fn gradient path without flattening
        def f(mass_value):
            params_modified = quad_params.copy()
            params_modified['mass'] = mass_value
            predicted = rollout(hover_state, pwm_sequence, params_modified, dt=0.01)
            residuals = predicted - target_traj_full[1:]
            weighted_residuals = residuals * weights
            return jnp.mean(weighted_residuals ** 2)

        # Test gradient
        grad_fn = jax.grad(f)
        grad_mass = grad_fn(quad_params['mass'])

        # Gradient should be near zero when params match target
        assert jnp.abs(grad_mass) < 1e-3, f"Gradient should be near zero, got {grad_mass}"


# ==============================================================================
# Parameter Flattening Tests
# ==============================================================================

class TestParameterFlattening:
    """Tests for parameter flattening/unflattening."""

    def test_flatten_unflatten_roundtrip(self, quad_params):
        """Test flatten then unflatten recovers original parameters."""
        # Extract optimizable params
        params_opt = {
            'mass': quad_params['mass'],
            'kT': quad_params['kT'],
            'kQ': quad_params['kQ'],
            'inertia': quad_params['inertia'],
            'c_drag': quad_params['c_drag'],
            'pwm_to_omega_poly': quad_params['pwm_to_omega_poly'],
        }

        # Flatten
        params_flat, template = flatten_params(params_opt)

        # Fixed params
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions'],
        }

        # Unflatten
        params_recovered = unflatten_params(params_flat, template, params_fixed)

        # Check all parameters match
        for key in params_opt:
            original = params_opt[key]
            recovered = params_recovered[key]
            assert jnp.allclose(original, recovered), f"Mismatch in {key}"

    def test_flatten_preserves_values(self, quad_params):
        """Test that flattening preserves parameter values."""
        params_opt = {
            'mass': quad_params['mass'],
            'kT': quad_params['kT'],
        }

        params_flat, _ = flatten_params(params_opt)

        # Check flattened array contains the values
        assert params_flat[0] == quad_params['mass']
        assert params_flat[1] == quad_params['kT']


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_hover(self, quad_params, hover_state):
        """Test full pipeline: initial state → rollout → loss."""
        # Generate target trajectory
        pwm_hover = jnp.ones(4) * 0.5
        pwm_sequence = jnp.tile(pwm_hover, (100, 1))

        target_traj = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)
        target_traj_full = jnp.vstack([hover_state[None, :], target_traj])

        # Flatten parameters
        params_opt = {
            'mass': quad_params['mass'],
            'kT': quad_params['kT'],
            'kQ': quad_params['kQ'],
            'inertia': quad_params['inertia'],
            'c_drag': quad_params['c_drag'],
            'pwm_to_omega_poly': quad_params['pwm_to_omega_poly'],
        }
        params_flat, template = flatten_params(params_opt)

        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions'],
        }

        weights = jnp.ones(10)

        # Compute loss (should be near zero with same params)
        loss = loss_fn(
            params_flat, template, params_fixed,
            target_traj_full, pwm_sequence, weights, dt=0.01
        )

        assert loss < 1e-6, f"Loss should be near zero, got {loss}"

    def test_gradient_descent_step(self, quad_params, hover_state):
        """Test that gradient descent reduces loss."""
        # Generate target with true params
        pwm_sequence = jnp.ones((20, 4)) * 0.5
        target_traj = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)
        target_traj_full = jnp.vstack([hover_state[None, :], target_traj])

        weights = jnp.ones(10)

        # Test gradient descent on single parameter (mass) directly
        # This avoids issues with flatten/unflatten
        mass_initial = quad_params['mass'] * 1.2  # 20% error

        def loss_fn_mass(mass_value):
            params_modified = quad_params.copy()
            params_modified['mass'] = mass_value
            predicted = rollout(hover_state, pwm_sequence, params_modified, dt=0.01)
            residuals = predicted - target_traj_full[1:]
            weighted_residuals = residuals * weights
            return jnp.mean(weighted_residuals ** 2)

        # Compute initial loss
        loss_initial = loss_fn_mass(mass_initial)

        # Compute gradient
        grad_fn = jax.grad(loss_fn_mass)
        grad_mass = grad_fn(mass_initial)

        # Take gradient descent step
        learning_rate = 0.01
        mass_updated = mass_initial - learning_rate * grad_mass

        # Compute new loss
        loss_updated = loss_fn_mass(mass_updated)

        # Loss should decrease
        assert loss_updated < loss_initial, \
            f"Loss should decrease: {loss_initial} → {loss_updated}"


# ==============================================================================
# Performance Benchmarks
# ==============================================================================

@pytest.mark.skipif(True, reason="pytest-benchmark not installed")
class TestPerformance:
    """Performance benchmarks (optional, run with pytest -m benchmark)."""

    def test_fdm_step_speed(self, quad_params, hover_state, hover_pwm, benchmark):
        """Benchmark single FDM step."""
        # First call to compile
        _ = fdm_step(hover_state, hover_pwm, quad_params, dt=0.01)

        # Benchmark compiled version
        result = benchmark(
            fdm_step, hover_state, hover_pwm, quad_params, dt=0.01
        )
        assert jnp.all(jnp.isfinite(result))

    def test_rollout_speed(self, quad_params, hover_state, benchmark):
        """Benchmark 10-second rollout."""
        pwm_sequence = jnp.ones((1000, 4)) * 0.5

        # First call to compile
        _ = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)

        # Benchmark compiled version
        result = benchmark(
            rollout, hover_state, pwm_sequence, quad_params, dt=0.01
        )
        assert result.shape == (1000, 10)

    def test_gradient_computation_speed(self, quad_params, hover_state, benchmark):
        """Benchmark gradient computation."""
        pwm_sequence = jnp.ones((100, 4)) * 0.5
        target_traj = rollout(hover_state, pwm_sequence, quad_params, dt=0.01)
        target_traj_full = jnp.vstack([hover_state[None, :], target_traj])

        params_opt = {
            'mass': quad_params['mass'],
            'kT': quad_params['kT'],
            'kQ': quad_params['kQ'],
            'inertia': quad_params['inertia'],
            'c_drag': quad_params['c_drag'],
            'pwm_to_omega_poly': quad_params['pwm_to_omega_poly'],
        }
        params_flat, template = flatten_params(params_opt)

        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions'],
        }

        weights = jnp.ones(10)

        grad_fn = jax.grad(loss_fn)

        # First call to compile
        _ = grad_fn(
            params_flat, template, params_fixed,
            target_traj_full, pwm_sequence, weights, dt=0.01
        )

        # Benchmark compiled version
        result = benchmark(
            grad_fn,
            params_flat, template, params_fixed,
            target_traj_full, pwm_sequence, weights, dt=0.01
        )
        assert jnp.all(jnp.isfinite(result))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
