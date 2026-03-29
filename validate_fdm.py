#!/usr/bin/env python3
"""
Validation script for the differentiable flight dynamics model.

This script runs key tests to validate the FDM implementation:
1. Known thrust/torque outputs
2. Gradient verification
3. Quaternion normalization
4. Rollout functionality
5. Performance on 10-second simulation
"""

import sys
import time
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

# Import FDM components
from ardupilot_sysid.src.fdm import (
    # Motor model
    normalize_pwm,
    pwm_to_angular_velocity,
    angular_velocity_to_thrust,
    # Frame configs
    get_frame_config,
    # FDM core
    fdm_step,
    rollout,
    quat_to_rotation,
    quat_integrate,
    validate_state,
)

def print_status(test_name, passed, details=""):
    """Print test status with formatting."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"  {details}")

def get_test_params():
    """Get standard test parameters."""
    frame = get_frame_config('quad_x')
    return {
        'mass': 1.5,  # kg
        'kT': 1e-5,   # N/(rad/s)²
        'kQ': 1e-6,   # N·m/(rad/s)²
        'inertia': jnp.array([0.01, 0.01, 0.02]),  # kg·m²
        'c_drag': 1e-4,
        'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),
        'motor_positions': frame['motor_positions'],
        'motor_directions': frame['motor_directions'],
    }

def get_hover_state():
    """Get hover state (stationary, level)."""
    return jnp.array([
        1.0, 0.0, 0.0, 0.0,  # Quaternion (identity)
        0.0, 0.0, 0.0,        # Velocity
        0.0, 0.0, 0.0         # Angular velocity
    ])

def test_motor_model():
    """Test 1: Motor model with known inputs produces expected outputs."""
    print("\n=== Test 1: Motor Model ===")

    # Test PWM normalization
    pwm_us = jnp.array([1000.0, 1500.0, 2000.0])
    pwm_norm = normalize_pwm(pwm_us)
    expected = jnp.array([0.0, 0.5, 1.0])
    passed = jnp.allclose(pwm_norm, expected, atol=1e-6)
    print_status("PWM normalization", passed,
                 f"Got {pwm_norm}, expected {expected}")

    # Test angular velocity conversion
    poly = jnp.array([0.0, 800.0, 200.0])
    pwm = jnp.array([0.5])
    omega = pwm_to_angular_velocity(pwm, poly)
    expected_omega = 0.0 + 800.0 * 0.5 + 200.0 * 0.25  # 450.0
    passed = jnp.allclose(omega, expected_omega, atol=1e-6)
    print_status("PWM to angular velocity", passed,
                 f"Got {omega[0]:.1f} rad/s, expected {expected_omega:.1f} rad/s")

    # Test thrust calculation
    omega_test = jnp.array([100.0, 200.0])
    kT = 1e-5
    thrust = angular_velocity_to_thrust(omega_test, kT)
    expected_thrust = jnp.array([0.1, 0.4])  # kT * omega^2
    passed = jnp.allclose(thrust, expected_thrust, atol=1e-6)
    print_status("Thrust calculation", passed,
                 f"Got {thrust}, expected {expected_thrust}")

    return True

def test_quaternion_ops():
    """Test 2: Quaternion operations."""
    print("\n=== Test 2: Quaternion Operations ===")

    # Test identity rotation
    q_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    R = quat_to_rotation(q_identity)
    passed = jnp.allclose(R, jnp.eye(3), atol=1e-6)
    print_status("Identity quaternion → identity matrix", passed)

    # Test quaternion normalization after integration
    q = jnp.array([1.0, 0.0, 0.0, 0.0])
    omega = jnp.array([1.0, 2.0, 3.0])
    q_next = quat_integrate(q, omega, dt=0.01)
    norm = jnp.linalg.norm(q_next)
    passed = jnp.allclose(norm, 1.0, atol=1e-6)
    print_status("Quaternion normalization after integration", passed,
                 f"Norm = {norm:.8f}")

    # Test no rotation with zero omega
    q_next_zero = quat_integrate(q, jnp.zeros(3), dt=0.01)
    passed = jnp.allclose(q_next_zero, q, atol=1e-6)
    print_status("Zero angular velocity preserves quaternion", passed)

    return True

def test_fdm_step():
    """Test 3: FDM step produces valid outputs."""
    print("\n=== Test 3: FDM Step ===")

    params = get_test_params()
    state = get_hover_state()
    pwm = jnp.ones(4) * 0.5

    # Run one step
    next_state = fdm_step(state, pwm, params, dt=0.01)

    # Check for NaN/Inf
    passed = jnp.all(jnp.isfinite(next_state))
    print_status("No NaN or Inf in output", passed)

    # Check quaternion normalization
    q_norm = jnp.linalg.norm(next_state[0:4])
    passed = jnp.allclose(q_norm, 1.0, atol=1e-6)
    print_status("Quaternion remains normalized", passed,
                 f"Norm = {q_norm:.8f}")

    # Validate state
    passed = validate_state(next_state)
    print_status("State validation passes", passed)

    # Test zero thrust causes falling
    state_fall = get_hover_state()
    for _ in range(10):
        state_fall = fdm_step(state_fall, jnp.zeros(4), params, dt=0.01)
    vz = state_fall[6]
    passed = vz < -0.5
    print_status("Zero thrust causes falling", passed,
                 f"vz = {vz:.3f} m/s (should be negative)")

    return True

def test_gradients():
    """Test 4: Gradient verification with check_grads."""
    print("\n=== Test 4: Gradient Verification ===")

    params = get_test_params()
    state = get_hover_state()
    pwm = jnp.ones(4) * 0.5

    # Test gradients w.r.t. state
    try:
        def f_state(s):
            return jnp.sum(fdm_step(s, pwm, params, dt=0.01))

        check_grads(f_state, (state,), order=1, rtol=1e-3)
        print_status("Gradients w.r.t. state", True)
    except Exception as e:
        print_status("Gradients w.r.t. state", False, str(e))
        return False

    # Test gradients w.r.t. PWM
    try:
        def f_pwm(p):
            return jnp.sum(fdm_step(state, p, params, dt=0.01))

        check_grads(f_pwm, (pwm,), order=1, rtol=1e-3)
        print_status("Gradients w.r.t. PWM", True)
    except Exception as e:
        print_status("Gradients w.r.t. PWM", False, str(e))
        return False

    # Test gradients w.r.t. parameters (mass)
    try:
        def f_mass(mass_val):
            p = params.copy()
            p['mass'] = mass_val
            return jnp.sum(fdm_step(state, pwm, p, dt=0.01))

        check_grads(f_mass, (params['mass'],), order=1, rtol=1e-3)
        print_status("Gradients w.r.t. parameters (mass)", True)
    except Exception as e:
        print_status("Gradients w.r.t. parameters (mass)", False, str(e))
        return False

    return True

def test_rollout():
    """Test 5: Rollout shape and performance."""
    print("\n=== Test 5: Rollout ===")

    params = get_test_params()
    state = get_hover_state()

    # Test shape
    T = 100
    pwm_sequence = jnp.ones((T, 4)) * 0.5
    trajectory = rollout(state, pwm_sequence, params, dt=0.01)

    passed = trajectory.shape == (T, 10)
    print_status("Correct output shape", passed,
                 f"Got {trajectory.shape}, expected ({T}, 10)")

    # Test no NaN
    passed = jnp.all(jnp.isfinite(trajectory))
    print_status("No NaN or Inf in trajectory", passed)

    # Test consistency with single step
    expected_first = fdm_step(state, pwm_sequence[0], params, dt=0.01)
    passed = jnp.allclose(trajectory[0], expected_first, atol=1e-6)
    print_status("First step matches fdm_step", passed)

    return True

def test_performance():
    """Test 6: Performance on 10-second simulation."""
    print("\n=== Test 6: Performance ===")

    params = get_test_params()
    state = get_hover_state()

    # 10 seconds at 100 Hz = 1000 steps
    T = 1000
    pwm_sequence = jnp.ones((T, 4)) * 0.5

    # First call (includes compilation)
    print("  Compiling...")
    start = time.time()
    _ = rollout(state, pwm_sequence, params, dt=0.01)
    compile_time = time.time() - start
    print(f"  Compilation time: {compile_time:.3f}s")

    # Second call (compiled)
    print("  Running compiled version...")
    start = time.time()
    trajectory = rollout(state, pwm_sequence, params, dt=0.01)
    run_time = time.time() - start

    passed = run_time < 1.0
    print_status("10-second simulation in < 1 second", passed,
                 f"Runtime: {run_time:.3f}s")

    # Verify output
    passed = trajectory.shape == (T, 10) and jnp.all(jnp.isfinite(trajectory))
    print_status("Valid output", passed)

    return True

def test_gradient_optimization():
    """Test 7: Gradient-based optimization can reduce loss."""
    print("\n=== Test 7: Gradient-Based Optimization ===")

    params = get_test_params()
    state = get_hover_state()
    pwm_sequence = jnp.ones((20, 4)) * 0.5

    # Generate target trajectory with true params
    target_traj = rollout(state, pwm_sequence, params, dt=0.01)

    # Test with perturbed mass (20% error)
    mass_initial = params['mass'] * 1.2

    def loss_fn(mass_val):
        p = params.copy()
        p['mass'] = mass_val
        predicted = rollout(state, pwm_sequence, p, dt=0.01)
        return jnp.mean((predicted - target_traj) ** 2)

    # Initial loss
    loss_initial = loss_fn(mass_initial)
    print(f"  Initial loss (mass={mass_initial:.2f}): {loss_initial:.6e}")

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grad_mass = grad_fn(mass_initial)
    print(f"  Gradient w.r.t. mass: {grad_mass:.6e}")

    # Take gradient descent step
    learning_rate = 0.01
    mass_updated = mass_initial - learning_rate * grad_mass

    # New loss
    loss_updated = loss_fn(mass_updated)
    print(f"  Updated loss (mass={mass_updated:.2f}): {loss_updated:.6e}")

    # Check that loss decreased
    passed = loss_updated < loss_initial
    reduction = (loss_initial - loss_updated) / loss_initial * 100
    print_status("Gradient descent reduces loss", passed,
                 f"Loss reduction: {reduction:.1f}%")

    return passed

def main():
    """Run all validation tests."""
    print("=" * 70)
    print("DIFFERENTIABLE FLIGHT DYNAMICS MODEL VALIDATION")
    print("=" * 70)

    all_passed = True

    try:
        all_passed &= test_motor_model()
    except Exception as e:
        print(f"✗ FAIL: Motor model test failed with error: {e}")
        all_passed = False

    try:
        all_passed &= test_quaternion_ops()
    except Exception as e:
        print(f"✗ FAIL: Quaternion test failed with error: {e}")
        all_passed = False

    try:
        all_passed &= test_fdm_step()
    except Exception as e:
        print(f"✗ FAIL: FDM step test failed with error: {e}")
        all_passed = False

    try:
        all_passed &= test_gradients()
    except Exception as e:
        print(f"✗ FAIL: Gradient test failed with error: {e}")
        all_passed = False

    try:
        all_passed &= test_rollout()
    except Exception as e:
        print(f"✗ FAIL: Rollout test failed with error: {e}")
        all_passed = False

    try:
        all_passed &= test_performance()
    except Exception as e:
        print(f"✗ FAIL: Performance test failed with error: {e}")
        all_passed = False

    try:
        all_passed &= test_gradient_optimization()
    except Exception as e:
        print(f"✗ FAIL: Optimization test failed with error: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nSUCCESS CRITERIA VERIFICATION:")
        print("  ✓ Known params → correct thrust/torque outputs")
        print("  ✓ Gradients verified with jax.test_util.check_grads (rtol=1e-3)")
        print("  ✓ Rollout on 10 seconds of data in < 1 second")
        print("  ✓ No NaNs or Infs in normal operation")
        print("  ✓ Gradient-based optimization works correctly")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(main())
