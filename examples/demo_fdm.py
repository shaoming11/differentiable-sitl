#!/usr/bin/env python3
"""
Demonstration of the differentiable flight dynamics model (FDM) in JAX.

This script shows:
1. Setting up a quadcopter model
2. Running a simple trajectory simulation
3. Computing gradients for parameter optimization
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from ardupilot_sysid.src.fdm import (
    get_frame_config,
    fdm_step,
    rollout,
    estimate_hover_pwm,
)


def main():
    print("=" * 70)
    print("Differentiable Flight Dynamics Model - Demonstration")
    print("=" * 70)
    print()

    # ==================================================================
    # 1. Setup quadcopter parameters
    # ==================================================================
    print("1. Setting up quadcopter model...")

    frame = get_frame_config('quad_x')

    params = {
        'mass': 1.5,  # kg
        'kT': 1e-5,   # N/(rad/s)²
        'kQ': 1e-6,   # N·m/(rad/s)²
        'inertia': jnp.array([0.01, 0.01, 0.02]),  # kg·m²
        'c_drag': 1e-4,  # Rotational drag coefficient
        'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),  # rad/s
        'motor_positions': frame['motor_positions'],
        'motor_directions': frame['motor_directions'],
    }

    print(f"   Mass: {params['mass']} kg")
    print(f"   Inertia: {params['inertia']} kg·m²")
    print(f"   Thrust coefficient: {params['kT']:.2e} N/(rad/s)²")
    print(f"   Number of motors: {len(params['motor_positions'])}")
    print()

    # ==================================================================
    # 2. Estimate hover PWM
    # ==================================================================
    print("2. Computing hover PWM...")

    hover_pwm = estimate_hover_pwm(
        mass=params['mass'],
        kT=params['kT'],
        poly_coeffs=params['pwm_to_omega_poly'],
        num_motors=4
    )

    print(f"   Estimated hover PWM: {hover_pwm:.3f} (normalized)")
    print(f"   Hover PWM (μs): {1000 + hover_pwm * 1000:.0f} μs")
    print()

    # ==================================================================
    # 3. Run simulation: hover → pitch forward → return to hover
    # ==================================================================
    print("3. Running trajectory simulation (5 seconds)...")

    dt = 0.01  # 100 Hz simulation
    T = int(5.0 / dt)  # 5 seconds

    # Initial state: level, at rest
    state_init = jnp.array([
        1.0, 0.0, 0.0, 0.0,  # Quaternion (level)
        0.0, 0.0, 0.0,        # Velocity (stationary)
        0.0, 0.0, 0.0         # Angular velocity (not rotating)
    ])

    # PWM sequence: hover → pitch forward → return
    pwm_sequence = jnp.ones((T, 4)) * hover_pwm

    # Add pitch command from t=1s to t=3s
    t1_idx = int(1.0 / dt)
    t2_idx = int(3.0 / dt)

    # Increase rear motors, decrease front motors for pitch forward
    pwm_sequence = pwm_sequence.at[t1_idx:t2_idx, [0, 2]].add(-0.05)  # Front motors
    pwm_sequence = pwm_sequence.at[t1_idx:t2_idx, [1, 3]].add(0.05)   # Rear motors

    # Run simulation
    print("   Simulating...")
    trajectory = rollout(state_init, pwm_sequence, params, dt)

    # Add initial state to trajectory
    trajectory_full = jnp.vstack([state_init[None, :], trajectory])

    print(f"   Simulation complete: {T} steps at {1/dt:.0f} Hz")
    print()

    # ==================================================================
    # 4. Analyze results
    # ==================================================================
    print("4. Analyzing trajectory...")

    # Extract components
    t = jnp.arange(T + 1) * dt
    pos_z = -trajectory_full[:, 6] * t  # Integrate vertical velocity (approximate)
    vel = trajectory_full[:, 4:7]
    omega = trajectory_full[:, 7:10]

    print(f"   Final altitude: {pos_z[-1]:.2f} m")
    print(f"   Max forward velocity: {jnp.max(vel[:, 0]):.2f} m/s")
    print(f"   Max pitch rate: {jnp.max(jnp.abs(omega[:, 1])):.2f} rad/s")
    print()

    # ==================================================================
    # 5. Demonstrate gradient computation
    # ==================================================================
    print("5. Computing gradients (for optimization)...")

    def loss_fn_demo(mass_value):
        """Simple loss function for demonstration."""
        params_mod = params.copy()
        params_mod['mass'] = mass_value
        traj = rollout(state_init, pwm_sequence, params_mod, dt)
        # Loss = sum of squared vertical velocities (want to minimize drift)
        return jnp.sum(traj[:, 6] ** 2)

    # Compute gradient of loss w.r.t. mass
    grad_fn = jax.grad(loss_fn_demo)
    grad_mass = grad_fn(params['mass'])

    print(f"   Loss w.r.t. mass: {loss_fn_demo(params['mass']):.4f}")
    print(f"   Gradient d(Loss)/d(mass): {grad_mass:.4f}")
    print()

    # ==================================================================
    # 6. Visualization
    # ==================================================================
    print("6. Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Linear velocities
    axes[0, 0].plot(t, vel[:, 0], label='vx (forward)')
    axes[0, 0].plot(t, vel[:, 1], label='vy (right)')
    axes[0, 0].plot(t, vel[:, 2], label='vz (down)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title('Linear Velocities')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Angular velocities
    axes[0, 1].plot(t, omega[:, 0], label='ωx (roll rate)')
    axes[0, 1].plot(t, omega[:, 1], label='ωy (pitch rate)')
    axes[0, 1].plot(t, omega[:, 2], label='ωz (yaw rate)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular velocity (rad/s)')
    axes[0, 1].set_title('Angular Velocities')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: PWM commands
    axes[1, 0].plot(t[:-1], pwm_sequence[:, 0], label='Motor 1 (FR)')
    axes[1, 0].plot(t[:-1], pwm_sequence[:, 1], label='Motor 2 (RL)')
    axes[1, 0].plot(t[:-1], pwm_sequence[:, 2], label='Motor 3 (RR)')
    axes[1, 0].plot(t[:-1], pwm_sequence[:, 3], label='Motor 4 (FL)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('PWM (normalized)')
    axes[1, 0].set_title('Motor Commands')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Quaternion norm (should stay at 1.0)
    q_norms = jnp.linalg.norm(trajectory_full[:, 0:4], axis=1)
    axes[1, 1].plot(t, q_norms)
    axes[1, 1].axhline(1.0, color='r', linestyle='--', label='Expected (1.0)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Quaternion norm')
    axes[1, 1].set_title('Quaternion Normalization Check')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim([0.999, 1.001])

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'fdm_demo_output.png'
    plt.savefig(output_path, dpi=150)
    print(f"   Plot saved to: {output_path}")

    # Show if in interactive mode
    # plt.show()

    print()
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
