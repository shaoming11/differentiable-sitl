"""
Demo: Parameter Excitation Analysis

This script demonstrates how to use the Fisher Information Matrix (FIM)
analysis to check if flight data can identify FDM parameters.

Usage:
    python examples/demo_excitation_analysis.py
"""

import jax.numpy as jnp
import numpy as np

from ardupilot_sysid.src.fdm import (
    get_frame_config,
    flatten_params,
    unflatten_params,
    rollout,
)

from ardupilot_sysid.src.analysis import (
    compute_fim,
    compute_excitation_scores,
    suggest_maneuvers,
    print_excitation_report,
    check_parameter_coupling,
    get_parameter_names_from_template,
    check_structural_identifiability,
    print_identifiability_report,
    assess_data_quality,
    suggest_data_improvements,
)


def generate_synthetic_trajectory(maneuver_type='hover', T=100, dt=0.01):
    """
    Generate synthetic flight trajectory for testing.

    Args:
        maneuver_type: 'hover', 'roll', 'pitch', 'yaw', or 'mixed'
        T: Number of timesteps
        dt: Timestep in seconds

    Returns:
        Tuple of (state_trajectory, pwm_sequence)
    """
    trajectories = []
    pwm_commands = []

    for t in range(T):
        time = t * dt

        if maneuver_type == 'hover':
            # Hover-only: no yaw, minimal roll/pitch
            roll = 0.01 * np.sin(2 * np.pi * time / 5.0)
            pitch = 0.01 * np.cos(2 * np.pi * time / 5.0)
            yaw = 0.0
            throttle = 0.5

        elif maneuver_type == 'roll':
            # Roll doublet
            roll = 0.3 * np.sin(2 * np.pi * time / 2.0)
            pitch = 0.0
            yaw = 0.0
            throttle = 0.5

        elif maneuver_type == 'pitch':
            # Pitch doublet
            roll = 0.0
            pitch = 0.3 * np.sin(2 * np.pi * time / 2.0)
            yaw = 0.0
            throttle = 0.5

        elif maneuver_type == 'yaw':
            # Yaw spin
            roll = 0.0
            pitch = 0.0
            yaw = 0.5 * time  # Accumulating yaw angle
            throttle = 0.5

        elif maneuver_type == 'mixed':
            # Mixed maneuver: roll, pitch, yaw, throttle variation
            roll = 0.2 * np.sin(2 * np.pi * time / 3.0)
            pitch = 0.15 * np.cos(2 * np.pi * time / 4.0)
            yaw = 0.3 * time
            throttle = 0.5 + 0.1 * np.sin(2 * np.pi * time / 6.0)

        else:
            raise ValueError(f"Unknown maneuver type: {maneuver_type}")

        # Construct quaternion from Euler angles (small angle approximation)
        q = jnp.array([
            np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2),
            np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2),
            np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2),
            np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        ])
        q = q / jnp.linalg.norm(q)

        # State: [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        state = jnp.concatenate([
            q,
            jnp.array([0.0, 0.0, 0.0]),  # Velocity (simplified)
            jnp.array([roll, pitch, yaw])  # Angular velocity (simplified)
        ])
        trajectories.append(state)

        # PWM command (4 motors)
        pwm = jnp.ones(4) * throttle
        pwm_commands.append(pwm)

    return jnp.array(trajectories), jnp.array(pwm_commands[:-1])  # T-1 PWM commands


def demo_hover_only():
    """Demonstrate excitation analysis on hover-only flight."""
    print("\n" + "=" * 70)
    print("DEMO 1: Hover-Only Flight")
    print("=" * 70)
    print("Hover-only flights lack yaw excitation → Izz poorly identified")
    print()

    # Generate hover-only trajectory
    state_trajectory, pwm_sequence = generate_synthetic_trajectory('hover', T=100)

    # Setup parameters
    frame = get_frame_config('quad_x')
    params = {
        'mass': 1.5,
        'kT': 1e-5,
        'kQ': 1e-6,
        'inertia': jnp.array([0.01, 0.01, 0.02]),
        'c_drag': 1e-4,
        'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),
        'motor_positions': frame['motor_positions'],
        'motor_directions': frame['motor_directions'],
    }

    params_flat, template = flatten_params(params)
    params_fixed = {
        'motor_positions': params['motor_positions'],
        'motor_directions': params['motor_directions']
    }

    # Compute FIM
    weights = jnp.ones(10)
    dt = 0.01
    fim = compute_fim(
        state_trajectory,
        pwm_sequence,
        params_flat,
        template,
        params_fixed,
        weights,
        dt
    )

    # Excitation analysis
    param_names = get_parameter_names_from_template(template)
    scores = compute_excitation_scores(fim, param_names, threshold=0.3)
    suggestions = suggest_maneuvers(scores)

    # Print report
    print_excitation_report(scores, suggestions, verbose=True)

    # Identifiability analysis
    id_info = check_structural_identifiability(fim, param_names)
    print_identifiability_report(id_info, param_names, verbose=False)

    # Data quality
    quality = assess_data_quality(scores, id_info)
    print(f"\nOverall Data Quality: {quality}")

    # Improvement suggestions
    improvements = suggest_data_improvements(scores, id_info)
    print("\nRecommendations:")
    for suggestion in improvements:
        print(f"  • {suggestion}")


def demo_mixed_maneuver():
    """Demonstrate excitation analysis on well-excited mixed maneuver."""
    print("\n" + "=" * 70)
    print("DEMO 2: Mixed Maneuver Flight")
    print("=" * 70)
    print("Mixed maneuvers excite all parameters → all parameters identifiable")
    print()

    # Generate mixed maneuver trajectory
    state_trajectory, pwm_sequence = generate_synthetic_trajectory('mixed', T=100)

    # Setup parameters
    frame = get_frame_config('quad_x')
    params = {
        'mass': 1.5,
        'kT': 1e-5,
        'kQ': 1e-6,
        'inertia': jnp.array([0.01, 0.01, 0.02]),
        'c_drag': 1e-4,
        'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),
        'motor_positions': frame['motor_positions'],
        'motor_directions': frame['motor_directions'],
    }

    params_flat, template = flatten_params(params)
    params_fixed = {
        'motor_positions': params['motor_positions'],
        'motor_directions': params['motor_directions']
    }

    # Compute FIM
    weights = jnp.ones(10)
    dt = 0.01
    fim = compute_fim(
        state_trajectory,
        pwm_sequence,
        params_flat,
        template,
        params_fixed,
        weights,
        dt
    )

    # Excitation analysis
    param_names = get_parameter_names_from_template(template)
    scores = compute_excitation_scores(fim, param_names, threshold=0.3)
    suggestions = suggest_maneuvers(scores)

    # Print report
    print_excitation_report(scores, suggestions, verbose=True)

    # Parameter coupling
    couplings = check_parameter_coupling(fim, param_names, correlation_threshold=0.7)
    if couplings:
        print("\nParameter Coupling Analysis:")
        print("-" * 60)
        for p1, p2, corr in couplings:
            print(f"  {p1} ↔ {p2}: correlation = {corr:.2f}")
    else:
        print("\n✓ No significant parameter coupling detected")

    # Identifiability analysis
    id_info = check_structural_identifiability(fim, param_names)
    print_identifiability_report(id_info, param_names, verbose=False)

    # Data quality
    quality = assess_data_quality(scores, id_info)
    print(f"\nOverall Data Quality: {quality}")


def demo_comparison():
    """Compare excitation across different maneuvers."""
    print("\n" + "=" * 70)
    print("DEMO 3: Maneuver Comparison")
    print("=" * 70)
    print("Compare excitation scores across different flight patterns")
    print()

    maneuvers = ['hover', 'roll', 'pitch', 'yaw', 'mixed']
    frame = get_frame_config('quad_x')
    params = {
        'mass': 1.5,
        'kT': 1e-5,
        'kQ': 1e-6,
        'inertia': jnp.array([0.01, 0.01, 0.02]),
        'c_drag': 1e-4,
        'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),
        'motor_positions': frame['motor_positions'],
        'motor_directions': frame['motor_directions'],
    }

    params_flat, template = flatten_params(params)
    params_fixed = {
        'motor_positions': params['motor_positions'],
        'motor_directions': params['motor_directions']
    }
    param_names = get_parameter_names_from_template(template)

    print(f"{'Maneuver':<15} {'Quality':<12} {'Excited Params':<20} {'Condition #':<15}")
    print("-" * 70)

    for maneuver in maneuvers:
        # Generate trajectory
        state_trajectory, pwm_sequence = generate_synthetic_trajectory(maneuver, T=50)

        # Compute FIM
        weights = jnp.ones(10)
        dt = 0.01
        fim = compute_fim(
            state_trajectory,
            pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        # Analysis
        scores = compute_excitation_scores(fim, param_names, threshold=0.3)
        id_info = check_structural_identifiability(fim, param_names)
        quality = assess_data_quality(scores, id_info)

        n_excited = sum(1 for info in scores.values() if info['excited'])
        n_total = len(scores)
        cond = id_info['condition_number']

        print(f"{maneuver:<15} {quality:<12} {n_excited}/{n_total:<17} {cond:.2e}")

    print("\nConclusion: Mixed maneuvers provide best parameter excitation.")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FISHER INFORMATION MATRIX (FIM) EXCITATION ANALYSIS DEMO")
    print("=" * 70)
    print()
    print("This demo shows how to use FIM analysis to:")
    print("  1. Check if flight data can identify each parameter")
    print("  2. Get excitation scores (0-1 scale)")
    print("  3. Receive actionable maneuver suggestions")
    print("  4. Detect parameter coupling")
    print("  5. Assess structural identifiability")

    demo_hover_only()
    demo_mixed_maneuver()
    demo_comparison()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Hover-only flights cannot identify yaw inertia (Izz)")
    print("  • Mixed maneuvers excite all parameters → better identification")
    print("  • FIM analysis guides data collection before optimization")
    print("  • High condition number indicates numerical challenges")
    print()
