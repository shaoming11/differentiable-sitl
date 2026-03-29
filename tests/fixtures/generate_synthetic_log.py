"""
Generate synthetic ArduPilot .bin logs for testing.

Creates realistic flight data with known ground truth parameters
for reproducible testing of the parameter identification pipeline.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

from ardupilot_sysid.src.fdm.multicopter_jax import rollout
from ardupilot_sysid.src.fdm.frame_configs import QUAD_X


@dataclass
class SyntheticLogConfig:
    """Configuration for synthetic log generation."""

    # Duration and rates
    duration_s: float = 60.0
    imu_rate_hz: float = 400.0
    gps_rate_hz: float = 10.0
    rcout_rate_hz: float = 50.0

    # True FDM parameters (ground truth)
    mass: float = 1.2  # kg
    kT: float = 1.5e-5  # N/(rad/s)^2
    kQ: float = 2e-7  # Nm/(rad/s)^2
    Ixx: float = 0.01  # kg·m²
    Iyy: float = 0.01  # kg·m²
    Izz: float = 0.02  # kg·m²
    c_drag: float = 0.001
    tau_motor: float = 0.05  # s

    # Sensor noise levels
    gyr_noise_std: float = 0.01  # rad/s
    acc_noise_std: float = 0.05  # m/s²
    gps_vel_noise_std: float = 0.05  # m/s
    gps_latency_s: float = 0.15  # GPS latency

    # Flight profile
    maneuver_type: str = 'hover_with_variation'  # 'hover', 'hover_with_variation', 'dynamic'

    # Frame configuration
    frame_type: str = 'quad_x'


class ManeuverGenerator:
    """Generate PWM command sequences for different flight maneuvers."""

    @staticmethod
    def hover(n_steps: int, dt: float, hover_pwm: float = 0.55) -> np.ndarray:
        """Generate constant hover PWM."""
        return np.ones((n_steps, 4)) * hover_pwm

    @staticmethod
    def hover_with_variation(n_steps: int, dt: float, hover_pwm: float = 0.55) -> np.ndarray:
        """Generate hover with small variations for excitation."""
        t = np.arange(n_steps) * dt
        pwm = np.ones((n_steps, 4)) * hover_pwm

        # Add sinusoidal variations to each motor
        pwm[:, 0] += 0.05 * np.sin(2*np.pi*0.3*t)
        pwm[:, 1] += 0.05 * np.sin(2*np.pi*0.3*t + np.pi/2)
        pwm[:, 2] += 0.05 * np.cos(2*np.pi*0.4*t)
        pwm[:, 3] += 0.05 * np.cos(2*np.pi*0.4*t + np.pi/2)

        return np.clip(pwm, 0.0, 1.0)

    @staticmethod
    def dynamic_maneuvers(n_steps: int, dt: float, hover_pwm: float = 0.55) -> np.ndarray:
        """Generate dynamic flight maneuvers (roll, pitch, yaw)."""
        t = np.arange(n_steps) * dt
        pwm = np.ones((n_steps, 4)) * hover_pwm

        # Roll maneuver (differential thrust front-back)
        roll_phase = np.where(t < 15, 1, 0)  # First 15 seconds
        pwm[:, 0] += roll_phase * 0.08 * np.sin(2*np.pi*0.5*t)
        pwm[:, 1] += roll_phase * -0.08 * np.sin(2*np.pi*0.5*t)

        # Pitch maneuver (differential thrust left-right)
        pitch_phase = np.where((t >= 15) & (t < 30), 1, 0)
        pwm[:, 0] += pitch_phase * 0.08 * np.sin(2*np.pi*0.4*t)
        pwm[:, 2] += pitch_phase * -0.08 * np.sin(2*np.pi*0.4*t)

        # Yaw maneuver (differential thrust diagonal)
        yaw_phase = np.where((t >= 30) & (t < 45), 1, 0)
        pwm[:, 0] += yaw_phase * 0.06 * np.sin(2*np.pi*0.3*t)
        pwm[:, 1] += yaw_phase * -0.06 * np.sin(2*np.pi*0.3*t)
        pwm[:, 2] += yaw_phase * 0.06 * np.sin(2*np.pi*0.3*t)
        pwm[:, 3] += yaw_phase * -0.06 * np.sin(2*np.pi*0.3*t)

        # Combined maneuver (last 15 seconds)
        combined_phase = np.where(t >= 45, 1, 0)
        pwm[:, 0] += combined_phase * 0.05 * np.sin(2*np.pi*0.5*t)
        pwm[:, 1] += combined_phase * 0.05 * np.sin(2*np.pi*0.5*t + np.pi/4)
        pwm[:, 2] += combined_phase * 0.05 * np.sin(2*np.pi*0.5*t + np.pi/2)
        pwm[:, 3] += combined_phase * 0.05 * np.sin(2*np.pi*0.5*t + 3*np.pi/4)

        return np.clip(pwm, 0.0, 1.0)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: (4,) or (N, 4) quaternion [qw, qx, qy, qz]

    Returns:
        (3,) or (N, 3) Euler angles [roll, pitch, yaw] in radians
    """
    if q.ndim == 1:
        qw, qx, qy, qz = q

        # Roll (x-axis)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # Yaw (z-axis)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
    else:
        return np.array([quaternion_to_euler(q[i]) for i in range(len(q))])


def generate_synthetic_log(
    config: SyntheticLogConfig,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Generate synthetic flight data with known ground truth.

    This generates pandas DataFrames in the same format as the parser output,
    which can be used for testing the full pipeline.

    Args:
        config: Configuration for log generation
        output_dir: Optional directory to save data files

    Returns:
        Dict containing:
            - 'imu': IMU DataFrame
            - 'gps': GPS DataFrame
            - 'rcout': RCOUT DataFrame
            - 'ekf': EKF DataFrame
            - 'true_params': Ground truth parameters
            - 'true_trajectory': Ground truth state trajectory
            - 'pwm_sequence': PWM command sequence
    """
    print(f"🔧 Generating synthetic log...")
    print(f"   Duration: {config.duration_s}s")
    print(f"   Maneuver: {config.maneuver_type}")
    print(f"   True mass: {config.mass} kg")
    print(f"   True kT: {config.kT:.2e} N/(rad/s)²")

    # Create FDM parameters
    params = {
        'mass': jnp.array(config.mass),
        'kT': jnp.array(config.kT),
        'kQ': jnp.array(config.kQ),
        'inertia': jnp.array([config.Ixx, config.Iyy, config.Izz]),
        'c_drag': jnp.array(config.c_drag),
        'pwm_to_omega_poly': jnp.array([0.0, 1000.0, 0.0]),
        'motor_positions': QUAD_X['motor_positions'],
        'motor_directions': QUAD_X['motor_directions']
    }

    # Generate PWM sequence based on maneuver type
    dt_sim = 1.0 / config.imu_rate_hz
    n_steps = int(config.duration_s / dt_sim)

    maneuver_gen = ManeuverGenerator()
    if config.maneuver_type == 'hover':
        pwm_sequence = maneuver_gen.hover(n_steps, dt_sim)
    elif config.maneuver_type == 'hover_with_variation':
        pwm_sequence = maneuver_gen.hover_with_variation(n_steps, dt_sim)
    elif config.maneuver_type == 'dynamic':
        pwm_sequence = maneuver_gen.dynamic_maneuvers(n_steps, dt_sim)
    else:
        raise ValueError(f"Unknown maneuver type: {config.maneuver_type}")

    # Roll out FDM to get ground truth trajectory
    print("   Running FDM simulation...")
    state_init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    trajectory = rollout(state_init, jnp.array(pwm_sequence), params, dt_sim)

    # Add initial state
    trajectory_with_init = np.vstack([np.array(state_init), np.array(trajectory)])

    # Extract components from trajectory
    # State: [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    quaternions = trajectory_with_init[:, 0:4]
    velocities = trajectory_with_init[:, 4:7]
    angular_velocities = trajectory_with_init[:, 7:10]

    # Convert quaternions to Euler angles for visualization
    euler_angles = quaternion_to_euler(quaternions)

    # Time vectors
    t_sim = np.arange(len(trajectory_with_init)) * dt_sim

    # Generate IMU data (gyro + accel)
    print("   Generating IMU data...")

    # Gyroscopes (body angular rates + noise)
    gyr_x = angular_velocities[:, 0] + np.random.randn(len(t_sim)) * config.gyr_noise_std
    gyr_y = angular_velocities[:, 1] + np.random.randn(len(t_sim)) * config.gyr_noise_std
    gyr_z = angular_velocities[:, 2] + np.random.randn(len(t_sim)) * config.gyr_noise_std

    # Accelerometers (body frame)
    # For simplicity, approximate as gravity + body accelerations + noise
    # (Real FDM would compute proper specific force)
    acc_x = np.gradient(velocities[:, 0], dt_sim) + np.random.randn(len(t_sim)) * config.acc_noise_std
    acc_y = np.gradient(velocities[:, 1], dt_sim) + np.random.randn(len(t_sim)) * config.acc_noise_std
    acc_z = 9.81 + np.gradient(velocities[:, 2], dt_sim) + np.random.randn(len(t_sim)) * config.acc_noise_std

    imu_df = pd.DataFrame({
        'timestamp': t_sim,
        'gyr_x': gyr_x,
        'gyr_y': gyr_y,
        'gyr_z': gyr_z,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
    })

    # Generate GPS data (with latency and lower rate)
    print("   Generating GPS data...")
    t_gps = np.arange(0, config.duration_s, 1.0/config.gps_rate_hz)

    # Apply GPS latency
    t_gps_delayed = t_gps + config.gps_latency_s
    t_gps_delayed = t_gps_delayed[t_gps_delayed < config.duration_s]

    # Interpolate velocities at GPS timestamps (with delay)
    vel_n = np.interp(t_gps_delayed, t_sim, velocities[:, 0]) + \
            np.random.randn(len(t_gps_delayed)) * config.gps_vel_noise_std
    vel_e = np.interp(t_gps_delayed, t_sim, velocities[:, 1]) + \
            np.random.randn(len(t_gps_delayed)) * config.gps_vel_noise_std
    vel_d = np.interp(t_gps_delayed, t_sim, velocities[:, 2]) + \
            np.random.randn(len(t_gps_delayed)) * config.gps_vel_noise_std

    gps_df = pd.DataFrame({
        'timestamp': t_gps[:len(t_gps_delayed)],
        'vel_n': vel_n,
        'vel_e': vel_e,
        'vel_d': vel_d,
        'lat': 37.7749 + np.cumsum(vel_n) * 1e-6,
        'lng': -122.4194 + np.cumsum(vel_e) * 1e-6,
        'alt': 100 + np.cumsum(vel_d) * 0.1,
        'gps_speed': np.sqrt(vel_n**2 + vel_e**2),
    })

    # Generate RCOUT data
    print("   Generating RCOUT data...")
    t_rcout = np.arange(0, config.duration_s, 1.0/config.rcout_rate_hz)

    # Interpolate PWM values at RCOUT rate
    pwm_rcout = np.zeros((len(t_rcout), 4))
    for i in range(4):
        pwm_rcout[:, i] = np.interp(t_rcout, t_sim[:len(pwm_sequence)], pwm_sequence[:, i])

    # Convert normalized PWM to microseconds (1000-2000 µs)
    pwm_us = 1000 + pwm_rcout * 1000

    rcout_df = pd.DataFrame({
        'timestamp': t_rcout,
        'pwm_1': pwm_us[:, 0],
        'pwm_2': pwm_us[:, 1],
        'pwm_3': pwm_us[:, 2],
        'pwm_4': pwm_us[:, 3],
    })

    # Generate EKF health data (all healthy for synthetic data)
    print("   Generating EKF data...")
    ekf_df = pd.DataFrame({
        'timestamp': t_gps,
        'innovation_ratio': 0.3 + 0.1 * np.random.rand(len(t_gps)),
    })

    # Save to files if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        imu_df.to_csv(output_dir / 'synthetic_imu.csv', index=False)
        gps_df.to_csv(output_dir / 'synthetic_gps.csv', index=False)
        rcout_df.to_csv(output_dir / 'synthetic_rcout.csv', index=False)
        ekf_df.to_csv(output_dir / 'synthetic_ekf.csv', index=False)

        # Save ground truth
        ground_truth = {
            'mass': config.mass,
            'kT': config.kT,
            'kQ': config.kQ,
            'Ixx': config.Ixx,
            'Iyy': config.Iyy,
            'Izz': config.Izz,
            'c_drag': config.c_drag,
            'tau_motor': config.tau_motor,
        }

        import json
        with open(output_dir / 'ground_truth.json', 'w') as f:
            json.dump(ground_truth, f, indent=2)

        print(f"   ✓ Saved data to {output_dir}")

    print(f"   ✓ Generated {len(imu_df)} IMU samples")
    print(f"   ✓ Generated {len(gps_df)} GPS samples")
    print(f"   ✓ Generated {len(rcout_df)} RCOUT samples")
    print("   ✓ Synthetic log generation complete!")

    return {
        'imu': imu_df,
        'gps': gps_df,
        'rcout': rcout_df,
        'ekf': ekf_df,
        'true_params': params,
        'true_trajectory': trajectory_with_init,
        'pwm_sequence': pwm_sequence,
        'ground_truth': {
            'mass': config.mass,
            'kT': config.kT,
            'kQ': config.kQ,
            'inertia': [config.Ixx, config.Iyy, config.Izz],
            'c_drag': config.c_drag,
            'tau_motor': config.tau_motor,
        }
    }


def generate_test_fixtures():
    """Generate standard test fixtures."""
    fixtures_dir = Path(__file__).parent

    print("\n" + "="*60)
    print("GENERATING TEST FIXTURES")
    print("="*60)

    # Fixture 1: Simple hover (30s)
    print("\n1. Simple Hover (30s)")
    config_hover = SyntheticLogConfig(
        duration_s=30.0,
        maneuver_type='hover'
    )
    generate_synthetic_log(config_hover, fixtures_dir / 'hover_30s')

    # Fixture 2: Hover with variation (60s)
    print("\n2. Hover with Variation (60s)")
    config_varied = SyntheticLogConfig(
        duration_s=60.0,
        maneuver_type='hover_with_variation'
    )
    generate_synthetic_log(config_varied, fixtures_dir / 'hover_varied_60s')

    # Fixture 3: Dynamic maneuvers (60s)
    print("\n3. Dynamic Maneuvers (60s)")
    config_dynamic = SyntheticLogConfig(
        duration_s=60.0,
        maneuver_type='dynamic'
    )
    generate_synthetic_log(config_dynamic, fixtures_dir / 'dynamic_60s')

    print("\n" + "="*60)
    print("✓ Test fixtures generated successfully!")
    print("="*60)


if __name__ == '__main__':
    # Generate standard test fixtures
    generate_test_fixtures()
