"""
Motor model for converting PWM commands to thrust and torque.

Implements the PWM-to-angular-velocity mapping and thrust/torque calculations
for multicopter motors.
"""

import jax.numpy as jnp


def normalize_pwm(pwm_us: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize PWM from microseconds to 0-1 range.

    Standard RC PWM range is 1000-2000 microseconds, where:
    - 1000 μs = 0% throttle (motor off or idle)
    - 2000 μs = 100% throttle (full power)

    Args:
        pwm_us: PWM values in microseconds, shape (N,) where N is number of motors

    Returns:
        Normalized PWM values in [0, 1] range, shape (N,)

    Example:
        >>> pwm_us = jnp.array([1000.0, 1500.0, 2000.0])
        >>> normalize_pwm(pwm_us)
        Array([0., 0.5, 1.], dtype=float32)
    """
    return jnp.clip((pwm_us - 1000.0) / 1000.0, 0.0, 1.0)


def denormalize_pwm(pwm_normalized: jnp.ndarray) -> jnp.ndarray:
    """
    Convert normalized PWM [0-1] back to microseconds [1000-2000].

    Args:
        pwm_normalized: Normalized PWM values in [0, 1], shape (N,)

    Returns:
        PWM values in microseconds [1000, 2000], shape (N,)
    """
    return jnp.clip(pwm_normalized * 1000.0 + 1000.0, 1000.0, 2000.0)


def pwm_to_angular_velocity(
    pwm_normalized: jnp.ndarray,
    poly_coeffs: jnp.ndarray
) -> jnp.ndarray:
    """
    Convert normalized PWM to motor angular velocity via polynomial mapping.

    Uses a quadratic polynomial model:
        ω = a₀ + a₁·PWM + a₂·PWM²

    where ω is in rad/s and PWM is normalized [0, 1].

    Args:
        pwm_normalized: (N,) array of normalized PWM values [0-1]
        poly_coeffs: (3,) array [a₀, a₁, a₂] polynomial coefficients

    Returns:
        (N,) array of angular velocities in rad/s

    Example:
        >>> pwm = jnp.array([0.0, 0.5, 1.0])
        >>> coeffs = jnp.array([0.0, 100.0, 50.0])  # Example coefficients
        >>> omega = pwm_to_angular_velocity(pwm, coeffs)
        >>> # omega[0] = 0, omega[1] = 62.5, omega[2] = 150.0 rad/s
    """
    a0, a1, a2 = poly_coeffs
    pwm_sq = pwm_normalized ** 2
    return a0 + a1 * pwm_normalized + a2 * pwm_sq


def angular_velocity_to_thrust(
    omega: jnp.ndarray,
    kT: float
) -> jnp.ndarray:
    """
    Convert motor angular velocity to thrust force.

    Uses the standard propeller thrust model:
        F = kT · ω²

    where:
    - F is thrust in Newtons
    - kT is thrust coefficient in N/(rad/s)²
    - ω is angular velocity in rad/s

    Args:
        omega: (N,) array of angular velocities in rad/s
        kT: Thrust coefficient in N/(rad/s)²

    Returns:
        (N,) array of thrust forces in Newtons

    Example:
        >>> omega = jnp.array([0.0, 100.0, 200.0])
        >>> kT = 1e-5
        >>> thrust = angular_velocity_to_thrust(omega, kT)
        >>> # thrust = [0.0, 0.1, 0.4] N
    """
    return kT * omega ** 2


def angular_velocity_to_torque(
    omega: jnp.ndarray,
    kQ: float
) -> jnp.ndarray:
    """
    Convert motor angular velocity to reaction torque.

    Uses the standard propeller torque model:
        Q = kQ · ω²

    where:
    - Q is torque magnitude in N·m
    - kQ is torque coefficient in N·m/(rad/s)²
    - ω is angular velocity in rad/s

    The sign of the torque (CW vs CCW) is handled separately based on
    motor rotation direction.

    Args:
        omega: (N,) array of angular velocities in rad/s
        kQ: Torque coefficient in N·m/(rad/s)²

    Returns:
        (N,) array of torque magnitudes in N·m

    Example:
        >>> omega = jnp.array([0.0, 100.0, 200.0])
        >>> kQ = 1e-6
        >>> torque = angular_velocity_to_torque(omega, kQ)
        >>> # torque = [0.0, 0.01, 0.04] N·m
    """
    return kQ * omega ** 2


def pwm_to_thrust_torque(
    pwm_normalized: jnp.ndarray,
    poly_coeffs: jnp.ndarray,
    kT: float,
    kQ: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Complete PWM to thrust and torque conversion pipeline.

    Combines PWM normalization, angular velocity conversion, and
    thrust/torque calculations into a single differentiable function.

    Args:
        pwm_normalized: (N,) array of normalized PWM values [0-1]
        poly_coeffs: (3,) array [a₀, a₁, a₂] polynomial coefficients
        kT: Thrust coefficient in N/(rad/s)²
        kQ: Torque coefficient in N·m/(rad/s)²

    Returns:
        Tuple of:
        - thrusts: (N,) array of thrust forces in Newtons
        - torques: (N,) array of torque magnitudes in N·m

    Example:
        >>> pwm = jnp.array([0.5, 0.6, 0.7, 0.8])
        >>> coeffs = jnp.array([0.0, 100.0, 50.0])
        >>> kT, kQ = 1e-5, 1e-6
        >>> thrusts, torques = pwm_to_thrust_torque(pwm, coeffs, kT, kQ)
    """
    omega = pwm_to_angular_velocity(pwm_normalized, poly_coeffs)
    thrusts = angular_velocity_to_thrust(omega, kT)
    torques = angular_velocity_to_torque(omega, kQ)
    return thrusts, torques


def estimate_hover_pwm(
    mass: float,
    kT: float,
    poly_coeffs: jnp.ndarray,
    num_motors: int,
    gravity: float = 9.81
) -> float:
    """
    Estimate the PWM value needed for hover.

    At hover, total thrust equals weight:
        N_motors · kT · ω² = m · g

    Solving for ω and then inverting the PWM polynomial gives hover PWM.

    Args:
        mass: Vehicle mass in kg
        kT: Thrust coefficient in N/(rad/s)²
        poly_coeffs: (3,) array [a₀, a₁, a₂] polynomial coefficients
        num_motors: Number of motors
        gravity: Gravitational acceleration (default 9.81 m/s²)

    Returns:
        Estimated hover PWM (normalized, 0-1 range)

    Note:
        This assumes all motors produce equal thrust and uses the quadratic
        formula to invert the PWM polynomial. May not be accurate for all
        coefficient combinations.
    """
    # Required thrust per motor
    thrust_per_motor = (mass * gravity) / num_motors

    # Solve for omega: kT * omega^2 = thrust_per_motor
    omega_hover = jnp.sqrt(thrust_per_motor / kT)

    # Invert polynomial: omega = a0 + a1*pwm + a2*pwm^2
    # Rearrange: a2*pwm^2 + a1*pwm + (a0 - omega) = 0
    a0, a1, a2 = poly_coeffs

    # Use quadratic formula
    a_coeff = a2
    b_coeff = a1
    c_coeff = a0 - omega_hover

    discriminant = b_coeff**2 - 4*a_coeff*c_coeff

    # Use jnp.where to handle edge cases
    pwm_hover = jnp.where(
        discriminant >= 0,
        (-b_coeff + jnp.sqrt(discriminant)) / (2 * a_coeff),
        0.5  # Default to 50% if no real solution
    )

    return jnp.clip(pwm_hover, 0.0, 1.0)
