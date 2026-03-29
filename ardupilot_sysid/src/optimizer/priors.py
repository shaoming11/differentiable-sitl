"""
Physical priors for MAP estimation.

Encodes physical measurements (mass, dimensions, component specs) as Gaussian
priors to regularize parameter identification and break degeneracies.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


@dataclass
class PhysicalPriors:
    """
    Physical measurements used as Gaussian priors.

    Each field is (mean, std) tuple representing:
    - mean: measured value
    - std: measurement uncertainty (1 sigma)
    """
    mass_kg: Tuple[float, float]              # (1.2, 0.05) - scale measurement
    arm_length_m: Tuple[float, float]         # (0.165, 0.005) - ruler measurement
    prop_diameter_in: Tuple[float, float]     # (5.1, 0.1) - spec sheet
    motor_kv: Tuple[float, float]             # (2300, 100) - manufacturer rating

    # Optional: direct inertia measurement (if available from CAD/swing test)
    Ixx_kgm2: Optional[Tuple[float, float]] = None
    Iyy_kgm2: Optional[Tuple[float, float]] = None
    Izz_kgm2: Optional[Tuple[float, float]] = None


def generate_parameter_priors(
    physical_priors: PhysicalPriors,
    frame_type: str = 'quad_x'
) -> Dict[str, Tuple]:
    """
    Generate Gaussian priors for all FDM parameters from physical measurements.

    Uses physics-based models to estimate parameters from measurable quantities.

    Args:
        physical_priors: Physical measurements
        frame_type: Vehicle frame type

    Returns:
        Dict mapping parameter name → (mean, std) or (mean_array, std_array)

    Example:
        >>> priors_spec = PhysicalPriors(
        ...     mass_kg=(1.2, 0.05),
        ...     arm_length_m=(0.165, 0.005),
        ...     prop_diameter_in=(5.1, 0.1),
        ...     motor_kv=(2300, 100)
        ... )
        >>> priors = generate_parameter_priors(priors_spec)
        >>> print(f"Mass prior: {priors['mass']}")
    """
    m_mean, m_std = physical_priors.mass_kg
    r_mean, r_std = physical_priors.arm_length_m
    d_prop_mean, d_prop_std = physical_priors.prop_diameter_in
    kv_mean, kv_std = physical_priors.motor_kv

    priors = {}

    # Mass: direct measurement
    priors['mass'] = (m_mean, m_std)

    # Inertia: point-mass approximation or direct measurement
    if physical_priors.Ixx_kgm2 is not None:
        # Use direct measurements if available
        priors['inertia'] = (
            jnp.array([
                physical_priors.Ixx_kgm2[0],
                physical_priors.Iyy_kgm2[0],
                physical_priors.Izz_kgm2[0]
            ]),
            jnp.array([
                physical_priors.Ixx_kgm2[1],
                physical_priors.Iyy_kgm2[1],
                physical_priors.Izz_kgm2[1]
            ])
        )
    else:
        # Estimate from point-mass model: I ≈ m_motor * r²
        # Assume 25% of total mass is motors (typical for racing quads)
        m_motor_each = m_mean * 0.25 / 4  # kg per motor

        if frame_type == 'quad_x':
            # X configuration: motors at ±45° in XY plane
            # Ixx = Iyy ≈ 2 * m * r² (two motors contribute to each axis)
            # Izz ≈ 4 * m * r² (all motors contribute to yaw)
            Ixx_mean = 2 * m_motor_each * r_mean**2
            Iyy_mean = Ixx_mean
            Izz_mean = 4 * m_motor_each * r_mean**2

            # Uncertainty propagation: σ_I ≈ I * sqrt((σ_m/m)² + (2σ_r/r)²)
            rel_unc_m = m_std / m_mean
            rel_unc_r = r_std / r_mean
            rel_unc_I = (rel_unc_m**2 + (2*rel_unc_r)**2)**0.5

            # 2x for model uncertainty (point mass is approximate)
            priors['inertia'] = (
                jnp.array([Ixx_mean, Iyy_mean, Izz_mean]),
                jnp.array([
                    Ixx_mean * rel_unc_I * 2,
                    Iyy_mean * rel_unc_I * 2,
                    Izz_mean * rel_unc_I * 2
                ])
            )

    # Thrust coefficient kT: estimate from propeller diameter
    # Empirical: kT ≈ 1e-5 * (D/5")² for 5" prop reference
    d_ref = 5.0  # inches
    kT_ref = 1.5e-5  # N/(rad/s)² for 5" prop
    kT_mean = kT_ref * (d_prop_mean / d_ref)**2
    kT_std = kT_mean * 0.5  # 50% uncertainty (rough estimate)
    priors['kT'] = (kT_mean, kT_std)

    # Torque coefficient kQ: typically kQ ≈ kT * (prop_diameter / 10)
    # For 5" prop, kQ ≈ 1e-5 * 0.127 / 10 ≈ 1.5e-7
    kQ_mean = kT_mean * (d_prop_mean * 0.0254) / 10  # Convert inches to meters
    kQ_std = kQ_mean * 0.5
    priors['kQ'] = (kQ_mean, kQ_std)

    # Rotational drag: very uncertain, wide prior
    # Typical range: 1e-4 to 1e-2
    priors['c_drag'] = (1e-3, 5e-3)

    # PWM→ω polynomial: [a0, a1, a2]
    # Linear model: ω = a1 * PWM, where a1 ≈ KV * V_max
    # For normalized PWM (0-1), a1 ≈ KV * 11.1V * (2π/60) for 3S LiPo
    omega_max = kv_mean * 11.1 * (2 * 3.14159 / 60)  # rad/s at full throttle
    priors['pwm_to_omega_poly'] = (
        jnp.array([0.0, omega_max, 0.0]),  # [a0, a1, a2] - linear model
        jnp.array([100.0, omega_max * 0.3, 100.0])  # Uncertainties
    )

    return priors


def prior_loss(
    params_flat: jnp.ndarray,
    param_names: List[str],
    priors: Dict[str, Tuple],
    template: Dict
) -> float:
    """
    Compute negative log-prior (Gaussian penalty).

    L_prior = Σ_i ((θ_i - μ_i) / σ_i)²

    Args:
        params_flat: Flattened parameter vector
        param_names: Parameter names in order (from template)
        priors: Dict of (mean, std) tuples
        template: Parameter template (for reconstructing shapes)

    Returns:
        Scalar prior loss

    Example:
        >>> params = jnp.array([1.2, 1.5e-5, 1.5e-7, 0.01, 0.01, 0.02, 0.001, 0.0, 2500.0, 0.0])
        >>> priors = {'mass': (1.2, 0.05), 'kT': (1.5e-5, 7.5e-6)}
        >>> template = {'mass': {'size': 1}, 'kT': {'size': 1}, ...}
        >>> loss = prior_loss(params, ['mass', 'kT', ...], priors, template)
    """
    loss = 0.0
    offset = 0

    for name in param_names:
        if name not in template:
            continue

        size = template[name]['size']
        param_value = params_flat[offset:offset+size]

        if name in priors:
            prior_mean, prior_std = priors[name]

            # Convert to arrays for uniform handling
            if isinstance(prior_mean, (int, float)):
                prior_mean = jnp.array([prior_mean])
                prior_std = jnp.array([prior_std])
                param_value = jnp.atleast_1d(param_value)
            else:
                prior_mean = jnp.atleast_1d(prior_mean)
                prior_std = jnp.atleast_1d(prior_std)

            # Gaussian penalty
            residual = (param_value - prior_mean) / prior_std
            loss += jnp.sum(residual**2)

        offset += size

    return loss


def describe_priors(priors: Dict[str, Tuple]) -> str:
    """
    Generate human-readable description of priors.

    Args:
        priors: Dict of (mean, std) tuples

    Returns:
        Formatted string describing all priors

    Example:
        >>> priors = generate_parameter_priors(PhysicalPriors(...))
        >>> print(describe_priors(priors))
        Parameter Priors:
          mass: 1.200 ± 0.050 kg
          kT: 1.50e-05 ± 7.50e-06 N/(rad/s)²
          ...
    """
    lines = ["Parameter Priors:"]

    for name, (mean, std) in priors.items():
        if isinstance(mean, jnp.ndarray):
            # Array parameters (inertia, polynomial coeffs)
            mean_np = mean.tolist()
            std_np = std.tolist()

            if name == 'inertia':
                lines.append(f"  Ixx: {mean_np[0]:.4e} ± {std_np[0]:.4e} kg·m²")
                lines.append(f"  Iyy: {mean_np[1]:.4e} ± {std_np[1]:.4e} kg·m²")
                lines.append(f"  Izz: {mean_np[2]:.4e} ± {std_np[2]:.4e} kg·m²")
            elif name == 'pwm_to_omega_poly':
                lines.append(f"  pwm_poly: [{mean_np[0]:.1f}, {mean_np[1]:.1f}, {mean_np[2]:.1f}] " +
                           f"± [{std_np[0]:.1f}, {std_np[1]:.1f}, {std_np[2]:.1f}] rad/s")
        else:
            # Scalar parameters
            units = {
                'mass': 'kg',
                'kT': 'N/(rad/s)²',
                'kQ': 'N·m/(rad/s)²',
                'c_drag': '(dimensionless)',
            }
            unit_str = units.get(name, '')

            if mean < 1e-3 or mean > 1e3:
                lines.append(f"  {name}: {mean:.2e} ± {std:.2e} {unit_str}")
            else:
                lines.append(f"  {name}: {mean:.3f} ± {std:.3f} {unit_str}")

    return '\n'.join(lines)
