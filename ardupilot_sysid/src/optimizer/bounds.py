"""
Parameter bounds for constrained optimization.

Defines hard physical bounds that parameters cannot exceed. These are absolute
limits based on physics, not soft priors.
"""

import jax.numpy as jnp
from typing import Dict, Tuple, List


def get_parameter_bounds(param_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Get hard physical bounds for each parameter.

    These are absolute limits based on physics - parameters outside
    these bounds are physically impossible or indicate model failure.

    Args:
        param_names: List of parameter names to get bounds for

    Returns:
        Dict mapping parameter name → (lower_bound, upper_bound)

    Example:
        >>> bounds = get_parameter_bounds(['mass', 'kT', 'kQ'])
        >>> print(bounds['mass'])  # (0.1, 50.0)
    """
    # Complete bounds for all possible parameters
    all_bounds = {
        # Scalar parameters
        'mass': (0.1, 50.0),                    # kg (100g to 50kg)
        'kT': (1e-7, 1e-3),                     # N/(rad/s)²
        'kQ': (1e-9, 1e-5),                     # Nm/(rad/s)²
        'c_drag': (1e-6, 1e-1),                 # dimensionless

        # Inertia components (when stored as 3-element array)
        'inertia_0': (1e-5, 1.0),               # Ixx: kg·m²
        'inertia_1': (1e-5, 1.0),               # Iyy: kg·m²
        'inertia_2': (1e-5, 1.0),               # Izz: kg·m²

        # PWM polynomial coefficients
        'pwm_to_omega_poly_0': (0.0, 1000.0),   # Offset (rad/s)
        'pwm_to_omega_poly_1': (0.0, 10000.0),  # Linear coeff
        'pwm_to_omega_poly_2': (-1000.0, 1000.0), # Quadratic coeff
    }

    # Handle array parameters (inertia, pwm_poly)
    # These get expanded into individual bounds per element
    bounds = {}

    for name in param_names:
        if name in all_bounds:
            bounds[name] = all_bounds[name]
        elif name == 'inertia':
            # Expand to individual components
            bounds['inertia_0'] = all_bounds['inertia_0']
            bounds['inertia_1'] = all_bounds['inertia_1']
            bounds['inertia_2'] = all_bounds['inertia_2']
        elif name == 'pwm_to_omega_poly':
            # Expand polynomial coefficients
            bounds['pwm_to_omega_poly_0'] = all_bounds['pwm_to_omega_poly_0']
            bounds['pwm_to_omega_poly_1'] = all_bounds['pwm_to_omega_poly_1']
            bounds['pwm_to_omega_poly_2'] = all_bounds['pwm_to_omega_poly_2']

    return bounds


def get_bounds_for_flattened(
    template: Dict,
    param_order: List[str]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get lower and upper bounds arrays for flattened parameter vector.

    Args:
        template: Parameter template from flatten_params
        param_order: Ordered list of parameter names

    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays matching flat parameter order

    Example:
        >>> template = {'mass': {'size': 1}, 'kT': {'size': 1}, 'inertia': {'size': 3}}
        >>> param_order = ['mass', 'kT', 'inertia']
        >>> lower, upper = get_bounds_for_flattened(template, param_order)
        >>> # lower.shape = (5,), upper.shape = (5,)
    """
    bounds_dict = get_parameter_bounds(param_order)

    lower_list = []
    upper_list = []

    for name in param_order:
        size = template[name]['size']

        if size == 1:
            # Scalar parameter
            if name in bounds_dict:
                lower, upper = bounds_dict[name]
                lower_list.append(jnp.array([lower]))
                upper_list.append(jnp.array([upper]))
            else:
                # No bounds specified - use very wide defaults
                lower_list.append(jnp.array([-1e6]))
                upper_list.append(jnp.array([1e6]))
        else:
            # Array parameter - lookup per-element bounds
            for i in range(size):
                key = f"{name}_{i}"
                if key in bounds_dict:
                    lower, upper = bounds_dict[key]
                    lower_list.append(jnp.array([lower]))
                    upper_list.append(jnp.array([upper]))
                else:
                    # No bounds specified
                    lower_list.append(jnp.array([-1e6]))
                    upper_list.append(jnp.array([1e6]))

    return jnp.concatenate(lower_list), jnp.concatenate(upper_list)


def project_to_bounds(
    params_flat: jnp.ndarray,
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray
) -> jnp.ndarray:
    """
    Project parameters onto feasible region (enforce bounds).

    Uses element-wise clipping to the nearest feasible value.

    Args:
        params_flat: Parameter vector (n_params,)
        lower_bounds: Lower bounds per parameter (n_params,)
        upper_bounds: Upper bounds per parameter (n_params,)

    Returns:
        Projected parameter vector (n_params,)

    Example:
        >>> params = jnp.array([0.5, 1e-3, 2.0])  # mass too small, others OK
        >>> lower = jnp.array([1.0, 1e-5, 1.0])
        >>> upper = jnp.array([10.0, 1e-3, 5.0])
        >>> projected = project_to_bounds(params, lower, upper)
        >>> # projected = [1.0, 1e-3, 2.0]  (first element clamped to 1.0)
    """
    return jnp.clip(params_flat, lower_bounds, upper_bounds)


def check_bounds_violation(
    params_flat: jnp.ndarray,
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray,
    template: Dict,
    param_order: List[str],
    tolerance: float = 1e-6
) -> Dict[str, Tuple[float, float, float]]:
    """
    Check which parameters violate their bounds.

    Args:
        params_flat: Parameter vector
        lower_bounds: Lower bounds
        upper_bounds: Upper bounds
        template: Parameter template (for reconstructing names)
        param_order: Parameter names in order
        tolerance: Small tolerance for numerical comparison

    Returns:
        Dict mapping parameter_name → (value, lower, upper) for violating params
        Empty dict if all parameters are within bounds

    Example:
        >>> violations = check_bounds_violation(params, lower, upper, template, param_order)
        >>> if violations:
        ...     print(f"Parameter 'mass' = {violations['mass'][0]:.3f} violates bounds")
    """
    violations = {}
    offset = 0

    for name in param_order:
        size = template[name]['size']

        for i in range(size):
            idx = offset + i
            value = float(params_flat[idx])
            lower = float(lower_bounds[idx])
            upper = float(upper_bounds[idx])

            if value < lower - tolerance or value > upper + tolerance:
                param_key = f"{name}_{i}" if size > 1 else name
                violations[param_key] = (value, lower, upper)

        offset += size

    return violations


def describe_bounds(bounds: Dict[str, Tuple[float, float]]) -> str:
    """
    Generate human-readable description of bounds.

    Args:
        bounds: Dict mapping parameter name → (lower, upper)

    Returns:
        Formatted string describing bounds

    Example:
        >>> bounds = get_parameter_bounds(['mass', 'kT', 'kQ'])
        >>> print(describe_bounds(bounds))
        Parameter Bounds:
          mass: [0.10, 50.00] kg
          kT: [1.00e-07, 1.00e-03] N/(rad/s)²
          ...
    """
    lines = ["Parameter Bounds:"]

    units = {
        'mass': 'kg',
        'kT': 'N/(rad/s)²',
        'kQ': 'N·m/(rad/s)²',
        'c_drag': '(dimensionless)',
        'inertia_0': 'kg·m² (Ixx)',
        'inertia_1': 'kg·m² (Iyy)',
        'inertia_2': 'kg·m² (Izz)',
        'pwm_to_omega_poly_0': 'rad/s (offset)',
        'pwm_to_omega_poly_1': 'rad/s (linear)',
        'pwm_to_omega_poly_2': 'rad/s (quadratic)',
    }

    for name, (lower, upper) in sorted(bounds.items()):
        unit_str = units.get(name, '')

        if lower < 1e-3 or upper > 1e3:
            lines.append(f"  {name}: [{lower:.2e}, {upper:.2e}] {unit_str}")
        else:
            lines.append(f"  {name}: [{lower:.2f}, {upper:.2f}] {unit_str}")

    return '\n'.join(lines)


def validate_bounds_consistency(
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray
) -> bool:
    """
    Check that lower bounds are strictly less than upper bounds.

    Args:
        lower_bounds: Lower bound array
        upper_bounds: Upper bound array

    Returns:
        True if all lower < upper, False otherwise

    Example:
        >>> lower = jnp.array([0.1, 1e-7])
        >>> upper = jnp.array([50.0, 1e-3])
        >>> validate_bounds_consistency(lower, upper)  # True
    """
    return jnp.all(lower_bounds < upper_bounds)
