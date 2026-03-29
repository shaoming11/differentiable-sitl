"""
Fisher Information Matrix (FIM) computation and excitation analysis.

The Fisher Information Matrix quantifies how much information the observed
flight data contains about each parameter. This module computes the FIM using
JAX automatic differentiation and provides excitation scores to guide users
in collecting better flight data.

Key concepts:
- FIM diagonal: Information content per parameter
- Excitation score: Normalized FIM diagonal (0 to 1 scale)
- Parameter coupling: Off-diagonal correlations in FIM
- Maneuver suggestions: Actionable flight patterns to excite parameters
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..fdm.multicopter_jax import rollout


def compute_fim(
    state_trajectory: jnp.ndarray,
    pwm_sequence: jnp.ndarray,
    params_flat: jnp.ndarray,
    template: Dict,
    params_fixed: Dict,
    weights: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Compute Fisher Information Matrix using JAX Jacobian.

    FIM quantifies how much information the observed data contains about
    each parameter. High FIM diagonal entries indicate well-identified parameters.

    The FIM is computed as:
        FIM = J^T W J
    where:
        J = Jacobian of predicted states w.r.t. parameters
        W = weighting matrix (diagonal, from RTS covariance)

    Args:
        state_trajectory: (T, 10) smoothed state trajectory from RTS
        pwm_sequence: (T-1, N) PWM commands
        params_flat: (n_params,) current parameter estimate
        template: Template for unflattening parameters
        params_fixed: Fixed parameters (geometry)
        weights: (10,) state dimension weights (from RTS covariance)
        dt: timestep

    Returns:
        (n_params, n_params) Fisher Information Matrix

    Algorithm:
        1. Define prediction function that maps params → predicted trajectory
        2. Compute Jacobian: J = ∂predicted/∂θ using jax.jacfwd
        3. Weight Jacobian by measurement uncertainties
        4. FIM = J^T W J

    Example:
        >>> fim = compute_fim(states, pwm, params, template, params_fixed, weights, dt)
        >>> print(f"FIM diagonal: {jnp.diag(fim)}")
        >>> # High diagonal values = well-identified parameters
    """
    from ..fdm.multicopter_jax import unflatten_params

    # Define prediction function for given parameters
    def predict_trajectory(params_flat_input):
        """Map parameter vector to predicted state trajectory."""
        params = unflatten_params(params_flat_input, template, params_fixed)
        predicted = rollout(state_trajectory[0], pwm_sequence, params, dt)
        return predicted

    # Compute Jacobian: ∂predicted/∂params
    # Shape: (T, 10, n_params) where T = trajectory length, 10 = state dim
    jacobian_fn = jax.jacfwd(predict_trajectory)
    J = jacobian_fn(params_flat)

    # Reshape to (T*10, n_params) - flatten over time and state dimensions
    T, state_dim = J.shape[0], J.shape[1]
    J_flat = J.reshape(T * state_dim, -1)

    # Weight matrix: repeat weights for each timestep
    # Each state dimension has its own weight based on measurement accuracy
    W = jnp.tile(weights, T)  # (T*10,)
    W_diag = jnp.diag(W)

    # Fisher Information Matrix: FIM = J^T W J
    # This is a Hessian-like matrix that captures parameter sensitivity
    fim = J_flat.T @ W_diag @ J_flat

    return fim


def compute_excitation_scores(
    fim: jnp.ndarray,
    param_names: List[str],
    threshold: float = 0.3
) -> Dict[str, Dict]:
    """
    Compute per-parameter excitation scores from FIM.

    Score = normalized FIM diagonal entry (0 to 1 scale).
    A score close to 1 means the parameter is well-excited (identifiable).
    A score close to 0 means the parameter is poorly excited (unidentifiable).

    Args:
        fim: (n_params, n_params) Fisher Information Matrix
        param_names: List of parameter names (must match FIM size)
        threshold: Minimum score for "well-excited" (default 0.3)

    Returns:
        Dict mapping param_name → {
            'score': float (0-1, higher is better),
            'excited': bool (score >= threshold),
            'rank': int (1 = most excited),
            'fim_diagonal': float (raw FIM diagonal value)
        }

    Example:
        >>> scores = compute_excitation_scores(fim, ['mass', 'kT', 'Ixx'])
        >>> for name, info in scores.items():
        ...     status = "✓" if info['excited'] else "⚠"
        ...     print(f"{status} {name}: {info['score']:.2f}")
        ✓ mass: 0.92
        ⚠ Ixx: 0.23
    """
    # Extract diagonal (information content per parameter)
    diag = jnp.diag(fim)

    # Normalize by maximum (scores from 0 to 1)
    max_diag = jnp.max(diag)
    if max_diag < 1e-10:
        # FIM is nearly zero - no parameters are identifiable
        # This usually means the data has no variation or wrong units
        scores = {name: {
            'score': 0.0,
            'excited': False,
            'rank': i + 1,
            'fim_diagonal': 0.0
        } for i, name in enumerate(param_names)}
        return scores

    normalized_scores = diag / max_diag

    # Rank parameters by excitation (1 = best)
    # argsort gives ascending order, so negate for descending
    ranks = jnp.argsort(-diag) + 1  # +1 for 1-based indexing

    # Build result dict
    result = {}
    for i, name in enumerate(param_names):
        score = float(normalized_scores[i])
        result[name] = {
            'score': score,
            'excited': score >= threshold,
            'rank': int(ranks[i]),
            'fim_diagonal': float(diag[i])
        }

    return result


def suggest_maneuvers(
    excitation_scores: Dict[str, Dict],
    frame_type: str = 'quad_x'
) -> List[str]:
    """
    Suggest flight maneuvers to excite poorly-identified parameters.

    Maps unexcited parameters to the specific maneuver that would excite them.
    Each suggestion includes duration and implementation details.

    Args:
        excitation_scores: Dict from compute_excitation_scores
        frame_type: Vehicle frame type (for context, currently unused)

    Returns:
        List of maneuver suggestions (human-readable strings)

    Note:
        Maneuvers are based on the observability structure of the FDM:
        - Inertia parameters require rotational maneuvers
        - Thrust coefficient requires vertical maneuvers
        - Drag requires high-speed flight
        - Motor time constant requires rapid throttle changes

    Example:
        >>> suggestions = suggest_maneuvers(scores)
        >>> for s in suggestions:
        ...     print(f"  - {s}")
        [Izz] Yaw spin: sustained 360°/s yaw rate
          Duration: 3s continuous rotation
          Details: Constant yaw rate to excite yaw inertia
          Current score: 0.23 (threshold: 0.30)
    """
    # Maneuver database: maps parameter names to excitation maneuvers
    # Each entry specifies the flight pattern that maximizes information
    # about that parameter
    PARAM_TO_MANEUVER = {
        'mass': {
            'maneuver': 'Vertical climbs and descents at various throttle levels',
            'duration': '3 × 10s segments',
            'details': 'Vary throttle 30% to 70% to excite mass and thrust coefficient coupling'
        },
        'kT': {
            'maneuver': 'Hover at 3 different altitudes with throttle steps',
            'duration': '5s per altitude',
            'details': 'Steady hover, then 20% throttle step up/down'
        },
        'kQ': {
            'maneuver': 'Yaw doublet: alternate yaw left/right at max rate',
            'duration': '5s',
            'details': 'Full yaw stick deflection, 2 Hz oscillation'
        },
        'Ixx': {
            'maneuver': 'Rapid roll doublet: ±45° roll at ~2 Hz',
            'duration': '5s',
            'details': 'Square wave roll input to excite roll inertia'
        },
        'Iyy': {
            'maneuver': 'Rapid pitch doublet: ±30° pitch at ~2 Hz',
            'duration': '5s',
            'details': 'Square wave pitch input to excite pitch inertia'
        },
        'Izz': {
            'maneuver': 'Yaw spin: sustained 360°/s yaw rate',
            'duration': '3s continuous rotation',
            'details': 'Constant yaw rate to excite yaw inertia'
        },
        'c_drag': {
            'maneuver': 'Fast forward flight (>10 m/s)',
            'duration': '10s',
            'details': 'High-speed translation to excite aerodynamic drag'
        },
        'tau_motor': {
            'maneuver': 'Rapid throttle steps from 30% to 70% and back',
            'duration': '10 steps, 0.5s each',
            'details': 'Square wave throttle input to excite motor time constant'
        },
        'pwm_to_omega_poly': {
            'maneuver': 'Throttle sweep from hover to full throttle',
            'duration': '15s linear ramp',
            'details': 'Characterize full PWM→RPM mapping'
        }
    }

    suggestions = []

    # Check each parameter for poor excitation
    for param, result in excitation_scores.items():
        if not result['excited']:
            # Parameter is poorly excited - suggest a maneuver

            # Handle compound parameter names (e.g., 'inertia[0]' → 'Ixx')
            # Try exact match first, then prefix match
            base_param = param.split('[')[0]  # Strip array indices

            # Map common variations
            param_aliases = {
                'inertia': ['Ixx', 'Iyy', 'Izz'],
                'pwm_to_omega_poly': ['pwm_to_omega_poly']
            }

            # Check for direct match or alias
            if param in PARAM_TO_MANEUVER:
                maneuver_info = PARAM_TO_MANEUVER[param]
            elif base_param in param_aliases:
                # For inertia components, suggest all rotational maneuvers
                if 'inertia' in base_param or param in ['Ixx', 'Iyy', 'Izz']:
                    # Determine which axis based on index or name
                    if '0' in param or param == 'Ixx':
                        maneuver_info = PARAM_TO_MANEUVER['Ixx']
                    elif '1' in param or param == 'Iyy':
                        maneuver_info = PARAM_TO_MANEUVER['Iyy']
                    elif '2' in param or param == 'Izz':
                        maneuver_info = PARAM_TO_MANEUVER['Izz']
                    else:
                        maneuver_info = None
                else:
                    maneuver_info = None
            else:
                maneuver_info = None

            if maneuver_info:
                suggestion = (
                    f"[{param}] {maneuver_info['maneuver']}\n"
                    f"  Duration: {maneuver_info['duration']}\n"
                    f"  Details: {maneuver_info['details']}\n"
                    f"  Current score: {result['score']:.2f} (threshold: 0.30)"
                )
                suggestions.append(suggestion)
            else:
                # Generic suggestion for unknown parameters
                suggestions.append(
                    f"[{param}] Increase variation in flight conditions "
                    f"(current score: {result['score']:.2f})"
                )

    if not suggestions:
        suggestions.append("✓ All parameters are well-excited! No additional maneuvers needed.")

    return suggestions


def print_excitation_report(
    excitation_scores: Dict[str, Dict],
    suggestions: List[str],
    verbose: bool = True
):
    """
    Print human-readable excitation report to stdout.

    Shows:
    - Excitation score for each parameter (bar chart + numeric)
    - Ranking by information content
    - Suggested maneuvers for poorly-excited parameters

    Args:
        excitation_scores: Dict from compute_excitation_scores
        suggestions: List from suggest_maneuvers
        verbose: Include detailed FIM diagonal values (default True)

    Example output:
        ==============================================================
        PARAMETER EXCITATION ANALYSIS
        ==============================================================

        Excitation Scores (0.0 = no info, 1.0 = maximum info):
        ------------------------------------------------------------
          ✓ kT                 [████████████████████] 1.000 (rank #1)
            └─ FIM diagonal: 2.34e+05
          ✓ mass               [████████████████░░░░] 0.821 (rank #2)
            └─ FIM diagonal: 1.92e+05
          ⚠ Izz                [█████░░░░░░░░░░░░░░░] 0.234 (rank #7)
            └─ FIM diagonal: 5.48e+03
    """
    print("\n" + "=" * 60)
    print("PARAMETER EXCITATION ANALYSIS")
    print("=" * 60)

    print("\nExcitation Scores (0.0 = no info, 1.0 = maximum info):")
    print("-" * 60)

    # Sort by rank (best first)
    sorted_params = sorted(
        excitation_scores.items(),
        key=lambda x: x[1]['rank']
    )

    for param, info in sorted_params:
        # Status indicator
        status = "✓" if info['excited'] else "⚠"

        # Progress bar (20 characters)
        bar_length = int(info['score'] * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        # Main line: status, name, bar, score, rank
        print(f"  {status} {param:20s} [{bar}] {info['score']:.3f} (rank #{info['rank']})")

        # Optional: show raw FIM diagonal value
        if verbose:
            print(f"    └─ FIM diagonal: {info['fim_diagonal']:.2e}")

    print("\nSuggested Maneuvers for Poorly-Excited Parameters:")
    print("-" * 60)

    if suggestions[0].startswith("✓"):
        # All parameters well-excited
        print(f"  {suggestions[0]}")
    else:
        # List each suggestion
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion}")

    print("\n" + "=" * 60)


def check_parameter_coupling(
    fim: jnp.ndarray,
    param_names: List[str],
    correlation_threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    Check for coupled parameters (high off-diagonal correlation in FIM).

    Coupled parameters are difficult to identify independently because
    they affect the observations similarly. Example: mass and thrust
    coefficient (kT) both affect total thrust, so they're coupled.

    High correlation means changes in one parameter can be compensated
    by changes in the other, making individual identification difficult.

    Args:
        fim: (n_params, n_params) Fisher Information Matrix
        param_names: List of parameter names
        correlation_threshold: Minimum |correlation| to report (default 0.7)

    Returns:
        List of (param1, param2, correlation) tuples for highly coupled pairs

    Note:
        The correlation matrix is computed as:
            Corr[i,j] = FIM[i,j] / sqrt(FIM[i,i] * FIM[j,j])

        This normalizes the FIM to have 1's on the diagonal, making
        off-diagonal entries interpretable as correlation coefficients.

    Example:
        >>> couplings = check_parameter_coupling(fim, param_names)
        >>> for p1, p2, corr in couplings:
        ...     print(f"{p1} ↔ {p2}: {corr:.2f}")
        mass ↔ kT: 0.87
        Ixx ↔ Iyy: 0.72
    """
    # Compute correlation matrix from FIM
    # Corr[i,j] = FIM[i,j] / sqrt(FIM[i,i] * FIM[j,j])
    diag = jnp.diag(fim)
    diag_sqrt = jnp.sqrt(diag)

    # Avoid division by zero for unexcited parameters
    # Replace near-zero values with 1.0 (will give correlation = 0)
    diag_sqrt = jnp.where(diag_sqrt < 1e-10, 1.0, diag_sqrt)

    # Compute correlation matrix
    correlation_matrix = fim / jnp.outer(diag_sqrt, diag_sqrt)

    # Find high correlations (off-diagonal only)
    n = len(param_names)
    couplings = []

    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only (symmetric matrix)
            corr = float(correlation_matrix[i, j])
            if abs(corr) >= correlation_threshold:
                couplings.append((param_names[i], param_names[j], corr))

    # Sort by absolute correlation (highest first)
    couplings.sort(key=lambda x: abs(x[2]), reverse=True)

    return couplings


def get_parameter_names_from_template(template: Dict) -> List[str]:
    """
    Extract ordered parameter names from flattening template.

    Args:
        template: Template dict from flatten_params

    Returns:
        List of parameter names in the order they appear in params_flat

    Example:
        >>> template = {
        ...     'mass': {'shape': (), 'size': 1},
        ...     'kT': {'shape': (), 'size': 1},
        ...     'inertia': {'shape': (3,), 'size': 3}
        ... }
        >>> names = get_parameter_names_from_template(template)
        >>> print(names)
        ['mass', 'kT', 'inertia[0]', 'inertia[1]', 'inertia[2]']
    """
    param_names = []

    # Must match order in flatten_params
    param_keys = ['mass', 'kT', 'kQ', 'inertia', 'c_drag', 'pwm_to_omega_poly']

    for key in param_keys:
        if key in template:
            size = template[key]['size']
            shape = template[key]['shape']

            if size == 1:
                # Scalar parameter
                param_names.append(key)
            else:
                # Array parameter - add indexed names
                for i in range(size):
                    param_names.append(f"{key}[{i}]")

    return param_names


def compute_condition_number(fim: jnp.ndarray) -> float:
    """
    Compute condition number of Fisher Information Matrix.

    The condition number is the ratio of largest to smallest singular value.
    High condition number (>1e6) indicates numerical instability or
    near-linear dependencies between parameters.

    Args:
        fim: (n_params, n_params) Fisher Information Matrix

    Returns:
        Condition number (scalar, >= 1)

    Example:
        >>> cond = compute_condition_number(fim)
        >>> if cond > 1e6:
        ...     print("Warning: FIM is ill-conditioned")
    """
    # Compute singular values (FIM is symmetric, so SVD = eigendecomposition)
    s = jnp.linalg.svd(fim, compute_uv=False)

    # Condition number = max / min singular value
    # Handle edge case where min singular value is zero
    if s[-1] < 1e-10:
        return jnp.inf

    return float(s[0] / s[-1])


def compute_weighted_fim_from_covariances(
    state_trajectory: jnp.ndarray,
    pwm_sequence: jnp.ndarray,
    params_flat: jnp.ndarray,
    template: Dict,
    params_fixed: Dict,
    covariances: List[np.ndarray],
    dt: float
) -> jnp.ndarray:
    """
    Compute FIM with time-varying weights from RTS covariances.

    Instead of constant weights, use the inverse covariance from each
    timestep of the RTS smoother. This properly accounts for varying
    measurement uncertainty over time.

    Args:
        state_trajectory: (T, 10) smoothed state trajectory
        pwm_sequence: (T-1, N) PWM commands
        params_flat: (n_params,) parameter estimate
        template: Template for unflattening
        params_fixed: Fixed parameters
        covariances: List of (10, 10) covariance matrices from RTS
        dt: timestep

    Returns:
        (n_params, n_params) Fisher Information Matrix

    Note:
        This is more accurate than compute_fim with constant weights,
        but requires the full RTS smoother output (not just mean states).
    """
    from ..fdm.multicopter_jax import unflatten_params

    # Define prediction function
    def predict_trajectory(params_flat_input):
        params = unflatten_params(params_flat_input, template, params_fixed)
        predicted = rollout(state_trajectory[0], pwm_sequence, params, dt)
        return predicted

    # Compute Jacobian
    jacobian_fn = jax.jacfwd(predict_trajectory)
    J = jacobian_fn(params_flat)  # (T, 10, n_params)

    # Build block-diagonal weight matrix from covariances
    # W[t] = inv(Covariance[t]) for each timestep
    T = len(covariances)
    n_params = params_flat.shape[0]

    # Initialize FIM
    fim = jnp.zeros((n_params, n_params))

    # Accumulate: FIM = sum_t J[t].T @ W[t] @ J[t]
    for t in range(T):
        # Inverse covariance (information matrix) at time t
        cov_t = jnp.array(covariances[t])

        # Add small regularization for numerical stability
        cov_t_reg = cov_t + jnp.eye(10) * 1e-6

        try:
            W_t = jnp.linalg.inv(cov_t_reg)  # (10, 10)
        except:
            # Singular covariance - skip this timestep
            continue

        J_t = J[t]  # (10, n_params)

        # FIM += J_t.T @ W_t @ J_t
        fim = fim + J_t.T @ W_t @ J_t

    return fim
