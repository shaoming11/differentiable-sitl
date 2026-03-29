"""
Structural identifiability checks for parameter identification.

This module analyzes whether parameters can be uniquely identified from
observed data, independent of the specific flight trajectory. Uses
singular value decomposition (SVD) of the Fisher Information Matrix (FIM)
to detect fundamental identification problems.

Key concepts:
- Structural identifiability: Can the parameter be identified in principle?
- Rank deficiency: Linear combinations of parameters are unidentifiable
- Condition number: Numerical stability of parameter identification
- Data quality assessment: Overall suitability for system identification
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional


def check_structural_identifiability(
    fim: jnp.ndarray,
    param_names: List[str],
    rank_tolerance: float = 1e-6
) -> Dict:
    """
    Check structural identifiability via FIM rank analysis.

    A parameter is structurally identifiable if the FIM has full rank.
    Rank deficiency indicates fundamental identification problems - certain
    parameters or parameter combinations cannot be determined from the data,
    no matter how good the optimizer is.

    Uses Singular Value Decomposition (SVD) to analyze the FIM:
        FIM = U Σ V^T
    where Σ contains singular values in descending order.

    Small singular values indicate directions in parameter space that
    are poorly constrained by the data.

    Args:
        fim: (n_params, n_params) Fisher Information Matrix
        param_names: List of parameter names
        rank_tolerance: Tolerance for singular value cutoff (default 1e-6)

    Returns:
        Dict with keys:
            - 'rank': int - Numerical rank of FIM
            - 'n_params': int - Total number of parameters
            - 'full_rank': bool - True if rank == n_params
            - 'singular_values': array - All singular values (descending)
            - 'condition_number': float - Ratio max/min singular value
            - 'well_conditioned': bool - True if condition number < 1e6
            - 'unidentifiable_directions': List[str] - Parameter combinations
              that are unidentifiable (if rank deficient)

    Note:
        Full rank is necessary but not sufficient for good identification.
        Even with full rank, high condition number indicates numerical issues.

    Example:
        >>> info = check_structural_identifiability(fim, param_names)
        >>> if not info['full_rank']:
        ...     print("Warning: Rank deficiency detected!")
        ...     for direction in info['unidentifiable_directions']:
        ...         print(f"  Unidentifiable: {direction}")
        >>> if not info['well_conditioned']:
        ...     print(f"Warning: High condition number {info['condition_number']:.2e}")
    """
    # Compute SVD: FIM = U Σ V^T
    # For symmetric positive semi-definite matrices, U = V
    U, s, Vt = jnp.linalg.svd(fim)

    # Determine numerical rank
    # Count singular values above tolerance
    rank = int(jnp.sum(s > rank_tolerance))
    n_params = len(param_names)

    # Condition number: ratio of largest to smallest singular value
    if s[-1] > rank_tolerance:
        cond = float(s[0] / s[-1])
    else:
        cond = jnp.inf

    # Identify unidentifiable directions (null space of FIM)
    # These are the right singular vectors corresponding to small singular values
    unidentifiable_directions = []

    if rank < n_params:
        # Rank deficient - some parameter combinations are unidentifiable
        for i in range(rank, n_params):
            # Right singular vector corresponding to small singular value
            direction = Vt[i]  # (n_params,)

            # Find dominant components (top 3 by absolute value)
            abs_direction = jnp.abs(direction)
            dominant_indices = jnp.argsort(-abs_direction)[:3]  # Top 3

            # Format as linear combination
            terms = []
            for idx in dominant_indices:
                idx = int(idx)
                coef = float(direction[idx])
                if abs(coef) > 0.1:  # Only show significant terms
                    sign = "+" if coef >= 0 else "-"
                    terms.append(f"{sign}{abs(coef):.2f}·{param_names[idx]}")

            if terms:
                direction_str = " ".join(terms)
                unidentifiable_directions.append(direction_str)

    return {
        'rank': rank,
        'n_params': n_params,
        'full_rank': rank == n_params,
        'singular_values': s,
        'condition_number': cond,
        'well_conditioned': cond < 1e6,
        'unidentifiable_directions': unidentifiable_directions
    }


def assess_data_quality(
    excitation_scores: Dict[str, Dict],
    identifiability_info: Dict
) -> str:
    """
    Provide overall assessment of data quality for parameter identification.

    Combines excitation analysis and structural identifiability to give
    a single quality rating. This helps users quickly understand if their
    flight data is suitable for system identification.

    Quality levels:
    - EXCELLENT: >90% parameters excited, full rank, well-conditioned
    - GOOD: >70% parameters excited, full rank, reasonably conditioned
    - FAIR: >50% parameters excited, or minor rank/conditioning issues
    - POOR: <50% parameters excited, or major structural problems

    Args:
        excitation_scores: Dict from compute_excitation_scores
        identifiability_info: Dict from check_structural_identifiability

    Returns:
        Quality rating: 'EXCELLENT', 'GOOD', 'FAIR', or 'POOR'

    Example:
        >>> quality = assess_data_quality(scores, id_info)
        >>> print(f"Data quality: {quality}")
        >>> if quality == 'POOR':
        ...     print("Collect more varied flight data before running optimization.")
    """
    # Count well-excited parameters
    n_excited = sum(1 for info in excitation_scores.values() if info['excited'])
    n_total = len(excitation_scores)
    excitation_ratio = n_excited / n_total if n_total > 0 else 0.0

    # Extract identifiability metrics
    full_rank = identifiability_info['full_rank']
    cond = identifiability_info['condition_number']

    # Decision tree for quality assessment
    if excitation_ratio >= 0.9 and full_rank and cond < 1e4:
        # Near-perfect data: almost all parameters excited, well-conditioned
        return 'EXCELLENT'

    elif excitation_ratio >= 0.7 and full_rank and cond < 1e6:
        # Good data: most parameters excited, acceptable conditioning
        return 'GOOD'

    elif excitation_ratio >= 0.5 and (full_rank or cond < 1e7):
        # Marginal data: some parameters excited, or minor issues
        return 'FAIR'

    else:
        # Poor data: major excitation or structural problems
        return 'POOR'


def print_identifiability_report(
    identifiability_info: Dict,
    param_names: List[str],
    verbose: bool = True
):
    """
    Print human-readable identifiability report.

    Shows:
    - Rank and full-rank status
    - Condition number and conditioning status
    - Singular value spectrum (if verbose)
    - Unidentifiable parameter combinations (if rank deficient)

    Args:
        identifiability_info: Dict from check_structural_identifiability
        param_names: List of parameter names
        verbose: Show detailed singular value spectrum (default True)

    Example output:
        ==============================================================
        STRUCTURAL IDENTIFIABILITY ANALYSIS
        ==============================================================

        Rank: 7 / 8 parameters
        Status: ⚠ RANK DEFICIENT (1 unidentifiable direction)

        Condition number: 2.34e+07
        Status: ⚠ ILL-CONDITIONED (numerical issues expected)

        Singular value spectrum:
          λ₁ = 2.34e+05 (100.0%)
          λ₂ = 1.92e+05 ( 82.1%)
          ...
          λ₈ = 1.00e-02 (  0.0%)

        Unidentifiable parameter combinations:
          1. +0.87·mass -0.89·kT
             (mass and thrust coefficient cannot be separated)
    """
    print("\n" + "=" * 60)
    print("STRUCTURAL IDENTIFIABILITY ANALYSIS")
    print("=" * 60)

    # Rank status
    rank = identifiability_info['rank']
    n_params = identifiability_info['n_params']
    full_rank = identifiability_info['full_rank']

    print(f"\nRank: {rank} / {n_params} parameters")
    if full_rank:
        print("Status: ✓ FULL RANK (all parameters are identifiable)")
    else:
        n_unidentifiable = n_params - rank
        print(f"Status: ⚠ RANK DEFICIENT ({n_unidentifiable} unidentifiable direction(s))")

    # Condition number
    cond = identifiability_info['condition_number']
    well_conditioned = identifiability_info['well_conditioned']

    print(f"\nCondition number: {cond:.2e}")
    if well_conditioned:
        print("Status: ✓ WELL-CONDITIONED (numerical stability expected)")
    elif cond < jnp.inf:
        print("Status: ⚠ ILL-CONDITIONED (numerical issues possible)")
    else:
        print("Status: ⚠ SINGULAR (FIM has zero eigenvalues)")

    # Singular value spectrum (if verbose)
    if verbose:
        s = identifiability_info['singular_values']
        print("\nSingular value spectrum:")
        print("-" * 60)

        # Show all singular values with percentage of maximum
        for i, sv in enumerate(s):
            percentage = 100.0 * sv / s[0] if s[0] > 1e-10 else 0.0
            subscript = chr(0x2080 + ((i + 1) % 10))  # Unicode subscript
            print(f"  λ{subscript if i < 9 else i+1} = {sv:.2e} ({percentage:5.1f}%)")

            # Mark the rank cutoff
            if i == rank - 1 and not full_rank:
                print("  " + "-" * 50 + " rank cutoff")

    # Unidentifiable directions (if rank deficient)
    if not full_rank:
        directions = identifiability_info['unidentifiable_directions']
        print("\nUnidentifiable parameter combinations:")
        print("-" * 60)

        if directions:
            for i, direction in enumerate(directions, 1):
                print(f"  {i}. {direction}")
                # Add interpretation for common couplings
                if 'mass' in direction and 'kT' in direction:
                    print("     (mass and thrust coefficient cannot be separated)")
                elif 'Ixx' in direction and 'Iyy' in direction:
                    print("     (roll and pitch inertia cannot be separated)")
        else:
            print("  (no clear unidentifiable directions found)")

    print("\n" + "=" * 60)


def compute_parameter_uncertainties(
    fim: jnp.ndarray,
    param_names: List[str]
) -> Dict[str, float]:
    """
    Compute parameter uncertainty estimates from FIM.

    Under Gaussian approximation (Laplace approximation), the parameter
    covariance matrix is the inverse of the FIM. The square root of the
    diagonal gives standard errors for each parameter.

    Args:
        fim: (n_params, n_params) Fisher Information Matrix
        param_names: List of parameter names

    Returns:
        Dict mapping param_name → standard_error

    Note:
        This assumes the FIM is invertible (full rank). If rank deficient,
        uses pseudo-inverse, which may give unreliable uncertainties.

    Example:
        >>> uncertainties = compute_parameter_uncertainties(fim, param_names)
        >>> for name, std_err in uncertainties.items():
        ...     print(f"{name}: ±{std_err:.3e}")
    """
    # Compute parameter covariance: Cov = FIM^{-1}
    try:
        # Try standard inverse (requires full rank)
        param_cov = jnp.linalg.inv(fim)
    except:
        # Fallback to pseudo-inverse (handles rank deficiency)
        param_cov = jnp.linalg.pinv(fim, rcond=1e-10)

    # Extract standard errors (square root of diagonal)
    std_errors = jnp.sqrt(jnp.diag(param_cov))

    # Build result dict
    uncertainties = {}
    for i, name in enumerate(param_names):
        uncertainties[name] = float(std_errors[i])

    return uncertainties


def compute_confidence_ellipsoid_volume(fim: jnp.ndarray) -> float:
    """
    Compute volume of 95% confidence ellipsoid in parameter space.

    The confidence ellipsoid is defined by:
        (θ - θ̂)^T FIM (θ - θ̂) ≤ χ²_{p,0.95}

    where p is the number of parameters. The volume is proportional
    to the product of singular values.

    Smaller volume = more confident parameter estimates.

    Args:
        fim: (n_params, n_params) Fisher Information Matrix

    Returns:
        Volume of confidence ellipsoid (arbitrary units)

    Note:
        Volume is proportional to 1/√det(FIM). We compute using singular
        values for numerical stability: volume ∝ 1/√(∏λᵢ)
    """
    # Compute singular values
    s = jnp.linalg.svd(fim, compute_uv=False)

    # Volume is proportional to 1 / sqrt(product of singular values)
    # = 1 / sqrt(det(FIM))
    # Use log to avoid overflow
    log_volume = -0.5 * jnp.sum(jnp.log(s + 1e-10))

    return float(jnp.exp(log_volume))


def suggest_data_improvements(
    excitation_scores: Dict[str, Dict],
    identifiability_info: Dict
) -> List[str]:
    """
    Suggest specific data collection improvements based on analysis.

    Provides actionable recommendations to improve parameter identification:
    - Which parameters need more excitation
    - Whether structural issues need addressing
    - If more data is needed overall

    Args:
        excitation_scores: Dict from compute_excitation_scores
        identifiability_info: Dict from check_structural_identifiability

    Returns:
        List of improvement suggestions (human-readable strings)

    Example:
        >>> improvements = suggest_data_improvements(scores, id_info)
        >>> for suggestion in improvements:
        ...     print(f"  • {suggestion}")
    """
    suggestions = []

    # Check for rank deficiency
    if not identifiability_info['full_rank']:
        suggestions.append(
            "⚠ CRITICAL: Rank deficiency detected. Some parameter combinations "
            "cannot be identified from this data. Consider adding priors or "
            "fixing some parameters based on physical measurements."
        )

    # Check for ill-conditioning
    cond = identifiability_info['condition_number']
    if cond > 1e8:
        suggestions.append(
            "⚠ WARNING: Very high condition number. Optimization may be "
            "numerically unstable. Consider rescaling parameters or adding "
            "regularization."
        )
    elif cond > 1e6:
        suggestions.append(
            "⚠ CAUTION: High condition number. Use a robust optimizer "
            "(e.g., Levenberg-Marquardt) and check convergence carefully."
        )

    # Check overall excitation
    n_excited = sum(1 for info in excitation_scores.values() if info['excited'])
    n_total = len(excitation_scores)
    excitation_ratio = n_excited / n_total if n_total > 0 else 0.0

    if excitation_ratio < 0.5:
        suggestions.append(
            f"⚠ LOW EXCITATION: Only {n_excited}/{n_total} parameters are "
            "well-excited. Collect more varied flight data with the suggested "
            "maneuvers before running optimization."
        )
    elif excitation_ratio < 0.7:
        suggestions.append(
            f"FAIR EXCITATION: {n_excited}/{n_total} parameters are excited. "
            "Results will be reasonable, but adding suggested maneuvers "
            "will improve accuracy."
        )

    # Check for specific problematic parameters
    poorly_excited = [name for name, info in excitation_scores.items()
                     if info['score'] < 0.1]

    if poorly_excited:
        suggestions.append(
            f"Parameters with almost no excitation: {', '.join(poorly_excited)}. "
            "Consider fixing these to known values or collecting targeted data."
        )

    # If everything is good
    if not suggestions:
        suggestions.append(
            "✓ Data quality is good! Proceed with optimization. All parameters "
            "are identifiable and well-excited."
        )

    return suggestions


def compare_pre_post_smoothing(
    fim_forward: jnp.ndarray,
    fim_smoothed: jnp.ndarray,
    param_names: List[str]
) -> Dict:
    """
    Compare FIM before and after RTS smoothing.

    Shows the improvement in parameter identifiability from using
    smoothed states instead of forward-only filtered states.

    Args:
        fim_forward: FIM computed from forward-only UKF states
        fim_smoothed: FIM computed from RTS smoothed states
        param_names: List of parameter names

    Returns:
        Dict with comparison metrics:
            - 'trace_ratio': ratio of FIM traces (smoothed / forward)
            - 'det_ratio': ratio of FIM determinants
            - 'per_param_improvement': Dict of improvement ratios per parameter

    Example:
        >>> comparison = compare_pre_post_smoothing(fim_fwd, fim_smooth, names)
        >>> print(f"Overall information gain: {comparison['trace_ratio']:.2f}x")
        >>> print("Per-parameter improvement:")
        >>> for name, ratio in comparison['per_param_improvement'].items():
        ...     print(f"  {name}: {ratio:.2f}x")
    """
    # Overall information content (trace of FIM)
    trace_forward = jnp.trace(fim_forward)
    trace_smoothed = jnp.trace(fim_smoothed)
    trace_ratio = float(trace_smoothed / trace_forward) if trace_forward > 1e-10 else 1.0

    # Determinant (overall uncertainty volume)
    det_forward = jnp.linalg.det(fim_forward + jnp.eye(fim_forward.shape[0]) * 1e-10)
    det_smoothed = jnp.linalg.det(fim_smoothed + jnp.eye(fim_smoothed.shape[0]) * 1e-10)
    det_ratio = float(det_smoothed / det_forward) if det_forward > 1e-10 else 1.0

    # Per-parameter improvement (diagonal ratios)
    diag_forward = jnp.diag(fim_forward)
    diag_smoothed = jnp.diag(fim_smoothed)

    per_param_improvement = {}
    for i, name in enumerate(param_names):
        if diag_forward[i] > 1e-10:
            ratio = float(diag_smoothed[i] / diag_forward[i])
        else:
            ratio = 1.0
        per_param_improvement[name] = ratio

    return {
        'trace_ratio': trace_ratio,
        'det_ratio': det_ratio,
        'per_param_improvement': per_param_improvement
    }
