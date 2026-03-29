"""
Maximum A Posteriori (MAP) optimizer with Laplace approximation.

Implements Levenberg-Marquardt optimization with exact JAX gradients for
parameter identification. Computes posterior covariance via Laplace approximation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .priors import PhysicalPriors, generate_parameter_priors, prior_loss
from .bounds import get_bounds_for_flattened, project_to_bounds, check_bounds_violation
from ..fdm.multicopter_jax import loss_fn as data_loss_fn, flatten_params, unflatten_params


@dataclass
class OptimizationResult:
    """Result from MAP optimization."""
    params_optimal: jnp.ndarray          # Flattened optimal parameters
    params_dict: Dict                     # Unflattened parameter dict
    loss_history: np.ndarray              # Loss at each iteration
    posterior_covariance: jnp.ndarray     # Posterior covariance (Laplace approx)
    confidence_intervals: Dict[str, Tuple[float, float]]  # 95% CIs per parameter
    n_iterations: int                     # Number of iterations run
    converged: bool                       # Whether convergence criterion was met
    final_loss: float                     # Final loss value
    final_data_loss: float                # Data loss component
    final_prior_loss: float               # Prior loss component


class MAPOptimizer:
    """
    Maximum A Posteriori optimizer with Laplace approximation.

    Uses Levenberg-Marquardt algorithm with exact JAX gradients.
    Posterior covariance is computed via Laplace approximation:
        P(θ|data) ≈ N(θ_MAP, H^{-1})
    where H is the Hessian at the optimum.
    """

    def __init__(
        self,
        physical_priors: Optional[PhysicalPriors] = None,
        lambda_init: float = 0.01,
        lambda_factor: float = 10.0,
        prior_weight: float = 0.1,
        frame_type: str = 'quad_x'
    ):
        """
        Initialize MAP optimizer.

        Args:
            physical_priors: Physical measurements for priors (None = no priors)
            lambda_init: Initial LM damping parameter
            lambda_factor: Factor to increase/decrease lambda
            prior_weight: Weight of prior term (0 = MLE, 1 = strong prior)
            frame_type: Vehicle frame type for prior generation
        """
        self.physical_priors = physical_priors
        self.lambda_init = lambda_init
        self.lambda_factor = lambda_factor
        self.prior_weight = prior_weight
        self.frame_type = frame_type

        if physical_priors is not None:
            self.priors = generate_parameter_priors(physical_priors, frame_type)
        else:
            self.priors = {}

    def optimize(
        self,
        params_init: Dict,
        params_fixed: Dict,
        state_trajectory: jnp.ndarray,
        pwm_sequence: jnp.ndarray,
        weights: jnp.ndarray,
        dt: float,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run MAP optimization with Levenberg-Marquardt.

        Args:
            params_init: Initial parameter guess (dict)
            params_fixed: Fixed parameters (geometry)
            state_trajectory: Target states from RTS smoother (T, 10)
            pwm_sequence: PWM commands (T-1, N_motors)
            weights: State dimension weights (10,)
            dt: Timestep
            max_iter: Maximum iterations
            tol: Convergence tolerance on ||Δθ||
            verbose: Print progress

        Returns:
            OptimizationResult with optimal parameters and uncertainties

        Example:
            >>> optimizer = MAPOptimizer(physical_priors=priors)
            >>> result = optimizer.optimize(
            ...     params_init, params_fixed, state_traj, pwm_seq, weights, dt=0.0025
            ... )
            >>> print(f"Mass: {result.params_dict['mass']:.3f} kg")
            >>> print(f"95% CI: {result.confidence_intervals['mass']}")
        """
        # Flatten initial parameters
        params_flat, template = flatten_params(params_init)
        param_names = list(template.keys())
        n_params = len(params_flat)

        # Get bounds for flattened parameters
        lower_bounds, upper_bounds = get_bounds_for_flattened(template, param_names)

        # Define total loss function
        def total_loss(params):
            """Combined data loss + prior loss."""
            data_loss = data_loss_fn(
                params, template, params_fixed,
                state_trajectory, pwm_sequence, weights, dt
            )

            if self.priors:
                prior_term = prior_loss(params, param_names, self.priors, template)
                return data_loss + self.prior_weight * prior_term
            else:
                return data_loss

        def data_loss_only(params):
            """Data loss without prior (for reporting)."""
            return data_loss_fn(
                params, template, params_fixed,
                state_trajectory, pwm_sequence, weights, dt
            )

        def prior_loss_only(params):
            """Prior loss only (for reporting)."""
            if self.priors:
                return prior_loss(params, param_names, self.priors, template)
            else:
                return 0.0

        # Compile gradient and Hessian
        grad_fn = jax.jit(jax.grad(total_loss))
        hessian_fn = jax.jit(jax.hessian(total_loss))

        # Initialize
        params = params_flat
        lambda_lm = self.lambda_init
        loss_history = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"MAP OPTIMIZATION (Levenberg-Marquardt)")
            print(f"{'='*70}")
            print(f"Parameters: {n_params}")
            print(f"Parameter names: {param_names}")
            print(f"Prior weight: {self.prior_weight}")
            print(f"Max iterations: {max_iter}")
            print(f"Convergence tolerance: {tol}")
            print(f"Timesteps: {len(state_trajectory)}")
            print(f"Timestep dt: {dt:.6f} s")
            print()

        # Optimization loop
        converged = False
        for iteration in range(max_iter):
            # Compute loss, gradient, Hessian
            loss = float(total_loss(params))
            grad = grad_fn(params)
            grad_norm = float(jnp.linalg.norm(grad))

            loss_history.append(loss)

            # Check for NaN/Inf
            if not jnp.all(jnp.isfinite(grad)):
                if verbose:
                    print(f"\n✗ Gradient contains NaN/Inf at iteration {iteration}")
                break

            # Compute Hessian (expensive - only when needed)
            H = hessian_fn(params)

            # LM update: (H + λI) Δθ = -g
            H_damped = H + lambda_lm * jnp.eye(n_params)

            try:
                # Solve for update direction
                delta = jnp.linalg.solve(H_damped, -grad)
            except:
                # Singular matrix - increase damping
                lambda_lm *= self.lambda_factor
                if verbose and iteration % 10 == 0:
                    print(f"Iter {iteration:3d}: Singular matrix, increasing λ to {lambda_lm:.2e}")
                continue

            # Candidate update
            params_new = params + delta

            # Project onto bounds
            params_new = project_to_bounds(params_new, lower_bounds, upper_bounds)

            # Evaluate candidate
            loss_new = float(total_loss(params_new))

            # Accept/reject step
            if loss_new < loss:
                # Accept - decrease damping (move toward Gauss-Newton)
                params = params_new
                lambda_lm = max(lambda_lm / self.lambda_factor, 1e-10)
                accepted = "✓"
            else:
                # Reject - increase damping (move toward gradient descent)
                lambda_lm = min(lambda_lm * self.lambda_factor, 1e10)
                accepted = "✗"

            # Check convergence
            delta_norm = float(jnp.linalg.norm(delta))
            converged = delta_norm < tol

            if verbose and (iteration % 10 == 0 or converged):
                print(f"Iter {iteration:3d}: loss={loss:.6e}, ||∇||={grad_norm:.2e}, "
                      f"||Δθ||={delta_norm:.2e}, λ={lambda_lm:.2e} {accepted}")

            if converged:
                if verbose:
                    print(f"\n✓ Converged after {iteration+1} iterations")
                break

        if not converged and verbose:
            print(f"\n⚠ Maximum iterations ({max_iter}) reached without convergence")

        # Compute final losses
        final_data_loss = float(data_loss_only(params))
        final_prior_loss = float(prior_loss_only(params))

        # Check for bound violations
        violations = check_bounds_violation(
            params, lower_bounds, upper_bounds, template, param_names
        )
        if violations and verbose:
            print("\n⚠ Warning: Some parameters violate bounds:")
            for name, (value, lower, upper) in violations.items():
                print(f"  {name}: {value:.3e} outside [{lower:.3e}, {upper:.3e}]")

        # Laplace approximation: Posterior ≈ N(θ_MAP, H^{-1})
        if verbose:
            print("\nComputing posterior covariance (Laplace approximation)...")

        H_final = hessian_fn(params)

        try:
            posterior_cov = jnp.linalg.inv(H_final)

            # Check if covariance is positive definite
            eigenvalues = jnp.linalg.eigvalsh(posterior_cov)
            if jnp.any(eigenvalues <= 0):
                if verbose:
                    print("⚠ Warning: Posterior covariance not positive definite, using pseudo-inverse")
                posterior_cov = jnp.linalg.pinv(H_final)
        except:
            # Singular Hessian - use pseudo-inverse
            posterior_cov = jnp.linalg.pinv(H_final)
            if verbose:
                print("⚠ Warning: Singular Hessian, using pseudo-inverse")

        # Confidence intervals (95% = 1.96 sigma)
        z = 1.96
        stds = jnp.sqrt(jnp.abs(jnp.diag(posterior_cov)))  # abs for numerical safety
        confidence_intervals = {}

        offset = 0
        for name in param_names:
            size = template[name]['size']

            if size == 1:
                # Scalar parameter
                confidence_intervals[name] = (
                    float(params[offset] - z * stds[offset]),
                    float(params[offset] + z * stds[offset])
                )
            else:
                # Array parameter - CI per element
                for i in range(size):
                    idx = offset + i
                    key = f"{name}_{i}"
                    confidence_intervals[key] = (
                        float(params[idx] - z * stds[idx]),
                        float(params[idx] + z * stds[idx])
                    )

            offset += size

        # Unflatten final parameters
        params_dict = unflatten_params(params, template, params_fixed)

        if verbose:
            print("\nOptimization complete!")
            print(f"Final total loss: {loss_history[-1]:.6e}")
            print(f"  Data loss: {final_data_loss:.6e}")
            print(f"  Prior loss: {final_prior_loss:.6e}")
            print(f"Iterations: {len(loss_history)}")
            print(f"Converged: {converged}")

        return OptimizationResult(
            params_optimal=params,
            params_dict=params_dict,
            loss_history=np.array(loss_history),
            posterior_covariance=posterior_cov,
            confidence_intervals=confidence_intervals,
            n_iterations=len(loss_history),
            converged=converged,
            final_loss=loss_history[-1],
            final_data_loss=final_data_loss,
            final_prior_loss=final_prior_loss
        )


def summarize_result(result: OptimizationResult, param_names: List[str]) -> str:
    """
    Generate human-readable summary of optimization result.

    Args:
        result: OptimizationResult from optimizer
        param_names: Parameter names

    Returns:
        Formatted summary string

    Example:
        >>> summary = summarize_result(result, ['mass', 'kT', 'kQ', ...])
        >>> print(summary)
    """
    lines = [
        "="*70,
        "OPTIMIZATION RESULT SUMMARY",
        "="*70,
        f"Converged: {result.converged}",
        f"Iterations: {result.n_iterations}",
        f"Final loss: {result.final_loss:.6e}",
        f"  - Data loss: {result.final_data_loss:.6e}",
        f"  - Prior loss: {result.final_prior_loss:.6e}",
        "",
        "Optimal Parameters (with 95% confidence intervals):",
    ]

    # Extract parameters from dict
    for name in param_names:
        if name in result.params_dict:
            value = result.params_dict[name]

            if isinstance(value, jnp.ndarray) or isinstance(value, np.ndarray):
                # Array parameter
                value_arr = np.atleast_1d(value)
                for i, v in enumerate(value_arr):
                    key = f"{name}_{i}"
                    if key in result.confidence_intervals:
                        ci_low, ci_high = result.confidence_intervals[key]
                        lines.append(f"  {key}: {v:.4e} [{ci_low:.4e}, {ci_high:.4e}]")
            else:
                # Scalar parameter
                if name in result.confidence_intervals:
                    ci_low, ci_high = result.confidence_intervals[name]
                    lines.append(f"  {name}: {value:.4e} [{ci_low:.4e}, {ci_high:.4e}]")

    lines.append("="*70)

    return '\n'.join(lines)
