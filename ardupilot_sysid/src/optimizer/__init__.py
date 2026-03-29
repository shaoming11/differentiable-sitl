"""MAP optimization with Laplace approximation for parameter identification."""

from .priors import (
    PhysicalPriors,
    generate_parameter_priors,
    prior_loss,
    describe_priors
)

from .bounds import (
    get_parameter_bounds,
    get_bounds_for_flattened,
    project_to_bounds,
    check_bounds_violation,
    describe_bounds,
    validate_bounds_consistency
)

from .map_optimizer import (
    MAPOptimizer,
    OptimizationResult,
    summarize_result
)

__all__ = [
    # Priors
    'PhysicalPriors',
    'generate_parameter_priors',
    'prior_loss',
    'describe_priors',

    # Bounds
    'get_parameter_bounds',
    'get_bounds_for_flattened',
    'project_to_bounds',
    'check_bounds_violation',
    'describe_bounds',
    'validate_bounds_consistency',

    # Optimizer
    'MAPOptimizer',
    'OptimizationResult',
    'summarize_result',
]
