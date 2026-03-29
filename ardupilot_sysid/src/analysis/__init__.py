"""Parameter excitation analysis and identifiability checks."""

from .excitation import (
    compute_fim,
    compute_excitation_scores,
    suggest_maneuvers,
    print_excitation_report,
    check_parameter_coupling,
    get_parameter_names_from_template,
    compute_condition_number,
    compute_weighted_fim_from_covariances,
)

from .identifiability import (
    check_structural_identifiability,
    assess_data_quality,
    print_identifiability_report,
    compute_parameter_uncertainties,
    compute_confidence_ellipsoid_volume,
    suggest_data_improvements,
    compare_pre_post_smoothing,
)

__all__ = [
    # Excitation analysis
    'compute_fim',
    'compute_excitation_scores',
    'suggest_maneuvers',
    'print_excitation_report',
    'check_parameter_coupling',
    'get_parameter_names_from_template',
    'compute_condition_number',
    'compute_weighted_fim_from_covariances',
    # Identifiability analysis
    'check_structural_identifiability',
    'assess_data_quality',
    'print_identifiability_report',
    'compute_parameter_uncertainties',
    'compute_confidence_ellipsoid_volume',
    'suggest_data_improvements',
    'compare_pre_post_smoothing',
]
