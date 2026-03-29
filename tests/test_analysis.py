"""
Tests for excitation analysis and identifiability checks.

Tests cover:
1. FIM computation with JAX autodiff
2. Excitation score calculation and normalization
3. Maneuver suggestions
4. Parameter coupling detection
5. Structural identifiability analysis
6. Data quality assessment
7. Edge cases (zero FIM, rank deficiency, ill-conditioning)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from ardupilot_sysid.src.analysis import (
    # Excitation
    compute_fim,
    compute_excitation_scores,
    suggest_maneuvers,
    check_parameter_coupling,
    get_parameter_names_from_template,
    compute_condition_number,
    # Identifiability
    check_structural_identifiability,
    assess_data_quality,
    compute_parameter_uncertainties,
    suggest_data_improvements,
)

from ardupilot_sysid.src.fdm import (
    get_frame_config,
    flatten_params,
    unflatten_params,
    rollout,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def quad_params():
    """Standard quadcopter parameters."""
    frame = get_frame_config('quad_x')
    return {
        'mass': 1.5,
        'kT': 1e-5,
        'kQ': 1e-6,
        'inertia': jnp.array([0.01, 0.01, 0.02]),
        'c_drag': 1e-4,
        'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),
        'motor_positions': frame['motor_positions'],
        'motor_directions': frame['motor_directions'],
    }


@pytest.fixture
def simple_trajectory():
    """Simple state trajectory for testing (10 timesteps)."""
    T = 10
    # Start at hover, slight roll and pitch variations
    trajectory = []
    for t in range(T):
        roll = 0.05 * jnp.sin(2 * jnp.pi * t / T)
        pitch = 0.03 * jnp.cos(2 * jnp.pi * t / T)

        # Simple quaternion (small angles)
        q = jnp.array([
            jnp.cos(roll/2) * jnp.cos(pitch/2),
            jnp.sin(roll/2) * jnp.cos(pitch/2),
            jnp.cos(roll/2) * jnp.sin(pitch/2),
            0.0
        ])
        q = q / jnp.linalg.norm(q)

        state = jnp.concatenate([
            q,                           # Quaternion
            jnp.array([0.0, 0.0, 0.0]),  # Velocity
            jnp.array([roll, pitch, 0.0]) # Angular velocity
        ])
        trajectory.append(state)

    return jnp.array(trajectory)


@pytest.fixture
def simple_pwm_sequence():
    """Simple PWM sequence for testing (9 timesteps)."""
    T = 9
    pwm = []
    for t in range(T):
        # Slight throttle variation
        throttle = 0.5 + 0.05 * jnp.sin(2 * jnp.pi * t / T)
        pwm.append(jnp.ones(4) * throttle)
    return jnp.array(pwm)


@pytest.fixture
def hover_only_trajectory():
    """Hover-only trajectory (no yaw, no translation)."""
    T = 20
    trajectory = []
    for t in range(T):
        # Only small roll/pitch, no yaw
        state = jnp.array([
            1.0, 0.0, 0.0, 0.0,     # Level quaternion
            0.0, 0.0, 0.0,           # No velocity
            0.01, 0.01, 0.0          # Tiny roll/pitch rates, NO yaw
        ])
        trajectory.append(state)
    return jnp.array(trajectory)


@pytest.fixture
def hover_pwm_sequence():
    """Hover PWM sequence (constant throttle)."""
    T = 19
    return jnp.ones((T, 4)) * 0.5  # 50% throttle


# ==============================================================================
# FIM Computation Tests
# ==============================================================================

class TestFIMComputation:
    """Tests for Fisher Information Matrix computation."""

    def test_fim_shape(self, simple_trajectory, simple_pwm_sequence, quad_params):
        """Test FIM has correct shape (n_params x n_params)."""
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.ones(10)
        dt = 0.01

        fim = compute_fim(
            simple_trajectory,
            simple_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        n_params = len(params_flat)
        assert fim.shape == (n_params, n_params)

    def test_fim_symmetric(self, simple_trajectory, simple_pwm_sequence, quad_params):
        """Test FIM is symmetric (within numerical precision)."""
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.ones(10)
        dt = 0.01

        fim = compute_fim(
            simple_trajectory,
            simple_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        # Check symmetry
        assert jnp.allclose(fim, fim.T, atol=1e-6)

    def test_fim_positive_semidefinite(
        self, simple_trajectory, simple_pwm_sequence, quad_params
    ):
        """Test FIM is positive semi-definite (all eigenvalues >= 0)."""
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.ones(10)
        dt = 0.01

        fim = compute_fim(
            simple_trajectory,
            simple_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        # Compute eigenvalues
        eigvals = jnp.linalg.eigvalsh(fim)

        # All eigenvalues should be non-negative
        assert jnp.all(eigvals >= -1e-6)  # Small tolerance for numerical error

    def test_fim_diagonal_nonzero(
        self, simple_trajectory, simple_pwm_sequence, quad_params
    ):
        """Test FIM diagonal has some non-zero entries (parameters are excited)."""
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.ones(10)
        dt = 0.01

        fim = compute_fim(
            simple_trajectory,
            simple_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        diag = jnp.diag(fim)

        # At least some parameters should have non-zero information
        assert jnp.sum(diag > 1e-6) > 0


# ==============================================================================
# Excitation Score Tests
# ==============================================================================

class TestExcitationScores:
    """Tests for excitation score computation."""

    def test_excitation_scores_normalized(self):
        """Test excitation scores are normalized to [0, 1] range."""
        fim = jnp.array([
            [10.0, 0.5,  0.0],
            [0.5,  5.0,  0.0],
            [0.0,  0.0,  0.1]
        ])
        param_names = ['mass', 'kT', 'Izz']

        scores = compute_excitation_scores(fim, param_names)

        for name, info in scores.items():
            assert 0.0 <= info['score'] <= 1.0

    def test_excitation_max_score_is_one(self):
        """Test maximum excitation score is exactly 1.0."""
        fim = jnp.array([
            [10.0, 0.5,  0.0],
            [0.5,  5.0,  0.0],
            [0.0,  0.0,  0.1]
        ])
        param_names = ['mass', 'kT', 'Izz']

        scores = compute_excitation_scores(fim, param_names)

        # Find max score
        max_score = max(info['score'] for info in scores.values())
        assert jnp.abs(max_score - 1.0) < 1e-6

    def test_excitation_rank_consistent(self):
        """Test rank ordering is consistent with scores."""
        fim = jnp.array([
            [10.0, 0.5,  0.0],
            [0.5,  5.0,  0.0],
            [0.0,  0.0,  0.1]
        ])
        param_names = ['mass', 'kT', 'Izz']

        scores = compute_excitation_scores(fim, param_names)

        # Sort by score (descending)
        sorted_by_score = sorted(
            scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        # Ranks should be 1, 2, 3, ...
        for i, (name, info) in enumerate(sorted_by_score):
            assert info['rank'] == i + 1

    def test_excitation_threshold_applied(self):
        """Test threshold correctly classifies excited/unexcited parameters."""
        fim = jnp.diag(jnp.array([10.0, 5.0, 0.1]))
        param_names = ['mass', 'kT', 'Izz']
        threshold = 0.3

        scores = compute_excitation_scores(fim, param_names, threshold)

        # Normalized: mass=1.0, kT=0.5, Izz=0.01
        assert scores['mass']['excited'] == True   # 1.0 > 0.3
        assert scores['kT']['excited'] == True     # 0.5 > 0.3
        assert scores['Izz']['excited'] == False   # 0.01 < 0.3

    def test_excitation_zero_fim(self):
        """Test handling of zero FIM (no excitation)."""
        fim = jnp.zeros((3, 3))
        param_names = ['mass', 'kT', 'Izz']

        scores = compute_excitation_scores(fim, param_names)

        # All scores should be zero
        for name, info in scores.items():
            assert info['score'] == 0.0
            assert info['excited'] == False


# ==============================================================================
# Maneuver Suggestion Tests
# ==============================================================================

class TestManeuverSuggestions:
    """Tests for maneuver suggestion generation."""

    def test_suggestions_for_unexcited_izz(self):
        """Test suggestion for unexcited Izz (yaw inertia)."""
        excitation_scores = {
            'mass': {'score': 0.9, 'excited': True},
            'kT': {'score': 0.8, 'excited': True},
            'Izz': {'score': 0.2, 'excited': False}
        }

        suggestions = suggest_maneuvers(excitation_scores)

        # Should suggest yaw maneuver
        assert len(suggestions) >= 1
        assert any('Izz' in s for s in suggestions)
        assert any('yaw' in s.lower() for s in suggestions)

    def test_suggestions_for_all_excited(self):
        """Test no suggestions when all parameters are excited."""
        excitation_scores = {
            'mass': {'score': 0.9, 'excited': True},
            'kT': {'score': 0.8, 'excited': True},
            'kQ': {'score': 0.7, 'excited': True}
        }

        suggestions = suggest_maneuvers(excitation_scores)

        # Should return success message
        assert len(suggestions) == 1
        assert suggestions[0].startswith("✓")

    def test_suggestions_include_details(self):
        """Test suggestions include duration and details."""
        excitation_scores = {
            'Ixx': {'score': 0.2, 'excited': False}
        }

        suggestions = suggest_maneuvers(excitation_scores)

        # Should include duration and details
        suggestion = suggestions[0]
        assert 'Duration:' in suggestion
        assert 'Details:' in suggestion
        assert 'Current score:' in suggestion


# ==============================================================================
# Parameter Coupling Tests
# ==============================================================================

class TestParameterCoupling:
    """Tests for parameter coupling detection."""

    def test_coupling_detection_high_correlation(self):
        """Test detection of highly coupled parameters."""
        # Create FIM with high correlation between param 0 and 1
        fim = jnp.array([
            [10.0,  8.0,  0.0],
            [ 8.0, 10.0,  0.0],
            [ 0.0,  0.0,  5.0]
        ])
        param_names = ['mass', 'kT', 'Izz']

        couplings = check_parameter_coupling(fim, param_names, correlation_threshold=0.7)

        # Should detect mass-kT coupling
        assert len(couplings) >= 1
        assert any(
            ('mass' in c[0] and 'kT' in c[1]) or
            ('kT' in c[0] and 'mass' in c[1])
            for c in couplings
        )

    def test_coupling_no_false_positives(self):
        """Test no coupling detected for diagonal FIM."""
        fim = jnp.diag(jnp.array([10.0, 5.0, 2.0]))
        param_names = ['mass', 'kT', 'Izz']

        couplings = check_parameter_coupling(fim, param_names, correlation_threshold=0.7)

        # Should detect no couplings (purely diagonal)
        assert len(couplings) == 0

    def test_coupling_sorted_by_magnitude(self):
        """Test couplings are sorted by correlation magnitude."""
        fim = jnp.array([
            [10.0,  8.0,  3.0],
            [ 8.0, 10.0,  2.0],
            [ 3.0,  2.0,  5.0]
        ])
        param_names = ['A', 'B', 'C']

        couplings = check_parameter_coupling(fim, param_names, correlation_threshold=0.3)

        # Correlations should be in descending order
        correlations = [abs(c[2]) for c in couplings]
        assert correlations == sorted(correlations, reverse=True)


# ==============================================================================
# Structural Identifiability Tests
# ==============================================================================

class TestStructuralIdentifiability:
    """Tests for structural identifiability analysis."""

    def test_identifiability_full_rank(self):
        """Test full rank detection for well-conditioned FIM."""
        fim = jnp.diag(jnp.array([10.0, 5.0, 2.0]))
        param_names = ['mass', 'kT', 'Izz']

        info = check_structural_identifiability(fim, param_names)

        assert info['rank'] == 3
        assert info['n_params'] == 3
        assert info['full_rank'] == True
        assert len(info['unidentifiable_directions']) == 0

    def test_identifiability_rank_deficient(self):
        """Test rank deficiency detection."""
        # Create rank-2 matrix (3x3 with one zero eigenvalue)
        fim = jnp.array([
            [10.0,  0.0,  0.0],
            [ 0.0,  5.0,  0.0],
            [ 0.0,  0.0,  0.0]  # Zero eigenvalue
        ])
        param_names = ['mass', 'kT', 'Izz']

        info = check_structural_identifiability(fim, param_names)

        assert info['rank'] == 2
        assert info['n_params'] == 3
        assert info['full_rank'] == False
        assert len(info['unidentifiable_directions']) >= 1

    def test_identifiability_condition_number(self):
        """Test condition number computation."""
        # Well-conditioned matrix
        fim_good = jnp.diag(jnp.array([10.0, 9.0, 8.0]))
        info_good = check_structural_identifiability(fim_good, ['A', 'B', 'C'])
        assert info_good['condition_number'] < 10.0
        assert info_good['well_conditioned'] == True

        # Ill-conditioned matrix
        fim_bad = jnp.diag(jnp.array([1e6, 1.0, 1e-3]))
        info_bad = check_structural_identifiability(fim_bad, ['A', 'B', 'C'])
        assert info_bad['condition_number'] > 1e6
        assert info_bad['well_conditioned'] == False

    def test_identifiability_singular_values_descending(self):
        """Test singular values are in descending order."""
        fim = jnp.diag(jnp.array([10.0, 5.0, 2.0]))
        param_names = ['mass', 'kT', 'Izz']

        info = check_structural_identifiability(fim, param_names)

        s = info['singular_values']
        # Should be descending
        assert all(s[i] >= s[i+1] for i in range(len(s) - 1))


# ==============================================================================
# Data Quality Assessment Tests
# ==============================================================================

class TestDataQualityAssessment:
    """Tests for overall data quality assessment."""

    def test_quality_excellent(self):
        """Test EXCELLENT rating for high excitation + full rank."""
        excitation_scores = {
            f'param{i}': {'score': 0.9 + i*0.01, 'excited': True}
            for i in range(10)
        }
        identifiability_info = {
            'full_rank': True,
            'condition_number': 100.0
        }

        quality = assess_data_quality(excitation_scores, identifiability_info)
        assert quality == 'EXCELLENT'

    def test_quality_good(self):
        """Test GOOD rating for decent excitation + full rank."""
        excitation_scores = {
            f'param{i}': {'score': 0.7 if i < 7 else 0.2, 'excited': i < 7}
            for i in range(10)
        }
        identifiability_info = {
            'full_rank': True,
            'condition_number': 1e5
        }

        quality = assess_data_quality(excitation_scores, identifiability_info)
        assert quality == 'GOOD'

    def test_quality_poor(self):
        """Test POOR rating for low excitation or rank deficiency."""
        excitation_scores = {
            f'param{i}': {'score': 0.1, 'excited': False}
            for i in range(10)
        }
        identifiability_info = {
            'full_rank': False,
            'condition_number': 1e8
        }

        quality = assess_data_quality(excitation_scores, identifiability_info)
        assert quality == 'POOR'


# ==============================================================================
# Edge Case Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_fim_with_zero_weights(self, simple_trajectory, simple_pwm_sequence, quad_params):
        """Test FIM computation with zero weights (should handle gracefully)."""
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.zeros(10)  # All zero weights
        dt = 0.01

        fim = compute_fim(
            simple_trajectory,
            simple_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        # FIM should be zero or near-zero
        assert jnp.max(jnp.abs(fim)) < 1e-6

    def test_hover_only_flags_izz_unexcited(
        self, hover_only_trajectory, hover_pwm_sequence, quad_params
    ):
        """Test hover-only flight flags Izz (yaw inertia) as poorly excited."""
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.ones(10)
        dt = 0.01

        fim = compute_fim(
            hover_only_trajectory,
            hover_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        param_names = get_parameter_names_from_template(template)
        scores = compute_excitation_scores(fim, param_names)

        # Find Izz in the parameter names
        izz_name = [name for name in param_names if 'inertia[2]' in name or name == 'Izz']

        if izz_name:
            # Izz should have low excitation (no yaw in hover-only)
            izz_score = scores[izz_name[0]]['score']
            # Note: May not be exactly 0 due to numerical effects,
            # but should be lower than other parameters
            assert izz_score < 0.5  # Relatively low

    def test_parameter_names_from_template(self, quad_params):
        """Test parameter name extraction from template."""
        params_flat, template = flatten_params(quad_params)
        param_names = get_parameter_names_from_template(template)

        # Should have correct number of parameters
        assert len(param_names) == len(params_flat)

        # Should contain expected parameter names
        assert 'mass' in param_names
        assert 'kT' in param_names
        assert 'kQ' in param_names

    def test_compute_condition_number_edge_cases(self):
        """Test condition number for various FIM conditions."""
        # Well-conditioned
        fim_good = jnp.eye(3)
        cond_good = compute_condition_number(fim_good)
        assert cond_good == 1.0

        # Singular (zero eigenvalue)
        fim_singular = jnp.zeros((3, 3))
        cond_singular = compute_condition_number(fim_singular)
        assert jnp.isinf(cond_singular)


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_excitation_analysis_pipeline(
        self, simple_trajectory, simple_pwm_sequence, quad_params
    ):
        """Test complete excitation analysis pipeline."""
        # 1. Compute FIM
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.ones(10)
        dt = 0.01

        fim = compute_fim(
            simple_trajectory,
            simple_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        # 2. Compute excitation scores
        param_names = get_parameter_names_from_template(template)
        scores = compute_excitation_scores(fim, param_names)

        # 3. Suggest maneuvers
        suggestions = suggest_maneuvers(scores)

        # 4. Check coupling
        couplings = check_parameter_coupling(fim, param_names)

        # 5. Check identifiability
        id_info = check_structural_identifiability(fim, param_names)

        # 6. Assess data quality
        quality = assess_data_quality(scores, id_info)

        # All should complete without errors
        assert fim is not None
        assert len(scores) == len(param_names)
        assert len(suggestions) >= 1
        assert quality in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']

    def test_data_improvement_suggestions(
        self, simple_trajectory, simple_pwm_sequence, quad_params
    ):
        """Test data improvement suggestion pipeline."""
        params_flat, template = flatten_params(quad_params)
        params_fixed = {
            'motor_positions': quad_params['motor_positions'],
            'motor_directions': quad_params['motor_directions']
        }
        weights = jnp.ones(10)
        dt = 0.01

        fim = compute_fim(
            simple_trajectory,
            simple_pwm_sequence,
            params_flat,
            template,
            params_fixed,
            weights,
            dt
        )

        param_names = get_parameter_names_from_template(template)
        scores = compute_excitation_scores(fim, param_names)
        id_info = check_structural_identifiability(fim, param_names)

        improvements = suggest_data_improvements(scores, id_info)

        # Should return at least one suggestion
        assert len(improvements) >= 1
        # Suggestions should be strings
        assert all(isinstance(s, str) for s in improvements)
