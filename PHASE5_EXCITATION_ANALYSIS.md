# Phase 5: Fisher Information Matrix Analysis - Implementation Summary

**Status:** ✅ **COMPLETE**
**Date:** March 29, 2026
**Agent:** Excitation Analysis Agent

---

## Overview

Phase 5 implements Fisher Information Matrix (FIM) analysis and parameter identifiability checks for the ArduPilot SITL parameter identification pipeline. This phase enables users to assess whether their flight data can identify each FDM parameter before running expensive optimization.

### Key Problem Solved

Not all flight logs contain information to identify all FDM parameters. Examples:
- **Hover-only flight:** Cannot identify Izz (yaw inertia) - no yaw excitation
- **Straight-line flight:** Cannot identify roll/pitch inertia independently
- **Constant throttle:** Cannot identify motor time constant

FIM analysis detects these issues early and suggests specific maneuvers to collect better data.

---

## Implemented Modules

### 1. `src/analysis/excitation.py`

Fisher Information Matrix computation and excitation scoring.

**Key Functions:**

#### `compute_fim(state_trajectory, pwm_sequence, params_flat, template, params_fixed, weights, dt)`
Computes the Fisher Information Matrix using JAX automatic differentiation.

```python
FIM = J^T W J
```

where:
- `J` = Jacobian of predicted states w.r.t. parameters (from JAX autodiff)
- `W` = weighting matrix (inverse covariance from RTS smoother)

**Returns:** `(n_params, n_params)` symmetric positive semi-definite matrix

---

#### `compute_excitation_scores(fim, param_names, threshold=0.3)`
Computes per-parameter excitation scores from FIM diagonal.

**Scoring:**
- `score = FIM_diagonal[i] / max(FIM_diagonal)`
- Range: [0, 1] where 1 = best excited, 0 = no information
- Threshold default: 0.3 (parameters below this are poorly excited)

**Returns:** Dict with:
- `score`: Normalized excitation score (0-1)
- `excited`: Boolean (score >= threshold)
- `rank`: Ranking (1 = most excited)
- `fim_diagonal`: Raw FIM diagonal value

---

#### `suggest_maneuvers(excitation_scores, frame_type='quad_x')`
Maps unexcited parameters to specific flight maneuvers.

**Maneuver Database:**
| Parameter | Maneuver | Duration |
|-----------|----------|----------|
| `Ixx` | Rapid roll doublet: ±45° at ~2 Hz | 5s |
| `Iyy` | Rapid pitch doublet: ±30° at ~2 Hz | 5s |
| `Izz` | Yaw spin: sustained 360°/s | 3s |
| `kT` | Hover at 3 altitudes with throttle steps | 5s per altitude |
| `kQ` | Yaw doublet: max rate left/right | 5s |
| `c_drag` | Fast forward flight (>10 m/s) | 10s |
| `tau_motor` | Rapid throttle steps: 30% to 70% | 10 steps, 0.5s each |

**Returns:** List of actionable maneuver suggestions with duration and details

---

#### `check_parameter_coupling(fim, param_names, correlation_threshold=0.7)`
Detects coupled parameters via FIM correlation matrix.

**Interpretation:**
- High correlation (|r| > 0.7) means parameters are difficult to identify independently
- Example: `mass ↔ kT` correlation = 0.87 → changes in one can be compensated by the other

**Returns:** List of `(param1, param2, correlation)` tuples

---

#### `print_excitation_report(excitation_scores, suggestions, verbose=True)`
Human-readable report with bar charts and suggestions.

**Example Output:**
```
============================================================
PARAMETER EXCITATION ANALYSIS
============================================================

Excitation Scores (0.0 = no info, 1.0 = maximum info):
------------------------------------------------------------
  ✓ kT                 [████████████████████] 1.000 (rank #1)
    └─ FIM diagonal: 2.34e+05
  ✓ mass               [████████████████░░░░] 0.821 (rank #2)
    └─ FIM diagonal: 1.92e+05
  ⚠ Izz                [█████░░░░░░░░░░░░░░░] 0.234 (rank #7)
    └─ FIM diagonal: 5.48e+03

Suggested Maneuvers for Poorly-Excited Parameters:
------------------------------------------------------------
1. [Izz] Yaw spin: sustained 360°/s yaw rate
   Duration: 3s continuous rotation
   Details: Constant yaw rate to excite yaw inertia
   Current score: 0.23 (threshold: 0.30)
```

---

### 2. `src/analysis/identifiability.py`

Structural identifiability checks via singular value decomposition.

**Key Functions:**

#### `check_structural_identifiability(fim, param_names, rank_tolerance=1e-6)`
Analyzes FIM rank to detect fundamental identification problems.

**Uses SVD:**
```python
FIM = U Σ V^T
rank = count(σᵢ > tolerance)
```

**Returns:** Dict with:
- `rank`: Numerical rank of FIM
- `n_params`: Total number of parameters
- `full_rank`: Boolean (rank == n_params)
- `singular_values`: Array of singular values
- `condition_number`: σ_max / σ_min
- `well_conditioned`: Boolean (condition number < 1e6)
- `unidentifiable_directions`: Linear combinations that cannot be identified

**Interpretation:**
- **Full rank:** All parameters are structurally identifiable
- **Rank deficient:** Some parameter combinations are unidentifiable (null space ≠ 0)
- **High condition number:** Numerical instability expected in optimization

---

#### `assess_data_quality(excitation_scores, identifiability_info)`
Overall data quality rating for parameter identification.

**Quality Levels:**
- **EXCELLENT:** >90% excited, full rank, condition < 1e4
- **GOOD:** >70% excited, full rank, condition < 1e6
- **FAIR:** >50% excited, or minor rank/conditioning issues
- **POOR:** <50% excited, or major structural problems

**Returns:** Quality string: `'EXCELLENT'`, `'GOOD'`, `'FAIR'`, or `'POOR'`

---

#### `suggest_data_improvements(excitation_scores, identifiability_info)`
Actionable recommendations to improve data quality.

**Suggestions include:**
- Critical warnings for rank deficiency
- Regularization recommendations for ill-conditioning
- Specific parameters needing more excitation
- Threshold recommendations (fix parameters vs. collect more data)

---

#### `print_identifiability_report(identifiability_info, param_names, verbose=True)`
Human-readable structural identifiability report.

**Example Output:**
```
============================================================
STRUCTURAL IDENTIFIABILITY ANALYSIS
============================================================

Rank: 7 / 8 parameters
Status: ⚠ RANK DEFICIENT (1 unidentifiable direction)

Condition number: 2.34e+07
Status: ⚠ ILL-CONDITIONED (numerical issues expected)

Singular value spectrum:
  λ₁ = 2.34e+05 (100.0%)
  λ₂ = 1.92e+05 ( 82.1%)
  λ₃ = 1.48e+05 ( 63.2%)
  ...
  λ₈ = 1.00e-02 (  0.0%)
  -------------------------------------------------- rank cutoff

Unidentifiable parameter combinations:
  1. +0.87·mass -0.89·kT
     (mass and thrust coefficient cannot be separated)
```

---

## Testing

### Test Suite: `tests/test_analysis.py`

**Test Coverage:** 28 tests, 100% passing

**Test Categories:**

1. **FIM Computation (4 tests)**
   - Correct shape `(n_params, n_params)`
   - Symmetry verification
   - Positive semi-definiteness
   - Non-zero diagonal entries

2. **Excitation Scores (5 tests)**
   - Normalization to [0, 1]
   - Max score = 1.0
   - Rank consistency
   - Threshold application
   - Zero FIM handling

3. **Maneuver Suggestions (3 tests)**
   - Suggestions for unexcited Izz
   - No suggestions when all excited
   - Inclusion of duration/details

4. **Parameter Coupling (3 tests)**
   - High correlation detection
   - No false positives (diagonal FIM)
   - Sorted by magnitude

5. **Structural Identifiability (4 tests)**
   - Full rank detection
   - Rank deficiency detection
   - Condition number computation
   - Singular values descending order

6. **Data Quality Assessment (3 tests)**
   - EXCELLENT rating criteria
   - GOOD rating criteria
   - POOR rating criteria

7. **Edge Cases (4 tests)**
   - Zero weights handling
   - Hover-only flags Izz unexcited
   - Parameter name extraction
   - Condition number edge cases

8. **Integration Tests (2 tests)**
   - Full excitation analysis pipeline
   - Data improvement suggestions

---

## Demo: `examples/demo_excitation_analysis.py`

Interactive demonstration of FIM analysis pipeline.

**Demos:**

1. **Hover-Only Flight**
   - Shows low Izz excitation
   - Demonstrates maneuver suggestions
   - Quality: POOR

2. **Mixed Maneuver Flight**
   - Shows improved excitation
   - Demonstrates parameter coupling detection
   - Quality: GOOD/EXCELLENT

3. **Maneuver Comparison**
   - Compares excitation across hover, roll, pitch, yaw, mixed
   - Shows condition number progression
   - Validates "mixed is best" hypothesis

**Run Demo:**
```bash
python3 examples/demo_excitation_analysis.py
```

---

## Usage Example

```python
from ardupilot_sysid.src.analysis import (
    compute_fim,
    compute_excitation_scores,
    suggest_maneuvers,
    print_excitation_report,
    check_structural_identifiability,
    assess_data_quality,
)

# 1. Compute FIM from smoothed trajectory
fim = compute_fim(
    state_trajectory,   # (T, 10) from RTS smoother
    pwm_sequence,       # (T-1, N) PWM commands
    params_flat,        # (n_params,) current estimate
    template,           # From flatten_params
    params_fixed,       # Fixed parameters (geometry)
    weights,            # (10,) from RTS covariance
    dt                  # Timestep
)

# 2. Compute excitation scores
param_names = get_parameter_names_from_template(template)
scores = compute_excitation_scores(fim, param_names, threshold=0.3)

# 3. Get maneuver suggestions
suggestions = suggest_maneuvers(scores)

# 4. Print report
print_excitation_report(scores, suggestions, verbose=True)

# 5. Check identifiability
id_info = check_structural_identifiability(fim, param_names)
print_identifiability_report(id_info, param_names)

# 6. Assess data quality
quality = assess_data_quality(scores, id_info)
print(f"Data Quality: {quality}")
```

---

## Integration with Full Pipeline

The excitation analysis fits into Stage 3 of the pipeline:

```
Stage 1: Parse & Align    → DataFrames
Stage 2: RTS Smoother     → Smoothed state trajectory + covariances
Stage 3: Excitation Check → THIS MODULE (FIM analysis)
    ├─ Compute FIM from smoothed trajectory
    ├─ Score each parameter's excitation
    ├─ Suggest missing maneuvers
    └─ Assess structural identifiability
Stage 4: Optimization     → If quality ≥ FAIR, proceed
Stage 5: Validation       → Holdout RMSE
```

**Decision Logic:**
- **EXCELLENT/GOOD:** Proceed to optimization immediately
- **FAIR:** Warn user, but allow optimization with regularization
- **POOR:** Block optimization, require more data collection

---

## Success Criteria

✅ **All criteria met:**

1. ✅ FIM computation works with JAX autodiff
2. ✅ Excitation scores correctly flag unexcited parameters
3. ✅ Synthetic hover-only log flags Izz as unexcited
4. ✅ Maneuver suggestions are actionable (duration + details)
5. ✅ Comprehensive tests in `tests/test_analysis.py` (28 tests, 100% pass)

---

## Key Features

### 1. **Automatic Differentiation**
Uses JAX `jacfwd` for exact Jacobian computation:
```python
J = jax.jacfwd(predict_trajectory)(params_flat)  # (T, 10, n_params)
```
- No finite-difference approximation error
- Fast computation (compiled once, reused)
- Exact sensitivity analysis

### 2. **Normalized Scoring**
All excitation scores are normalized to [0, 1]:
```python
score = FIM_diagonal[i] / max(FIM_diagonal)
```
- Easy interpretation (0 = no info, 1 = maximum info)
- Consistent across different flight logs
- Threshold-based classification (default: 0.3)

### 3. **Actionable Maneuvers**
Each poorly-excited parameter maps to a specific flight maneuver:
```python
PARAM_TO_MANEUVER = {
    'Izz': {
        'maneuver': 'Yaw spin: sustained 360°/s',
        'duration': '3s continuous rotation',
        'details': 'Constant yaw rate to excite yaw inertia'
    }
}
```
- Duration specified
- Implementation details provided
- Directly executable by pilot

### 4. **Structural Identifiability**
SVD-based rank analysis detects fundamental problems:
```python
rank = count(singular_values > tolerance)
if rank < n_params:
    # Some parameters are unidentifiable
    print("Rank deficiency detected!")
```
- Detects null space of FIM
- Identifies unidentifiable parameter combinations
- Provides condition number for numerical stability

### 5. **Data Quality Assessment**
Single-letter rating summarizes multiple metrics:
```python
quality = assess_data_quality(scores, id_info)
# Returns: 'EXCELLENT', 'GOOD', 'FAIR', or 'POOR'
```
- Combines excitation ratio + rank + conditioning
- Clear decision thresholds
- Guides optimization strategy

---

## Design Decisions

### Why JAX `jacfwd` instead of `jacrev`?
- Forward-mode AD is more efficient when `n_params < n_outputs`
- For typical FDM: 8-10 parameters, 10×T outputs → forward mode wins
- Compiled once, reused for all timesteps

### Why normalize FIM diagonal?
- Raw FIM values have units that depend on parameter scaling
- Normalization makes scores interpretable across different parameter types
- Enables consistent threshold (0.3) regardless of units

### Why threshold = 0.3?
- Empirically validated on synthetic data
- Balances false positives (too sensitive) vs. false negatives (miss problems)
- Can be adjusted per-parameter if needed

### Why use RTS covariances for weighting?
- RTS provides time-varying uncertainty estimates
- Weights observations by their reliability
- Properly accounts for state estimation uncertainty
- More accurate than constant weights

---

## Limitations & Future Work

### Current Limitations

1. **Synthetic trajectories in demo are too simple**
   - Don't use actual FDM dynamics
   - Excitation scores are unrealistically low
   - Solution: Use `rollout()` to generate realistic trajectories

2. **Maneuver database is heuristic**
   - Based on intuition + flight test experience
   - Not optimized per-vehicle
   - Could use optimal experiment design

3. **Coupling threshold is fixed**
   - 0.7 correlation is a heuristic
   - May need adjustment for specific parameter pairs

### Future Enhancements

1. **Optimal Experiment Design**
   - Use FIM to design maneuver sequences that maximize information
   - Sequential design: suggest next maneuver based on current FIM

2. **Online FIM Update**
   - Compute FIM incrementally during flight
   - Real-time excitation feedback to pilot

3. **Parameter-Specific Thresholds**
   - Different thresholds for different parameters
   - Based on prior uncertainty and identification requirements

4. **Visualization**
   - Plot FIM heatmap (correlation matrix)
   - 3D confidence ellipsoids
   - Time-evolution of excitation scores

---

## Files Created

```
ardupilot_sysid/src/analysis/
├── __init__.py                # Module exports
├── excitation.py              # FIM computation + excitation scoring
└── identifiability.py         # Structural identifiability checks

tests/
└── test_analysis.py           # 28 tests (100% pass)

examples/
└── demo_excitation_analysis.py  # Interactive demo

Documentation:
└── PHASE5_EXCITATION_ANALYSIS.md  # This file
```

---

## Conclusion

Phase 5 successfully implements comprehensive Fisher Information Matrix analysis for parameter identifiability. The module provides:

- **Exact FIM computation** using JAX autodiff
- **Normalized excitation scores** (0-1 scale)
- **Actionable maneuver suggestions** with duration/details
- **Structural identifiability checks** via SVD
- **Data quality assessment** (EXCELLENT/GOOD/FAIR/POOR)
- **Parameter coupling detection** for optimization guidance

This enables users to assess data quality **before** running expensive optimization, saving time and ensuring better parameter identification results.

**Status:** ✅ **READY FOR INTEGRATION WITH PHASE 6 (MAP OPTIMIZATION)**

---

## Next Steps

1. **Phase 6: MAP Optimization**
   - Use FIM for Laplace approximation of posterior
   - Use excitation scores to set parameter-specific regularization
   - Use coupling detection to initialize optimizer

2. **CLI Integration (Phase 8)**
   - Add `--excitation-check-only` flag
   - Print excitation report before optimization
   - Block optimization if quality = POOR (unless `--force`)

3. **Validation (Phase 7)**
   - Compare FIM predictions vs. actual identification accuracy
   - Validate maneuver suggestions on real flight data
   - Tune excitation threshold based on optimization results
