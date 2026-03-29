"""
Demonstration of validation and output generation.

This example shows how to:
1. Validate identified parameters on hold-out data
2. Compute comprehensive validation metrics
3. Generate SITL parameter files
4. Create JSON and text reports
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.rollout import hold_out_validation, split_train_test
from src.validation.metrics import summarize_validation_metrics
from src.output.parm_writer import convert_to_sitl_params, write_parm_file
from src.output.report import generate_json_report, generate_text_report


def main():
    """Demonstrate validation and output generation workflow."""

    print("=" * 70)
    print("VALIDATION AND OUTPUT GENERATION DEMO")
    print("=" * 70)
    print()

    # 1. Split segments into training and testing
    print("1. Splitting segments into train/test sets...")
    segments = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 25.0)]
    train_segments, test_segments = split_train_test(segments, test_ratio=0.2)
    print(f"   Training segments: {len(train_segments)}")
    print(f"   Test segments: {len(test_segments)}")
    print()

    # 2. Simulate identified parameters (from optimization)
    print("2. Identified parameters:")
    params_identified = {
        'mass': 1.2,
        'inertia': np.array([0.015, 0.018, 0.025]),
        'kT': 1.2e-5,
        'pwm_to_omega_poly': np.array([0.0, 900.0]),
        'tau_motor': 0.04,
        'c_drag': 0.0008
    }

    for key, value in params_identified.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: {value.tolist()}")
        else:
            print(f"   {key}: {value}")
    print()

    # 3. Simulate confidence intervals (from bootstrap/Hessian)
    print("3. Confidence intervals (95%):")
    confidence_intervals = {
        'mass': (1.15, 1.25),
        'kT': (1.1e-5, 1.3e-5),
        'tau_motor': (0.035, 0.045)
    }

    for param, (low, high) in confidence_intervals.items():
        print(f"   {param}: [{low:.6g}, {high:.6g}]")
    print()

    # 4. Simulate excitation scores
    print("4. Excitation analysis:")
    excitation_scores = {
        'mass': {'score': 0.95, 'excited': True},
        'kT': {'score': 0.88, 'excited': True},
        'tau_motor': {'score': 0.65, 'excited': True},
        'c_drag': {'score': 0.42, 'excited': False}
    }

    for param, scores in excitation_scores.items():
        status = "✓" if scores['excited'] else "⚠"
        print(f"   {status} {param}: score={scores['score']:.2f}")
    print()

    # 5. Simulate validation (with synthetic data)
    print("5. Hold-out validation (synthetic data):")
    T = 100
    dt = 0.01

    # Create synthetic ground truth trajectory
    actual_trajectory = np.zeros((T, 10))
    actual_trajectory[:, 0] = 1.0  # Identity quaternion
    actual_trajectory[:, 4:7] = np.random.randn(T, 3) * 0.1  # Small velocities

    # Add small prediction errors to simulate FDM rollout
    predicted_trajectory = actual_trajectory.copy()
    predicted_trajectory[:, 0:4] += np.random.randn(T, 4) * 0.01
    predicted_trajectory[:, 0:4] /= np.linalg.norm(
        predicted_trajectory[:, 0:4], axis=1, keepdims=True
    )
    predicted_trajectory[:, 4:7] += np.random.randn(T, 3) * 0.05

    validation_result = {
        'predicted': predicted_trajectory,
        'actual': actual_trajectory,
        'residuals': predicted_trajectory - actual_trajectory,
        'timestamps': np.arange(T) * dt
    }

    # Compute validation metrics
    validation_metrics = summarize_validation_metrics(validation_result)

    print(f"   Attitude RMSE:")
    print(f"     Roll:  {validation_metrics['attitude']['roll_deg']:.2f}°")
    print(f"     Pitch: {validation_metrics['attitude']['pitch_deg']:.2f}°")
    print(f"     Yaw:   {validation_metrics['attitude']['yaw_deg']:.2f}°")
    print(f"   Velocity RMSE: {validation_metrics['velocity']['v_total']:.3f} m/s")
    print(f"   Duration: {validation_metrics['duration_s']:.2f}s")
    print()

    # 6. Convert to SITL parameters
    print("6. Converting to SITL parameters:")
    sitl_params = convert_to_sitl_params(params_identified)

    for param, value in sorted(sitl_params.items()):
        print(f"   {param}: {value:.6g}")
    print()

    # 7. Write SITL parameter file
    print("7. Writing SITL parameter file...")
    metadata = {
        'log_file': 'example_flight.bin',
        'frame_type': 'quad_x',
        'validation_rmse': validation_metrics['attitude']
    }

    write_parm_file('output_params.parm', sitl_params, metadata)
    print()

    # 8. Generate JSON report
    print("8. Generating JSON report...")
    report_metadata = {
        'log_file': 'example_flight.bin',
        'frame_type': 'quad_x',
        'flight_duration_s': 25.0,
        'usable_duration_s': 20.0,
        'preprocessing': {
            'filter_cutoff_hz': 10.0,
            'segments_extracted': 5
        }
    }

    generate_json_report(
        params_identified,
        confidence_intervals,
        excitation_scores,
        validation_metrics,
        report_metadata,
        'output_report.json'
    )
    print()

    # 9. Generate text report
    print("9. Generating text report...")
    report_text = generate_text_report(
        params_identified,
        confidence_intervals,
        validation_metrics,
        'output_report.txt'
    )
    print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  - output_params.parm   : SITL parameter file")
    print("  - output_report.json   : Machine-readable report")
    print("  - output_report.txt    : Human-readable report")
    print()


if __name__ == '__main__':
    main()
