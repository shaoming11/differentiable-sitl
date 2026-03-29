"""
JSON report generation for parameter identification results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def generate_json_report(
    identified_params: Dict[str, float],
    confidence_intervals: Dict[str, tuple],
    excitation_scores: Dict[str, float],
    validation_metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Generate comprehensive JSON report of identification results.

    Args:
        identified_params: Dictionary of identified parameter values
        confidence_intervals: Dictionary of (lower, upper) confidence intervals per parameter
        excitation_scores: Dictionary of excitation scores per parameter
        validation_metrics: Validation metrics (RMSE, etc.)
        metadata: Metadata about the identification run
        output_path: Path to output JSON file
    """
    report = {
        "metadata": {
            "sysid_version": "0.1.0",
            "generated": datetime.now().isoformat(),
            **metadata
        },
        "identified_parameters": {},
        "excitation_scores": excitation_scores,
        "validation": validation_metrics,
        "warnings": []
    }

    # Format identified parameters with confidence intervals
    for param_name, value in identified_params.items():
        param_data = {"value": float(value)}

        if param_name in confidence_intervals:
            ci = confidence_intervals[param_name]
            param_data["ci_95"] = [float(ci[0]), float(ci[1])]
            param_data["uncertainty_percent"] = float(
                100 * (ci[1] - ci[0]) / (2 * value) if value != 0 else 0
            )

        report["identified_parameters"][param_name] = param_data

    # Generate warnings based on excitation scores
    poorly_excited_threshold = 0.3
    for param_name, score in excitation_scores.items():
        if score < poorly_excited_threshold:
            report["warnings"].append(
                f"{param_name} poorly excited (score={score:.2f}). "
                f"Confidence interval may be unreliable."
            )

    # Write to file
    output_path = Path(output_path)
    with output_path.open('w') as f:
        json.dump(report, f, indent=2)


def print_report_summary(report_path: Path) -> None:
    """
    Print a human-readable summary of a JSON report.

    Args:
        report_path: Path to JSON report file
    """
    with open(report_path) as f:
        report = json.load(f)

    print("\n" + "="*70)
    print("Parameter Identification Report Summary")
    print("="*70)

    # Metadata
    print(f"\nLog file: {report['metadata'].get('log_file', 'N/A')}")
    print(f"Frame type: {report['metadata'].get('frame_type', 'N/A')}")
    print(f"Duration: {report['metadata'].get('flight_duration_s', 0):.1f}s")

    # Parameters
    print("\nIdentified Parameters:")
    print("-" * 70)
    for param_name, data in report['identified_parameters'].items():
        value = data['value']
        if 'ci_95' in data:
            ci = data['ci_95']
            print(f"  {param_name:20s} = {value:.6f}  [{ci[0]:.6f}, {ci[1]:.6f}]")
        else:
            print(f"  {param_name:20s} = {value:.6f}")

    # Validation
    if 'validation' in report and report['validation']:
        print("\nValidation Metrics:")
        print("-" * 70)
        val = report['validation']
        if 'attitude' in val:
            att = val['attitude']
            print(f"  Roll RMSE:  {att.get('roll_deg', 0):.2f}°")
            print(f"  Pitch RMSE: {att.get('pitch_deg', 0):.2f}°")
            print(f"  Yaw RMSE:   {att.get('yaw_deg', 0):.2f}°")

    # Warnings
    if report.get('warnings'):
        print("\nWarnings:")
        print("-" * 70)
        for warning in report['warnings']:
            print(f"  ⚠ {warning}")

    print("\n" + "="*70)
