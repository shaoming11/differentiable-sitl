import json
from typing import Dict
from datetime import datetime
from pathlib import Path


def generate_json_report(
    params_identified: Dict,
    confidence_intervals: Dict,
    excitation_scores: Dict,
    validation_metrics: Dict,
    metadata: Dict,
    output_path: str
) -> None:
    """
    Generate comprehensive JSON report.

    Includes:
    - Identified parameters with confidence intervals
    - Excitation scores and warnings
    - Validation metrics
    - Preprocessing metadata

    Args:
        params_identified: Optimal parameter values
        confidence_intervals: 95% CIs per parameter
        excitation_scores: Excitation analysis results
        validation_metrics: Hold-out validation RMSE
        metadata: Log info, preprocessing stats
        output_path: Path to output JSON file
    """
    report = {
        "metadata": {
            "log_file": metadata.get('log_file', 'unknown'),
            "frame_type": metadata.get('frame_type', 'quad_x'),
            "flight_duration_s": metadata.get('flight_duration_s', 0),
            "usable_duration_s": metadata.get('usable_duration_s', 0),
            "sysid_version": "0.1.0",
            "generation_date": datetime.now().isoformat()
        },

        "identified_parameters": {},
        "excitation_scores": excitation_scores,
        "warnings": [],
        "validation": validation_metrics,
        "preprocessing": metadata.get('preprocessing', {})
    }

    # Add parameters with confidence intervals
    for param_name, value in params_identified.items():
        # Convert value to JSON-serializable format
        if hasattr(value, 'tolist'):
            # NumPy array
            json_value = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Already a list or tuple
            json_value = list(value)
        else:
            # Scalar
            json_value = float(value)

        if param_name in confidence_intervals:
            ci_low, ci_high = confidence_intervals[param_name]
            report["identified_parameters"][param_name] = {
                "value": json_value,
                "ci_95": [float(ci_low), float(ci_high)]
            }
        else:
            report["identified_parameters"][param_name] = {
                "value": json_value
            }

    # Add warnings for poorly excited parameters
    for param, scores in excitation_scores.items():
        if not scores.get('excited', True):
            report["warnings"].append(
                f"{param} poorly excited (score={scores['score']:.2f}). "
                f"Consider adding maneuvers to improve identification."
            )

    # Write JSON
    path = Path(output_path)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ JSON report written to: {path}")


def generate_text_report(
    params_identified: Dict,
    confidence_intervals: Dict,
    validation_metrics: Dict,
    output_path: str = None
) -> str:
    """
    Generate human-readable text report.

    Returns formatted string (and optionally writes to file).
    """
    lines = []
    lines.append("="*60)
    lines.append("ARDUPILOT SITL PARAMETER IDENTIFICATION REPORT")
    lines.append("="*60)
    lines.append("")

    lines.append("IDENTIFIED PARAMETERS:")
    lines.append("-"*60)
    for param, value in params_identified.items():
        if param in confidence_intervals:
            ci = confidence_intervals[param]
            lines.append(f"  {param:20s} = {value:.6g} [{ci[0]:.6g}, {ci[1]:.6g}]")
        else:
            lines.append(f"  {param:20s} = {value}")
    lines.append("")

    lines.append("VALIDATION METRICS:")
    lines.append("-"*60)
    if 'attitude' in validation_metrics:
        att = validation_metrics['attitude']
        lines.append(f"  Roll RMSE:   {att['roll_deg']:.2f}°")
        lines.append(f"  Pitch RMSE:  {att['pitch_deg']:.2f}°")
        lines.append(f"  Yaw RMSE:    {att['yaw_deg']:.2f}°")
    if 'velocity' in validation_metrics:
        vel = validation_metrics['velocity']
        lines.append(f"  Velocity RMSE: {vel['v_total']:.3f} m/s")
    lines.append("")
    lines.append("="*60)

    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Text report written to: {output_path}")

    return report_text
