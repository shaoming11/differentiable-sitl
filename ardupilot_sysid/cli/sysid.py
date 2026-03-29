"""
Command-line interface for ArduPilot SITL parameter identification.
"""

import click


@click.command()
@click.version_option(version="0.1.0")
def main():
    """
    ArduPilot SITL Parameter Identification Tool

    Identifies Flight Dynamics Model parameters from ArduPilot DataFlash logs
    using differentiable optimization with JAX.
    """
    click.echo("ArduPilot SITL Parameter Identification v0.1.0")
    click.echo("Full implementation coming soon...")
    click.echo("\nPlanned usage:")
    click.echo("  ardupilot-sysid --log flight.bin --frame quad_x --mass 1.2 --output sitl_params.parm")


if __name__ == "__main__":
    main()
