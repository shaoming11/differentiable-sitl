"""
Tests for CLI interface.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.sysid import main


def test_cli_help():
    """Test that CLI help works."""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'ArduPilot SITL Parameter Identification' in result.output


def test_cli_version():
    """Test that version flag works."""
    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert result.exit_code == 0
    assert '0.1.0' in result.output


def test_cli_missing_required():
    """Test that missing required arguments are caught."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert 'Missing option' in result.output or 'Error' in result.output


def test_cli_invalid_log_path():
    """Test that invalid log path is caught."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "output.parm"
        result = runner.invoke(main, [
            '--log', 'nonexistent.bin',
            '--frame', 'quad_x',
            '--mass', '1.2',
            '--output', str(output)
        ])
        assert result.exit_code != 0


def test_cli_invalid_mass():
    """Test that invalid mass values are rejected."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy log file
        log_file = Path(tmpdir) / "test.bin"
        log_file.write_bytes(b'test')
        output = Path(tmpdir) / "output.parm"

        # Test negative mass
        result = runner.invoke(main, [
            '--log', str(log_file),
            '--frame', 'quad_x',
            '--mass', '-1.0',
            '--output', str(output)
        ])
        assert result.exit_code != 0


def test_cli_invalid_frame():
    """Test that invalid frame type is rejected."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.bin"
        log_file.write_bytes(b'test')
        output = Path(tmpdir) / "output.parm"

        result = runner.invoke(main, [
            '--log', str(log_file),
            '--frame', 'invalid_frame',
            '--mass', '1.2',
            '--output', str(output)
        ])
        assert result.exit_code != 0
        assert 'Invalid value' in result.output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
