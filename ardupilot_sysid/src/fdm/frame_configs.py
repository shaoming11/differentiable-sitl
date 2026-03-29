"""
Frame configurations for different multicopter geometries.

Defines motor positions (relative to CG) and rotation directions for standard
multicopter frames used in ArduPilot.
"""

import jax.numpy as jnp
from typing import Dict, Any

# Quad X configuration (most common)
# Motor numbering follows ArduPilot convention:
# 1: Front-right (CW)
# 2: Rear-left (CW)
# 3: Rear-right (CCW)
# 4: Front-left (CCW)
QUAD_X = {
    'motor_positions': jnp.array([
        [ 1.0,  1.0, 0.0],  # Motor 1: Front-right
        [-1.0, -1.0, 0.0],  # Motor 2: Rear-left
        [ 1.0, -1.0, 0.0],  # Motor 3: Rear-right
        [-1.0,  1.0, 0.0],  # Motor 4: Front-left
    ]) * 0.165,  # Typical arm length: 165mm (scaled to meters)
    'motor_directions': jnp.array([1, 1, -1, -1])  # +1=CW, -1=CCW
}

# Hexa X configuration
# 6 motors in X pattern
HEXA_X = {
    'motor_positions': jnp.array([
        [ 1.0,  0.0, 0.0],  # Motor 1: Front
        [ 0.5,  0.866, 0.0],  # Motor 2: Front-left (60 degrees)
        [-0.5,  0.866, 0.0],  # Motor 3: Rear-left (120 degrees)
        [-1.0,  0.0, 0.0],  # Motor 4: Rear
        [-0.5, -0.866, 0.0],  # Motor 5: Rear-right (240 degrees)
        [ 0.5, -0.866, 0.0],  # Motor 6: Front-right (300 degrees)
    ]) * 0.2,  # Typical arm length: 200mm
    'motor_directions': jnp.array([1, -1, 1, -1, 1, -1])  # Alternating
}

# Quad + configuration
QUAD_PLUS = {
    'motor_positions': jnp.array([
        [ 1.0,  0.0, 0.0],  # Motor 1: Front
        [ 0.0, -1.0, 0.0],  # Motor 2: Right
        [-1.0,  0.0, 0.0],  # Motor 3: Rear
        [ 0.0,  1.0, 0.0],  # Motor 4: Left
    ]) * 0.165,
    'motor_directions': jnp.array([1, -1, 1, -1])
}

# Octocopter X configuration
OCTO_X = {
    'motor_positions': jnp.array([
        [ 1.0,  1.0, 0.0],  # Motor 1: Front-right
        [-1.0, -1.0, 0.0],  # Motor 2: Rear-left
        [ 1.0, -1.0, 0.0],  # Motor 3: Rear-right
        [-1.0,  1.0, 0.0],  # Motor 4: Front-left
        [ 1.0,  0.5, 0.0],  # Motor 5: Front-right-inner
        [-1.0, -0.5, 0.0],  # Motor 6: Rear-left-inner
        [ 1.0, -0.5, 0.0],  # Motor 7: Rear-right-inner
        [-1.0,  0.5, 0.0],  # Motor 8: Front-left-inner
    ]) * 0.22,  # Larger frame
    'motor_directions': jnp.array([1, 1, -1, -1, -1, -1, 1, 1])
}

# Central registry of all frame configurations
FRAME_CONFIGS: Dict[str, Dict[str, Any]] = {
    'quad_x': QUAD_X,
    'quad_plus': QUAD_PLUS,
    'hexa_x': HEXA_X,
    'octo_x': OCTO_X,
}


def get_frame_config(frame_type: str) -> Dict[str, jnp.ndarray]:
    """
    Retrieve frame configuration by name.

    Args:
        frame_type: Frame type identifier (e.g., 'quad_x', 'hexa_x')

    Returns:
        Dictionary with 'motor_positions' and 'motor_directions' arrays

    Raises:
        ValueError: If frame_type is not recognized
    """
    if frame_type not in FRAME_CONFIGS:
        available = ', '.join(FRAME_CONFIGS.keys())
        raise ValueError(
            f"Unknown frame type '{frame_type}'. Available: {available}"
        )
    return FRAME_CONFIGS[frame_type]


def validate_frame_config(config: Dict[str, jnp.ndarray]) -> None:
    """
    Validate that a frame configuration has correct structure.

    Args:
        config: Frame configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    if 'motor_positions' not in config:
        raise ValueError("Frame config missing 'motor_positions'")
    if 'motor_directions' not in config:
        raise ValueError("Frame config missing 'motor_directions'")

    positions = config['motor_positions']
    directions = config['motor_directions']

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"motor_positions must be (N, 3), got {positions.shape}"
        )

    if directions.ndim != 1:
        raise ValueError(
            f"motor_directions must be 1D, got {directions.shape}"
        )

    if positions.shape[0] != directions.shape[0]:
        raise ValueError(
            f"Number of motors mismatch: {positions.shape[0]} positions, "
            f"{directions.shape[0]} directions"
        )

    # Check that directions are only +1 or -1
    if not jnp.all((directions == 1) | (directions == -1)):
        raise ValueError("motor_directions must be +1 (CW) or -1 (CCW)")
