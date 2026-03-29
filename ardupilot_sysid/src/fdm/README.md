# Differentiable Flight Dynamics Model (FDM) in JAX

Pure JAX implementation of multicopter flight dynamics with exact gradients via automatic differentiation for system identification.

## Overview

This module implements a physics-based multicopter simulator optimized for gradient-based parameter optimization. All computations are written in JAX, enabling:

- **Exact gradients** via automatic differentiation (no finite differences)
- **JIT compilation** for near-C performance
- **Vectorization** for batch processing
- **GPU acceleration** (when available)

## Architecture

```
fdm/
├── frame_configs.py      # Motor geometries (quad, hexa, octo, etc.)
├── motor_model.py        # PWM → thrust/torque conversions
├── multicopter_jax.py    # Core FDM physics engine
└── README.md             # This file
```

## Quick Start

```python
import jax.numpy as jnp
from ardupilot_sysid.src.fdm import (
    get_frame_config,
    fdm_step,
    rollout,
)

# Setup quadcopter
frame = get_frame_config('quad_x')
params = {
    'mass': 1.5,
    'kT': 1e-5,
    'kQ': 1e-6,
    'inertia': jnp.array([0.01, 0.01, 0.02]),
    'c_drag': 1e-4,
    'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),
    'motor_positions': frame['motor_positions'],
    'motor_directions': frame['motor_directions'],
}

# Initial state: level, at rest
state = jnp.array([1, 0, 0, 0,  # quaternion
                   0, 0, 0,      # velocity
                   0, 0, 0])     # angular velocity

# PWM commands (4 motors, 50% throttle)
pwm = jnp.ones(4) * 0.5

# Single timestep
next_state = fdm_step(state, pwm, params, dt=0.01)

# Multi-step rollout
pwm_sequence = jnp.ones((1000, 4)) * 0.5
trajectory = rollout(state, pwm_sequence, params, dt=0.01)
```

## Physics Model

### State Representation

State vector (10 dimensions):
```
[qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
```

- **q** (qw, qx, qy, qz): Unit quaternion (world-to-body rotation)
- **v** (vx, vy, vz): Linear velocity in world frame (m/s)
- **ω** (wx, wy, wz): Angular velocity in body frame (rad/s)

### Thrust Model

Each motor produces thrust proportional to angular velocity squared:

```
F_i = kT · ω_i²
```

where:
- **F_i**: Thrust force in Newtons
- **kT**: Thrust coefficient (N/(rad/s)²)
- **ω_i**: Motor angular velocity (rad/s)

Angular velocity is mapped from PWM via quadratic polynomial:
```
ω = a₀ + a₁·PWM + a₂·PWM²
```

### Torque Model

Total torque on the body consists of three components:

1. **Spatial torques** (moment arm effects):
   ```
   τ_spatial = Σ(r_i × F_i)
   ```
   where r_i is motor position vector from CG.

2. **Reaction torques** (motor drag):
   ```
   τ_reaction = Σ(±kQ · ω_i² · ẑ)
   ```
   Sign depends on rotation direction (+1 for CW, -1 for CCW).

3. **Rotational drag**:
   ```
   τ_drag = -c_drag · ω · |ω|
   ```

### Equations of Motion

**Translational dynamics:**
```
F_total = R(q)ᵀ · F_body - [0, 0, mg]
a = F_total / m
v(t+dt) = v(t) + a·dt
```

**Rotational dynamics (Euler's equation):**
```
I · α = τ_total - ω × (I · ω)
ω(t+dt) = ω(t) + α·dt
```

**Quaternion kinematics:**
```
q̇ = 0.5 · q ⊗ [0, ω]
q(t+dt) = normalize(q(t) + q̇·dt)
```

## Frame Configurations

Supported multicopter frames:

- **quad_x**: X-configuration quadcopter (most common)
- **quad_plus**: +-configuration quadcopter
- **hexa_x**: X-configuration hexacopter
- **octo_x**: X-configuration octocopter

Each frame defines:
- `motor_positions`: (N, 3) array of motor positions relative to CG (meters)
- `motor_directions`: (N,) array of rotation directions (+1=CW, -1=CCW)

## Gradient Computation

The FDM is fully differentiable. Compute gradients w.r.t. any parameter:

```python
import jax

# Define loss function
def loss_fn(params_to_optimize):
    trajectory = rollout(state_init, pwm_sequence, params_to_optimize, dt=0.01)
    # Compare to target trajectory
    residuals = trajectory - target_trajectory
    return jnp.mean(residuals ** 2)

# Compute gradients
grad_fn = jax.grad(loss_fn)
gradients = grad_fn(params)
```

## Optimization

Use JAX optimizers (optax) for parameter identification:

```python
import optax

# Initialize optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params_flat)

# Optimization loop
for step in range(num_steps):
    grads = jax.grad(loss_fn)(params_flat)
    updates, opt_state = optimizer.update(grads, opt_state)
    params_flat = optax.apply_updates(params_flat, updates)
```

## Performance

On typical hardware (M1 Mac / modern CPU):

- **Single step**: ~0.1 ms (compiled)
- **1000 steps** (10 seconds @ 100Hz): ~50-100 ms
- **Gradient computation**: ~2-3x forward pass time

JIT compilation overhead (~1-2 seconds) occurs on first call only.

## Testing

Run comprehensive test suite:

```bash
pytest tests/test_fdm.py -v
```

Tests include:
- Motor model functions
- Frame configurations
- Quaternion operations
- FDM core physics
- Gradient correctness (via `jax.test_util.check_grads`)
- Rollout functionality
- Integration tests

## Examples

See `examples/demo_fdm.py` for a complete demonstration:

```bash
python examples/demo_fdm.py
```

This demonstrates:
1. Setting up a quadcopter model
2. Running trajectory simulation
3. Computing gradients
4. Visualization

## Parameter Reference

### Physical Parameters

| Parameter | Units | Typical Range | Description |
|-----------|-------|---------------|-------------|
| `mass` | kg | 0.5 - 5.0 | Vehicle mass |
| `kT` | N/(rad/s)² | 1e-6 - 1e-4 | Thrust coefficient |
| `kQ` | N·m/(rad/s)² | 1e-7 - 1e-5 | Torque coefficient |
| `inertia[3]` | kg·m² | 0.001 - 0.1 | Ixx, Iyy, Izz |
| `c_drag` | - | 1e-5 - 1e-3 | Rotational drag coefficient |

### PWM Polynomial

`pwm_to_omega_poly = [a₀, a₁, a₂]` defines quadratic mapping:
```
ω = a₀ + a₁·PWM + a₂·PWM²
```

Typical values for standard ESC/motor combinations:
- **a₀**: 0 - 100 rad/s (idle speed)
- **a₁**: 500 - 1500 rad/s (linear term)
- **a₂**: 0 - 500 rad/s (quadratic term)

## Limitations & Future Work

Current limitations:
- Simple Euler integration (consider RK4 for higher accuracy)
- No aerodynamic drag on body
- No ground effect
- Diagonal inertia tensor only

Planned improvements:
- Higher-order integrators
- Aerodynamic drag model
- Wind disturbances
- Ground effect near surface
- Full inertia tensor support

## References

1. Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*. Princeton University Press.
2. Stevens, B. L., Lewis, F. L., & Johnson, E. N. (2015). *Aircraft Control and Simulation* (3rd ed.). Wiley.
3. Bradbury, J., et al. (2018). *JAX: Composable transformations of Python+NumPy programs*. http://github.com/google/jax

## License

Part of the ArduPilot SITL Parameter Identification project.
