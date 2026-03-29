# FDM Quick Start Guide

**5-minute guide to using the differentiable flight dynamics model.**

## Installation

```bash
pip install jax jaxlib numpy
```

## Basic Usage

### 1. Simple Simulation

```python
import jax.numpy as jnp
from ardupilot_sysid.src.fdm import get_frame_config, fdm_step

# Get quadcopter frame
frame = get_frame_config('quad_x')

# Define parameters
params = {
    'mass': 1.5,  # kg
    'kT': 1e-5,   # thrust coefficient
    'kQ': 1e-6,   # torque coefficient
    'inertia': jnp.array([0.01, 0.01, 0.02]),
    'c_drag': 1e-4,
    'pwm_to_omega_poly': jnp.array([0.0, 800.0, 200.0]),
    'motor_positions': frame['motor_positions'],
    'motor_directions': frame['motor_directions'],
}

# Initial state: [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
state = jnp.array([1, 0, 0, 0,  # level orientation
                   0, 0, 0,      # zero velocity
                   0, 0, 0])     # zero angular velocity

# Motor commands (normalized PWM: 0-1)
pwm = jnp.ones(4) * 0.5  # 50% throttle

# Simulate one timestep
next_state = fdm_step(state, pwm, params, dt=0.01)
```

### 2. Trajectory Simulation

```python
from ardupilot_sysid.src.fdm import rollout

# Create PWM sequence (1000 steps = 10 seconds at 100 Hz)
pwm_sequence = jnp.ones((1000, 4)) * 0.5

# Run full simulation
trajectory = rollout(state, pwm_sequence, params, dt=0.01)
# Returns: (1000, 10) array of states
```

### 3. Computing Gradients

```python
import jax

# Define loss function
def my_loss(mass):
    params_mod = params.copy()
    params_mod['mass'] = mass
    traj = rollout(state, pwm_sequence, params_mod, dt=0.01)
    return jnp.sum(traj[:, 6]**2)  # minimize vertical velocity

# Compute gradient
grad_fn = jax.grad(my_loss)
gradient = grad_fn(params['mass'])
print(f"Gradient: {gradient}")
```

### 4. Parameter Optimization

```python
import optax

# Setup optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params['mass'])

# Optimization loop
current_mass = params['mass']
for step in range(100):
    grad = grad_fn(current_mass)
    updates, opt_state = optimizer.update(grad, opt_state)
    current_mass = optax.apply_updates(current_mass, updates)

    if step % 10 == 0:
        loss = my_loss(current_mass)
        print(f"Step {step}: Loss = {loss:.6f}")
```

## State Vector Layout

```
Index | Name | Description              | Units
------|------|--------------------------|-------
0-3   | q    | Quaternion (qw,qx,qy,qz) | -
4-6   | v    | Velocity (vx,vy,vz)      | m/s
7-9   | ω    | Angular vel (wx,wy,wz)   | rad/s
```

## Parameter Dictionary

```python
params = {
    # Physical properties
    'mass': 1.5,                              # kg
    'inertia': jnp.array([Ixx, Iyy, Izz]),   # kg·m²

    # Motor/propeller
    'kT': 1e-5,                               # N/(rad/s)²
    'kQ': 1e-6,                               # N·m/(rad/s)²
    'pwm_to_omega_poly': jnp.array([a0, a1, a2]),  # rad/s coefficients

    # Aerodynamics
    'c_drag': 1e-4,                           # rotational drag

    # Frame geometry
    'motor_positions': (N, 3) array,          # meters from CG
    'motor_directions': (N,) array,           # +1=CW, -1=CCW
}
```

## Typical Parameter Ranges

| Parameter | Symbol | Typical Range | Units |
|-----------|--------|---------------|-------|
| Mass | m | 0.5 - 5.0 | kg |
| Thrust coeff | kT | 1e-6 - 1e-4 | N/(rad/s)² |
| Torque coeff | kQ | 1e-7 - 1e-5 | N·m/(rad/s)² |
| Inertia | I | 0.001 - 0.1 | kg·m² |
| Drag | c_drag | 1e-5 - 1e-3 | - |

## Available Frames

```python
# Get frame configuration
frame = get_frame_config('quad_x')      # 4 motors, X configuration
frame = get_frame_config('quad_plus')   # 4 motors, + configuration
frame = get_frame_config('hexa_x')      # 6 motors, X configuration
frame = get_frame_config('octo_x')      # 8 motors, X configuration
```

## Utilities

### Estimate Hover PWM

```python
from ardupilot_sysid.src.fdm import estimate_hover_pwm

hover_pwm = estimate_hover_pwm(
    mass=1.5,
    kT=1e-5,
    poly_coeffs=jnp.array([0.0, 800.0, 200.0]),
    num_motors=4
)
print(f"Hover PWM: {hover_pwm:.3f}")
```

### Convert PWM

```python
from ardupilot_sysid.src.fdm import normalize_pwm, denormalize_pwm

# Microseconds → normalized
pwm_normalized = normalize_pwm(jnp.array([1500.0, 1600.0]))
# Returns: [0.5, 0.6]

# Normalized → microseconds
pwm_us = denormalize_pwm(jnp.array([0.5, 0.6]))
# Returns: [1500.0, 1600.0]
```

### Validate State

```python
from ardupilot_sysid.src.fdm import validate_state

is_valid = validate_state(state)
if not is_valid:
    print("Warning: Invalid state detected!")
```

## Performance Tips

1. **First call is slow** (JIT compilation): Cache compiled functions
2. **Use batching** for multiple simulations: `jax.vmap(rollout, ...)`
3. **Avoid Python loops**: Let JAX handle iteration with `lax.scan`
4. **Pre-allocate arrays**: Use fixed-size arrays when possible

## Common Patterns

### Batch Processing

```python
# Simulate multiple initial conditions
initial_states = jnp.stack([state1, state2, state3])
trajectories = jax.vmap(
    lambda s: rollout(s, pwm_sequence, params, dt=0.01)
)(initial_states)
# Returns: (3, 1000, 10) array
```

### Custom Loss Functions

```python
def tracking_loss(params_to_opt, target_trajectory, pwm_seq):
    """Loss = MSE between simulated and target trajectory."""
    predicted = rollout(state_init, pwm_seq, params_to_opt, dt=0.01)

    # Weight different state components
    weights = jnp.array([1, 1, 1, 1,  # quaternion
                         1, 1, 2,       # velocity (weight Z more)
                         5, 5, 5])      # angular velocity (high priority)

    residuals = (predicted - target_trajectory) * weights
    return jnp.mean(residuals**2)
```

## Troubleshooting

### NaN or Inf in Output

**Problem**: `jnp.any(jnp.isnan(trajectory))`

**Solutions**:
1. Check quaternion normalization: `jnp.linalg.norm(state[0:4]) ≈ 1.0`
2. Reduce timestep: Try `dt=0.005` instead of `dt=0.01`
3. Check parameter values: Very large/small values can cause instability

### Slow Performance

**Problem**: Simulation taking too long

**Solutions**:
1. Make sure first call completed (JIT compilation)
2. Use `@jax.jit` on custom functions
3. Avoid `.copy()` in tight loops (JAX handles immutability)
4. Profile with `jax.profiler.trace()`

### Gradient Issues

**Problem**: `check_grads()` fails or gradients are NaN

**Solutions**:
1. Check for discontinuities (use `jnp.where` not `if`)
2. Avoid in-place operations
3. Ensure all operations are JAX primitives
4. Test on smaller sequences first

## Examples

Full working examples in `examples/demo_fdm.py`:

```bash
python examples/demo_fdm.py
```

## Testing

Run test suite:

```bash
# All tests
pytest tests/test_fdm.py -v

# Gradient tests only
pytest tests/test_fdm.py::TestGradients -v

# Quick validation
python validate_fdm.py
```

## Further Reading

- Full documentation: `ardupilot_sysid/src/fdm/README.md`
- Implementation summary: `FDM_IMPLEMENTATION_SUMMARY.md`
- Test file: `tests/test_fdm.py` (good examples)
- JAX documentation: https://jax.readthedocs.io/

## Questions?

Check the test suite (`tests/test_fdm.py`) for working examples of every feature.
