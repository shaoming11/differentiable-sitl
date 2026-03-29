# RTS Smoother for State Estimation

This module implements the Rauch-Tung-Striebel (RTS) smoother for optimal state estimation in the ArduPilot SITL parameter identification pipeline.

## Overview

### Why RTS Smoothing?

- **ArduPilot's EKF is causal**: It only uses data up to time `t` (past observations)
- **RTS smoother is non-causal**: Uses the FULL log (forward + backward) for minimum variance estimates
- **Better state estimates** → better parameter identification
- **Provides per-timestep covariance** for weighting in optimization

### Algorithm

The RTS smoother operates in two passes:

1. **Forward pass (UKF)**: Process measurements chronologically, saving intermediate estimates
2. **Backward pass (RTS)**: Refine estimates using future information

### State Vector

The 13-dimensional state vector represents:

```
x = [qw, qx, qy, qz,    # Quaternion (4) - attitude
     vx, vy, vz,        # Velocity in world/NED frame (3)
     wx, wy, wz,        # Angular velocity in body frame (3)
     px, py, pz]        # Position in world/NED frame (3)
```

### Observation Models

Three sensor types are supported:

- **IMU**: `[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]` (6D) @ 400 Hz
- **GPS**: `[vel_n, vel_e, vel_d, lat, lng, alt]` (6D) @ 10 Hz
- **Barometer**: `[altitude]` (1D) @ 10 Hz

## Module Structure

### Files

```
smoother/
├── __init__.py           # Module exports
├── state_space.py        # State transition and observation models
├── ukf.py                # Unscented Kalman Filter implementation
├── rts.py                # RTS smoother backward pass
└── README.md             # This file
```

### Key Classes

#### `UnscentedKalmanFilter` (`ukf.py`)

Implements the Unscented Kalman Filter for nonlinear state estimation.

**Key advantages over EKF:**
- No Jacobian calculations required
- More accurate for highly nonlinear dynamics
- Captures mean and covariance to second order (Taylor expansion)

**Tuning parameters:**
- `alpha`: Sigma point spread (typically `1e-3` to `1`)
- `beta`: Prior distribution knowledge (`2.0` for Gaussian)
- `kappa`: Secondary scaling (`0` or `3-n`)

**Main methods:**
- `predict()`: Propagate state forward in time
- `update()`: Incorporate measurement
- `forward_pass()`: Process full dataset

#### `RTSSmoother` (`rts.py`)

Implements the backward smoothing pass.

**Main methods:**
- `backward_pass()`: Refine forward estimates using future data
- `extract_angular_velocity()`: Get angular velocity trajectory
- `compare_forward_vs_smoothed()`: Compute improvement metrics

#### State Space Models (`state_space.py`)

**Dynamics:**
- `state_transition_model()`: Constant angular velocity model with quaternion integration
- `imu_observation_model()`: Maps state to IMU measurements
- `gps_observation_model()`: Maps state to GPS measurements
- `baro_observation_model()`: Maps state to barometer measurements

**Utilities:**
- `quaternion_to_euler()`: Convert quaternion to Euler angles
- `euler_to_quaternion()`: Convert Euler angles to quaternion
- `rotate_vector_body_to_world()`: Frame transformation
- `rotate_vector_world_to_body()`: Inverse frame transformation

## Usage

### Basic Example

```python
import numpy as np
from ardupilot_sysid.src.smoother import (
    UnscentedKalmanFilter,
    RTSSmoother
)

# Initial state and covariance
x_init = np.array([1, 0, 0, 0,  # quaternion (identity)
                   0, 0, 0,     # velocity
                   0, 0, 0,     # angular velocity
                   0, 0, 0])    # position
P_init = np.eye(13) * 0.1

# Process and measurement noise covariances
Q = np.eye(13) * 1e-4       # Process noise
R_imu = np.eye(6) * 1e-3    # IMU noise
R_gps = np.eye(6) * 0.5     # GPS noise
R_baro = np.array([[0.5]])  # Barometer noise

# Measurements: dict with 'imu', 'gps', 'baro' DataFrames
# Each DataFrame must have a 'timestamp' column
measurements = {
    'imu': imu_dataframe,
    'gps': gps_dataframe,
    'baro': baro_dataframe
}

# Forward pass
ukf = UnscentedKalmanFilter(state_dim=13)
forward_states = ukf.forward_pass(
    x_init, P_init,
    measurements,
    dt=0.0025,  # 400 Hz
    Q=Q,
    R_imu=R_imu,
    R_gps=R_gps,
    R_baro=R_baro
)

# Backward pass
smoother = RTSSmoother()
smoothed_states = smoother.backward_pass(forward_states)

# Extract results
omega = smoother.extract_angular_velocity(smoothed_states)
timestamps = smoother.get_timestamps(smoothed_states)

# Compare forward vs smoothed
metrics = smoother.compare_forward_vs_smoothed(
    forward_states,
    smoothed_states
)
print(f"Variance reduction: {metrics['variance_reduction']:.2f}%")
```

### Full Example

See `example_smoother_usage.py` in the project root for a complete demonstration with visualization.

```bash
python3 example_smoother_usage.py
```

## Tuning Guidelines

### Process Noise Covariance (`Q`)

Represents uncertainty in the dynamics model:

- **Quaternion**: Low noise (`1e-6`) - assuming smooth attitude changes
- **Velocity**: Medium noise (`1e-3`) - no aerodynamic model yet
- **Angular velocity**: Medium-high noise (`1e-3` to `1e-2`) - piecewise constant assumption
- **Position**: Low-medium noise (`1e-4`) - integrated from velocity

### Measurement Noise Covariances (`R_*`)

Based on sensor specifications:

- **IMU accelerometer**: `1e-2` (m/s²)²
- **IMU gyroscope**: `1e-3` (rad/s)²
- **GPS velocity**: `0.5` (m/s)²
- **GPS position**: `1e-6` to `1.0` (degrees/meters)²
- **Barometer altitude**: `0.5` (m)²

### Initial Covariance (`P_init`)

High uncertainty for unknown initial state:

```python
P_init = np.diag([
    1.0, 1.0, 1.0, 1.0,  # quaternion
    10.0, 10.0, 10.0,    # velocity
    1.0, 1.0, 1.0,       # angular velocity
    100.0, 100.0, 100.0  # position
])
```

### UKF Parameters

Default values work well for most cases:

- `alpha=1e-3`: Tight sigma point distribution
- `beta=2.0`: Optimal for Gaussian
- `kappa=0.0`: Standard choice

## Integration with Parameter Identification

The smoothed state estimates feed into the optimization pipeline:

1. **RTS Smoother** → smoothed state trajectory with covariances
2. **Differentiable FDM** → predicted dynamics from parameters
3. **Loss function** → weighted by inverse covariance from smoother
4. **JAX autodiff** → compute parameter gradients
5. **Optimizer** → update parameter estimates

## Testing

Run the test suite:

```bash
python3 -m pytest tests/test_smoother.py -v
```

### Test Coverage

- ✅ State transition models
- ✅ Observation models
- ✅ Quaternion conversions and normalization
- ✅ UKF sigma point generation
- ✅ UKF predict/update steps
- ✅ RTS backward pass
- ✅ Variance reduction verification
- ✅ Full integration pipeline
- ✅ Quaternion normalization preservation

## Known Limitations

1. **Simplified dynamics**: Constant velocity/angular velocity model
   - No aerodynamic forces yet
   - Will be refined with FDM integration

2. **No EKF comparison**: Current implementation only uses UKF
   - Could add EKF baseline for comparison

3. **Local linearization**: Position/velocity in NED frame
   - For long flights, need geodetic coordinate handling

4. **Tuning sensitivity**: Performance depends on noise covariances
   - Automatic tuning (e.g., EM algorithm) could be added

## Future Enhancements

1. **Adaptive noise estimation**: Learn Q and R from data
2. **Constrained smoothing**: Enforce physical constraints (e.g., altitude > 0)
3. **Multiple UKF passes**: Use different tuning for different flight phases
4. **Magnetometer fusion**: Add compass measurements
5. **Optical flow**: Integrate vision-based velocity estimates

## References

1. Rauch, H. E., Tung, F., & Striebel, C. T. (1965). "Maximum likelihood estimates of linear dynamic systems." *AIAA Journal*.

2. Julier, S. J., & Uhlmann, J. K. (2004). "Unscented filtering and nonlinear estimation." *Proceedings of the IEEE*.

3. Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press.

4. ArduPilot EKF Documentation: https://ardupilot.org/dev/docs/extended-kalman-filter.html

## Authors

- Implementation: Claude (Anthropic) via Claude Code
- Architecture: ArduPilot SITL Parameter Identification Project

## License

Same as parent project (ArduPilot SITL).
