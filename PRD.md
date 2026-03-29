# PRD: Differentiable SITL Model Generation from ArduPilot Flight Logs

**Project:** ArduPilot GSoC — Automatic SITL Parameter Identification  
**Owner:** Shao (Altura Robotics / ArduPilot GSoC candidate)  
**Status:** Ready for implementation  
**Target:** Claude Code autonomous implementation

---

## 1. Problem Statement

ArduPilot's SITL (Software-In-The-Loop) simulator uses a Flight Dynamics Model (FDM) whose
parameters — thrust coefficient, drag coefficient, inertia tensor, motor time constant — are
hardcoded generic approximations. For any custom vehicle (racing quad, heavy-lift hex, long-range
fixed-wing), these defaults diverge from reality by meaningful margins. Gains tuned in SITL
oscillate differently on the real vehicle; maneuvers SITL says are feasible may be impossible
with the real thrust-to-weight ratio.

The goal is a CLI tool that ingests a real ArduPilot DataFlash `.bin` log, identifies the FDM
parameters that best explain the observed flight dynamics, and emits a ready-to-use SITL
parameter file.

### Core bottlenecks to solve

| Bottleneck | Naive approach | This project's solution |
|---|---|---|
| No RPM in logs (only PWM) | Ignore or require ESC telemetry | Identify lumped PWM→thrust polynomial; optionally ingest DSHOT telemetry if present |
| Parameter coupling (kT ↔ mass, Ixx ↔ Iyy) | Hope optimizer finds global min | Structured identifiability pre-check + regularization toward physical measurements |
| Poor excitation in calm hover logs | Fail silently or produce garbage params | Per-parameter excitation scoring; warn user; generate an excitation flight profile |
| Timestamp jitter between IMU/GPS/RCOUT | Ignore, accept systematic error | Cross-correlation alignment on known-good signal pairs before optimization |
| Finite-difference Jacobians are slow and noisy | Accept slow convergence | Differentiable FDM in JAX — exact gradients via autodiff |
| No uncertainty estimate on identified params | Output point estimates only | Laplace approximation posterior → confidence intervals per parameter |

---

## 2. Architecture

### 2.1 High-level pipeline

```
[ .bin log file ]
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Parse & Align     │  pymavlink DFReader → pandas DataFrames
│  - Extract IMU/RCOUT/ATT/   │  Resample all streams to 400 Hz grid
│    EKF/GPS/BARO/PARM        │  Cross-correlation timestamp alignment
│  - Detect EKF health flags  │  Segment by EKF innovation < threshold
└────────────┬────────────────┘
             │  aligned, segmented DataFrames
             ▼
┌─────────────────────────────┐
│  Stage 2: RTS Smoother      │  Forward UKF pass → save (x̂, P) at each step
│  - Forward UKF on IMU+GPS+  │  Backward RTS pass → minimum-variance state
│    BARO+MAG                 │  trajectory x̂(t) with covariance P(t)
│  - Backward RTS pass        │
└────────────┬────────────────┘
             │  smooth state trajectory
             ▼
┌─────────────────────────────┐
│  Stage 3: Excitation Check  │  Compute Fisher Information Matrix (FIM)
│  - Per-segment FIM analysis │  diagonal → per-parameter excitability score
│  - Warn on low-rank dims    │  Suggest which maneuvers are missing
│  - Select best segments     │
└────────────┬────────────────┘
             │  selected segments + scores
             ▼
┌─────────────────────────────┐
│  Stage 4: Differentiable    │  JAX implementation of ArduPilot FDM
│  Optimization               │  Levenberg-Marquardt with exact Jacobians
│  - Nonlinear least squares  │  MAP with Gaussian priors on all params
│  - Laplace posterior        │  Posterior covariance → confidence intervals
└────────────┬────────────────┘
             │  θ̂ + covariance
             ▼
┌─────────────────────────────┐
│  Stage 5: Validation &      │  Hold-out rollout on unused flight segment
│  Output                     │  RMSE vs. IMU/ATT ground truth
│  - Rollout comparison       │  Emit .parm file + JSON report
│  - Write SITL .parm file    │
└─────────────────────────────┘
```

### 2.2 Differentiable FDM (the core innovation)

Implement the ArduPilot multicopter FDM as a pure JAX function so that `jax.grad` and
`jax.jacfwd` produce exact Jacobians for free. This eliminates finite-difference approximation
error and makes optimization dramatically faster.

```python
# src/fdm/multicopter_jax.py

import jax
import jax.numpy as jnp

def fdm_step(state: jnp.ndarray, pwm: jnp.ndarray, params: dict, dt: float) -> jnp.ndarray:
    """
    One step of the multicopter FDM.
    
    state: [qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]  (quaternion + velocities + rates)
    pwm:   [pwm_1, ..., pwm_N]  (normalized 0–1 per motor)
    params: {
        mass: scalar,
        kT: scalar,             # thrust coefficient (N / (rad/s)^2)
        kQ: scalar,             # torque coefficient
        tau_motor: scalar,      # ESC+motor first-order time constant (s)
        Ixx, Iyy, Izz: scalars, # principal moments of inertia
        c_drag: scalar,         # rotational drag coefficient
        pwm_to_omega_poly: [a0, a1, a2],  # PWM→RPM polynomial coefficients
        motor_positions: (N, 3) array,    # motor arm vectors
        motor_directions: (N,) array,     # +1 CW, -1 CCW
    }
    
    Returns: next_state
    """
    q = state[0:4]
    v = state[4:7]
    omega = state[7:10]
    
    # PWM → angular velocity via identified polynomial
    poly = params['pwm_to_omega_poly']
    omega_motors = poly[0] + poly[1] * pwm + poly[2] * pwm**2  # rad/s per motor
    
    # Thrust per motor: F_i = kT * omega_i^2
    thrusts = params['kT'] * omega_motors**2                    # (N,)
    
    # Total force in body frame (all motors thrust along +z_body)
    F_total_body = jnp.array([0., 0., jnp.sum(thrusts)])
    
    # Rotate to world frame, subtract gravity
    R = quat_to_rotation(q)                                     # (3,3)
    F_world = R @ F_total_body - jnp.array([0., 0., params['mass'] * 9.81])
    accel = F_world / params['mass']                            # (3,)
    
    # Torques from each motor
    torques = jnp.stack([
        jnp.cross(params['motor_positions'][i], jnp.array([0., 0., thrusts[i]]))
        + params['motor_directions'][i] * params['kQ'] * omega_motors[i]**2 * jnp.array([0., 0., 1.])
        for i in range(len(thrusts))
    ]).sum(axis=0)                                              # (3,)
    
    # Rotational drag
    torques = torques - params['c_drag'] * omega * jnp.abs(omega)
    
    # Euler's equation: I * alpha = torques - omega × (I * omega)
    I = jnp.diag(jnp.array([params['Ixx'], params['Iyy'], params['Izz']]))
    alpha = jnp.linalg.solve(I, torques - jnp.cross(omega, I @ omega))
    
    # Integrate
    omega_next = omega + alpha * dt
    q_next = quat_integrate(q, omega, dt)
    v_next = v + accel * dt
    
    return jnp.concatenate([q_next, v_next, omega_next])


# Vectorized rollout — vmapped over time
def rollout(states_init, pwm_sequence, params, dt):
    """Run FDM forward for T steps. Returns predicted state trajectory."""
    def step_fn(carry, pwm_t):
        state = carry
        next_state = fdm_step(state, pwm_t, params, dt)
        return next_state, next_state
    _, predicted = jax.lax.scan(step_fn, states_init, pwm_sequence)
    return predicted


# Loss for optimization
def loss_fn(params_flat, params_template, state_trajectory, pwm_sequence, weights, dt):
    params = unflatten_params(params_flat, params_template)
    predicted = rollout(state_trajectory[0], pwm_sequence[:-1], params, dt)
    residuals = predicted - state_trajectory[1:]
    return jnp.sum(weights * residuals**2)

# Jacobian is now free: jax.jacfwd(loss_fn)(params_flat, ...)
```

### 2.3 RTS Smoother

```python
# src/smoother/rts.py

class RTSSmoother:
    """
    Rauch-Tung-Striebel smoother on top of any UKF implementation.
    
    Forward pass: standard UKF, store (x̂_k|k, P_k|k, x̂_k|k-1, P_k|k-1, F_k) at each step.
    Backward pass: refine estimates using future observations.
    """
    
    def forward_pass(self, imu_data, gps_data, baro_data) -> list[KFState]:
        """Run UKF forward, return all intermediate states."""
        ...
    
    def backward_pass(self, forward_states: list[KFState]) -> list[KFState]:
        """RTS backward pass. Computes smoother gain at each step."""
        smoothed = [None] * len(forward_states)
        smoothed[-1] = forward_states[-1]
        
        for k in reversed(range(len(forward_states) - 1)):
            fwd = forward_states[k]
            # Smoother gain: K_s = P_{k|k} F_k^T P_{k+1|k}^{-1}
            Ks = fwd.P_posterior @ fwd.F.T @ jnp.linalg.inv(forward_states[k+1].P_prior)
            x_smooth = fwd.x_posterior + Ks @ (smoothed[k+1].x_posterior - fwd.x_prior_next)
            P_smooth = fwd.P_posterior + Ks @ (smoothed[k+1].P_posterior - fwd.P_prior_next) @ Ks.T
            smoothed[k] = KFState(x=x_smooth, P=P_smooth)
        
        return smoothed
```

### 2.4 Excitation Scoring via Fisher Information Matrix

Before optimization, check whether the observed flight data can actually identify each parameter.
The Fisher Information Matrix (FIM) diagonal gives the information content about each parameter
given the observed data — a near-zero diagonal entry means that parameter is unidentifiable from
this flight.

```python
# src/analysis/excitation.py

def compute_fim(state_trajectory, pwm_sequence, params_nominal, dt):
    """
    Compute Fisher Information Matrix using the JAX Jacobian of the FDM.
    FIM ≈ J^T W J where J is the Jacobian of predicted states w.r.t. params.
    """
    J = jax.jacfwd(rollout, argnums=2)(state_trajectory[0], pwm_sequence, params_nominal, dt)
    # J shape: (T, state_dim, n_params)
    J_flat = J.reshape(-1, n_params)
    FIM = J_flat.T @ W @ J_flat  # (n_params, n_params)
    return FIM

def excitation_report(FIM, param_names, threshold=1e-4):
    """
    Returns dict mapping param_name → excitability score and warning flag.
    Score = normalized FIM diagonal entry. Below threshold → 'poorly excited'.
    """
    diag = jnp.diag(FIM)
    scores = diag / jnp.max(diag)
    return {
        name: {'score': float(scores[i]), 'excited': scores[i] > threshold}
        for i, name in enumerate(param_names)
    }

def suggest_maneuvers(excitation_report: dict) -> list[str]:
    """Map unexcited parameters to the maneuver that would excite them."""
    PARAM_TO_MANEUVER = {
        'Ixx': 'Rapid roll doublet: ±45° roll at ~2 Hz for 5 s',
        'Iyy': 'Rapid pitch doublet: ±30° pitch at ~2 Hz for 5 s',
        'Izz': 'Yaw spin: sustained 360°/s yaw rate for 3 s',
        'kT':  'Hover at 3 different altitudes with 5 s each + throttle steps',
        'kQ':  'Yaw doublet: alternate yaw left/right at max rate',
        'c_drag': 'Fast forward flight (>10 m/s) segment',
        'tau_motor': 'Rapid throttle steps from 30% to 70% and back',
    }
    suggestions = []
    for param, result in excitation_report.items():
        if not result['excited'] and param in PARAM_TO_MANEUVER:
            suggestions.append(f"[{param}] {PARAM_TO_MANEUVER[param]}")
    return suggestions
```

### 2.5 MAP Optimization with Laplace Posterior

```python
# src/optimizer/map_optimizer.py

@dataclass
class PhysicalPriors:
    """User-measured physical properties as Gaussian priors."""
    mass_kg: tuple[float, float]            # (mean, std)
    arm_length_m: tuple[float, float]       # informs inertia prior
    prop_diameter_in: tuple[float, float]   # informs kT prior
    motor_kv: tuple[float, float]           # informs PWM→RPM prior

class MAPOptimizer:
    def __init__(self, priors: PhysicalPriors):
        self.priors = priors
    
    def build_prior_loss(self, params_flat):
        """Gaussian prior penalty on each parameter."""
        prior_means, prior_stds = self._priors_from_physics()
        return jnp.sum(((params_flat - prior_means) / prior_stds)**2)
    
    def optimize(self, state_trajectory, pwm_sequence, dt, max_iter=500):
        """
        Minimize: data_loss + lambda * prior_loss
        Using Levenberg-Marquardt with JAX exact Jacobians.
        Returns: (params_optimal, posterior_covariance)
        """
        def total_loss(params_flat):
            data_loss = loss_fn(params_flat, ...)
            prior_loss = self.build_prior_loss(params_flat)
            return data_loss + 0.1 * prior_loss
        
        # LM optimization loop
        params = self._initialize_from_priors()
        for i in range(max_iter):
            grad = jax.grad(total_loss)(params)
            hessian = jax.hessian(total_loss)(params)
            # LM update: (H + λI) Δθ = -g
            delta = jnp.linalg.solve(hessian + self.lm_lambda * jnp.eye(len(params)), -grad)
            params = params + delta
            self._update_lm_lambda(total_loss, params, delta)
        
        # Laplace approximation: posterior covariance = H^{-1} at optimum
        H_opt = jax.hessian(total_loss)(params)
        posterior_cov = jnp.linalg.inv(H_opt)
        
        return params, posterior_cov
    
    def confidence_intervals(self, params, posterior_cov, alpha=0.95):
        """95% confidence intervals per parameter from Laplace posterior."""
        z = 1.96  # for 95%
        stds = jnp.sqrt(jnp.diag(posterior_cov))
        return {
            name: (float(params[i] - z * stds[i]), float(params[i] + z * stds[i]))
            for i, name in enumerate(PARAM_NAMES)
        }
```

### 2.6 Timestamp Cross-Correlation Alignment

```python
# src/preprocessing/align.py

def align_timestamps(imu_df, gps_df, rcout_df):
    """
    Align sensor streams by cross-correlating known-good signal pairs.
    
    Strategy: The integrated IMU velocity (dead-reckoning) and GPS velocity
    are measuring the same physical quantity with different latencies.
    Cross-correlate vx_imu_integrated vs vx_gps to find GPS latency offset.
    Then apply the same offset to RCOUT (ESC command latency is separate,
    handled by tau_motor in the FDM).
    """
    # Resample GPS velocity to IMU grid via linear interpolation
    gps_vx_resampled = np.interp(imu_df.timestamp, gps_df.timestamp, gps_df.vx)
    
    # Integrate IMU accel to get velocity estimate
    imu_vx_integrated = cumtrapz(imu_df.ax, imu_df.timestamp, initial=0)
    
    # Cross-correlate: peak lag = GPS hardware latency
    correlation = np.correlate(imu_vx_integrated, gps_vx_resampled, mode='full')
    lag_samples = np.argmax(correlation) - len(imu_vx_integrated) + 1
    lag_seconds = lag_samples / IMU_RATE_HZ
    
    # Shift GPS timestamps to compensate
    gps_df['timestamp'] = gps_df['timestamp'] - lag_seconds
    return imu_df, gps_df, rcout_df, {'gps_latency_ms': lag_seconds * 1000}
```

---

## 3. Module Structure

```
ardupilot_sysid/
├── src/
│   ├── parser/
│   │   ├── dflog_reader.py        # pymavlink DFReader wrapper → DataFrames
│   │   ├── message_types.py       # IMU, RCOUT, ATT, EKF, GPS, BARO, PARM constants
│   │   └── ekf_health.py          # EKF innovation threshold filtering
│   ├── preprocessing/
│   │   ├── align.py               # Cross-correlation timestamp alignment
│   │   ├── resample.py            # Multi-rate → 400 Hz grid
│   │   └── segment.py             # Split log into EKF-healthy segments
│   ├── smoother/
│   │   ├── ukf.py                 # Unscented Kalman Filter (forward pass)
│   │   ├── rts.py                 # Rauch-Tung-Striebel backward pass
│   │   └── state_space.py         # State/observation models for smoother
│   ├── fdm/
│   │   ├── multicopter_jax.py     # Differentiable FDM (JAX)
│   │   ├── fixed_wing_jax.py      # Fixed-wing FDM (JAX) — v2 scope
│   │   ├── motor_model.py         # PWM→thrust polynomial model
│   │   └── frame_configs.py       # Motor positions for X4, X6, X8, H, Y6 frames
│   ├── analysis/
│   │   ├── excitation.py          # FIM-based excitation scoring
│   │   └── identifiability.py     # Structural identifiability checks
│   ├── optimizer/
│   │   ├── map_optimizer.py       # MAP optimization + Laplace posterior
│   │   ├── priors.py              # Physical prior distributions
│   │   └── bounds.py              # Hard physical bounds on each parameter
│   ├── validation/
│   │   ├── rollout.py             # Hold-out segment rollout comparison
│   │   └── metrics.py             # RMSE, MAE per state dimension
│   └── output/
│       ├── parm_writer.py         # Write ArduPilot .parm file
│       └── report.py              # JSON + human-readable report
├── cli/
│   └── sysid.py                   # CLI entry point (argparse)
├── tests/
│   ├── test_fdm.py                # FDM unit tests (known inputs → known outputs)
│   ├── test_smoother.py           # RTS smoother on synthetic data
│   ├── test_optimizer.py          # Recovery of known params from synthetic logs
│   └── fixtures/                  # Sample .bin log files for testing
├── notebooks/
│   └── explore_log.ipynb          # Interactive exploration notebook
├── requirements.txt
└── README.md
```

---

## 4. CLI Interface

```bash
# Basic usage — identify parameters from a flight log
python -m ardupilot_sysid.cli.sysid \
    --log flight_2024_01_15.bin \
    --frame quad_x \
    --mass 1.2 \              # measured vehicle mass in kg (Gaussian prior mean)
    --mass-std 0.05 \         # measurement uncertainty in kg
    --arm-length 0.165 \      # motor-to-center distance in meters
    --output sitl_params.parm

# With optional physical priors for tighter regularization
python -m ardupilot_sysid.cli.sysid \
    --log flight.bin \
    --frame quad_x \
    --mass 1.2 --mass-std 0.05 \
    --prop-diameter 5.1 \
    --motor-kv 2300 \
    --output sitl_params.parm \
    --report report.json \
    --verbose

# Generate an excitation flight profile if log is under-excited
python -m ardupilot_sysid.cli.sysid \
    --log flight.bin \
    --frame quad_x \
    --excitation-check-only

# Output example:
# [PARSE]  Loaded 8.3 min flight, 199,200 IMU samples
# [ALIGN]  GPS latency: 143 ms, RCOUT latency: 0 ms (reference)
# [SEGMENT] 3 EKF-healthy segments found (4.1 min usable)
# [SMOOTH]  RTS smoother converged in 2 forward + backward passes
# [EXCITE]  Parameter excitation scores:
#            kT:      0.92  ✓
#            kQ:      0.71  ✓
#            Ixx:     0.84  ✓
#            Iyy:     0.79  ✓
#            Izz:     0.23  ⚠ LOW — add yaw doublet maneuver
#            tau_motor: 0.61 ✓
#            c_drag:  0.18  ⚠ LOW — add forward flight segment
# [OPTIM]  Converged in 87 iterations. Final loss: 2.34e-4
# [VALID]  Hold-out RMSE — roll: 0.8°, pitch: 0.6°, yaw: 1.2°, vz: 0.04 m/s
# [OUTPUT] Written: sitl_params.parm, report.json
```

---

## 5. Output Format

### 5.1 SITL Parameter File (`.parm`)

```
# Generated by ardupilot-sysid v0.1.0
# Source log: flight_2024_01_15.bin
# Vehicle: quad_x, 4 motors
# Identification date: 2024-01-16T09:23:11Z
# Validation RMSE: roll=0.8deg pitch=0.6deg yaw=1.2deg vz=0.04m/s

SIM_MASS,1.187
SIM_MOI_X,0.0089
SIM_MOI_Y,0.0091
SIM_MOI_Z,0.0163
SIM_THST_MAX,14.2
SIM_THST_EXPO,0.62
SIM_MOT_DRAG,0.0034
SIM_ESC_TCONST,0.047
SIM_WIND_SPD,0
SIM_WIND_DIR,0
```

### 5.2 JSON Report

```json
{
  "metadata": {
    "log_file": "flight_2024_01_15.bin",
    "frame_type": "quad_x",
    "flight_duration_s": 498,
    "usable_duration_s": 246,
    "sysid_version": "0.1.0"
  },
  "identified_parameters": {
    "mass_kg":       {"value": 1.187, "ci_95": [1.162, 1.212]},
    "Ixx_kgm2":      {"value": 0.0089, "ci_95": [0.0081, 0.0097]},
    "Iyy_kgm2":      {"value": 0.0091, "ci_95": [0.0083, 0.0099]},
    "Izz_kgm2":      {"value": 0.0163, "ci_95": [0.0131, 0.0195]},
    "kT_N_per_rad2s": {"value": 4.2e-6, "ci_95": [3.9e-6, 4.5e-6]},
    "kQ_Nm_per_rad2s": {"value": 6.1e-8, "ci_95": [5.5e-8, 6.7e-8]},
    "tau_motor_s":   {"value": 0.047, "ci_95": [0.038, 0.056]},
    "c_drag":        {"value": 0.0034, "ci_95": [0.0021, 0.0047]}
  },
  "excitation_scores": {
    "kT": 0.92, "kQ": 0.71, "Ixx": 0.84, "Iyy": 0.79,
    "Izz": 0.23, "tau_motor": 0.61, "c_drag": 0.18
  },
  "warnings": [
    "Izz poorly excited (score=0.23). Add yaw doublet maneuver.",
    "c_drag poorly excited (score=0.18). Add forward flight segment >10 m/s."
  ],
  "validation": {
    "hold_out_segment_s": 45,
    "rmse": {"roll_deg": 0.8, "pitch_deg": 0.6, "yaw_deg": 1.2, "vz_ms": 0.04}
  },
  "preprocessing": {
    "gps_latency_ms": 143,
    "n_ekf_segments": 3,
    "timestamp_jitter_ms": 2.1
  }
}
```

---

## 6. Implementation Sequence

### Phase 1: Parser + Preprocessor (Week 1–2)
- [ ] `dflog_reader.py` — ingest `.bin`, extract all relevant message types to DataFrames
- [ ] `align.py` — cross-correlation timestamp alignment (IMU vs GPS velocity)
- [ ] `resample.py` — multi-rate → 400 Hz interpolation
- [ ] `ekf_health.py` — segment by EKF innovation ratio < 1.0
- [ ] Unit tests with a real `.bin` log fixture

### Phase 2: RTS Smoother (Week 3)
- [ ] `ukf.py` — UKF forward pass on IMU/GPS/BARO state space
- [ ] `rts.py` — backward smoothing pass
- [ ] Validate: smoothed angular velocity should match `ATT.GyrX/Y/Z` from log

### Phase 3: Differentiable FDM (Week 4–5)
- [ ] `multicopter_jax.py` — full JAX FDM for quad_x frame
- [ ] `motor_model.py` — PWM→thrust polynomial (degree 2)
- [ ] `frame_configs.py` — motor positions for X4, X6, Y6 standard frames
- [ ] Unit test: known params → known thrust/torque output; verify gradients with `jax.test_util`

### Phase 4: Excitation + Optimizer (Week 6–7)
- [ ] `excitation.py` — FIM computation + per-parameter scoring
- [ ] `priors.py` — physical prior distributions from user measurements
- [ ] `map_optimizer.py` — LM optimizer with JAX exact Jacobians + Laplace posterior
- [ ] Integration test: synthesize a log from known params, recover params within 5%

### Phase 5: Output + CLI (Week 8)
- [ ] `parm_writer.py` — ArduPilot-compatible `.parm` file
- [ ] `report.py` — JSON + human-readable summary
- [ ] `sysid.py` — CLI glue with argparse
- [ ] End-to-end test: real `.bin` → SITL params → load in SITL → compare hover behavior

### Phase 6: Validation (Week 9–10)
- [ ] Fly 3 different vehicles (if hardware available) or use 3 different public logs
- [ ] Compare SITL-with-identified-params vs SITL-with-defaults vs real flight
- [ ] Document methodology and post to ArduPilot forums for review

---

## 7. Dependencies

```
# requirements.txt
pymavlink>=2.4.40
jax[cpu]>=0.4.25          # or jax[cuda12] for GPU
jaxlib>=0.4.25
numpy>=1.26
pandas>=2.1
scipy>=1.12               # for cumtrapz, signal processing
matplotlib>=3.8           # for validation plots
tqdm>=4.66
click>=8.1                # CLI
```

---

## 8. Key Design Decisions and Rationale

**Why JAX instead of PyTorch or numpy?**  
JAX's `jax.grad` / `jax.jacfwd` works on arbitrary Python/numpy-style code with no special
annotation. The FDM can be written exactly as it reads physically (with cross products, matrix
solves, quaternion integration) and differentiation is free. PyTorch requires careful use of
tensor operations and loses expressibility; finite differences in numpy are slow and noisy.

**Why Laplace approximation instead of MCMC?**  
Full MCMC posterior sampling on a nonlinear FDM with 8+ parameters and 400 Hz data would
require days of computation. The Laplace approximation (Gaussian centered at the MAP estimate,
covariance = inverse Hessian) is accurate when the posterior is roughly quadratic near the mode
— which is true for well-identified FDM parameters. The Hessian is free from JAX. This gives
meaningful confidence intervals in seconds.

**Why RTS smoother instead of just using the ArduPilot EKF output?**  
The ArduPilot EKF is a causal filter — its estimate at time t only uses observations up to t.
The RTS smoother uses the full log, forward and backward, producing lower-variance state
estimates. Better state estimates → better parameter identification. The smoother also produces
per-timestep covariance estimates, which are used to weight the optimization (don't trust
high-variance state estimates as strongly as low-variance ones).

**Why MAP with Gaussian priors instead of pure MLE?**  
Pure maximum likelihood is unregularized and can overfit to noise or land in physically
impossible regions (e.g., negative inertia) when parameters are poorly excited. The MAP
formulation encodes physical knowledge (you can weigh the vehicle; props have known diameters)
as soft constraints. It also prevents the kT↔mass coupling degeneracy from producing
compensating errors — if you tell the optimizer the mass is 1.2 ± 0.05 kg (you measured it),
it cannot compensate a wrong kT by also being wrong about mass.

---

## 9. Out of Scope (v1)

- Fixed-wing FDM identification (architecture supports it; implementation deferred)
- VTOL tiltrotor identification
- Aerodynamic drag at high forward speeds (>15 m/s)
- Wind estimation (requires airspeed sensor in log)
- Real-time in-flight identification
- Web UI or GUI (CLI only)
- ArduPilot firmware modifications (this is pure tooling)