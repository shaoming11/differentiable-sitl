"""
Unscented Kalman Filter (UKF) implementation for nonlinear state estimation.

The UKF uses the unscented transform to handle nonlinear dynamics and observations
without requiring Jacobian calculations (unlike the Extended Kalman Filter).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict, Optional
import pandas as pd
from .state_space import (
    state_transition_model,
    imu_observation_model,
    gps_observation_model,
    baro_observation_model
)


@dataclass
class UKFState:
    """State estimate from UKF at a single timestep."""
    x_posterior: np.ndarray      # x̂_{k|k} - state estimate after update
    P_posterior: np.ndarray      # P_{k|k} - covariance after update
    x_prior: np.ndarray          # x̂_{k|k-1} - state prediction
    P_prior: np.ndarray          # P_{k|k-1} - covariance prediction
    timestamp: float             # Time in seconds


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear state estimation.

    Uses the unscented transform to propagate mean and covariance through
    nonlinear transformations, providing better accuracy than linearization
    for highly nonlinear systems.

    Key advantages over EKF:
    - No need to compute Jacobians
    - More accurate for highly nonlinear dynamics
    - Captures mean and covariance to second order (Taylor expansion)
    """

    def __init__(
        self,
        state_dim: int = 13,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        """
        Initialize UKF with tuning parameters.

        Args:
            state_dim: Dimension of state vector (default: 13)
            alpha: Spread of sigma points, typically 1e-3 to 1
                   Small values (1e-4 to 1e-3) for tight distributions
            beta: Incorporates prior knowledge of distribution
                  2.0 is optimal for Gaussian distributions
            kappa: Secondary scaling parameter
                   Typical values: 0 or 3-n (where n is state_dim)
        """
        self.n = state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Compute lambda (scaling parameter)
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim

        # Compute sigma point weights
        self.weights_mean = self._compute_weights_mean()
        self.weights_cov = self._compute_weights_cov()

    def _compute_weights_mean(self) -> np.ndarray:
        """Compute sigma point weights for mean calculation."""
        W_m = np.zeros(2 * self.n + 1)
        W_m[0] = self.lambda_ / (self.n + self.lambda_)
        W_m[1:] = 0.5 / (self.n + self.lambda_)
        return W_m

    def _compute_weights_cov(self) -> np.ndarray:
        """Compute sigma point weights for covariance calculation."""
        W_c = np.zeros(2 * self.n + 1)
        W_c[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        W_c[1:] = 0.5 / (self.n + self.lambda_)
        return W_c

    def _generate_sigma_points(
        self,
        x: np.ndarray,
        P: np.ndarray
    ) -> np.ndarray:
        """
        Generate sigma points for unscented transform.

        Creates 2n+1 sigma points symmetrically distributed around the mean,
        with spread determined by the covariance and scaling parameters.

        Args:
            x: (n,) mean state vector
            P: (n, n) covariance matrix

        Returns:
            sigma_points: (2n+1, n) array where each row is a sigma point
        """
        n = len(x)
        sigma_points = np.zeros((2*n + 1, n))

        # Central point (mean)
        sigma_points[0] = x

        # Compute matrix square root using Cholesky decomposition
        # We need: L such that L @ L.T = (n + lambda) * P
        try:
            L = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            # If Cholesky fails (matrix not positive definite),
            # use eigenvalue decomposition as fallback
            eigvals, eigvecs = np.linalg.eigh(P)
            # Ensure all eigenvalues are positive (add small regularization)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt((n + self.lambda_) * eigvals))

        # Generate positive and negative perturbations
        for i in range(n):
            sigma_points[i+1] = x + L[:, i]
            sigma_points[n+i+1] = x - L[:, i]

        return sigma_points

    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        transition_fn: Callable,
        Q: np.ndarray,
        dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        UKF predict step: propagate state and covariance forward in time.

        Args:
            x: (n,) current state estimate
            P: (n, n) current covariance estimate
            transition_fn: Function that maps state x to next state
            Q: (n, n) process noise covariance
            dt: time step in seconds

        Returns:
            x_pred: (n,) predicted state
            P_pred: (n, n) predicted covariance
        """
        # Generate sigma points around current estimate
        sigma_points = self._generate_sigma_points(x, P)

        # Propagate each sigma point through state transition
        sigma_points_pred = np.array([
            transition_fn(sp, dt) for sp in sigma_points
        ])

        # Compute predicted mean as weighted sum of propagated sigma points
        x_pred = np.sum(self.weights_mean[:, None] * sigma_points_pred, axis=0)

        # Compute predicted covariance
        diff = sigma_points_pred - x_pred
        P_pred = sum(
            self.weights_cov[i] * np.outer(diff[i], diff[i])
            for i in range(2*self.n + 1)
        ) + Q

        # Ensure symmetry (numerical stability)
        P_pred = 0.5 * (P_pred + P_pred.T)

        return x_pred, P_pred

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        measurement: np.ndarray,
        observation_fn: Callable,
        R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        UKF update step: incorporate measurement to refine state estimate.

        Args:
            x_pred: (n,) predicted state
            P_pred: (n, n) predicted covariance
            measurement: (m,) actual measurement
            observation_fn: Function that maps state to expected measurement
            R: (m, m) measurement noise covariance

        Returns:
            x_updated: (n,) updated state estimate
            P_updated: (n, n) updated covariance
        """
        # Generate sigma points around predicted state
        sigma_points = self._generate_sigma_points(x_pred, P_pred)

        # Propagate sigma points through observation model
        sigma_observations = np.array([
            observation_fn(sp) for sp in sigma_points
        ])

        # Predicted observation mean
        z_pred = np.sum(self.weights_mean[:, None] * sigma_observations, axis=0)

        # Innovation (measurement residual)
        innovation = measurement - z_pred

        # Innovation covariance
        diff_z = sigma_observations - z_pred
        P_zz = sum(
            self.weights_cov[i] * np.outer(diff_z[i], diff_z[i])
            for i in range(2*self.n + 1)
        ) + R

        # Cross-covariance between state and observation
        diff_x = sigma_points - x_pred
        P_xz = sum(
            self.weights_cov[i] * np.outer(diff_x[i], diff_z[i])
            for i in range(2*self.n + 1)
        )

        # Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)

        # Update state estimate
        x_updated = x_pred + K @ innovation

        # Normalize quaternion (first 4 elements) to maintain unit constraint
        q_norm = np.linalg.norm(x_updated[0:4])
        if q_norm > 0:
            x_updated[0:4] = x_updated[0:4] / q_norm

        # Update covariance
        P_updated = P_pred - K @ P_zz @ K.T

        # Ensure symmetry
        P_updated = 0.5 * (P_updated + P_updated.T)

        return x_updated, P_updated

    def forward_pass(
        self,
        x_init: np.ndarray,
        P_init: np.ndarray,
        measurements: Dict[str, pd.DataFrame],
        dt: float,
        Q: np.ndarray,
        R_imu: np.ndarray,
        R_gps: np.ndarray,
        R_baro: np.ndarray
    ) -> List[UKFState]:
        """
        Run forward UKF pass through entire dataset.

        Processes measurements in chronological order, performing predict-update
        cycles and storing all intermediate states for the RTS backward pass.

        Args:
            x_init: (13,) initial state estimate
            P_init: (13, 13) initial covariance
            measurements: Dictionary with keys 'imu', 'gps', 'baro'
                         Each value is a DataFrame with 'timestamp' column
            dt: nominal timestep in seconds (e.g., 1/400 for 400 Hz IMU)
            Q: (13, 13) process noise covariance
            R_imu: (6, 6) IMU measurement noise covariance
            R_gps: (6, 6) GPS measurement noise covariance
            R_baro: (1, 1) Barometer measurement noise covariance

        Returns:
            List of UKFState objects, one per timestep
        """
        # Initialize state
        x = x_init.copy()
        P = P_init.copy()

        # Get all unique timestamps and sort
        all_timestamps = set()
        for sensor_name, df in measurements.items():
            if df is not None and not df.empty:
                all_timestamps.update(df['timestamp'].values)
        timestamps = sorted(all_timestamps)

        states = []

        # Process each timestep
        for t in timestamps:
            # Predict step
            x_pred, P_pred = self.predict(
                x, P,
                state_transition_model,
                Q,
                dt
            )

            # Update step: check which sensors have measurements at this time
            x_updated = x_pred.copy()
            P_updated = P_pred.copy()

            # IMU update (highest rate sensor)
            if 'imu' in measurements and measurements['imu'] is not None:
                imu_data = measurements['imu']
                imu_at_t = imu_data[np.isclose(imu_data['timestamp'], t, atol=dt/2)]
                if len(imu_at_t) > 0:
                    # Extract IMU measurement [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
                    measurement = imu_at_t.iloc[0][['acc_x', 'acc_y', 'acc_z',
                                                     'gyr_x', 'gyr_y', 'gyr_z']].values
                    x_updated, P_updated = self.update(
                        x_updated, P_updated,
                        measurement,
                        imu_observation_model,
                        R_imu
                    )

            # GPS update (lower rate)
            if 'gps' in measurements and measurements['gps'] is not None:
                gps_data = measurements['gps']
                gps_at_t = gps_data[np.isclose(gps_data['timestamp'], t, atol=0.05)]
                if len(gps_at_t) > 0:
                    # Extract GPS measurement [vel_n, vel_e, vel_d, lat, lng, alt]
                    measurement = gps_at_t.iloc[0][['vel_n', 'vel_e', 'vel_d',
                                                     'lat', 'lng', 'alt']].values
                    x_updated, P_updated = self.update(
                        x_updated, P_updated,
                        measurement,
                        gps_observation_model,
                        R_gps
                    )

            # Barometer update (lower rate)
            if 'baro' in measurements and measurements['baro'] is not None:
                baro_data = measurements['baro']
                baro_at_t = baro_data[np.isclose(baro_data['timestamp'], t, atol=0.05)]
                if len(baro_at_t) > 0:
                    # Extract barometer measurement [altitude]
                    measurement = baro_at_t.iloc[0][['baro_alt']].values
                    x_updated, P_updated = self.update(
                        x_updated, P_updated,
                        measurement,
                        baro_observation_model,
                        R_baro
                    )

            # Store state
            states.append(UKFState(
                x_posterior=x_updated.copy(),
                P_posterior=P_updated.copy(),
                x_prior=x_pred.copy(),
                P_prior=P_pred.copy(),
                timestamp=t
            ))

            # Update for next iteration
            x = x_updated
            P = P_updated

        return states
