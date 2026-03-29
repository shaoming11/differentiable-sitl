"""
Rauch-Tung-Striebel (RTS) smoother for optimal state estimation.

The RTS smoother performs a backward pass over forward UKF estimates to produce
minimum-variance state estimates using all available data (past and future).

Key insight: ArduPilot's EKF only uses data up to time t (causal).
The RTS smoother uses the FULL log (forward + backward) for better estimates.
"""

import numpy as np
from typing import List, Callable, Optional
from .ukf import UKFState


class RTSSmoother:
    """
    Rauch-Tung-Striebel smoother implementation.

    Algorithm overview:
    1. Forward pass: Run UKF through data, saving all intermediate estimates
    2. Backward pass: Refine each estimate using information from the future

    The backward pass equations:
        K_s = P_{k|k} @ F_k^T @ inv(P_{k+1|k})
        x̂_{k|N} = x̂_{k|k} + K_s @ (x̂_{k+1|N} - x̂_{k+1|k})
        P_{k|N} = P_{k|k} + K_s @ (P_{k+1|N} - P_{k+1|k}) @ K_s^T

    where:
        - k|k denotes filtered estimate at time k using data up to k
        - k|N denotes smoothed estimate at time k using ALL data (0 to N)
        - K_s is the smoother gain
        - F_k is the state transition Jacobian (implicit in UKF via prior covariance)
    """

    def __init__(self):
        """Initialize RTS smoother."""
        pass

    def backward_pass(
        self,
        forward_states: List[UKFState],
        transition_fn: Optional[Callable] = None
    ) -> List[UKFState]:
        """
        Perform RTS backward smoothing pass.

        Processes states in reverse chronological order, refining each estimate
        by incorporating information from future measurements.

        Args:
            forward_states: List of UKFState from forward UKF pass
            transition_fn: State transition function (not used in basic RTS,
                          kept for potential future extensions)

        Returns:
            List of smoothed UKFState objects in chronological order

        Raises:
            ValueError: If forward_states is empty
        """
        if not forward_states:
            raise ValueError("forward_states cannot be empty")

        N = len(forward_states)
        smoothed = [None] * N

        # Initialize with final forward estimate (already optimal - no future data)
        smoothed[-1] = UKFState(
            x_posterior=forward_states[-1].x_posterior.copy(),
            P_posterior=forward_states[-1].P_posterior.copy(),
            x_prior=forward_states[-1].x_prior.copy(),
            P_prior=forward_states[-1].P_prior.copy(),
            timestamp=forward_states[-1].timestamp
        )

        # Backward pass: work backwards from N-2 to 0
        for k in reversed(range(N - 1)):
            fwd = forward_states[k]         # x̂_{k|k}, P_{k|k}
            fwd_next = forward_states[k + 1]  # x̂_{k+1|k}, P_{k+1|k}
            smooth_next = smoothed[k + 1]    # x̂_{k+1|N}, P_{k+1|N}

            # Check for numerical issues in covariances
            if np.any(np.isnan(fwd.P_posterior)) or np.any(np.isnan(fwd_next.P_prior)):
                # Skip this step, use forward estimate
                smoothed[k] = fwd
                continue

            # Compute smoother gain using stable solve
            # K_s = P_{k|k} @ inv(P_{k+1|k})
            # Equivalent to: solve(P_{k+1|k}.T @ K_s.T = P_{k|k}.T) for K_s.T

            # Add small regularization to ensure numerical stability
            P_prior_reg = fwd_next.P_prior + np.eye(fwd_next.P_prior.shape[0]) * 1e-6

            try:
                # Use solve instead of inv for numerical stability
                # K_s.T = solve(P_{k+1|k}, P_{k|k}.T)
                K_s = np.linalg.solve(P_prior_reg, fwd.P_posterior.T).T
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse if matrix is singular
                try:
                    K_s = fwd.P_posterior @ np.linalg.pinv(P_prior_reg, rcond=1e-8)
                except:
                    # If all else fails, skip smoothing for this timestep
                    smoothed[k] = fwd
                    continue

            # Clip smoother gain to prevent numerical instability
            K_s = np.clip(K_s, -10, 10)

            # Smoothed state: x̂_{k|N} = x̂_{k|k} + K_s @ (x̂_{k+1|N} - x̂_{k+1|k})
            state_diff = smooth_next.x_posterior - fwd_next.x_prior
            x_smooth = fwd.x_posterior + K_s @ state_diff

            # Check for NaN
            if np.any(np.isnan(x_smooth)):
                smoothed[k] = fwd
                continue

            # Normalize quaternion (first 4 elements) to maintain unit constraint
            q_norm = np.linalg.norm(x_smooth[0:4])
            if q_norm > 1e-6:
                x_smooth[0:4] = x_smooth[0:4] / q_norm
            else:
                # Quaternion became degenerate, use forward estimate
                smoothed[k] = fwd
                continue

            # Smoothed covariance: P_{k|N} = P_{k|k} + K_s @ (P_{k+1|N} - P_{k+1|k}) @ K_s^T
            cov_diff = smooth_next.P_posterior - fwd_next.P_prior
            P_smooth = fwd.P_posterior + K_s @ cov_diff @ K_s.T

            # Ensure symmetry (numerical stability)
            P_smooth = 0.5 * (P_smooth + P_smooth.T)

            # Ensure positive definiteness
            min_eigval = np.min(np.linalg.eigvalsh(P_smooth))
            if min_eigval < 1e-6:
                P_smooth += (1e-6 - min_eigval) * np.eye(P_smooth.shape[0])

            # Check for covariance explosion
            max_trace = 100.0 * np.trace(fwd.P_posterior)
            if np.trace(P_smooth) > max_trace or np.any(np.isnan(P_smooth)):
                # Don't smooth if covariance explodes
                smoothed[k] = fwd
                continue

            # Store smoothed state
            smoothed[k] = UKFState(
                x_posterior=x_smooth,
                P_posterior=P_smooth,
                x_prior=fwd.x_prior.copy(),  # Keep original prior for reference
                P_prior=fwd.P_prior.copy(),
                timestamp=fwd.timestamp
            )

        return smoothed

    def get_covariance_trace(self, states: List[UKFState]) -> np.ndarray:
        """
        Extract trace of covariance matrices over time.

        The trace gives a scalar measure of total uncertainty.
        Useful for comparing forward-only vs. smoothed estimates.

        Args:
            states: List of UKFState objects

        Returns:
            traces: (N,) array of covariance traces
        """
        return np.array([np.trace(state.P_posterior) for state in states])

    def get_state_trajectory(self, states: List[UKFState]) -> np.ndarray:
        """
        Extract state trajectory as a 2D array.

        Args:
            states: List of UKFState objects

        Returns:
            trajectory: (N, 13) array where each row is a state vector
        """
        return np.array([state.x_posterior for state in states])

    def get_timestamps(self, states: List[UKFState]) -> np.ndarray:
        """
        Extract timestamps from state list.

        Args:
            states: List of UKFState objects

        Returns:
            timestamps: (N,) array of timestamps in seconds
        """
        return np.array([state.timestamp for state in states])

    def compute_rmse(
        self,
        states: List[UKFState],
        ground_truth: np.ndarray,
        state_indices: Optional[List[int]] = None
    ) -> float:
        """
        Compute Root Mean Square Error against ground truth.

        Args:
            states: List of UKFState objects
            ground_truth: (N, k) array of ground truth values
            state_indices: Indices of state components to compare (default: all)

        Returns:
            rmse: Root mean square error
        """
        trajectory = self.get_state_trajectory(states)

        if state_indices is not None:
            trajectory = trajectory[:, state_indices]

        if trajectory.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: trajectory {trajectory.shape} "
                f"vs ground_truth {ground_truth.shape}"
            )

        mse = np.mean((trajectory - ground_truth) ** 2)
        return np.sqrt(mse)

    def extract_angular_velocity(self, states: List[UKFState]) -> np.ndarray:
        """
        Extract angular velocity trajectory (body frame).

        Args:
            states: List of UKFState objects

        Returns:
            omega: (N, 3) array of angular velocities [wx, wy, wz] in rad/s
        """
        trajectory = self.get_state_trajectory(states)
        return trajectory[:, 7:10]  # Indices 7, 8, 9 are wx, wy, wz

    def extract_velocity(self, states: List[UKFState]) -> np.ndarray:
        """
        Extract velocity trajectory (world frame).

        Args:
            states: List of UKFState objects

        Returns:
            velocity: (N, 3) array of velocities [vx, vy, vz] in m/s
        """
        trajectory = self.get_state_trajectory(states)
        return trajectory[:, 4:7]  # Indices 4, 5, 6 are vx, vy, vz

    def extract_position(self, states: List[UKFState]) -> np.ndarray:
        """
        Extract position trajectory (world frame).

        Args:
            states: List of UKFState objects

        Returns:
            position: (N, 3) array of positions [px, py, pz] in meters
        """
        trajectory = self.get_state_trajectory(states)
        return trajectory[:, 10:13]  # Indices 10, 11, 12 are px, py, pz

    def extract_quaternion(self, states: List[UKFState]) -> np.ndarray:
        """
        Extract quaternion trajectory.

        Args:
            states: List of UKFState objects

        Returns:
            quaternions: (N, 4) array of quaternions [qw, qx, qy, qz]
        """
        trajectory = self.get_state_trajectory(states)
        return trajectory[:, 0:4]  # Indices 0, 1, 2, 3 are qw, qx, qy, qz

    def compare_forward_vs_smoothed(
        self,
        forward_states: List[UKFState],
        smoothed_states: List[UKFState]
    ) -> dict:
        """
        Compare forward-only UKF vs RTS smoothed estimates.

        Args:
            forward_states: List of forward UKF states
            smoothed_states: List of RTS smoothed states

        Returns:
            Dictionary with comparison metrics:
            - 'forward_trace_mean': Mean trace of forward covariance
            - 'smoothed_trace_mean': Mean trace of smoothed covariance
            - 'variance_reduction': Percentage reduction in uncertainty
            - 'state_diff_norm': RMS difference between forward and smoothed states
        """
        forward_trace = self.get_covariance_trace(forward_states)
        smoothed_trace = self.get_covariance_trace(smoothed_states)

        forward_trajectory = self.get_state_trajectory(forward_states)
        smoothed_trajectory = self.get_state_trajectory(smoothed_states)

        state_diff = forward_trajectory - smoothed_trajectory
        state_diff_norm = np.sqrt(np.mean(np.sum(state_diff ** 2, axis=1)))

        return {
            'forward_trace_mean': np.mean(forward_trace),
            'smoothed_trace_mean': np.mean(smoothed_trace),
            'variance_reduction': 100 * (1 - np.mean(smoothed_trace) / np.mean(forward_trace)),
            'state_diff_norm': state_diff_norm,
            'forward_trace': forward_trace,
            'smoothed_trace': smoothed_trace
        }
