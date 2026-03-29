"""
Unit tests for the parser module.

Tests cover:
- Message type constants and field mappings
- DFLogReader functionality (with synthetic data)
- EKF health filtering and segment extraction
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock, patch

from ardupilot_sysid.src.parser import message_types as mt
from ardupilot_sysid.src.parser.dflog_reader import DFLogReader
from ardupilot_sysid.src.parser import ekf_health


# =============================================================================
# Test message_types.py
# =============================================================================

class TestMessageTypes:
    """Tests for message type constants and utilities."""

    def test_message_type_constants(self):
        """Verify all message type constants are defined."""
        assert mt.IMU_MSG == 'IMU'
        assert mt.RCOUT_MSG == 'RCOUT'
        assert mt.ATT_MSG == 'ATT'
        assert mt.GPS_MSG == 'GPS'
        assert mt.BARO_MSG == 'BARO'
        assert mt.EKF_MSG == 'EKF1'
        assert mt.PARM_MSG == 'PARM'

    def test_field_mappings(self):
        """Verify field mappings contain expected fields."""
        assert 'TimeUS' in mt.IMU_FIELDS
        assert 'GyrX' in mt.IMU_FIELDS
        assert 'AccX' in mt.IMU_FIELDS

        assert 'C1' in mt.RCOUT_FIELDS
        assert 'C14' in mt.RCOUT_FIELDS

        assert 'Roll' in mt.ATT_FIELDS
        assert 'Pitch' in mt.ATT_FIELDS
        assert 'Yaw' in mt.ATT_FIELDS

    def test_normalized_columns(self):
        """Verify normalized column names."""
        assert mt.TIMESTAMP_COL == 'timestamp'
        assert 'gyr_x' in mt.GYRO_COLS
        assert 'acc_x' in mt.ACCEL_COLS
        assert 'roll' in mt.ATTITUDE_COLS

    def test_get_message_fields(self):
        """Test get_message_fields function."""
        fields = mt.get_message_fields('IMU')
        assert 'TimeUS' in fields
        assert 'GyrX' in fields

        with pytest.raises(ValueError):
            mt.get_message_fields('INVALID_MSG_TYPE')

    def test_get_normalized_columns(self):
        """Test get_normalized_columns function."""
        cols = mt.get_normalized_columns('IMU')
        assert mt.TIMESTAMP_COL in cols
        assert 'gyr_x' in cols

        with pytest.raises(ValueError):
            mt.get_normalized_columns('INVALID_MSG_TYPE')

    def test_get_message_rate(self):
        """Test get_message_rate function."""
        assert mt.get_message_rate('IMU') == 400
        assert mt.get_message_rate('GPS') == 10
        assert mt.get_message_rate('UNKNOWN') == 1.0  # Default

    def test_unit_conversion_constants(self):
        """Test unit conversion constants."""
        assert np.isclose(mt.DEG_TO_RAD, np.pi / 180.0)
        assert np.isclose(mt.RAD_TO_DEG, 180.0 / np.pi)
        assert mt.US_TO_SEC == 1e-6


# =============================================================================
# Test ekf_health.py
# =============================================================================

class TestEKFHealth:
    """Tests for EKF health filtering."""

    def test_filter_healthy_segments_basic(self):
        """Test basic segment filtering with synthetic data."""
        # Create synthetic EKF data with alternating healthy/unhealthy periods
        timestamps = np.linspace(0, 100, 1000)
        innovation_ratio = np.ones(1000) * 0.5  # Start healthy

        # Make some segments unhealthy
        innovation_ratio[200:300] = 1.5  # Unhealthy from 20-30s
        innovation_ratio[500:600] = 2.0  # Unhealthy from 50-60s

        ekf_df = pd.DataFrame({
            'timestamp': timestamps,
            'innovation_ratio': innovation_ratio,
        })

        segments = ekf_health.filter_ekf_healthy_segments(
            ekf_df,
            innovation_threshold=1.0,
            min_segment_duration=5.0
        )

        # Should get 3 segments: [0-20], [30-50], [60-100]
        assert len(segments) == 3
        assert segments[0][0] < 1.0  # First segment starts near 0
        assert segments[0][1] > 19.0  # First segment ends near 20
        assert segments[2][1] > 99.0  # Last segment ends near 100

    def test_filter_healthy_segments_edge_cases(self):
        """Test edge cases: empty data, all healthy, all unhealthy."""
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['timestamp', 'innovation_ratio'])
        segments = ekf_health.filter_ekf_healthy_segments(empty_df)
        assert len(segments) == 0

        # All healthy
        healthy_df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'innovation_ratio': np.ones(100) * 0.5,
        })
        segments = ekf_health.filter_ekf_healthy_segments(
            healthy_df,
            min_segment_duration=1.0
        )
        assert len(segments) == 1
        assert segments[0][0] < 1.0
        assert segments[0][1] > 9.0

        # All unhealthy
        unhealthy_df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'innovation_ratio': np.ones(100) * 2.0,
        })
        segments = ekf_health.filter_ekf_healthy_segments(unhealthy_df)
        assert len(segments) == 0

    def test_filter_segments_minimum_duration(self):
        """Test that short segments are filtered out."""
        timestamps = np.linspace(0, 100, 1000)
        innovation_ratio = np.ones(1000) * 1.5  # Start unhealthy

        # Create a very short healthy period (1 second)
        innovation_ratio[500:510] = 0.5

        ekf_df = pd.DataFrame({
            'timestamp': timestamps,
            'innovation_ratio': innovation_ratio,
        })

        # With min_segment_duration=5.0, this should be filtered out
        segments = ekf_health.filter_ekf_healthy_segments(
            ekf_df,
            min_segment_duration=5.0
        )
        assert len(segments) == 0

        # With min_segment_duration=0.5, this should be included
        segments = ekf_health.filter_ekf_healthy_segments(
            ekf_df,
            min_segment_duration=0.5
        )
        assert len(segments) == 1

    def test_apply_segment_filter(self):
        """Test applying segment filter to a DataFrame."""
        # Create test data
        df = pd.DataFrame({
            'timestamp': np.linspace(0, 100, 1000),
            'value': np.arange(1000),
        })

        # Filter to segments [10-20] and [50-60]
        segments = [(10.0, 20.0), (50.0, 60.0)]
        filtered_df = ekf_health.apply_segment_filter(df, segments)

        # Check that only data within segments is included
        assert len(filtered_df) < len(df)
        assert filtered_df['timestamp'].min() >= 10.0
        assert filtered_df['timestamp'].max() <= 60.0

        # Check that there's a gap in the middle
        mid_data = filtered_df[(filtered_df['timestamp'] > 20) & (filtered_df['timestamp'] < 50)]
        assert len(mid_data) == 0

    def test_apply_segment_filter_edge_cases(self):
        """Test edge cases for apply_segment_filter."""
        df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'value': np.arange(100),
        })

        # Empty segments
        filtered = ekf_health.apply_segment_filter(df, [])
        assert len(filtered) == 0

        # Empty DataFrame
        empty_df = pd.DataFrame(columns=['timestamp', 'value'])
        filtered = ekf_health.apply_segment_filter(empty_df, [(0, 10)])
        assert len(filtered) == 0

    def test_compute_segment_statistics(self):
        """Test segment statistics computation."""
        ekf_df = pd.DataFrame({
            'timestamp': np.linspace(0, 100, 1000),
            'innovation_ratio': np.ones(1000) * 0.5,
        })

        segments = [(0.0, 30.0), (50.0, 80.0)]

        stats = ekf_health.compute_segment_statistics(ekf_df, segments)

        assert stats['num_segments'] == 2
        assert np.isclose(stats['total_duration'], 60.0)  # 30 + 30
        assert np.isclose(stats['mean_duration'], 30.0)
        assert np.isclose(stats['min_duration'], 30.0)
        assert np.isclose(stats['max_duration'], 30.0)
        assert np.isclose(stats['coverage'], 0.6)  # 60/100

    def test_segment_statistics_empty(self):
        """Test statistics with no segments."""
        ekf_df = pd.DataFrame({
            'timestamp': np.linspace(0, 100, 1000),
            'innovation_ratio': np.ones(1000) * 0.5,
        })

        stats = ekf_health.compute_segment_statistics(ekf_df, [])

        assert stats['num_segments'] == 0
        assert stats['total_duration'] == 0.0
        assert stats['coverage'] == 0.0


# =============================================================================
# Test dflog_reader.py
# =============================================================================

class TestDFLogReader:
    """Tests for DFLogReader."""

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DFLogReader('/path/to/nonexistent/file.bin')

    def test_init_wrong_extension(self, tmp_path):
        """Test initialization with wrong file extension."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.touch()

        with pytest.raises(ValueError):
            DFLogReader(str(wrong_file))

    def test_normalize_timestamps(self, tmp_path):
        """Test timestamp normalization."""
        # Create a temporary .bin file (just for initialization)
        test_file = tmp_path / "test.bin"
        test_file.touch()

        reader = DFLogReader(str(test_file))

        # Create test DataFrame with TimeUS
        df = pd.DataFrame({
            'TimeUS': [1000000, 2000000, 3000000],  # 1, 2, 3 seconds in microseconds
            'value': [1, 2, 3],
        })

        normalized = reader._normalize_timestamps(df)

        assert 'timestamp' in normalized.columns
        assert 'TimeUS' not in normalized.columns
        assert np.isclose(normalized['timestamp'].iloc[0], 0.0)  # Normalized to start at 0
        assert np.isclose(normalized['timestamp'].iloc[1], 1.0)
        assert np.isclose(normalized['timestamp'].iloc[2], 2.0)

    def test_normalize_timestamps_empty(self, tmp_path):
        """Test timestamp normalization with empty DataFrame."""
        test_file = tmp_path / "test.bin"
        test_file.touch()

        reader = DFLogReader(str(test_file))

        empty_df = pd.DataFrame()
        normalized = reader._normalize_timestamps(empty_df)
        assert len(normalized) == 0

    @patch('ardupilot_sysid.src.parser.dflog_reader.mavutil')
    def test_extract_messages_mock(self, mock_mavutil, tmp_path):
        """Test message extraction with mocked pymavlink."""
        test_file = tmp_path / "test.bin"
        test_file.touch()

        # Create mock message
        mock_msg1 = Mock()
        mock_msg1.TimeUS = 1000000
        mock_msg1.GyrX = 0.1
        mock_msg1.GyrY = 0.2

        mock_msg2 = Mock()
        mock_msg2.TimeUS = 2000000
        mock_msg2.GyrX = 0.3
        mock_msg2.GyrY = 0.4

        # Setup mock connection
        mock_conn = Mock()
        mock_conn.recv_match.side_effect = [mock_msg1, mock_msg2, None]
        mock_conn.rewind.return_value = None
        mock_mavutil.mavlink_connection.return_value = mock_conn

        reader = DFLogReader(str(test_file))
        reader.mlog = mock_conn

        # Extract messages
        df = reader._extract_messages('IMU', ['TimeUS', 'GyrX', 'GyrY'])

        assert len(df) == 2
        assert df['TimeUS'].iloc[0] == 1000000
        assert df['GyrX'].iloc[0] == 0.1
        assert df['GyrY'].iloc[1] == 0.4

    def test_get_log_summary(self, tmp_path):
        """Test log summary generation."""
        test_file = tmp_path / "test.bin"
        test_file.touch()

        reader = DFLogReader(str(test_file))

        # Create mock parsed data
        parsed_data = {
            'imu': pd.DataFrame({
                'timestamp': np.linspace(0, 100, 1000),
                'gyr_x': np.random.randn(1000),
            }),
            'gps': pd.DataFrame({
                'timestamp': np.linspace(0, 100, 100),
                'lat': np.random.randn(100),
            }),
            'params': {'PARAM1': 1.0, 'PARAM2': 2.0},
        }

        summary = reader.get_log_summary(parsed_data)

        assert 'log_file' in summary
        assert summary['imu_count'] == 1000
        assert summary['gps_count'] == 100
        assert summary['params_count'] == 2
        assert 'total_duration_s' in summary
        assert np.isclose(summary['total_duration_s'], 100.0, atol=1.0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestParserIntegration:
    """Integration tests combining multiple modules."""

    def test_timestamp_normalization_and_filtering(self):
        """Test full pipeline: normalize timestamps and filter by EKF health."""
        # Create synthetic IMU data
        imu_df = pd.DataFrame({
            'TimeUS': np.arange(0, 10000000, 10000),  # 0-10s at 100Hz
            'GyrX': np.random.randn(1000) * 0.1,
            'GyrY': np.random.randn(1000) * 0.1,
        })

        # Create synthetic EKF data
        ekf_df = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'innovation_ratio': np.concatenate([
                np.ones(30) * 0.5,   # Healthy
                np.ones(20) * 1.5,   # Unhealthy
                np.ones(50) * 0.5,   # Healthy
            ]),
        })

        # Normalize IMU timestamps
        imu_df['timestamp'] = (imu_df['TimeUS'] - imu_df['TimeUS'].iloc[0]) * mt.US_TO_SEC
        imu_df = imu_df.drop(columns=['TimeUS'])

        # Find healthy segments
        segments = ekf_health.filter_ekf_healthy_segments(
            ekf_df,
            innovation_threshold=1.0,
            min_segment_duration=1.0
        )

        # Should get 2 segments
        assert len(segments) >= 1

        # Filter IMU data
        filtered_imu = ekf_health.apply_segment_filter(imu_df, segments)

        # Filtered data should be smaller than original
        assert len(filtered_imu) < len(imu_df)


# =============================================================================
# Test Utilities
# =============================================================================

def test_print_functions_no_crash():
    """Test that print functions don't crash."""
    # These are just smoke tests to ensure the print functions work

    ekf_df = pd.DataFrame({
        'timestamp': np.linspace(0, 100, 1000),
        'innovation_ratio': np.ones(1000) * 0.5,
    })
    segments = [(0.0, 50.0), (60.0, 100.0)]

    # Should not crash
    ekf_health.print_segment_report(ekf_df, segments, 1.0)

    # Test with parsed data
    parsed_data = {
        'imu': pd.DataFrame({'timestamp': [0, 1, 2]}),
        'params': {'TEST': 1.0},
    }

    # Create a dummy reader
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_path = f.name

    try:
        reader = DFLogReader(temp_path)
        summary = reader.get_log_summary(parsed_data)

        from ardupilot_sysid.src.parser.dflog_reader import print_log_summary
        print_log_summary(summary)  # Should not crash

    finally:
        Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
