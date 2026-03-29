# ArduPilot Log Parser Module

This module provides functionality for parsing ArduPilot DataFlash `.bin` logs and extracting sensor data into pandas DataFrames with normalized timestamps and SI units.

## Modules

### `message_types.py`
Constants and field mappings for ArduPilot message types.

**Key Features:**
- Message type names (IMU, RCOUT, ATT, GPS, BARO, EKF, PARM)
- Field mappings for each message type
- Normalized column names (timestamp, gyr_x, acc_x, roll, etc.)
- Unit conversion constants (DEG_TO_RAD, US_TO_SEC)
- Helper functions for message metadata

### `dflog_reader.py`
Main log parser using pymavlink.

**Key Features:**
- Parse `.bin` logs into pandas DataFrames
- Automatic unit conversion (degrees→radians, microseconds→seconds)
- Normalized timestamps starting at 0.0
- Graceful handling of missing message types
- Parameter extraction
- Log summary generation

### `ekf_health.py`
EKF health monitoring and segment filtering.

**Key Features:**
- Identify time segments where EKF is healthy (innovation ratio < threshold)
- Filter sensor data to only include healthy segments
- Configurable thresholds and minimum segment duration
- Segment statistics and reporting

## Quick Start

### Parse a log file

```python
from ardupilot_sysid.src.parser.dflog_reader import DFLogReader

# Initialize reader
reader = DFLogReader('flight.bin')

# Parse log
data = reader.parse()

# Access data
imu_df = data['imu']        # IMU data (400 Hz)
rcout_df = data['rcout']    # PWM outputs (50 Hz)
att_df = data['att']        # Attitude (400 Hz)
gps_df = data['gps']        # GPS (5-10 Hz)
baro_df = data['baro']      # Barometer (10 Hz)
ekf_df = data['ekf']        # EKF health (10 Hz)
params = data['params']     # Parameters (dict)
```

### Filter by EKF health

```python
from ardupilot_sysid.src.parser import ekf_health

# Find healthy segments
segments = ekf_health.filter_ekf_healthy_segments(
    data['ekf'],
    innovation_threshold=1.0,
    min_segment_duration=5.0
)

# Filter sensor data
imu_healthy = ekf_health.apply_segment_filter(data['imu'], segments)
gps_healthy = ekf_health.apply_segment_filter(data['gps'], segments)

# Print report
ekf_health.print_segment_report(data['ekf'], segments)
```

### Generate log summary

```python
from ardupilot_sysid.src.parser.dflog_reader import print_log_summary

summary = reader.get_log_summary(data)
print_log_summary(summary)
```

## Data Format

### IMU DataFrame
Columns: `[timestamp, gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z]`
- `timestamp`: seconds (float)
- `gyr_x/y/z`: rad/s (body frame)
- `acc_x/y/z`: m/s² (body frame)

### RCOUT DataFrame
Columns: `[timestamp, pwm_1, pwm_2, ..., pwm_14]`
- `timestamp`: seconds (float)
- `pwm_*`: microseconds (1000-2000 typical)

### ATT DataFrame
Columns: `[timestamp, roll, pitch, yaw]`
- `timestamp`: seconds (float)
- `roll/pitch/yaw`: radians (Euler angles)

### GPS DataFrame
Columns: `[timestamp, lat, lng, alt, gps_speed, vel_n, vel_e, vel_d]`
- `timestamp`: seconds (float)
- `lat/lng`: degrees
- `alt`: meters
- `gps_speed`: m/s (ground speed)
- `vel_n/e/d`: m/s (NED frame)

### BARO DataFrame
Columns: `[timestamp, baro_alt, baro_press]`
- `timestamp`: seconds (float)
- `baro_alt`: meters (relative altitude)
- `baro_press`: Pascals

### EKF DataFrame
Columns: `[timestamp, innovation_ratio]`
- `timestamp`: seconds (float)
- `innovation_ratio`: dimensionless (should be < 1.0 for healthy)

### Parameters
Dictionary: `{param_name: value}`
- All parameter values as floats

## Unit Conversions

The parser automatically performs the following conversions:

| Quantity | Input | Output | Conversion |
|----------|-------|--------|------------|
| Time | TimeUS (µs) | timestamp (s) | ÷ 1,000,000 |
| Angles | degrees | radians | × π/180 |
| Gyro | rad/s | rad/s | (no change) |
| Accel | m/s² | m/s² | (no change) |
| Velocity | m/s | m/s | (no change) |
| PWM | µs | µs | (no change) |

## EKF Health

The EKF (Extended Kalman Filter) health is monitored using the innovation ratio (`SV` field in EKF1 messages).

**Innovation ratio interpretation:**
- `< 0.5`: Excellent - high confidence in state estimates
- `0.5-1.0`: Good - filter is tracking well
- `1.0-2.0`: Marginal - filter is stressed but functional
- `> 2.0`: Poor - filter is diverging, state estimates unreliable

**Causes of high innovation:**
- GPS glitches or multipath
- Compass interference (power lines, metal structures)
- Vibration-induced accelerometer clipping
- Rapid maneuvers exceeding model assumptions
- Sensor failures

**Recommendation:** Only use data where innovation ratio < 1.0 for parameter identification.

## Error Handling

The parser handles the following error conditions gracefully:

1. **File not found:** Raises `FileNotFoundError`
2. **Wrong file extension:** Raises `ValueError`
3. **Missing message types:** Returns empty DataFrame
4. **Missing fields:** Fills with NaN
5. **Malformed messages:** Skips and continues
6. **Empty log:** Returns empty DataFrames

## Testing

Run unit tests:

```bash
# Full test suite (requires pytest)
python -m pytest tests/test_parser.py -v

# Standalone verification (no pytest required)
python verify_parser.py

# Integration tests
python test_parser_integration.py

# Usage examples
python example_parser_usage.py
```

## Dependencies

- `pymavlink>=2.4.40` - ArduPilot log parsing
- `pandas>=2.1` - DataFrame operations
- `numpy>=1.26` - Numerical operations

## See Also

- [ArduPilot DataFlash Message Format](https://ardupilot.org/dev/docs/code-overview-adding-a-new-log-message.html)
- [pymavlink Documentation](https://github.com/ArduPilot/pymavlink)
- [Project PRD](../../../PRD.md)
