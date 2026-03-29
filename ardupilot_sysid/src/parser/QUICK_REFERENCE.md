# Parser Module - Quick Reference

## One-Line Usage

```python
from ardupilot_sysid.src.parser.dflog_reader import DFLogReader
data = DFLogReader('flight.bin').parse()
```

## Common Operations

### Parse a log file
```python
reader = DFLogReader('flight.bin')
data = reader.parse()
```

### Access data
```python
imu_df = data['imu']      # IMU data (400 Hz)
rcout_df = data['rcout']  # PWM outputs (50 Hz)
att_df = data['att']      # Attitude (400 Hz)
gps_df = data['gps']      # GPS (5-10 Hz)
baro_df = data['baro']    # Barometer (10 Hz)
ekf_df = data['ekf']      # EKF health (10 Hz)
params = data['params']   # Parameters (dict)
```

### Print summary
```python
from ardupilot_sysid.src.parser.dflog_reader import print_log_summary
summary = reader.get_log_summary(data)
print_log_summary(summary)
```

### Filter by EKF health
```python
from ardupilot_sysid.src.parser import ekf_health

segments = ekf_health.filter_ekf_healthy_segments(
    data['ekf'],
    innovation_threshold=1.0,
    min_segment_duration=5.0
)

imu_healthy = ekf_health.apply_segment_filter(data['imu'], segments)
```

### Check data availability
```python
for msg_type, df in data.items():
    if msg_type != 'params':
        print(f"{msg_type}: {len(df)} samples")
```

### Get column names
```python
from ardupilot_sysid.src.parser import message_types as mt

print(mt.GYRO_COLS)    # ['gyr_x', 'gyr_y', 'gyr_z']
print(mt.ACCEL_COLS)   # ['acc_x', 'acc_y', 'acc_z']
print(mt.ATTITUDE_COLS)  # ['roll', 'pitch', 'yaw']
```

## DataFrame Schemas

### IMU
```
timestamp: float (seconds)
gyr_x, gyr_y, gyr_z: float (rad/s, body frame)
acc_x, acc_y, acc_z: float (m/s², body frame)
```

### RCOUT
```
timestamp: float (seconds)
pwm_1, ..., pwm_14: int (microseconds, 1000-2000)
```

### ATT
```
timestamp: float (seconds)
roll, pitch, yaw: float (radians, Euler angles)
```

### GPS
```
timestamp: float (seconds)
lat, lng: float (degrees)
alt: float (meters)
gps_speed: float (m/s, ground speed)
vel_n, vel_e, vel_d: float (m/s, NED frame)
```

### BARO
```
timestamp: float (seconds)
baro_alt: float (meters, relative altitude)
baro_press: float (Pascals)
```

### EKF
```
timestamp: float (seconds)
innovation_ratio: float (dimensionless, <1.0 is healthy)
```

## Unit Conversions

| Input | Output | Note |
|-------|--------|------|
| TimeUS (µs) | timestamp (s) | Normalized to start at 0 |
| Angles (deg) | Angles (rad) | × π/180 |
| Gyro (rad/s) | Gyro (rad/s) | No change |
| Accel (m/s²) | Accel (m/s²) | No change |
| PWM (µs) | PWM (µs) | No change (1000-2000) |

## Constants Reference

```python
from ardupilot_sysid.src.parser import message_types as mt

# Message types
mt.IMU_MSG, mt.RCOUT_MSG, mt.ATT_MSG, mt.GPS_MSG, mt.BARO_MSG, mt.EKF_MSG

# Standard columns
mt.TIMESTAMP_COL          # 'timestamp'
mt.GYRO_COLS              # ['gyr_x', 'gyr_y', 'gyr_z']
mt.ACCEL_COLS             # ['acc_x', 'acc_y', 'acc_z']
mt.ATTITUDE_COLS          # ['roll', 'pitch', 'yaw']
mt.GPS_VELOCITY_COLS      # ['vel_n', 'vel_e', 'vel_d']
mt.PWM_COLS               # ['pwm_1', ..., 'pwm_14']

# Unit conversions
mt.DEG_TO_RAD             # π/180
mt.US_TO_SEC              # 1e-6

# Typical rates
mt.IMU_RATE_HZ            # 400
mt.RCOUT_RATE_HZ          # 50
mt.GPS_RATE_HZ            # 10
```

## Error Handling

```python
try:
    reader = DFLogReader('flight.bin')
    data = reader.parse()
except FileNotFoundError:
    print("Log file not found")
except ValueError as e:
    print(f"Invalid file: {e}")
except RuntimeError as e:
    print(f"Parse error: {e}")
```

## EKF Health

Innovation ratio interpretation:
- `< 0.5`: Excellent
- `0.5 - 1.0`: Good (use this data)
- `1.0 - 2.0`: Marginal
- `> 2.0`: Poor (discard)

## Demo Script

```bash
python examples/demo_parser.py /path/to/flight.bin
```

## Testing

```bash
python -m pytest tests/test_parser.py -v
```

## Documentation

- `README.md` - Full documentation
- `IMPLEMENTATION_NOTES.md` - Technical details
- `VERIFICATION_REPORT.md` - Test results
- `QUICK_REFERENCE.md` - This file
