# Data Preprocessing Module

This module implements Phase 2, Part 2 of the ArduPilot SITL parameter identification project: data preprocessing (alignment, resampling, segmentation).

## Overview

The preprocessing module prepares raw sensor data for system identification by:
1. **Timestamp Alignment**: Compensating for hardware latencies using cross-correlation
2. **Resampling**: Converting all sensors to a uniform time grid
3. **Segmentation**: Filtering data based on EKF health metrics

## Module Structure

```
ardupilot_sysid/src/preprocessing/
├── __init__.py          # Package exports
├── align.py             # Timestamp alignment via cross-correlation
├── resample.py          # Resampling to uniform time grid
└── segment.py           # EKF health-based segmentation
```

## Key Concepts

### 1. Timestamp Alignment

Different sensors have different hardware latencies:
- **IMU**: ~1ms (fastest, used as reference)
- **GPS**: 100-200ms (signal propagation + processing delay)
- **RCOUT**: ~5ms (negligible, accounted for in motor model)

**Cross-Correlation Strategy:**
```python
# Integrate IMU acceleration → velocity estimate
imu_velocity = integrate(imu_acceleration)

# GPS provides direct velocity measurements
gps_velocity = measured

# Cross-correlate to find lag
lag = argmax(correlate(imu_velocity, gps_velocity))

# Shift GPS timestamps backward to compensate
gps_aligned = gps - lag
```

**Why it works:**
- GPS velocity is accurate but delayed
- Integrated IMU gives velocity estimate but drifts
- High-pass filtering removes DC offset/drift
- Peak correlation reveals hardware latency

### 2. Resampling

Sensor data arrives at different rates:
- IMU: 400 Hz
- GPS: 10 Hz
- RCOUT: 50 Hz
- Baro: 10 Hz

All sensors are resampled to a common **400 Hz** grid using linear interpolation.

**Benefits:**
- Simplifies downstream processing
- Enables element-wise operations
- Consistent time base for derivatives/integrals

### 3. Segmentation

Only use data where EKF (Extended Kalman Filter) is healthy:

**EKF Innovation Ratio:**
- < 0.5: Excellent
- 0.5-1.0: Good (acceptable)
- \> 1.0: Poor (reject)

**Unhealthy EKF indicates:**
- GPS glitches/multipath
- Compass interference
- Excessive vibration
- Sensor failures

## Usage Examples

### Basic Pipeline

```python
from ardupilot_sysid.src.preprocessing import (
    align_timestamps,
    resample_to_uniform_grid,
    segment_by_ekf_health,
    apply_segments,
)

# Step 1: Align timestamps
imu_aligned, gps_aligned, rcout_aligned, metadata = align_timestamps(
    imu_df,
    gps_df,
    rcout_df
)

print(f"GPS latency detected: {metadata['gps_latency_ms']:.1f} ms")

# Step 2: Resample to uniform grid
dataframes = {
    'imu': imu_aligned,
    'gps': gps_aligned,
    'rcout': rcout_aligned,
    'ekf': ekf_df,
}

resampled = resample_to_uniform_grid(dataframes, target_rate_hz=400.0)

# Step 3: Segment by EKF health
segments = segment_by_ekf_health(
    resampled['ekf'],
    innovation_threshold=1.0,
    min_segment_duration_s=5.0
)

# Step 4: Apply segments to all data
imu_segments = apply_segments(resampled['imu'], segments)
gps_segments = apply_segments(resampled['gps'], segments)

# Now you have list of healthy data segments
for i, (imu_seg, gps_seg) in enumerate(zip(imu_segments, gps_segments)):
    print(f"Segment {i}: {len(imu_seg)} samples")
```

### Timestamp Jitter Analysis

```python
from ardupilot_sysid.src.preprocessing import check_timestamp_jitter

stats = check_timestamp_jitter(imu_df, name='IMU')

print(f"Sample rate: {stats['sample_rate_hz']:.1f} Hz")
print(f"Max jitter: {stats['max_jitter_ms']:.3f} ms")
print(f"Outliers: {stats['num_outliers']}")
```

### Advanced Segmentation

```python
from ardupilot_sysid.src.preprocessing import (
    segment_by_ekf_health,
    merge_close_segments,
    filter_segments_by_criteria,
    summarize_segments,
)

# Find healthy segments
segments = segment_by_ekf_health(ekf_df, innovation_threshold=1.0)

# Merge segments with small gaps
segments = merge_close_segments(segments, max_gap_s=2.0)

# Keep only the longest segments
segments = filter_segments_by_criteria(
    segments,
    min_duration_s=10.0,
    max_count=3  # Keep top 3
)

# Print statistics
stats = summarize_segments(segments)
print(f"Found {stats['n_segments']} segments")
print(f"Total duration: {stats['total_duration_s']:.1f} s")
print(f"Mean duration: {stats['mean_duration_s']:.1f} s")
```

## Function Reference

### align.py

#### `align_timestamps(imu_df, gps_df, rcout_df, max_lag_ms=500.0)`
Align sensor streams by cross-correlating IMU and GPS velocity.

**Returns:**
- `(imu_df, gps_aligned_df, rcout_df, metadata)`

**Metadata keys:**
- `gps_latency_ms`: Detected GPS latency in milliseconds
- `correlation_peak`: Peak correlation value
- `lag_samples`: Lag in samples
- `quality`: Signal-to-noise ratio (> 5.0 is good)

#### `check_timestamp_jitter(df, name='data')`
Analyze timestamp jitter (variation in sample intervals).

**Returns:**
- Dict with `sample_rate_hz`, `mean_dt`, `std_dt`, `max_jitter_ms`

### resample.py

#### `resample_to_uniform_grid(dataframes, target_rate_hz=400.0, method='linear')`
Resample all sensor streams to a uniform time grid.

**Args:**
- `dataframes`: Dict of DataFrames (e.g., `{'imu': imu_df, 'gps': gps_df}`)
- `target_rate_hz`: Target sample rate (default 400 Hz)
- `method`: Interpolation method ('linear', 'cubic', 'nearest')

**Returns:**
- Dict of resampled DataFrames on uniform grid

#### `resample_single_stream(df, target_rate_hz, method='linear')`
Resample a single DataFrame to a uniform time grid.

### segment.py

#### `segment_by_ekf_health(ekf_df, innovation_threshold=1.0, min_segment_duration_s=5.0)`
Find time ranges where EKF is healthy.

**Returns:**
- List of `(start_time, end_time)` tuples

#### `apply_segments(df, segments)`
Split DataFrame into multiple DataFrames, one per segment.

**Returns:**
- List of DataFrames

#### `merge_close_segments(segments, max_gap_s=2.0)`
Merge segments that are close together in time.

#### `filter_segments_by_criteria(segments, min_duration_s=None, max_duration_s=None, max_count=None)`
Filter segments by duration and count criteria.

#### `summarize_segments(segments)`
Compute summary statistics for segments.

**Returns:**
- Dict with `n_segments`, `total_duration_s`, `mean_duration_s`, etc.

## Quality Metrics

### Alignment Quality

**Good alignment:**
- Quality > 5.0
- GPS latency: 100-250 ms
- Correlation peak > 1e2

**Poor alignment:**
- Quality < 2.0
- Latency outside 50-500 ms range
- May indicate:
  - Poor GPS signal
  - Excessive vibration
  - Bad sensor calibration

### Segmentation Coverage

**Good coverage:**
- \> 50% of log has healthy EKF
- Segments > 10s each
- 3+ segments available

**Poor coverage:**
- < 30% healthy
- Segments < 5s
- May need different flight log

## Testing

Run the test suite:

```bash
pytest tests/test_preprocessing.py -v
```

**Test coverage:**
- ✅ Timestamp alignment with synthetic data
- ✅ Timestamp jitter analysis
- ✅ Resampling (up/down sampling)
- ✅ Signal preservation
- ✅ EKF health segmentation
- ✅ Segment merging/filtering
- ✅ Full integration pipeline

## Example Output

See `notebooks/preprocessing_example.py` for a complete demonstration.

```bash
python3 notebooks/preprocessing_example.py
```

**Expected output:**
```
======================================================================
PREPROCESSING PIPELINE EXAMPLE
======================================================================

Generated data:
  IMU: 8000 samples @ ~400 Hz
  GPS: 200 samples @ ~10 Hz (with 150ms artificial latency)
  RCOUT: 1000 samples @ ~50 Hz
  EKF: 200 samples @ ~10 Hz

----------------------------------------------------------------------
STEP 2: Timestamp Alignment
----------------------------------------------------------------------

============================================================
TIMESTAMP ALIGNMENT REPORT
============================================================
IMU sample rate: 400.0 Hz
GPS latency detected: 150.2 ms
...
```

## Performance

**Typical processing time (20s flight log):**
- Alignment: ~50 ms
- Resampling: ~100 ms
- Segmentation: ~10 ms

**Memory usage:**
- ~5 MB per 10 seconds of flight data at 400 Hz
- Peak: ~20 MB during resampling

## Limitations

1. **Cross-correlation assumes:**
   - Reasonable body-to-NED frame alignment
   - GPS velocity is fairly accurate
   - IMU accelerometer is not saturated

2. **Linear interpolation:**
   - Not suitable for very high-frequency content
   - May smooth sharp transitions
   - Consider 'cubic' for smoother signals

3. **EKF segmentation:**
   - Relies on ArduPilot's EKF implementation
   - Innovation ratio definition may vary
   - Manual inspection recommended for critical flights

## Next Steps

After preprocessing, data is ready for:

1. **Phase 3**: RTS (Rauch-Tung-Striebel) Smoothing
2. **Phase 4**: Differentiable Flight Dynamics Model
3. **Phase 5**: Excitation Analysis
4. **Phase 6**: MAP Optimization

## References

- ArduPilot DataFlash Log Format: https://ardupilot.org/copter/docs/logmessages.html
- EKF Innovation Ratio: Gelb, A. (1974). Applied Optimal Estimation
- Cross-correlation: Oppenheim & Schafer (2009). Discrete-Time Signal Processing
