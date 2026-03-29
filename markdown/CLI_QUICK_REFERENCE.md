# CLI Quick Reference Card

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```bash
python -m cli.sysid \
    --log flight.bin \
    --frame quad_x \
    --mass 1.2 \
    --output sitl_params.parm
```

## Common Commands

### Check Excitation
```bash
python -m cli.sysid \
    --log flight.bin \
    --frame quad_x \
    --mass 1.2 \
    --output dummy.parm \
    --excitation-check-only
```

### Full Identification with Report
```bash
python -m cli.sysid \
    --log flight.bin \
    --frame quad_x \
    --mass 1.2 \
    --output sitl_params.parm \
    --report results.json \
    --verbose
```

### With Physical Measurements
```bash
python -m cli.sysid \
    --log flight.bin \
    --frame quad_x \
    --mass 1.2 \
    --mass-std 0.05 \
    --arm-length 0.165 \
    --prop-diameter 5.1 \
    --motor-kv 2300 \
    --output sitl_params.parm
```

## Frame Types

- `quad_x` - Quadcopter X configuration
- `quad_plus` - Quadcopter + configuration
- `hexa_x` - Hexacopter X configuration
- `octo_x` - Octocopter X configuration

## Required Options

| Option | Description | Example |
|--------|-------------|---------|
| `--log` | Path to .bin log file | `flight.bin` |
| `--frame` | Vehicle frame type | `quad_x` |
| `--mass` | Vehicle mass (kg) | `1.2` |
| `--output` | Output .parm file | `sitl.parm` |

## Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mass-std` | 0.05 | Mass uncertainty (kg) |
| `--arm-length` | 0.165 | Motor arm length (m) |
| `--arm-length-std` | 0.005 | Arm length uncertainty (m) |
| `--prop-diameter` | 5.1 | Propeller diameter (in) |
| `--motor-kv` | 2300 | Motor KV rating |
| `--report` | None | JSON report output path |
| `--verbose` | Off | Show detailed progress |
| `--excitation-check-only` | Off | Only check excitation |

## Help & Version

```bash
python -m cli.sysid --help     # Show help
python -m cli.sysid --version  # Show version
```

## Output Files

### SITL Parameter File (.parm)
```
SIM_MASS,1.187
SIM_MOI_X,0.0089
SIM_MOI_Y,0.0091
SIM_MOI_Z,0.0163
...
```

### JSON Report (optional)
```json
{
  "identified_parameters": {...},
  "excitation_scores": {...},
  "validation": {...}
}
```

## Typical Workflow

1. **Check data quality**
   ```bash
   python -m cli.sysid --log flight.bin --frame quad_x --mass 1.2 \
       --output dummy.parm --excitation-check-only
   ```

2. **Run identification**
   ```bash
   python -m cli.sysid --log flight.bin --frame quad_x --mass 1.2 \
       --output sitl_params.parm --report results.json --verbose
   ```

3. **Load into SITL**
   ```bash
   sim_vehicle.py -v ArduCopter -P sitl_params.parm
   ```

## Tips

- Weigh your vehicle precisely
- Fly with varied maneuvers (not just hover)
- Ensure good GPS lock during flight
- Check validation RMSE (< 2° is good)
- Use `--verbose` for debugging

## Troubleshooting

| Error | Solution |
|-------|----------|
| "No healthy EKF segments" | Fly with better GPS reception |
| "Data quality is POOR" | Fly suggested maneuvers |
| "No IMU data found" | Check log file integrity |
| Import errors | Run from project root |

## Documentation

- Tutorial: `QUICKSTART_CLI.md`
- Full docs: `cli/README.md`
- Technical: `CLI_IMPLEMENTATION_SUMMARY.md`

## Example: Complete Workflow

```bash
# 1. Check data quality
python -m cli.sysid \
    --log my_flight.bin \
    --frame quad_x \
    --mass 1.2 \
    --output check.parm \
    --excitation-check-only

# 2. If quality is good, run full identification
python -m cli.sysid \
    --log my_flight.bin \
    --frame quad_x \
    --mass 1.2 \
    --mass-std 0.05 \
    --arm-length 0.165 \
    --prop-diameter 5.1 \
    --motor-kv 2300 \
    --output sitl_params.parm \
    --report results.json \
    --verbose

# 3. Review results
cat sitl_params.parm
python -c "import json; print(json.dumps(json.load(open('results.json')), indent=2))"

# 4. Load into SITL
sim_vehicle.py -v ArduCopter --console --map -P sitl_params.parm
```

## Status Indicators

- ✓ Green checkmark = Success
- ⚠ Yellow warning = Non-critical issue
- ✗ Red X = Error (stopped)

## Exit Codes

- `0` - Success
- `1` - Error
- `130` - Interrupted by user (Ctrl+C)
