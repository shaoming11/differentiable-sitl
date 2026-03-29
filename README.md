# ArduPilot SITL Parameter Identification

Automatic identification of Flight Dynamics Model (FDM) parameters from ArduPilot DataFlash logs using differentiable optimization with JAX.

## Overview

ArduPilot's SITL (Software-In-The-Loop) simulator uses hardcoded generic FDM parameters that don't match real vehicles. This causes gains tuned in simulation to behave differently on real hardware. This tool solves that problem by:

1. **Parsing** real ArduPilot `.bin` flight logs
2. **Identifying** vehicle-specific FDM parameters through differentiable optimization
3. **Outputting** ready-to-use SITL parameter files

### Core Innovation

The FDM is implemented as a pure JAX function, enabling `jax.grad` to produce exact Jacobians for free. This eliminates finite-difference approximation error and dramatically accelerates convergence.

## Features

- **5-stage pipeline**: Parse & align → RTS smoother → Excitation analysis → MAP optimization → Validation & output
- **Differentiable FDM in JAX**: Exact gradients via automatic differentiation
- **Excitation analysis**: Fisher Information Matrix checks parameter identifiability
- **MAP optimization**: Levenberg-Marquardt with physical priors and Laplace approximation
- **Confidence intervals**: Per-parameter uncertainty quantification
- **Output formats**: ArduPilot-compatible `.parm` files and detailed JSON reports

## Installation

```bash
# Clone the repository
git clone https://github.com/shaoming/differentiable-sitl.git
cd differentiable-sitl

# Install in editable mode with dependencies
pip install -e .
```

## Usage (Planned)

```bash
# Basic usage
ardupilot-sysid --log flight.bin --frame quad_x --mass 1.2 --output sitl_params.parm

# With physical priors
ardupilot-sysid --log flight.bin --frame quad_x \
    --mass 1.2 --mass-std 0.05 \
    --arm-length 0.165 \
    --prop-diameter 5.1 \
    --motor-kv 2300 \
    --output sitl_params.parm \
    --report report.json

# Excitation check only
ardupilot-sysid --log flight.bin --frame quad_x --excitation-check-only
```

## Development Status

**Phase 1: Project Setup** ✅ (Current)
- Package structure established
- Dependencies defined
- Basic CLI skeleton

**Phase 2-9: Implementation** 🚧 (In Progress)
- See [PRD.md](PRD.md) for detailed implementation plan

## Project Structure

```
ardupilot_sysid/
├── src/
│   ├── parser/          # DataFlash log parsing
│   ├── preprocessing/   # Alignment, resampling, segmentation
│   ├── smoother/        # RTS smoother for state estimation
│   ├── fdm/            # Differentiable flight dynamics models
│   ├── analysis/       # Excitation & identifiability
│   ├── optimizer/      # MAP optimization
│   ├── validation/     # Model validation
│   └── output/         # .parm and report generation
├── cli/                # Command-line interface
├── tests/              # Test suite
└── notebooks/          # Jupyter notebooks for exploration
```

## Requirements

- Python >= 3.8
- JAX >= 0.4.25 (CPU or GPU)
- pymavlink >= 2.4.40
- pandas >= 2.1
- scipy >= 1.12
- click >= 8.1

See [requirements.txt](requirements.txt) for full dependency list.

## Author

Shaoming Wu

## References

- [ArduPilot SITL Documentation](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)
- [JAX Documentation](https://jax.readthedocs.io/)
- [PRD: Detailed Project Specification](PRD.md)
