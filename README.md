# Power-System-Informed Cybersecurity Investment for Power-Grid Control Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code for a Stackelberg cybersecurity investment model for transmission-grid control networks. The project combines MATPOWER case parsing, power-system-informed consequence weighting, topology-driven propagation, and an exact greedy attacker best response.

## What this repository includes

- reusable model components in `src/power_grid`
- experiment runners in `src/scripts`
- regression tests in `tests`
- packaging metadata for local installation and reuse

Large benchmark files, generated results, and manuscript sources are intentionally excluded from the public repository view. They are created or maintained locally as needed.

## Core components

- `src/power_grid/matpower.py` — MATPOWER parser and case container
- `src/power_grid/criticality.py` — consequence, vulnerability, and propagation builders
- `src/power_grid/model.py` — Stackelberg defender-attacker model with exact attacker response
- `src/scripts/run_power_grid_case_study.py` — single-case runner
- `src/scripts/run_full_experiments.py` — benchmark, ablation, and sensitivity runner
- `src/scripts/generate_paper_plots.py` — figure generation from experiment JSON outputs

## Installation

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Quick start

### Run the default IEEE 118 case study

```bash
python -m src.scripts.run_power_grid_case_study --output outputs/case118
```

If no case path is provided, the script downloads the IEEE 118-bus benchmark locally and writes outputs to the specified directory.

### Run the benchmark suite

```bash
python src/scripts/run_full_experiments.py
```

Missing PGLib benchmark cases are downloaded automatically into a local ignored `data/` directory. Experiment JSON outputs are written to a local ignored `results/` directory.

### Run tests

```bash
pytest
```

## Repository policy

The public GitHub repository is code-focused. The following directories are intentionally not tracked:

- `paper/`
- `data/`
- `results/`

This keeps the repository lightweight while preserving full local reproducibility.

## License

This project is licensed under the MIT License. See `LICENSE`.
