from __future__ import annotations

import json
from pathlib import Path

import pytest


SMALL_MATPOWER_CASE = """function mpc = case3
mpc.version = '2';
mpc.baseMVA = 100;

mpc.bus = [
1 3 50 30 0 0 1 1 0 230 1 1.05 0.95;
2 1 70 20 0 0 1 1 0 230 1 1.05 0.95;
3 2 10 5 0 0 1 1 0 230 1 1.05 0.95;
];

mpc.gen = [
1 80 0 100 -100 1 100 1 150 0;
3 40 0 100 -100 1 100 1 60 0;
];

mpc.branch = [
1 2 0.01 0.03 0.0 100 100 100 0 0 1 -360 360;
2 3 0.02 0.04 0.0 90 90 90 0 0 1 -360 360;
1 3 0.03 0.05 0.0 70 70 70 0 0 0 -360 360;
];
"""


@pytest.fixture
def small_matpower_case(tmp_path: Path) -> Path:
    case_path = tmp_path / "case3.m"
    case_path.write_text(SMALL_MATPOWER_CASE, encoding="utf-8")
    return case_path


@pytest.fixture
def sample_results_dir(tmp_path: Path) -> Path:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    main_results = {
        "14": {
            "stackelberg": {"loss": 0.30, "solve_time": 0.20},
            "baselines": {
                "Uniform": {"loss": 0.40, "stackelberg_improvement_pct": 25.0},
                "Load-weighted": {"loss": 0.35, "stackelberg_improvement_pct": 14.3},
                "Degree-based": {"loss": 0.38},
                "Generation-weighted": {"loss": 0.34},
                "Vulnerability-weighted": {"loss": 0.37},
                "Marginal-consequence-weighted": {"loss": 0.33},
            },
        },
        "39": {
            "stackelberg": {"loss": 0.28, "solve_time": 0.90},
            "baselines": {
                "Uniform": {"loss": 0.36, "stackelberg_improvement_pct": 22.2},
                "Load-weighted": {"loss": 0.32, "stackelberg_improvement_pct": 12.5},
                "Degree-based": {"loss": 0.35},
                "Generation-weighted": {"loss": 0.31},
                "Vulnerability-weighted": {"loss": 0.34},
                "Marginal-consequence-weighted": {"loss": 0.30},
            },
        },
        "57": {
            "stackelberg": {"loss": 0.26, "solve_time": 1.40},
            "baselines": {
                "Uniform": {"loss": 0.42, "stackelberg_improvement_pct": 38.1},
                "Load-weighted": {"loss": 0.31, "stackelberg_improvement_pct": 16.1},
                "Degree-based": {"loss": 0.36},
                "Generation-weighted": {"loss": 0.30},
                "Vulnerability-weighted": {"loss": 0.35},
                "Marginal-consequence-weighted": {"loss": 0.29},
            },
        },
        "118": {
            "stackelberg": {"loss": 0.2618, "solve_time": 6.64},
            "baselines": {
                "Uniform": {"loss": 0.3265, "stackelberg_improvement_pct": 19.8},
                "Load-weighted": {"loss": 0.3135, "stackelberg_improvement_pct": 16.5},
                "Degree-based": {"loss": 0.3200},
                "Generation-weighted": {"loss": 0.3000},
                "Vulnerability-weighted": {"loss": 0.3198},
                "Marginal-consequence-weighted": {"loss": 0.3132},
            },
        },
    }
    ablation_results = {
        "no_attacker": {"loss": 0.2691},
        "no_propagation": {"loss": 0.2637},
        "no_physics_weights": {"loss": 0.2817},
    }
    sensitivity_results = {
        "defender_budget": {
            "2": {"stackelberg": 0.42, "uniform": 0.45},
            "5": {"stackelberg": 0.34, "uniform": 0.39},
            "10": {"stackelberg": 0.2618, "uniform": 0.3265},
        },
        "attacker_budget": {
            "1": {"stackelberg": 0.21, "uniform": 0.25},
            "3": {"stackelberg": 0.2618, "uniform": 0.3265},
            "5": {"stackelberg": 0.31, "uniform": 0.39},
        },
        "propagation_strength": {
            "0.0": {"stackelberg": 0.18, "uniform": 0.22},
            "0.35": {"stackelberg": 0.2618, "uniform": 0.3265},
            "0.45": {"stackelberg": 0.30, "uniform": 0.37},
        },
    }

    (results_dir / "main_results.json").write_text(json.dumps(main_results), encoding="utf-8")
    (results_dir / "ablation_results.json").write_text(json.dumps(ablation_results), encoding="utf-8")
    (results_dir / "sensitivity_results.json").write_text(json.dumps(sensitivity_results), encoding="utf-8")
    return results_dir
