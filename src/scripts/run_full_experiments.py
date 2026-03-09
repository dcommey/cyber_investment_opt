"""Comprehensive experiment runner for the IEEE TSG paper.

Downloads missing IEEE test cases, runs the Stackelberg solver and baselines,
performs ablation studies and sensitivity analysis, and saves results as JSON.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.power_grid import (
    AttackParameters,
    DefenseParameters,
    InfluenceParameters,
    PowerGridStackelbergModel,
    build_baseline_vulnerability,
    build_bus_criticality,
    build_propagation_matrix,
    parse_matpower_case,
)


PGLIB_BASE = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master"
CASE_FILES = {
    14: "pglib_opf_case14_ieee.m",
    39: "pglib_opf_case39_epri.m",
    57: "pglib_opf_case57_ieee.m",
    118: "pglib_opf_case118_ieee.m",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "real_world"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODEL_CONFIG_KEYS = {
    "budget_w",
    "budget_a",
    "eta",
    "alpha",
    "lam",
    "a_cap",
    "load_wt",
    "gen_wt",
    "cent_wt",
    "consequence_mode",
}
SOLVER_CONFIG_KEYS = {"num_random_starts", "random_seed", "maxiter"}


def ensure_case_files() -> dict[int, Path]:
    """Download missing MATPOWER case files from PGLib-OPF."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for n_bus, fname in CASE_FILES.items():
        local = DATA_DIR / fname
        if not local.exists():
            url = f"{PGLIB_BASE}/{fname}"
            print(f"Downloading {fname} ...")
            try:
                urllib.request.urlretrieve(url, local)
                print(f"  Saved to {local}")
            except Exception as e:
                print(f"  Failed to download {fname}: {e}")
                continue
        paths[n_bus] = local
    return paths


def build_model(
    case_path: Path,
    budget_w: float = 10.0,
    budget_a: float = 3.0,
    eta: float = 0.35,
    alpha: float = 0.35,
    lam: float = 1e-3,
    a_cap: float = 1.0,
    load_wt: float = 0.50,
    gen_wt: float = 0.30,
    cent_wt: float = 0.20,
    consequence_mode: str = "physics",
):
    """Build a PowerGridStackelbergModel from a MATPOWER case file."""
    case = parse_matpower_case(case_path)
    bus_order = case.bus_ids

    if consequence_mode == "physics":
        c = build_bus_criticality(
            case,
            bus_order=bus_order,
            load_weight=load_wt,
            generation_weight=gen_wt,
            centrality_weight=cent_wt,
        )
    elif consequence_mode == "uniform":
        c = np.full(len(bus_order), 1.0 / len(bus_order), dtype=float)
    else:
        raise ValueError(f"Unsupported consequence_mode: {consequence_mode}")

    v = build_baseline_vulnerability(case, bus_order=bus_order)
    P = build_propagation_matrix(case, bus_order=bus_order)

    model = PowerGridStackelbergModel(
        propagation_matrix=P,
        baseline_vulnerability=v,
        consequence_weights=c,
        defense=DefenseParameters(budget=budget_w, alpha=alpha, regularization=lam),
        attack=AttackParameters(budget=budget_a, max_attack_per_bus=a_cap),
        influence=InfluenceParameters(propagation_strength=eta),
        bus_ids=bus_order,
    )
    return model, case


def _normalized_allocation(weights: np.ndarray, budget: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be a one-dimensional vector")
    weights = np.maximum(weights, 0.0)
    if weights.size == 0:
        return weights

    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, budget / weights.size, dtype=float)
    return budget * weights / total


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in MODEL_CONFIG_KEYS if key in config}


def _solver_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in SOLVER_CONFIG_KEYS if key in config}


def evaluate_allocation(model, z):
    """Evaluate a defense allocation against the optimal attacker."""
    a = model.attacker_best_response(z)
    prop = model.propagated_compromise(z, a)
    loss = float(model.consequence_weights @ prop)
    reg = 0.5 * model.regularization * float(np.dot(z, z))
    return {
        "loss": loss,
        "loss_with_reg": loss + reg,
        "total_propagated": float(np.sum(prop)),
        "max_defense": float(np.max(z)),
        "total_attack": float(np.sum(a)),
    }


def build_baseline_allocations(model, case) -> dict[str, np.ndarray]:
    """Construct baseline defense allocations using case-derived signals."""
    n = model.n
    W = model.defense_budget
    bus_order = list(model.bus_ids)
    graph = case.physical_graph()

    degrees = np.array([graph.degree(bus_id) for bus_id in bus_order], dtype=float)
    loads = case.load_vector(bus_order)
    generation = case.generation_capacity_vector(bus_order)
    vulnerability = np.maximum(model.baseline_vulnerability, 0.0)
    marginal = np.maximum(model._marginal_consequence, 0.0)

    return {
        "Uniform": np.full(n, W / n, dtype=float),
        "Degree-based": _normalized_allocation(degrees, W),
        "Load-weighted": _normalized_allocation(loads, W),
        "Generation-weighted": _normalized_allocation(generation, W),
        "Vulnerability-weighted": _normalized_allocation(vulnerability, W),
        "Marginal-consequence-weighted": _normalized_allocation(marginal, W),
    }


def run_baselines(model, case):
    """Run all baseline allocations."""
    results = {}
    for name, allocation in build_baseline_allocations(model, case).items():
        baseline_result = evaluate_allocation(model, allocation)
        baseline_result["budget"] = float(np.sum(allocation))
        results[name] = baseline_result

    return results


def greedy_marginal(model, steps=100):
    """Greedy marginal risk reduction heuristic."""
    n = model.n
    W = model.defense_budget
    z = np.zeros(n)
    step_size = W / steps

    for _ in range(steps):
        best_i = -1
        best_reduction = -np.inf
        current_obj = model.defender_objective(z)
        for i in range(n):
            z_trial = z.copy()
            z_trial[i] += step_size
            if np.sum(z_trial) > W + 1e-10:
                continue
            trial_obj = model.defender_objective(z_trial)
            reduction = current_obj - trial_obj
            if reduction > best_reduction:
                best_reduction = reduction
                best_i = i
        if best_i >= 0 and np.sum(z) + step_size <= W + 1e-10:
            z[best_i] += step_size
        else:
            break
    return z


def run_case(case_path: Path, n_bus: int, config: dict):
    """Run full evaluation for one benchmark case."""
    print(f"\n{'='*60}")
    print(f"IEEE {n_bus}-bus system")
    print(f"{'='*60}")

    model, case = build_model(case_path, **_model_kwargs(config))
    solver_kwargs = _solver_kwargs(config)
    print(f"  {len(case.buses)} buses, {len(case.branches)} branches, "
          f"{len(case.generators)} generators, "
          f"total demand = {case.total_demand():.0f} MW")

    # Stackelberg solve
    t0 = time.time()
    sol = model.solve_stackelberg(**solver_kwargs)
    solve_time = time.time() - t0
    stack_eval = evaluate_allocation(model, sol.defense)
    stack_eval["solve_time"] = solve_time
    print(f"  Stackelberg loss: {stack_eval['loss']:.6f} ({solve_time:.2f}s)")

    # Baselines
    baselines = run_baselines(model, case)
    for name, res in baselines.items():
        imp = (res["loss"] - stack_eval["loss"]) / res["loss"] * 100
        res["stackelberg_improvement_pct"] = imp
        print(f"  {name}: {res['loss']:.6f} (Stackelberg +{imp:.1f}%)")

    return {
        "n_bus": n_bus,
        "n_branches": len(case.branches),
        "n_generators": len(case.generators),
        "total_demand_mw": case.total_demand(),
        "stackelberg": stack_eval,
        "baselines": baselines,
    }


def run_ablation(case_path: Path, config: dict):
    """Ablation study: remove one modeling layer at a time."""
    print(f"\nAblation study (IEEE 118-bus)")

    results = {}
    model_config = _model_kwargs(config)
    solver_kwargs = _solver_kwargs(config)

    # Full model
    reference_model, _ = build_model(case_path, **model_config)
    sol = reference_model.solve_stackelberg(**solver_kwargs)
    full = evaluate_allocation(reference_model, sol.defense)
    results["full"] = full
    print(f"  Full model: {full['loss']:.6f}")

    # No adaptive attacker during design; evaluate under the full threat model.
    cfg_no_atk = {**model_config, "budget_a": 0.0}
    model_no_atk, _ = build_model(case_path, **cfg_no_atk)
    sol_no_atk = model_no_atk.solve_stackelberg(**solver_kwargs)
    no_atk = evaluate_allocation(reference_model, sol_no_atk.defense)
    results["no_attacker"] = no_atk
    print(f"  No attacker: {no_atk['loss']:.6f}")

    # No propagation during design; evaluate under the full propagation model.
    cfg_no_prop = {**model_config, "eta": 0.0}
    model_no_prop, _ = build_model(case_path, **cfg_no_prop)
    sol_no_prop = model_no_prop.solve_stackelberg(**solver_kwargs)
    no_prop = evaluate_allocation(reference_model, sol_no_prop.defense)
    results["no_propagation"] = no_prop
    print(f"  No propagation: {no_prop['loss']:.6f}")

    # Uniform consequence weights during design; evaluate under the full power-system-informed proxy model.
    cfg_no_phys = {**model_config, "consequence_mode": "uniform"}
    model_uniform_c, _ = build_model(case_path, **cfg_no_phys)
    sol_uniform_c = model_uniform_c.solve_stackelberg(**solver_kwargs)
    no_phys = evaluate_allocation(reference_model, sol_uniform_c.defense)
    results["no_physics_weights"] = no_phys
    print(f"  No physics weights: {no_phys['loss']:.6f}")

    return results


def run_sensitivity(case_path: Path, base_config: dict):
    """Sensitivity analysis on the 118-bus system."""
    print(f"\nSensitivity analysis (IEEE 118-bus)")
    results = {}
    solver_kwargs = _solver_kwargs(base_config)

    # Defender budget sweep
    budget_results = {}
    for W in [2, 5, 10, 15, 20]:
        cfg = {**base_config, "budget_w": W}
        model, _ = build_model(case_path, **_model_kwargs(cfg))
        sol = model.solve_stackelberg(**solver_kwargs)
        stack = evaluate_allocation(model, sol.defense)
        z_uni = np.full(model.n, W / model.n)
        uni = evaluate_allocation(model, z_uni)
        imp = (uni["loss"] - stack["loss"]) / uni["loss"] * 100
        budget_results[str(W)] = {"stackelberg": stack["loss"], "uniform": uni["loss"], "improvement": imp}
        print(f"  W={W}: Stack={stack['loss']:.4f}, Uni={uni['loss']:.4f}, Imp={imp:.1f}%")
    results["defender_budget"] = budget_results

    # Attacker budget sweep
    attack_results = {}
    for A in [1, 2, 3, 4, 5]:
        cfg = {**base_config, "budget_a": A}
        model, _ = build_model(case_path, **_model_kwargs(cfg))
        sol = model.solve_stackelberg(**solver_kwargs)
        stack = evaluate_allocation(model, sol.defense)
        z_uni = np.full(model.n, base_config["budget_w"] / model.n)
        uni = evaluate_allocation(model, z_uni)
        imp = (uni["loss"] - stack["loss"]) / uni["loss"] * 100
        attack_results[str(A)] = {"stackelberg": stack["loss"], "uniform": uni["loss"], "improvement": imp}
        print(f"  A={A}: Stack={stack['loss']:.4f}, Uni={uni['loss']:.4f}, Imp={imp:.1f}%")
    results["attacker_budget"] = attack_results

    # Propagation strength sweep
    eta_results = {}
    for eta_val in [0.0, 0.10, 0.20, 0.35, 0.45]:
        try:
            cfg = {**base_config, "eta": eta_val}
            model, _ = build_model(case_path, **_model_kwargs(cfg))
            sol = model.solve_stackelberg(**solver_kwargs)
            stack = evaluate_allocation(model, sol.defense)
            z_uni = np.full(model.n, base_config["budget_w"] / model.n)
            uni = evaluate_allocation(model, z_uni)
            imp = (uni["loss"] - stack["loss"]) / uni["loss"] * 100
            eta_results[str(eta_val)] = {"stackelberg": stack["loss"], "uniform": uni["loss"], "improvement": imp}
            print(f"  η={eta_val}: Stack={stack['loss']:.4f}, Uni={uni['loss']:.4f}, Imp={imp:.1f}%")
        except ValueError as e:
            print(f"  η={eta_val}: Skipped ({e})")
    results["propagation_strength"] = eta_results

    return results


def main():
    case_paths = ensure_case_files()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_config = {
        "budget_w": 10.0,
        "budget_a": 3.0,
        "eta": 0.35,
        "alpha": 0.35,
        "lam": 1e-3,
        "a_cap": 1.0,
        "num_random_starts": 4,
        "random_seed": 7,
        "maxiter": 300,
    }

    # Main results across all benchmarks
    all_results = {}
    for n_bus in sorted(case_paths.keys()):
        try:
            result = run_case(case_paths[n_bus], n_bus, base_config)
            all_results[str(n_bus)] = result
        except Exception as e:
            print(f"  ERROR on {n_bus}-bus: {e}")

    # Save main results
    with open(RESULTS_DIR / "main_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nMain results saved to {RESULTS_DIR / 'main_results.json'}")

    # Ablation study (118-bus)
    if 118 in case_paths:
        ablation = run_ablation(case_paths[118], base_config)
        with open(RESULTS_DIR / "ablation_results.json", "w") as f:
            json.dump(ablation, f, indent=2, default=float)
        print(f"Ablation results saved to {RESULTS_DIR / 'ablation_results.json'}")

    # Sensitivity analysis (118-bus)
    if 118 in case_paths:
        sensitivity = run_sensitivity(case_paths[118], base_config)
        with open(RESULTS_DIR / "sensitivity_results.json", "w") as f:
            json.dump(sensitivity, f, indent=2, default=float)
        print(f"Sensitivity results saved to {RESULTS_DIR / 'sensitivity_results.json'}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
