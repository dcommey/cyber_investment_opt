import argparse
import json
from pathlib import Path

import pandas as pd

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
from src.scripts.run_full_experiments import ensure_case_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TSG-oriented power-grid cyber investment case study."
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Path to a MATPOWER case file; if omitted, the IEEE 118-bus case is downloaded locally",
    )
    parser.add_argument("--budget", type=float, default=10.0, help="Defender budget")
    parser.add_argument("--attack-budget", type=float, default=3.0, help="Attacker budget")
    parser.add_argument(
        "--propagation-strength",
        type=float,
        default=0.35,
        help="Network propagation strength eta with eta*rho(P) < 1",
    )
    parser.add_argument(
        "--output",
        default="results/power_grid_case_study",
        help="Directory for summary outputs",
    )
    args = parser.parse_args()

    if args.case is None:
        case_paths = ensure_case_files()
        case_path = case_paths.get(118)
        if case_path is None:
            raise RuntimeError("Unable to download or locate the default IEEE 118-bus case")
    else:
        case_path = Path(args.case)

    case = parse_matpower_case(case_path)
    bus_order = case.bus_ids

    baseline_vulnerability = build_baseline_vulnerability(case, bus_order=bus_order)
    consequence_weights = build_bus_criticality(case, bus_order=bus_order)
    propagation_matrix = build_propagation_matrix(case, bus_order=bus_order)

    model = PowerGridStackelbergModel(
        propagation_matrix=propagation_matrix,
        baseline_vulnerability=baseline_vulnerability,
        consequence_weights=consequence_weights,
        defense=DefenseParameters(budget=args.budget, alpha=0.35, regularization=1e-3),
        attack=AttackParameters(budget=args.attack_budget, max_attack_per_bus=1.0),
        influence=InfluenceParameters(propagation_strength=args.propagation_strength),
        bus_ids=bus_order,
    )
    solution = model.solve_stackelberg()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    demand = case.load_vector(bus_order)
    generation = case.generation_capacity_vector(bus_order)
    bus_frame = pd.DataFrame(
        {
            "bus_id": bus_order,
            "load_mw": demand,
            "generation_capacity_mw": generation,
            "baseline_vulnerability": baseline_vulnerability,
            "consequence_weight": consequence_weights,
            "defense": solution.defense,
            "attack": solution.attack,
            "attacker_score": solution.attacker_scores,
            "propagated_compromise": solution.propagated_compromise,
        }
    ).sort_values(["defense", "attacker_score"], ascending=False)
    bus_frame.to_csv(output_dir / "bus_level_solution.csv", index=False)

    summary = {
        "case": case.name,
        "num_buses": len(bus_order),
        "defense_budget": args.budget,
        "attack_budget": args.attack_budget,
        "propagation_strength": args.propagation_strength,
        "objective": solution.objective,
        "max_defense": float(bus_frame["defense"].max()),
        "max_attack": float(bus_frame["attack"].max()),
        "total_propagated_compromise": float(bus_frame["propagated_compromise"].sum()),
        "total_demand_mw": float(demand.sum()),
        "total_generation_capacity_mw": float(generation.sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved case-study outputs to {output_dir}")


if __name__ == "__main__":
    main()
