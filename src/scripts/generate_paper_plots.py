"""Generate publication-quality figures for the TSG manuscript."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "paper" / "figures"


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_case_keys(main_results: Mapping[str, Any]) -> list[str]:
    return sorted(main_results.keys(), key=lambda value: int(value))


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cross_system_summary(main_results: Mapping[str, Any], output_dir: Path) -> Path:
    case_keys = _ordered_case_keys(main_results)
    labels = [f"IEEE {key}" for key in case_keys]
    x = np.arange(len(case_keys), dtype=float)
    width = 0.24

    stack = np.array([main_results[key]["stackelberg"]["loss"] for key in case_keys], dtype=float)
    uniform = np.array([main_results[key]["baselines"]["Uniform"]["loss"] for key in case_keys], dtype=float)
    load = np.array([main_results[key]["baselines"]["Load-weighted"]["loss"] for key in case_keys], dtype=float)
    uniform_gain = np.array(
        [main_results[key]["baselines"]["Uniform"]["stackelberg_improvement_pct"] for key in case_keys],
        dtype=float,
    )
    load_gain = np.array(
        [main_results[key]["baselines"]["Load-weighted"]["stackelberg_improvement_pct"] for key in case_keys],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)

    ax = axes[0]
    bars_stack = ax.bar(x - width, stack, width=width, label="Stackelberg", color="#1f77b4")
    bars_uniform = ax.bar(x, uniform, width=width, label="Uniform", color="#9aa0a6")
    bars_load = ax.bar(x + width, load, width=width, label="Load-weighted", color="#ff7f0e")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Worst-case propagated loss")
    ax.set_title("Absolute loss across benchmark systems")
    ax.legend(loc="upper right", frameon=True)
    ax.bar_label(bars_stack, fmt="%.3f", padding=2)
    ax.bar_label(bars_uniform, fmt="%.3f", padding=2)
    ax.bar_label(bars_load, fmt="%.3f", padding=2)

    ax = axes[1]
    bars_uniform_gain = ax.bar(x - width / 2, uniform_gain, width=width, label="vs Uniform", color="#2ca02c")
    bars_load_gain = ax.bar(x + width / 2, load_gain, width=width, label="vs Load-weighted", color="#9467bd")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Relative gain of the Stackelberg allocation")
    ax.legend(loc="upper right", frameon=True)
    ax.bar_label(bars_uniform_gain, fmt="%.1f%%", padding=2)
    ax.bar_label(bars_load_gain, fmt="%.1f%%", padding=2)

    return _save_figure(fig, output_dir / "cross_system_summary.pdf")


def plot_ieee118_baselines(main_results: Mapping[str, Any], output_dir: Path) -> Path:
    case_data = main_results["118"]
    items = [
        ("Stackelberg", case_data["stackelberg"]["loss"], "#1f77b4"),
        ("Uniform", case_data["baselines"]["Uniform"]["loss"], "#9aa0a6"),
        ("Degree-based", case_data["baselines"]["Degree-based"]["loss"], "#8c564b"),
        ("Load-weighted", case_data["baselines"]["Load-weighted"]["loss"], "#ff7f0e"),
        ("Generation-weighted", case_data["baselines"]["Generation-weighted"]["loss"], "#2ca02c"),
        ("Vulnerability-weighted", case_data["baselines"]["Vulnerability-weighted"]["loss"], "#17becf"),
        (
            "Marginal-consequence-weighted",
            case_data["baselines"]["Marginal-consequence-weighted"]["loss"],
            "#9467bd",
        ),
    ]
    items = sorted(items, key=lambda item: item[1], reverse=True)

    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    colors = [item[2] for item in items]
    y = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(3.45, 2.9), constrained_layout=True)
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Worst-case propagated loss")
    ax.set_title("IEEE 118-bus heuristic comparison")
    ax.bar_label(bars, fmt="%.3f", padding=3)

    return _save_figure(fig, output_dir / "ieee118_baselines.pdf")


def plot_ablation(main_results: Mapping[str, Any], ablation_results: Mapping[str, Any], output_dir: Path) -> Path:
    full_loss = float(main_results["118"]["stackelberg"]["loss"])
    variants = [
        ("Full model", full_loss, "#1f77b4"),
        ("No attacker in design", float(ablation_results["no_attacker"]["loss"]), "#d62728"),
        ("No propagation in design", float(ablation_results["no_propagation"]["loss"]), "#ff9896"),
        (
            "Uniform consequence in design",
            float(ablation_results["no_physics_weights"]["loss"]),
            "#c5b0d5",
        ),
    ]

    labels = [item[0] for item in variants]
    values = np.array([item[1] for item in variants], dtype=float)
    deltas = (values - full_loss) / full_loss * 100.0
    colors = [item[2] for item in variants]
    y = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Evaluated loss under full model")
    ax.set_title("IEEE 118-bus ablation study")

    annotations = ["baseline"] + [f"+{delta:.1f}%" for delta in deltas[1:]]
    for bar, note in zip(bars, annotations):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, note, va="center")

    return _save_figure(fig, output_dir / "ablation_118.pdf")


def _extract_sensitivity_series(series: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(series.keys(), key=lambda value: float(value))
    x = np.array([float(key) for key in keys], dtype=float)
    stack = np.array([float(series[key]["stackelberg"]) for key in keys], dtype=float)
    uniform = np.array([float(series[key]["uniform"]) for key in keys], dtype=float)
    return x, stack, uniform


def plot_sensitivity_suite(sensitivity_results: Mapping[str, Any], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), constrained_layout=True)
    panels: Sequence[tuple[str, str, str]] = (
        ("defender_budget", "Defender budget $W$", "Budget sensitivity"),
        ("attacker_budget", "Attacker budget $A$", "Attacker sensitivity"),
        ("propagation_strength", "Propagation strength $\\eta$", "Propagation sensitivity"),
    )

    for ax, (key, xlabel, title) in zip(axes, panels):
        x, stack, uniform = _extract_sensitivity_series(sensitivity_results[key])
        ax.plot(x, stack, marker="o", linewidth=1.8, color="#1f77b4", label="Stackelberg")
        ax.plot(x, uniform, marker="s", linewidth=1.8, color="#9aa0a6", label="Uniform")
        ax.fill_between(x, stack, uniform, color="#1f77b4", alpha=0.12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.set_title(title)

    axes[0].legend(loc="upper left", frameon=True)
    return _save_figure(fig, output_dir / "sensitivity_suite.pdf")


def plot_runtime_scaling(main_results: Mapping[str, Any], output_dir: Path) -> Path:
    case_keys = _ordered_case_keys(main_results)
    buses = np.array([int(key) for key in case_keys], dtype=float)
    runtimes = np.array([main_results[key]["stackelberg"]["solve_time"] for key in case_keys], dtype=float)

    fig, ax = plt.subplots(figsize=(3.45, 2.7), constrained_layout=True)
    ax.plot(buses, runtimes, marker="o", linewidth=2.0, color="#1f77b4")
    for bus, runtime in zip(buses, runtimes):
        ax.annotate(f"{runtime:.2f}s", (bus, runtime), textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xlabel("Number of buses")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Solver scaling across benchmark systems")

    return _save_figure(fig, output_dir / "runtime_scaling.pdf")


def generate_all_figures(results_dir: Path = RESULTS_DIR, output_dir: Path = FIGURES_DIR) -> list[Path]:
    configure_matplotlib()

    main_results = load_json(results_dir / "main_results.json")
    ablation_results = load_json(results_dir / "ablation_results.json")
    sensitivity_results = load_json(results_dir / "sensitivity_results.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_cross_system_summary(main_results, output_dir),
        plot_ieee118_baselines(main_results, output_dir),
        plot_ablation(main_results, ablation_results, output_dir),
        plot_sensitivity_suite(sensitivity_results, output_dir),
        plot_runtime_scaling(main_results, output_dir),
    ]
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality figures for the TSG manuscript")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing main_results.json, ablation_results.json, and sensitivity_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to write the generated figure PDFs",
    )
    args = parser.parse_args()

    generated = generate_all_figures(results_dir=args.results_dir, output_dir=args.output_dir)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()