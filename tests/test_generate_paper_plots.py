from pathlib import Path

from src.scripts.generate_paper_plots import generate_all_figures


def test_generate_all_figures_smoke(tmp_path, sample_results_dir) -> None:
    results_dir = sample_results_dir
    generated = generate_all_figures(results_dir=results_dir, output_dir=tmp_path)

    expected = {
        "cross_system_summary.pdf",
        "ieee118_baselines.pdf",
        "ablation_118.pdf",
        "sensitivity_suite.pdf",
        "runtime_scaling.pdf",
    }

    assert {path.name for path in generated} == expected
    for path in generated:
        assert path.exists()
        assert path.stat().st_size > 0