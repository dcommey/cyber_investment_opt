import numpy as np

from src.scripts.run_full_experiments import (
    build_baseline_allocations,
    build_model,
)


def test_baseline_allocations_follow_case_metrics(small_matpower_case) -> None:
    case_path = small_matpower_case
    model, case = build_model(case_path, budget_w=5.0, budget_a=1.0)
    allocations = build_baseline_allocations(model, case)
    bus_order = list(model.bus_ids)

    graph = case.physical_graph()
    degree_expected = np.array([graph.degree(bus_id) for bus_id in bus_order], dtype=float)
    degree_expected = 5.0 * degree_expected / np.sum(degree_expected)

    load_expected = case.load_vector(bus_order)
    load_expected = 5.0 * load_expected / np.sum(load_expected)

    generation_expected = case.generation_capacity_vector(bus_order)
    generation_expected = 5.0 * generation_expected / np.sum(generation_expected)

    vulnerability_expected = 5.0 * model.baseline_vulnerability / np.sum(model.baseline_vulnerability)

    marginal_expected = 5.0 * model._marginal_consequence / np.sum(model._marginal_consequence)

    assert np.isclose(np.sum(allocations["Uniform"]), 5.0)
    assert np.isclose(np.sum(allocations["Degree-based"]), 5.0)
    assert np.isclose(np.sum(allocations["Load-weighted"]), 5.0)
    assert np.isclose(np.sum(allocations["Generation-weighted"]), 5.0)
    assert np.isclose(np.sum(allocations["Vulnerability-weighted"]), 5.0)
    assert np.isclose(np.sum(allocations["Marginal-consequence-weighted"]), 5.0)

    assert np.allclose(allocations["Degree-based"], degree_expected)
    assert np.allclose(allocations["Load-weighted"], load_expected)
    assert np.allclose(allocations["Generation-weighted"], generation_expected)
    assert np.allclose(allocations["Vulnerability-weighted"], vulnerability_expected)
    assert np.allclose(allocations["Marginal-consequence-weighted"], marginal_expected)
    assert not np.allclose(allocations["Degree-based"], allocations["Uniform"])
    assert not np.allclose(allocations["Load-weighted"], allocations["Marginal-consequence-weighted"])
