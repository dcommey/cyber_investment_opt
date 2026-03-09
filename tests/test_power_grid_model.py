import numpy as np
import pytest

from src.power_grid import (
    AttackParameters,
    DefenseParameters,
    InfluenceParameters,
    PowerGridStackelbergModel,
)


def make_small_model() -> PowerGridStackelbergModel:
    propagation_matrix = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    baseline_vulnerability = np.array([0.40, 0.30, 0.20])
    consequence_weights = np.array([0.60, 0.25, 0.15])
    return PowerGridStackelbergModel(
        propagation_matrix=propagation_matrix,
        baseline_vulnerability=baseline_vulnerability,
        consequence_weights=consequence_weights,
        defense=DefenseParameters(budget=1.2, alpha=0.8, regularization=1e-3),
        attack=AttackParameters(budget=1.5, max_attack_per_bus=1.0),
        influence=InfluenceParameters(propagation_strength=0.35),
        bus_ids=["A", "B", "C"],
    )


def test_attacker_best_response_respects_budget() -> None:
    model = make_small_model()
    defense = np.zeros(3)

    attack = model.attacker_best_response(defense)
    scores = model.attacker_scores(defense)
    order = np.argsort(-scores)

    assert np.isclose(np.sum(attack), 1.5)
    assert np.all(attack >= 0.0)
    assert np.all(attack <= 1.0 + 1e-12)
    assert attack[order[0]] == pytest.approx(1.0)
    assert attack[order[1]] == pytest.approx(0.5)
    assert attack[order[2]] == pytest.approx(0.0)


def test_stackelberg_solution_is_feasible() -> None:
    model = make_small_model()
    solution = model.solve_stackelberg(num_random_starts=2, maxiter=100)

    assert solution.defense.shape == (3,)
    assert solution.attack.shape == (3,)
    assert np.all(solution.defense >= 0.0)
    assert np.sum(solution.defense) <= 1.2 + 1e-8
    assert np.all(solution.attack >= 0.0)
    assert np.sum(solution.attack) <= 1.5 + 1e-8
    assert np.isfinite(solution.objective)
    assert np.all(solution.propagated_compromise >= 0.0)


def test_unstable_propagation_is_rejected() -> None:
    propagation_matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError):
        PowerGridStackelbergModel(
            propagation_matrix=propagation_matrix,
            baseline_vulnerability=np.array([0.2, 0.2]),
            consequence_weights=np.array([0.5, 0.5]),
            defense=DefenseParameters(budget=1.0),
            attack=AttackParameters(budget=1.0),
            influence=InfluenceParameters(propagation_strength=1.0),
        )
