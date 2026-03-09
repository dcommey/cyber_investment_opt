from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class DefenseParameters:
    budget: float
    alpha: float | np.ndarray = 0.35
    regularization: float = 1e-3


@dataclass(frozen=True)
class AttackParameters:
    budget: float
    max_attack_per_bus: float = 1.0


@dataclass(frozen=True)
class InfluenceParameters:
    propagation_strength: float = 0.35


@dataclass(frozen=True)
class StackelbergSolution:
    defense: np.ndarray
    attack: np.ndarray
    propagated_compromise: np.ndarray
    objective: float
    attacker_scores: np.ndarray
    bus_ids: tuple[str, ...]


class PowerGridStackelbergModel:
    """Bilevel security investment model tailored to power-grid control assets.

    The model is built around three ingredients:
    1. A power-grid-informed physical consequence vector `c`.
    2. A linear propagation operator `(I - eta P)^{-1}` with `eta rho(P) < 1`.
    3. An adaptive attacker whose inner problem has an exact greedy solution.

    For fixed defender investments `z`, the attacker solves the linear program

        max_a q(z)^T a
        s.t. 0 <= a_i <= a_max,  sum_i a_i <= A,

    where `q_i(z) = s_i(z) [M^T c]_i`, `M = (I - eta P)^{-1}`, and
    `s_i(z) = v_i^0 exp(-alpha_i z_i)`.

    This closed form is the key proof hook for the accompanying TSG manuscript.
    """

    def __init__(
        self,
        propagation_matrix: np.ndarray,
        baseline_vulnerability: np.ndarray,
        consequence_weights: np.ndarray,
        defense: DefenseParameters,
        attack: AttackParameters,
        influence: InfluenceParameters | None = None,
        bus_ids: Sequence[str] | None = None,
    ):
        self.propagation_matrix = _as_square_matrix(propagation_matrix, "propagation_matrix")
        self.n = self.propagation_matrix.shape[0]
        self.baseline_vulnerability = _as_vector(baseline_vulnerability, self.n, "baseline_vulnerability")
        self.consequence_weights = _normalize_weights(
            _as_vector(consequence_weights, self.n, "consequence_weights")
        )
        self.alpha = _as_vector(defense.alpha, self.n, "alpha", lower_bound=0.0)
        self.defense_budget = float(defense.budget)
        self.attack_budget = float(attack.budget)
        self.attack_cap = float(attack.max_attack_per_bus)
        self.regularization = float(defense.regularization)
        self.propagation_strength = float((influence or InfluenceParameters()).propagation_strength)
        self.bus_ids = tuple(bus_ids or [str(i) for i in range(self.n)])

        if len(self.bus_ids) != self.n:
            raise ValueError("bus_ids length must match the state dimension")
        if self.defense_budget < 0:
            raise ValueError("Defender budget must be non-negative")
        if self.attack_budget < 0:
            raise ValueError("Attacker budget must be non-negative")
        if self.attack_cap <= 0:
            raise ValueError("max_attack_per_bus must be positive")
        if np.any(self.baseline_vulnerability < 0):
            raise ValueError("baseline_vulnerability must be non-negative")
        if self.propagation_strength < 0:
            raise ValueError("propagation_strength must be non-negative")

        rho = _spectral_radius(self.propagation_matrix)
        if self.propagation_strength * rho >= 1.0:
            raise ValueError(
                "Propagation operator is unstable: require propagation_strength * spectral_radius(P) < 1"
            )

        identity = np.eye(self.n)
        self.influence_matrix = np.linalg.inv(identity - self.propagation_strength * self.propagation_matrix)
        self._marginal_consequence = self.influence_matrix.T @ self.consequence_weights

    def post_hardening_vulnerability(self, defense_allocation: np.ndarray) -> np.ndarray:
        defense_allocation = self._sanitize_defense(defense_allocation)
        return self.baseline_vulnerability * np.exp(-self.alpha * defense_allocation)

    def attacker_scores(self, defense_allocation: np.ndarray) -> np.ndarray:
        vulnerabilities = self.post_hardening_vulnerability(defense_allocation)
        return vulnerabilities * self._marginal_consequence

    def attacker_best_response(self, defense_allocation: np.ndarray) -> np.ndarray:
        scores = self.attacker_scores(defense_allocation)
        attack = np.zeros(self.n, dtype=float)

        remaining = min(self.attack_budget, self.attack_cap * self.n)
        for index in np.argsort(-scores):
            if remaining <= 1e-12:
                break
            if scores[index] <= 0:
                break
            allocation = min(self.attack_cap, remaining)
            attack[index] = allocation
            remaining -= allocation

        return attack

    def propagated_compromise(self, defense_allocation: np.ndarray, attack_allocation: np.ndarray) -> np.ndarray:
        defense_allocation = self._sanitize_defense(defense_allocation)
        attack_allocation = _as_vector(attack_allocation, self.n, "attack_allocation", lower_bound=0.0)
        vulnerabilities = self.post_hardening_vulnerability(defense_allocation)
        local_intrusion = vulnerabilities * (1.0 + attack_allocation)
        return self.influence_matrix @ local_intrusion

    def defender_objective(self, defense_allocation: np.ndarray) -> float:
        defense_allocation = self._sanitize_defense(defense_allocation)
        attack_allocation = self.attacker_best_response(defense_allocation)
        propagated = self.propagated_compromise(defense_allocation, attack_allocation)
        physical_loss = float(self.consequence_weights @ propagated)
        regularization = 0.5 * self.regularization * float(np.dot(defense_allocation, defense_allocation))
        return physical_loss + regularization

    def solve_defender_problem(
        self,
        initial_guess: np.ndarray | None = None,
        num_random_starts: int = 4,
        random_seed: int = 7,
        maxiter: int = 300,
    ) -> tuple[np.ndarray, float]:
        guesses = self._initial_guesses(initial_guess, num_random_starts, random_seed)
        best_result = None

        constraints = [{"type": "ineq", "fun": lambda z: self.defense_budget - np.sum(z)}]
        bounds = [(0.0, self.defense_budget) for _ in range(self.n)]

        for guess in guesses:
            result = minimize(
                self.defender_objective,
                x0=guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": maxiter, "ftol": 1e-8, "disp": False},
            )
            if not result.success and best_result is not None:
                continue
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        if best_result is None or best_result.x is None:
            raise RuntimeError("Failed to solve the defender problem")

        defense = self._sanitize_defense(best_result.x)
        return defense, float(self.defender_objective(defense))

    def solve_stackelberg(
        self,
        initial_guess: np.ndarray | None = None,
        num_random_starts: int = 4,
        random_seed: int = 7,
        maxiter: int = 300,
    ) -> StackelbergSolution:
        defense, objective = self.solve_defender_problem(
            initial_guess=initial_guess,
            num_random_starts=num_random_starts,
            random_seed=random_seed,
            maxiter=maxiter,
        )
        attack = self.attacker_best_response(defense)
        propagated = self.propagated_compromise(defense, attack)
        return StackelbergSolution(
            defense=defense,
            attack=attack,
            propagated_compromise=propagated,
            objective=objective,
            attacker_scores=self.attacker_scores(defense),
            bus_ids=self.bus_ids,
        )

    def _initial_guesses(
        self,
        initial_guess: np.ndarray | None,
        num_random_starts: int,
        random_seed: int,
    ) -> list[np.ndarray]:
        guesses: list[np.ndarray] = []

        if initial_guess is not None:
            guesses.append(self._sanitize_defense(initial_guess))
        else:
            guesses.append(np.full(self.n, self.defense_budget / max(self.n, 1), dtype=float))

        if self.defense_budget <= 0:
            return [np.zeros(self.n, dtype=float)]

        rng = np.random.default_rng(random_seed)
        for _ in range(max(num_random_starts, 0)):
            sample = rng.random(self.n)
            sample_sum = float(np.sum(sample))
            if sample_sum <= 0:
                guesses.append(np.full(self.n, self.defense_budget / self.n, dtype=float))
                continue
            guesses.append(self.defense_budget * sample / sample_sum)

        consequence_seed = self.consequence_weights / np.sum(self.consequence_weights)
        guesses.append(self.defense_budget * consequence_seed)
        return guesses

    def _sanitize_defense(self, defense_allocation: np.ndarray) -> np.ndarray:
        defense_allocation = _as_vector(defense_allocation, self.n, "defense_allocation", lower_bound=0.0)
        total = float(np.sum(defense_allocation))
        if total <= self.defense_budget + 1e-10:
            return defense_allocation
        if total <= 0.0:
            return np.zeros(self.n, dtype=float)
        return defense_allocation * (self.defense_budget / total)


def _as_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if np.any(matrix < 0):
        raise ValueError(f"{name} must be elementwise non-negative")
    return matrix


def _as_vector(
    values: float | np.ndarray,
    size: int,
    name: str,
    lower_bound: float | None = None,
) -> np.ndarray:
    if np.isscalar(values):
        vector = np.full(size, float(values), dtype=float)
    else:
        vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.shape[0] != size:
        raise ValueError(f"{name} must be a one-dimensional vector of length {size}")
    if lower_bound is not None and np.any(vector < lower_bound):
        raise ValueError(f"{name} must be elementwise >= {lower_bound}")
    return vector


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("consequence_weights must be non-negative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("consequence_weights must have positive mass")
    return values / total


def _spectral_radius(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))
