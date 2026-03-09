from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

from .matpower import PowerGridCase


def build_bus_criticality(
    case: PowerGridCase,
    bus_order: Sequence[str] | None = None,
    load_weight: float = 0.50,
    generation_weight: float = 0.30,
    centrality_weight: float = 0.20,
) -> np.ndarray:
    """Build normalized physical consequence weights for substation-level assets.

    The default proxy blends active demand, generation capacity, and graph
    betweenness centrality. These weights are intended as a transparent baseline
    for paper development and should be replaced with EMS/SCADA asset data when
    such information is available.
    """
    order = list(bus_order or case.bus_ids)
    graph = case.physical_graph()

    weight_sum = load_weight + generation_weight + centrality_weight
    if weight_sum <= 0:
        raise ValueError("At least one criticality component must have positive weight")

    load_component = _normalize_nonnegative(case.load_vector(order))
    generation_component = _normalize_nonnegative(case.generation_capacity_vector(order))
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    centrality_component = _normalize_nonnegative(
        np.array([betweenness.get(bus_id, 0.0) for bus_id in order], dtype=float)
    )

    raw = (
        load_weight * load_component
        + generation_weight * generation_component
        + centrality_weight * centrality_component
    ) / weight_sum
    return _normalize_positive(raw)


def build_baseline_vulnerability(
    case: PowerGridCase,
    bus_order: Sequence[str] | None = None,
    degree_weight: float = 0.45,
    load_weight: float = 0.35,
    generator_weight: float = 0.20,
    v_min: float = 0.05,
    v_max: float = 0.45,
) -> np.ndarray:
    """Construct a bounded vulnerability prior for each cyber asset.

    This proxy is intentionally conservative: it treats highly connected,
    high-load, and generator-attached substations as more exposed. The final
    journal version should replace this prior with data from asset inventories,
    patch cadence, exposure pathways, or incident logs.
    """
    order = list(bus_order or case.bus_ids)
    graph = case.physical_graph()

    degrees = np.array([graph.degree(bus_id) for bus_id in order], dtype=float)
    degree_component = _normalize_nonnegative(degrees)
    load_component = _normalize_nonnegative(case.load_vector(order))
    generator_presence = (case.generation_capacity_vector(order) > 0.0).astype(float)
    generator_component = _normalize_nonnegative(generator_presence)

    weight_sum = degree_weight + load_weight + generator_weight
    if weight_sum <= 0:
        raise ValueError("At least one vulnerability component must have positive weight")

    exposure = (
        degree_weight * degree_component
        + load_weight * load_component
        + generator_weight * generator_component
    ) / weight_sum

    exposure = np.clip(exposure, 0.0, 1.0)
    return v_min + (v_max - v_min) * exposure


def build_propagation_matrix(
    case: PowerGridCase,
    bus_order: Sequence[str] | None = None,
    self_loop_weight: float = 0.0,
) -> np.ndarray:
    """Build a row-stochastic propagation matrix from the physical topology."""
    order = list(bus_order or case.bus_ids)
    graph = case.physical_graph()
    adjacency = nx.to_numpy_array(graph, nodelist=order, weight=None, dtype=float)

    if self_loop_weight > 0:
        adjacency = adjacency + self_loop_weight * np.eye(len(order))

    row_sums = adjacency.sum(axis=1)
    row_sums[row_sums == 0.0] = 1.0
    return adjacency / row_sums[:, np.newaxis]


def _normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0.0)
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(values)
    return values / max_value


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional vector")
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        if values.size == 0:
            raise ValueError("Cannot normalize an empty vector")
        return np.full(values.shape, 1.0 / values.size, dtype=float)
    return values / total
