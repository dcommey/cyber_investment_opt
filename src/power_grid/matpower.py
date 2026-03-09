from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class BusRecord:
    bus_id: str
    bus_type: int
    pd: float
    qd: float
    base_kv: float
    vmax: float
    vmin: float


@dataclass(frozen=True)
class GeneratorRecord:
    bus_id: str
    pg: float
    qg: float
    pmax: float
    pmin: float
    status: int


@dataclass(frozen=True)
class BranchRecord:
    from_bus: str
    to_bus: str
    resistance: float
    reactance: float
    line_charging: float
    rate_a: float
    status: int


@dataclass(frozen=True)
class PowerGridCase:
    name: str
    base_mva: float
    buses: Dict[str, BusRecord]
    generators: List[GeneratorRecord]
    branches: List[BranchRecord]

    @property
    def bus_ids(self) -> List[str]:
        return sorted(self.buses.keys(), key=_numeric_bus_sort_key)

    def physical_graph(self) -> nx.Graph:
        graph = nx.Graph(name=self.name)

        generation_by_bus = self.generation_capacity_by_bus()
        for bus_id in self.bus_ids:
            bus = self.buses[bus_id]
            graph.add_node(
                bus_id,
                bus_type=bus.bus_type,
                pd=bus.pd,
                qd=bus.qd,
                base_kv=bus.base_kv,
                vmax=bus.vmax,
                vmin=bus.vmin,
                pmax=generation_by_bus.get(bus_id, 0.0),
            )

        for branch in self.branches:
            if branch.status <= 0:
                continue
            if graph.has_edge(branch.from_bus, branch.to_bus):
                edge = graph[branch.from_bus][branch.to_bus]
                edge["parallel_lines"] += 1
                edge["rate_a"] += max(branch.rate_a, 0.0)
                edge["susceptance_proxy"] += _safe_inverse(abs(branch.reactance))
            else:
                graph.add_edge(
                    branch.from_bus,
                    branch.to_bus,
                    parallel_lines=1,
                    rate_a=max(branch.rate_a, 0.0),
                    susceptance_proxy=_safe_inverse(abs(branch.reactance)),
                )

        return graph

    def load_vector(self, bus_order: Sequence[str] | None = None) -> np.ndarray:
        order = list(bus_order or self.bus_ids)
        return np.array([max(self.buses[bus_id].pd, 0.0) for bus_id in order], dtype=float)

    def generation_capacity_by_bus(self) -> Dict[str, float]:
        capacities: Dict[str, float] = {bus_id: 0.0 for bus_id in self.buses}
        for generator in self.generators:
            if generator.status > 0:
                capacities[generator.bus_id] = capacities.get(generator.bus_id, 0.0) + max(generator.pmax, 0.0)
        return capacities

    def generation_capacity_vector(self, bus_order: Sequence[str] | None = None) -> np.ndarray:
        order = list(bus_order or self.bus_ids)
        capacities = self.generation_capacity_by_bus()
        return np.array([capacities.get(bus_id, 0.0) for bus_id in order], dtype=float)

    def total_demand(self) -> float:
        return float(np.sum(self.load_vector()))


def parse_matpower_case(file_path: str | Path) -> PowerGridCase:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    bus_rows = _parse_matrix_block(content, "bus")
    gen_rows = _parse_matrix_block(content, "gen")
    branch_rows = _parse_matrix_block(content, "branch")
    base_mva = _parse_scalar_assignment(content, "baseMVA", default=100.0)

    buses = {
        str(int(row[0])): BusRecord(
            bus_id=str(int(row[0])),
            bus_type=int(row[1]),
            pd=float(row[2]),
            qd=float(row[3]),
            base_kv=float(row[9]),
            vmax=float(row[11]),
            vmin=float(row[12]),
        )
        for row in bus_rows
    }

    generators = [
        GeneratorRecord(
            bus_id=str(int(row[0])),
            pg=float(row[1]),
            qg=float(row[2]),
            pmax=float(row[8]),
            pmin=float(row[9]),
            status=int(row[7]),
        )
        for row in gen_rows
    ]

    branches = [
        BranchRecord(
            from_bus=str(int(row[0])),
            to_bus=str(int(row[1])),
            resistance=float(row[2]),
            reactance=float(row[3]),
            line_charging=float(row[4]),
            rate_a=float(row[5]),
            status=int(row[10]) if len(row) > 10 else 1,
        )
        for row in branch_rows
    ]

    return PowerGridCase(
        name=path.stem,
        base_mva=base_mva,
        buses=buses,
        generators=generators,
        branches=branches,
    )


def _parse_scalar_assignment(content: str, name: str, default: float) -> float:
    match = re.search(rf"mpc\.{re.escape(name)}\s*=\s*([-+0-9.eE]+)\s*;", content)
    if not match:
        return float(default)
    return float(match.group(1))


def _parse_matrix_block(content: str, name: str) -> List[List[float]]:
    match = re.search(
        rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\];",
        content,
        flags=re.DOTALL | re.MULTILINE,
    )
    if not match:
        raise ValueError(f"MATPOWER case is missing the '{name}' data block")

    rows: List[List[float]] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue
        if line.endswith(";"):
            line = line[:-1].strip()
        if not line:
            continue
        values = [float(token) for token in re.split(r"\s+", line) if token]
        rows.append(values)

    if not rows:
        raise ValueError(f"MATPOWER case has an empty '{name}' data block")
    return rows


def _numeric_bus_sort_key(bus_id: str) -> tuple[int, str]:
    try:
        return (0, int(bus_id))
    except ValueError:
        return (1, bus_id)


def _safe_inverse(value: float, default: float = 1.0) -> float:
    if value <= 1e-12:
        return default
    return 1.0 / value
