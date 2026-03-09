from src.power_grid import parse_matpower_case


def test_parse_small_case_counts(small_matpower_case) -> None:
    case = parse_matpower_case(small_matpower_case)

    assert case.name == "case3"
    assert len(case.buses) == 3
    assert len(case.generators) == 2
    assert len(case.branches) == 3
    assert case.total_demand() == 130.0

    graph = case.physical_graph()
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert "1" in graph.nodes
    assert "3" in graph.nodes
