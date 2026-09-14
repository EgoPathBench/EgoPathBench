from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_geometry_contract_sensitivity.py"
spec = spec_from_file_location("audit_geometry_contract_sensitivity", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
original_sys_path = list(sys.path)
try:
    spec.loader.exec_module(mod)
finally:
    sys.path[:] = original_sys_path


def test_edge_set_normalizes_undirected_neighbors():
    neighbors = {
        "1": [{"display_id": 2}],
        "2": [{"display_id": 1}, {"display_id": 3}],
        "3": [{"display_id": 2}],
    }
    assert mod.edge_set(neighbors) == {(1, 2), (2, 3)}


def test_evaluate_route_separates_endpoint_from_legal_path():
    question = {"ground_truth": {"start_id": 1}}
    contract = {
        "visible_ids": [1, 2, 3],
        "direct_neighbors": {
            "1": [{"display_id": 2}],
            "2": [{"display_id": 1}, {"display_id": 3}],
            "3": [{"display_id": 2}],
        },
        "acceptable_goal_ids": [3],
    }
    legal_wrong_goal = mod.evaluate_route(question, [1, 2], contract)
    assert legal_wrong_goal["valid_path"] is True
    assert legal_wrong_goal["success"] is False

    successful = mod.evaluate_route(question, [1, 2, 3], contract)
    assert successful["valid_path"] is True
    assert successful["endpoint_hit"] is True
    assert successful["success"] is True


def test_evaluate_route_rejects_illegal_edge_even_at_goal():
    question = {"ground_truth": {"start_id": 1}}
    contract = {
        "visible_ids": [1, 2, 3],
        "direct_neighbors": {"1": [{"display_id": 2}], "2": [], "3": []},
        "acceptable_goal_ids": [3],
    }
    result = mod.evaluate_route(question, [1, 3], contract)
    assert result["endpoint_hit"] is True
    assert result["edge_legal"] is False
    assert result["success"] is False


def test_fixed_contract_waypoints_preserves_original_candidate_set():
    waypoints = [
        {"display_id": 1, "embodied_feasible": True},
        {"display_id": 2, "embodied_feasible": False},
    ]
    fixed = mod.fixed_contract_waypoints(waypoints, "embodied_feasible")
    assert [row["sensitivity_feasible"] for row in fixed] == [True, False]
