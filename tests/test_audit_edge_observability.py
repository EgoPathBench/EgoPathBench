from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_edge_observability.py"
spec = spec_from_file_location("audit_edge_observability", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_default_audit_contract_uses_formal_embodied_radius():
    assert mod.DEFAULT_EMBODIED_RADIUS_M == 0.30
    assert mod.DEFAULT_OUTPUT_DIR.name == "edge_observability_audit_20260718_radius030"


def test_fit_view_calibration_recovers_planar_projection_and_depth():
    waypoints = []
    for waypoint_id, (x, y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1), (2, 1)), start=1):
        waypoints.append({
            "display_id": waypoint_id,
            "world_xyz": [x, y, 0.0],
            "image_xy": [100 + 20 * x, 200 + 30 * y],
            "depth": 1.0 + 0.5 * x + 0.25 * y,
        })

    calibration = mod.fit_view_calibration(waypoints)
    px, py, depth = mod.project_ground_point(calibration, (0.5, 0.5))

    assert abs(px - 110.0) < 1e-3
    assert abs(py - 215.0) < 1e-3
    assert abs(depth - 1.375) < 1e-6


def test_visible_foreground_separates_safe_and_blocked_edges():
    calibration = {
        "homography": np.asarray([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]),
        "depth_coefficients": np.asarray([0.0, 0.0, 2.0]),
    }
    samples = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
    clear_depth = np.full((32, 32), 2.0, dtype=np.float32)
    blocked_depth = clear_depth.copy()
    blocked_depth[5, 10] = 1.0

    clear = mod.inspect_ground_samples(samples, calibration, clear_depth, None, 0.08)
    blocked = mod.inspect_ground_samples(samples, calibration, blocked_depth, None, 0.08)

    assert mod.classify_edge_support(True, clear, clear, False, 0.95, 0.80) == "observable_safe"
    assert mod.classify_edge_support(False, blocked, blocked, False, 0.95, 0.80) == "depth_obstruction_evidence"


def test_embodied_corridor_adds_lateral_samples():
    center = mod.sample_segment((0.0, 0.0), (1.0, 0.0), step_m=0.5)
    corridor = mod.sample_corridor((0.0, 0.0), (1.0, 0.0), step_m=0.5, radius_m=0.4)

    assert len(center) == 3
    assert len(corridor) == 15
    assert (0.5, 0.4) in corridor
