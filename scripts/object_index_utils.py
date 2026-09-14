from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OBJECT_INDEX_PNG_MAX_VALUE = 65535.0


def max_pass_id_from_mapping(mapping: dict | None) -> int | None:
    if not mapping:
        return None
    max_pass_id = 0
    for key in mapping.keys():
        try:
            max_pass_id = max(max_pass_id, int(key))
        except Exception:
            continue
    return max_pass_id or None


def load_object_index_mapping(mapping_path: Path) -> dict[int, dict]:
    if not mapping_path.exists():
        return {}
    try:
        raw = json.load(open(mapping_path))
    except Exception:
        return {}
    mapping: dict[int, dict] = {}
    for key, value in raw.items():
        try:
            mapping[int(key)] = value
        except Exception:
            continue
    return mapping


def decode_object_index_array(arr, max_pass_id: int | None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if max_pass_id is None or int(max_pass_id) <= 0:
        return arr.astype(np.int32)

    max_pass_id = int(max_pass_id)
    arr_max = float(np.max(arr)) if arr.size else 0.0

    if arr_max <= float(max_pass_id):
        return np.rint(arr).astype(np.int32)

    if np.issubdtype(arr.dtype, np.floating):
        scaled = np.clip(arr.astype(np.float32), 0.0, 1.0)
    else:
        scaled = np.clip(arr.astype(np.float32) / OBJECT_INDEX_PNG_MAX_VALUE, 0.0, 1.0)
    return np.rint(scaled * float(max_pass_id)).astype(np.int32)
