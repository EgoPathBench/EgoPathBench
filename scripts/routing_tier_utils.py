#!/usr/bin/env python3
"""
Legacy compatibility wrapper for routing tier classification.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tier_policy import (  # noqa: E402
    LEGACY_ROUTING_REQUIRED_FIELDS,
    TIER_ORDER,
    classify_legacy_routing_tier,
)


TIER_REQUIRED_FIELDS = LEGACY_ROUTING_REQUIRED_FIELDS


# New family-aware tier logic lives in tier_policy.py. This file remains only
# so existing callers can continue importing classify_routing_tier while the
# rest of the pipeline migrates to navigation/affordance tier families.
def classify_routing_tier(routing_complexity: dict | None) -> str:
    return classify_legacy_routing_tier(routing_complexity)
