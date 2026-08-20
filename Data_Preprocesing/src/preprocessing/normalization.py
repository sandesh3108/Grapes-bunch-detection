"""
Stage 7 — Normalization Policy Enforcer.
Enforces and documents framework normalization policies.
For Ultralytics YOLO, pixel scaling is handled internally at train time, so images remain UINT8.
"""

from typing import Dict, Any


def get_normalization_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns normalization policy summary.
    """
    norm_config = config.get("normalization", {})
    enabled = norm_config.get("enabled", False)

    policy = {
        "enabled": enabled,
        "framework": "Ultralytics YOLO (YOLOv11/v12)",
        "internal_normalization": True,
        "pixel_range": "0-255 (UINT8 saved format)",
        "note": "Ultralytics YOLO normalizes pixels to [0,1] internally during batch collation.",
    }
    return policy
