import json
import os
from functools import lru_cache
from typing import Dict, List


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CALIBRATION_FILE = os.path.join(DATA_DIR, "calibration_samples.json")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def load_calibration_samples() -> List[Dict]:
    if not os.path.exists(CALIBRATION_FILE):
        return []
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def derive_calibration_params() -> Dict[str, float]:
    """
    Fit a tiny linear mapping on toy labeled samples:
    calibrated = scale * raw + offset
    """
    samples = load_calibration_samples()
    if not samples:
        return {"scale": 1.0, "offset": 0.0}

    raw_values = [float(s.get("raw_score", 0.0)) for s in samples]
    target_values = [float(s.get("target_score", 0.0)) for s in samples]
    n = len(samples)
    mean_raw = sum(raw_values) / n
    mean_target = sum(target_values) / n

    numerator = sum((r - mean_raw) * (t - mean_target) for r, t in zip(raw_values, target_values))
    denominator = sum((r - mean_raw) ** 2 for r in raw_values)
    scale = numerator / denominator if denominator else 1.0
    offset = mean_target - scale * mean_raw
    return {"scale": scale, "offset": offset}


def calibrate_score(raw_score: float) -> float:
    params = derive_calibration_params()
    calibrated = params["scale"] * raw_score + params["offset"]
    return _clamp(calibrated)

