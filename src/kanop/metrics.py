"""Evaluation metrics for model comparison."""

from __future__ import annotations

import math


def absolute_error(estimate: float, target: float) -> float:
    return abs(estimate - target)


def relative_error(estimate: float, target: float) -> float:
    if target == 0:
        return math.nan
    return abs(estimate - target) / abs(target)
