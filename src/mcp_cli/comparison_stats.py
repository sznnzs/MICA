from __future__ import annotations

import math
import random
from statistics import fmean
from typing import Iterable, Sequence


ANALYSIS_SEED_VERSION = "formal-comparison-seed.v1"


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated empirical percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stable_analysis_seed(parts: Iterable[str]) -> int:
    """Build a deterministic analysis-only seed without affecting task sampling."""

    value = 20_260_805
    for part in (ANALYSIS_SEED_VERSION, *tuple(str(part) for part in parts)):
        for character in part:
            value = (value * 131 + ord(character)) % 2_147_483_647
    return value


def paired_bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Return a task-resampling percentile CI for a paired mean difference."""

    if not differences:
        raise ValueError("paired bootstrap requires at least one difference")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    values = tuple(float(value) for value in differences)
    generator = random.Random(seed)
    size = len(values)
    means = [
        fmean(values[generator.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def exact_mcnemar_p(discordant_a_wins: int, discordant_b_wins: int) -> float:
    """Return the two-sided exact binomial McNemar p-value."""

    if discordant_a_wins < 0 or discordant_b_wins < 0:
        raise ValueError("discordant counts cannot be negative")
    total = discordant_a_wins + discordant_b_wins
    if total == 0:
        return 1.0
    smaller = min(discordant_a_wins, discordant_b_wins)
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def holm_adjust(values: Sequence[float]) -> list[float]:
    """Apply the monotone Holm step-down adjustment in original row order."""

    if any(not 0.0 <= float(value) <= 1.0 for value in values):
        raise ValueError("p-values must be in [0, 1]")
    indexed = sorted(enumerate(float(value) for value in values), key=lambda pair: pair[1])
    adjusted = [1.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted
