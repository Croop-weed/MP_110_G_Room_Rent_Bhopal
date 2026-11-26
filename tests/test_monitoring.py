"""Synthetic tests for PSI, chronology, promotion, and holdout utilities."""

import numpy as np
import pandas as pd

from pg_rent.monitoring import (
    calculate_psi,
    chronological_split,
    promotion_decision,
)


def test_psi_zero_frequency_bin_is_safe():
    reference = pd.Series([1.0, 1.0, 1.0, 2.0])
    current = pd.Series([10.0, 10.0, 10.0, 10.0])
    psi = calculate_psi(reference, current, bins=3)
    assert np.isfinite(psi)
    assert psi >= 0


def test_chronological_split_uses_sorted_dates():
    later = pd.DataFrame(
        {
            "date": ["2026-08-03", "2026-07-01", "2026-07-31", "2026-07-15"],
            "record_id": ["R4", "R1", "R3", "R2"],
            "rent_rs": [4000, 1000, 3000, 2000],
        }
    )
    first, final = chronological_split(later, 0.75)
    assert first["record_id"].tolist() == ["R1", "R2", "R3"]
    assert final["record_id"].tolist() == ["R4"]


def test_promotion_threshold_requires_15_percent_improvement():
    keep = promotion_decision(1000.0, 900.0)
    promote = promotion_decision(1000.0, 840.0)
    assert keep["decision"] == "keep"
    assert promote["decision"] == "promote"
