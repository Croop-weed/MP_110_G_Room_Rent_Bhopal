"""Synthetic tests for the seven data-audit functions."""

import pandas as pd

from pg_rent.data_audit import (
    audit_duplicate_rows,
    audit_leaking_column,
    audit_sentinel_value,
    audit_spelling_variants,
    audit_stuck_sensor,
    audit_target_typos,
    audit_unit_mixup,
)


def test_leaking_column_distinguishes_missing_from_leakage():
    train = pd.DataFrame({"rent_rs": [100, 200, 300], "post_event": [100, 200, 300]})
    holdout = pd.DataFrame({"other": [1, 2, 3]})
    result = audit_leaking_column(train, holdout)
    assert result["present"] is True
    assert result["missing_at_inference_columns"] == ["post_event"]
    assert result["candidates"][0]["unique_values"] == 3


def test_unit_mixup_requires_confirmation():
    train = pd.DataFrame(
        {
            "locality": ["A"] * 4 + ["B"] * 4,
            "building_age_yr": [12, 24, 36, 48, 144, 288, 432, 576],
        }
    )
    result = audit_unit_mixup(train)
    assert result["present"] is True
    assert result["group_value"] == "B"
    assert result["factor"] == 12.0


def test_sentinel_value_is_detected():
    train = pd.DataFrame({"sharing": [2, 999, 2, 999, 1, 999, 1, 999, 3, 999, 3, 999]})
    result = audit_sentinel_value(train)
    assert result["present"] is True
    assert result["column"] == "sharing"
    assert result["count"] == 6


def test_duplicate_rows_reports_extra_copies():
    train = pd.DataFrame({"a": [1, 1, 2], "b": [x * 2 for x in [1, 1, 2]]})
    result = audit_duplicate_rows(train)
    assert result["present"] is True
    assert result["count"] == 1


def test_stuck_sensor_reports_only_suspicious_continuous_runs():
    train = pd.DataFrame(
        {
            "record_id": [f"R{i}" for i in range(8)],
            "date": pd.date_range("2026-01-01", periods=8).astype(str),
            "wifi_mbps": [100] * 8,
        }
    )
    result = audit_stuck_sensor(train)
    assert result["status"] == "detected"
    assert result["column"] == "wifi_mbps"
    assert result["start_date"] == "2026-01-01"


def test_target_typos_reports_row_level_values():
    train = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3", "R4", "R5"],
            "rent_rs": [10000, 12000, 11000, 13000, 130000],
        }
    )
    result = audit_target_typos(train)
    assert result["present"] is True
    assert result["count"] == 1
    assert result["anomalies"][0]["record_id"] == "R5"
    assert result["anomalies"][0]["corrected_value"] == 13000


def test_spelling_variants_normalizes_clean_categories():
    train = pd.DataFrame(
        {"furnishing": ["full", " Full", "FUL.", "semi", "SEMI", "unf."]}
    )
    later = pd.DataFrame({"furnishing": ["full", "semi", "unfurnished"]})
    result = audit_spelling_variants(train, later)
    assert result["present"] is True
    assert result["column"] == "furnishing"
    assert result["true_levels"] == 3
