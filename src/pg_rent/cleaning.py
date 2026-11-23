"""Verified-data cleaning for PG room rent datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_TYPO_RECORD_IDS = {
    "R00224",
    "R00279",
    "R00385",
    "R00423",
    "R00466",
    "R00616",
    "R00637",
    "R00642",
    "R00730",
    "R00783",
    "R00803",
    "R00807",
}


def _normalize_furnishing(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower().replace(".", "").replace("_", "")
    return {"ful": "full", "sem": "semi", "unf": "unfurnished"}.get(text, text)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return an independent copy with only verified audit corrections applied."""
    cleaned = df.copy(deep=True)

    if {"locality", "building_age_yr"}.issubset(cleaned.columns):
        mask = cleaned["locality"] == "Sector West"
        cleaned.loc[mask, "building_age_yr"] = (
            pd.to_numeric(cleaned.loc[mask, "building_age_yr"], errors="coerce") / 12.0
        )

    if "sharing" in cleaned:
        cleaned.loc[
            pd.to_numeric(cleaned["sharing"], errors="coerce") == 999, "sharing"
        ] = pd.NA

    if "rent_rs" in cleaned and "record_id" in cleaned:
        mask = cleaned["record_id"].isin(TARGET_TYPO_RECORD_IDS)
        cleaned.loc[mask, "rent_rs"] = (
            pd.to_numeric(cleaned.loc[mask, "rent_rs"], errors="coerce") / 10.0
        )

    if "furnishing" in cleaned:
        cleaned["furnishing"] = cleaned["furnishing"].map(_normalize_furnishing)

    cleaned = cleaned.drop_duplicates()
    return cleaned


def generate_cleaning_report(
    before: pd.DataFrame, after: pd.DataFrame, report_path: str | Path
) -> None:
    """Write before/after cleaning statistics and verified correction counts."""
    report = report_path if isinstance(report_path, Path) else Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)

    west_before = (
        before.loc[before["locality"] == "Sector West", "building_age_yr"]
        if "locality" in before
        else pd.Series(dtype="float64")
    )
    west_after = (
        after.loc[after["locality"] == "Sector West", "building_age_yr"]
        if "locality" in after
        else pd.Series(dtype="float64")
    )
    sentinel_count = (
        int((pd.to_numeric(before["sharing"], errors="coerce") == 999).sum())
        if "sharing" in before
        else 0
    )
    typo_mask = (
        before["record_id"].isin(TARGET_TYPO_RECORD_IDS)
        if "record_id" in before and "rent_rs" in before
        else pd.Series(False, index=before.index)
    )

    lines = [
        "PG ROOM RENT - CLEANING REPORT",
        f"Rows before: {len(before)}",
        f"Rows after: {len(after)}",
        f"Duplicate rows removed: {int(len(before) - len(before.drop_duplicates()))}",
        f"Sector West building_age_yr rows divided by 12: {len(west_before)}",
        f"Sector West building_age median before: {west_before.median()}",
        f"Sector West building_age median after: {west_after.median()}",
        f"sharing 999 values replaced with NaN: {sentinel_count}",
        f"Target typo records corrected by /10: {int(typo_mask.sum())}",
        f"Furnishing variants normalized: {int(before['furnishing'].notna().sum()) if 'furnishing' in before else 0}",
        "Verified target record IDs: " + ", ".join(sorted(TARGET_TYPO_RECORD_IDS)),
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
