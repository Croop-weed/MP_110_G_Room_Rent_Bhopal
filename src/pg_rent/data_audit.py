"""Simpler seven-problem audit for the PG rent training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "train.csv"
LATER_PATH = ROOT / "data" / "later.csv"
HOLDOUT_PATH = ROOT / "data" / "holdout_features.csv"
FINDINGS_PATH = ROOT / "findings.yaml"
REPORT_PATH = ROOT / "reports" / "data_audit_report.txt"
RESULTS_PATH = ROOT / "reports" / "data_audit_results.json"


def _to_float(val: Any) -> float | None:
    return None if pd.isna(val) else float(val)


def audit_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, Any]:
    """Separate missing-at-inference columns from strongly target-correlated leakage."""
    missing = sorted((set(train.columns) - set(holdout.columns)) - {"rent_rs"})
    candidates: list[dict[str, Any]] = []
    target = (
        pd.to_numeric(train["rent_rs"], errors="coerce") if "rent_rs" in train else None
    )

    for col in missing:
        series = pd.to_numeric(train[col], errors="coerce")
        correlation = (
            None
            if target is None or not series.notna().any()
            else float(series.corr(target))
        )
        unique_count = int(train[col].nunique(dropna=True))
        unique_ratio = unique_count / max(len(train), 1)
        strong_correlation = correlation is not None and abs(correlation) >= 0.95
        constant_evidence = unique_count <= 1
        likely_leak = bool(strong_correlation or constant_evidence)
        candidates.append(
            {
                "column": col,
                "missing_from_holdout": True,
                "unique_values": unique_count,
                "unique_ratio": round(unique_ratio, 6),
                "target_correlation": None
                if correlation is None or pd.isna(correlation)
                else round(correlation, 6),
                "strong_correlation_ge_0_95": strong_correlation,
                "actual_leak_evidence": likely_leak,
            }
        )

    leaked = [item["column"] for item in candidates if item["actual_leak_evidence"]]
    return {
        "status": "detected"
        if leaked
        else ("inconclusive" if missing else "not_detected"),
        "present": bool(leaked),
        "missing_at_inference_columns": missing,
        "columns": leaked,
        "candidates": candidates,
        "evidence": "A candidate must be absent at inference and show strong target correlation or a single value; absence alone is not target leakage.",
    }


def audit_unit_mixup(train: pd.DataFrame) -> dict[str, Any]:
    """Report unit candidates separately; confirm only plausible distribution-aligned factors."""
    if not {"building_age_yr", "locality"}.issubset(train.columns):
        return {
            "status": "inconclusive",
            "present": False,
            "column": None,
            "group_value": None,
            "factor": None,
            "statistics": {},
            "affected_rows": 0,
            "suspicious_candidates": [],
            "confirmed": [],
        }

    stats = train.groupby("locality")["building_age_yr"].agg(
        ["count", "min", "median", "mean", "max"]
    )
    stats_json = {
        str(loc): {k: _to_float(v) for k, v in row.items()}
        for loc, row in stats.iterrows()
    }
    suspicious: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []

    for loc, row in stats.iterrows():
        others = stats.drop(index=loc)
        if pd.isna(row["median"]) or others["median"].dropna().empty:
            continue
        others_median = float(others["median"].median())
        if others_median == 0:
            continue
        ratio = float(row["median"] / others_median)
        factors = [
            factor
            for factor in (12.0, 10.0, 100.0, 1000.0)
            if 0.75 <= ratio / factor <= 1.25
        ]
        if ratio >= 10 or ratio <= 0.1:
            vals = pd.to_numeric(
                train.loc[train["locality"] == loc, "building_age_yr"], errors="coerce"
            )
            candidate = {
                "column": "building_age_yr",
                "group_value": str(loc),
                "median_ratio": round(ratio, 3),
                "plausible_factors": factors,
                "median_before": _to_float(vals.median()),
                "max_before": _to_float(vals.max()),
            }
            suspicious.append(candidate)

            for factor in factors:
                corrected_median = float(vals.median() / factor)
                corrected_max = float(vals.max() / factor)
                plausible_median = 0 <= corrected_median <= 50
                plausible_max = 0 <= corrected_max <= 100
                after_range_aligned = abs(corrected_median - others_median) <= 10
                if plausible_median and plausible_max and after_range_aligned:
                    confirmed.append(
                        {
                            **candidate,
                            "factor": factor,
                            "median_after": round(corrected_median, 3),
                            "max_after": round(corrected_max, 3),
                            "affected_rows": int((train["locality"] == loc).sum()),
                            "evidence": "Median ratio is near a known factor, and corrected median/max fit the other-locality range.",
                        }
                    )

    if confirmed:
        best = confirmed[0]
        return {
            "status": "detected",
            "present": True,
            "column": best["column"],
            "group_value": best["group_value"],
            "factor": best["factor"],
            "statistics": stats_json,
            "affected_rows": best["affected_rows"],
            "suspicious_candidates": suspicious,
            "confirmed": confirmed,
            "verification": {
                "median_before": best["median_before"],
                "median_after": best["median_after"],
                "max_before": best["max_before"],
                "max_after": best["max_after"],
            },
        }
    status = "inconclusive" if suspicious else "not_detected"
    return {
        "status": status,
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
        "statistics": stats_json,
        "affected_rows": 0,
        "suspicious_candidates": suspicious,
        "confirmed": confirmed,
    }


def audit_sentinel_value(train: pd.DataFrame) -> dict[str, Any]:
    sentinels = (-1, -999, 999, 9999)
    matches = [
        {"column": col, "value": val, "count": int((train[col] == val).sum())}
        for col in train.select_dtypes(include="number")
        for val in sentinels
        if int((train[col] == val).sum()) > 0
    ]
    matches.sort(key=lambda item: item["count"], reverse=True)
    if not matches:
        return {
            "status": "not_detected",
            "present": False,
            "column": None,
            "value": None,
            "count": 0,
            "evidence": [],
        }
    # Ambiguous one-off sentinel matches (for example, 9999 in a target) are reported but need count evidence.
    strongest = matches[0]
    if strongest["count"] < 3:
        return {
            "status": "inconclusive",
            "present": False,
            "column": None,
            "value": None,
            "count": 0,
            "evidence": matches,
            "evidence_note": "A sentinel-like value occurred fewer than 3 times and was not confirmed.",
        }
    return {"status": "detected", "present": True, **strongest, "evidence": matches}


def audit_duplicate_rows(train: pd.DataFrame) -> dict[str, Any]:
    count = int(train.duplicated().sum())
    return {
        "status": "not_detected" if count == 0 else "detected",
        "present": count > 0,
        "count": count,
        "evidence": {"duplicate_mask_rows": count},
    }


def audit_stuck_sensor(train: pd.DataFrame) -> dict[str, Any]:
    """Report repeated-read runs conservatively and require evidence beyond ordinary equal counts."""
    if not {"date", "record_id"}.issubset(train.columns):
        return {
            "status": "inconclusive",
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None,
            "value": None,
            "suspicious_runs": [],
            "confirmed_runs": [],
        }

    df = train.assign(date=pd.to_datetime(train["date"], errors="coerce")).dropna(
        subset=["date"]
    )
    df = df.sort_values(["date", "record_id"])
    suspicious: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []

    num_cols = [c for c in df.select_dtypes(include="number") if c not in ("rent_rs",)]
    for col in num_cols:
        i = 0
        while i < len(df):
            value = df[col].iloc[i]
            if pd.isna(value):
                i += 1
                continue
            j = i
            while j + 1 < len(df) and df[col].iloc[j + 1] == value:
                j += 1
            length = j - i + 1
            dates = df["date"].iloc[i : j + 1]
            distinct_dates = int(dates.nunique())
            date_span = int((dates.max() - dates.min()).days)
            if length >= 7 or distinct_dates >= 7:
                run = {
                    "column": col,
                    "value": value.item() if hasattr(value, "item") else value,
                    "length": length,
                    "distinct_dates": distinct_dates,
                    "start_date": dates.min().strftime("%Y-%m-%d"),
                    "end_date": dates.max().strftime("%Y-%m-%d"),
                    "date_span_days": date_span,
                }
                suspicious.append(run)
                # Conservative sensor confirmation: exactly the same numeric reading spans a week
                # and occurs on at least seven consecutive calendar dates.
                unique_date_index = pd.DatetimeIndex(
                    dates.drop_duplicates().sort_values()
                )
                consecutive = (
                    len(unique_date_index) >= 7
                    and (unique_date_index[1:] - unique_date_index[:-1]).days.min() == 1
                )
                if consecutive:
                    confirmed.append(
                        {
                            **run,
                            "evidence": "Repeated value spans at least seven consecutive calendar dates.",
                        }
                    )
            i = j + 1

    suspicious.sort(key=lambda x: (x["length"], x["distinct_dates"]), reverse=True)
    confirmed.sort(key=lambda x: (x["length"], x["distinct_dates"]), reverse=True)
    if confirmed:
        first = confirmed[0]
        return {
            "status": "detected",
            "present": True,
            **first,
            "suspicious_runs": suspicious,
            "confirmed_runs": confirmed,
        }
    status = "inconclusive" if suspicious else "not_detected"
    return {
        "status": status,
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
        "value": None,
        "suspicious_runs": suspicious,
        "confirmed_runs": confirmed,
        "evidence": "Repeated values were inspected, but were not on enough continuous dates to confirm a sensor fault.",
    }


def audit_target_typos(train: pd.DataFrame) -> dict[str, Any]:
    """Detect individual 10x anomalies using row-level evidence and normal-target plausibility."""
    if "rent_rs" not in train:
        return {
            "status": "inconclusive",
            "present": False,
            "count": 0,
            "candidate_count": 0,
            "anomalies": [],
            "evidence": "rent_rs is unavailable.",
        }

    values = pd.to_numeric(train["rent_rs"], errors="coerce")
    normal = values.dropna()
    if normal.empty:
        return {
            "status": "inconclusive",
            "present": False,
            "count": 0,
            "candidate_count": 0,
            "anomalies": [],
            "evidence": "No usable target values.",
        }

    q1, q3 = float(normal.quantile(0.25)), float(normal.quantile(0.75))
    iqr = q3 - q1
    upper_outlier = q3 + 1.5 * iqr
    normal_floor = float(normal.quantile(0.05))
    normal_ceiling = float(normal.quantile(0.95))
    anomalies: list[dict[str, Any]] = []

    for index, original in values.items():
        if pd.isna(original):
            continue
        original_float = float(original)
        corrected = original_float / 10.0
        corrected_in_normal_range = 0 <= corrected <= upper_outlier
        original_is_statistical_outlier = original_float > upper_outlier
        gap_evidence = original_float / max(normal.median(), 1.0) >= 3
        if (
            corrected_in_normal_range
            and original_is_statistical_outlier
            and gap_evidence
        ):
            anomalies.append(
                {
                    "record_id": str(train.loc[index, "record_id"]),
                    "original_value": int(original_float)
                    if original_float.is_integer()
                    else original_float,
                    "corrected_value": int(corrected)
                    if corrected.is_integer()
                    else round(corrected, 2),
                    "evidence": {
                        "corrected_within_iqr_normal_range": True,
                        "original_above_iqr_upper_bound": True,
                        "iqr_upper_bound": round(upper_outlier, 2),
                        "normal_p05_p95": [
                            round(normal_floor, 2),
                            round(normal_ceiling, 2),
                        ],
                        "multiplication_ratio": 10.0,
                    },
                }
            )

    status = "detected" if anomalies else "not_detected"
    return {
        "status": status,
        "present": bool(anomalies),
        "count": len(anomalies),
        "candidate_count": len(anomalies),
        "anomalies": anomalies,
        "normal_statistics": {
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr_upper_bound": round(upper_outlier, 2),
            "median": round(float(normal.median()), 2),
            "p05": round(normal_floor, 2),
            "p95": round(normal_ceiling, 2),
        },
    }


def audit_spelling_variants(train: pd.DataFrame, later: pd.DataFrame) -> dict[str, Any]:
    """Detect case, whitespace, punctuation, and known abbreviation variants, without frequency differences."""
    cat_cols = [
        c
        for c in later.select_dtypes(include=["object", "string"]).columns
        if c not in ("record_id", "date") and c in train
    ]
    evidence: list[dict[str, Any]] = []

    for col in cat_cols:
        raw_series = train[col].dropna().astype(str)
        raw_values = set(raw_series.unique())
        clean_values = set(
            later[col].dropna().astype(str).str.strip().str.lower().unique()
        )
        clean_raw_values = set(later[col].dropna().astype(str).str.strip().unique())

        normalized = set()
        variant_types: dict[str, list[str]] = {
            "case": [],
            "whitespace": [],
            "punctuation": [],
            "abbreviation": [],
        }
        for raw in raw_values:
            value = raw.strip().lower()
            if raw != raw.strip():
                variant_types["whitespace"].append(raw)
            if any(ch.isupper() for ch in raw):
                variant_types["case"].append(raw)
            if any(ch in ".,_-/" for ch in raw):
                variant_types["punctuation"].append(raw)
            compact = value.replace(".", "").replace("_", "")
            abbreviation_map = {
                "ful": "full",
                "semi": "semi",
                "sem": "semi",
                "unf": "unfurnished",
                "unfur": "unfurnished",
            }
            expanded = abbreviation_map.get(compact, compact)
            if expanded != compact:
                variant_types["abbreviation"].append(raw)
            normalized.add(expanded)

        unmatched = sorted(normalized - clean_values)
        substantive_variant_types = set(k for k, vals in variant_types.items() if vals)
        ordinary_case_only = (
            substantive_variant_types == {"case"} and raw_values == clean_raw_values
        )
        has_raw_variant = bool(substantive_variant_types) and not ordinary_case_only
        has_real_variant = bool(
            has_raw_variant and not unmatched and normalized.issubset(clean_values)
        )
        if has_real_variant:
            evidence.append(
                {
                    "column": col,
                    "raw_levels": len(raw_values),
                    "normalized_levels": len(normalized),
                    "clean_later_levels": len(clean_values),
                    "raw_values": sorted(raw_values),
                    "normalized_values": sorted(normalized),
                    "clean_values": sorted(clean_values),
                    "unmatched_normalized_values": unmatched,
                    "variant_types": {
                        k: sorted(v) for k, v in variant_types.items() if v
                    },
                }
            )

    if not evidence:
        return {
            "status": "not_detected",
            "present": False,
            "column": None,
            "true_levels": None,
            "evidence": [],
        }
    first = evidence[0]
    return {
        "status": "detected",
        "present": True,
        "column": first["column"],
        "true_levels": first["clean_later_levels"],
        "evidence": evidence,
    }


def _summary_fields(name: str, res: dict[str, Any]) -> dict[str, Any]:
    present = res["present"]
    if name == "leaking_column":
        return {
            "present": present,
            "column": res["columns"][0] if res["columns"] else None,
        }
    if name == "unit_mixup":
        return {
            k: res.get(k) if present else None
            for k in ("present", "column", "group_value", "factor")
        } | {"present": present}
    if name == "sentinel_value":
        return {
            k: res.get(k) if present else None
            for k in ("present", "column", "value", "count")
        } | {"present": present}
    if name == "duplicate_rows":
        return {"present": present, "count": res["count"]}
    if name == "stuck_sensor":
        return {
            k: res.get(k) if present else None
            for k in ("present", "column", "start_date", "end_date")
        } | {"present": present}
    if name == "target_typos":
        return {"present": present, "count": res["count"] if present else None}
    if name == "spelling_variants":
        return {
            k: res.get(k) if present else None
            for k in ("present", "column", "true_levels")
        } | {"present": present}
    return {"present": present}


def run_audit() -> dict[str, Any]:
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    results = {
        "leaking_column": audit_leaking_column(train, holdout),
        "unit_mixup": audit_unit_mixup(train),
        "sentinel_value": audit_sentinel_value(train),
        "duplicate_rows": audit_duplicate_rows(train),
        "stuck_sensor": audit_stuck_sensor(train),
        "target_typos": audit_target_typos(train),
        "spelling_variants": audit_spelling_variants(train, later),
    }

    lines = [
        "PG ROOM RENT - DATA AUDIT REPORT",
        f"Training rows: {len(train)}",
        f"Training SHA-256 source: {TRAIN_PATH}",
        "",
    ]
    for name, res in results.items():
        lines += [
            f"{name}: present={res['present']}",
            json.dumps(res, indent=2, sort_keys=True, default=str),
            "",
        ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )

    findings = yaml.safe_load(FINDINGS_PATH.read_text(encoding="utf-8"))
    data_issues = findings.setdefault("data_issues", {})
    for name, res in results.items():
        if res.get("status") == "inconclusive":
            continue
        data_issues[name] = _summary_fields(name, res)
    FINDINGS_PATH.write_text(
        yaml.safe_dump(findings, sort_keys=False), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_audit()
