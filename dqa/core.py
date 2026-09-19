"""Read-only profiling and baseline-trained anomaly detection."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(os.environ.get("DQA_HOME", Path(__file__).resolve().parents[1])).resolve()
MAX_ROWS = 100_000
MAX_BYTES = 50 * 1024 * 1024


def safe_file(folder: str, name: str, suffix: str) -> Path:
    base = (ROOT / folder).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or path.suffix.lower() != suffix:
        raise ValueError("File must be inside the configured folder with the expected extension.")
    if not path.is_file():
        raise ValueError(f"File not found: {name}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("File exceeds the 50 MB local MVP limit.")
    return path


def load(name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(safe_file("data", name, ".csv"), nrows=MAX_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError("Please provide a valid UTF-8 CSV with a header.") from exc
    if len(frame) > MAX_ROWS:
        raise ValueError("Dataset exceeds the 100,000 row MVP limit; export a smaller partition.")
    if frame.empty:
        raise ValueError("Dataset has no rows.")
    return frame


def scalar(value):
    return None if pd.isna(value) or not np.isfinite(value) else float(value)


def datasets() -> list[str]:
    base = (ROOT / "data").resolve()
    return sorted(p.name for p in base.glob("*.csv") if p.resolve().is_relative_to(base))


def profile(name: str) -> dict:
    df = load(name)
    columns = []
    for column in df:
        s = df[column]
        item = {"name": column, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 2), "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            finite = s.replace([np.inf, -np.inf], np.nan)
            item.update({"mean": scalar(finite.mean()), "min": scalar(finite.min()), "max": scalar(finite.max()),
                         "non_finite": int(np.isinf(s).sum())})
        columns.append(item)
    return {"dataset": name, "rows": len(df), "column_count": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()), "columns": columns}


def compare(current: str, baseline: str) -> dict:
    now, before = load(current), load(baseline)
    changes = []
    for col in sorted(set(now) & set(before)):
        a, b = before[col], now[col]
        item = {"column": col, "baseline_dtype": str(a.dtype), "current_dtype": str(b.dtype),
                "missing_change_pp": round(float((b.isna().mean() - a.isna().mean()) * 100), 2)}
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.replace([np.inf, -np.inf], np.nan), b.replace([np.inf, -np.inf], np.nan)
            item.update({"baseline_mean": scalar(a.mean()), "current_mean": scalar(b.mean())})
            std = a.std()
            item["mean_shift_in_baseline_std"] = scalar((b.mean() - a.mean()) / std) if std > 0 else None
            item["constant_baseline_changed"] = bool(a.nunique() == 1 and (b.dropna() != a.dropna().iloc[0]).any())
        changes.append(item)
    return {"baseline": baseline, "current": current, "baseline_rows": len(before), "current_rows": len(now),
            "row_count_change_pct": round((len(now) / len(before) - 1) * 100, 2),
            "added_columns": sorted(set(now) - set(before)), "removed_columns": sorted(set(before) - set(now)),
            "changes": changes, "note": "Mean shifts are descriptive, not statistical significance or proof of cause."}


def anomalies(current: str, baseline: str, columns: list[str] | None = None) -> dict:
    now, before = load(current), load(baseline)
    if len(before) < 20:
        raise ValueError("At least 20 baseline rows are required for anomaly detection.")
    if columns is not None and not columns:
        raise ValueError("Select at least one numeric feature.")
    selected = columns if columns is not None else [c for c in before.select_dtypes(include="number")
        if c in now.select_dtypes(include="number") and not (c.lower() == "id" or c.lower().endswith("_id"))]
    for col in selected:
        if col not in before or col not in now or not pd.api.types.is_numeric_dtype(before[col]) or not pd.api.types.is_numeric_dtype(now[col]):
            raise ValueError(f"Feature must be numeric in both datasets: {col}")
    train = before[selected].replace([np.inf, -np.inf], np.nan)
    selected = [c for c in selected if train[c].notna().any()]
    if not selected:
        raise ValueError("No usable shared numeric features. Check types or select numeric columns.")
    medians = train[selected].median()
    train = train[selected].fillna(medians)
    test = now[selected].replace([np.inf, -np.inf], np.nan).fillna(medians)
    model = IsolationForest(n_estimators=150, contamination="auto", random_state=42, n_jobs=1).fit(train)
    scores = model.decision_function(test)
    positions = np.flatnonzero(scores < 0)
    ranked = sorted(positions, key=lambda i: scores[i])[:50]
    rows = [{"data_row": int(i + 1), "score": round(float(scores[i]), 6),
             "values": json.loads(now.iloc[[i]][selected].to_json(orient="records"))[0]} for i in ranked]
    return {"features": selected, "flagged_rows": len(positions), "total_rows": len(now), "rows": rows,
            "note": "Fit on baseline only; negative scores are unusual, not confirmed errors. Top 50 shown. Missing values use baseline medians. Data rows exclude the header."}

