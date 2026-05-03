"""Column profiling — produces `ColumnProfile` records the schema agent reads."""
from __future__ import annotations

import pandas as pd

from ada.state import ColumnProfile


def profile_columns(df: pd.DataFrame, sample_n: int = 5) -> list[ColumnProfile]:
    """One ColumnProfile per column. Samples are non-null distinct values."""
    n = max(len(df), 1)
    profiles: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        non_null = non_null[non_null.astype(str).str.strip() != ""]
        samples = (
            non_null.astype(str)
            .drop_duplicates()
            .head(sample_n)
            .tolist()
        )
        profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(s.dtype),
                null_pct=round((1 - len(non_null) / n) * 100, 2),
                unique_pct=round(s.nunique(dropna=True) / n * 100, 2),
                sample_values=[v[:120] for v in samples],  # truncate long strings
            )
        )
    return profiles
