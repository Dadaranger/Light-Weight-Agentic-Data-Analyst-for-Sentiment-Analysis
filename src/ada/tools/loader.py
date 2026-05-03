"""Format-aware file loading. Returns a pandas DataFrame + light metadata.

Supported: .csv (utf-8-sig auto-tried), .xlsx/.xls, .json, .jsonl, .parquet.
Reads everything as string for safe profiling; type inference is the schema
agent's job, not the loader's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

Encoding = Literal["utf-8", "utf-8-sig", "cp950", "big5"]


def _try_csv(path: Path) -> pd.DataFrame:
    """Try common encodings for CSVs (utf-8-sig handles Excel-exported zh-TW)."""
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "csv", b"", 0, 1, f"could not decode {path} with utf-8-sig/utf-8/cp950/big5"
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a tabular dataset by extension. Returns all-string DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _try_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False).astype(str)
    if suffix == ".json":
        return pd.read_json(path, dtype=False).astype(str)
    if suffix == ".parquet":
        return pd.read_parquet(path).astype(str)
    raise ValueError(f"unsupported file type: {suffix}")
