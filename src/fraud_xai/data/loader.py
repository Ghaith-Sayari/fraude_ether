"""Dataset loading (upload, path or demo) and column-name hygiene."""
from __future__ import annotations

import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace (e.g. ' ERC20 avg val rec') and de-duplicate names."""
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        c = str(c).strip()
        n = seen.get(c, 0)
        seen[c] = n + 1
        cols.append(c if n == 0 else f"{c}.{n}")
    out = df.copy()
    out.columns = cols
    return out


def load_table(source, filename: str | None = None) -> pd.DataFrame:
    """Load CSV/TSV/Parquet/Excel from a path or a file-like object."""
    name = (filename or getattr(source, "name", str(source))).lower()
    if name.endswith((".parquet", ".pq")):
        df = pd.read_parquet(source)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)
        if df.shape[1] == 1:  # wrong delimiter -> sniff
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, sep=None, engine="python")
    return clean_columns(df)
