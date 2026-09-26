"""Load raw CSVs and parse datetimes (explicit ISO format, dayfirst cross-check)."""
import warnings

import numpy as np
import pandas as pd

from .config import RAW


def parse_dt(q, df, col, fmt, fname):
    """Parse with the observed explicit format; cross-check against dayfirst=True; fail loudly."""
    raw = df[col].astype(str)
    strict = pd.to_datetime(raw, format=fmt, errors="coerce")
    n_bad = strict.isna().sum()
    if n_bad:
        q.log("parse", f"{fname}.{col}: {n_bad} unparseable values", "STOP - raised error", n_bad)
        raise ValueError(f"{fname}.{col}: {n_bad} values failed format {fmt}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dayfirst = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    nat = dayfirst.isna().sum()
    swapped = (dayfirst.notna() & (dayfirst != strict)).sum()
    q.log("parse", f"{fname}.{col}: file uses ISO {fmt}, NOT DD-MM-YYYY as specified. "
        f"Parsing with dayfirst=True would corrupt {swapped + nat} of {len(df)} rows "
        f"({swapped} silently month/day-swapped, {nat} -> NaT)",
        "parsed with explicit ISO format instead (0 failures); dayfirst=True NOT used",
        swapped + nat, f"range {strict.min()} .. {strict.max()}")
    df[col] = strict
    return df


def load(q):
    print("Loading raw files")
    f = {
        "bb": "booknow_booking.csv", "bt": "booknow_theaters.csv", "vi": "booknow_visits.csv",
        "cb": "cinePOS_booking.csv", "ct": "cinePOS_theaters.csv", "rel": "movie_theater_id_relation.csv",
    }
    spec_names = {"vi": "booknow_visit.csv", "ct": "CinPOS_theaters.csv", "cb": "cinPOS_booking.csv"}
    for k, v in spec_names.items():
        q.log("load", f"file name differs from spec: expected {v}", f"used {f[k]}", None)
    q.log("load", "data folder is ./raw, not ./data/raw", "read from ./raw (files not moved)")

    d = {k: pd.read_csv(RAW / v, dtype={"book_theater_id": str, "cine_theater_id": str}) for k, v in f.items()}
    for k, v in f.items():
        q.count(f"load {v}", None, len(d[k]))
    for k in ("bb", "cb"):
        for c in ("show_datetime", "booking_datetime"):
            parse_dt(q, d[k], c, "%Y-%m-%d %H:%M:%S", f[k])
    parse_dt(q, d["vi"], "show_date", "%Y-%m-%d", f["vi"])
    return d
