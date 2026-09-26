"""Shared loaders for the Phase 1 outputs, used by Phases 2-5."""
import pandas as pd

from .config import MODEL_START, OUT_D


def load_daily(model_window=True):
    """Daily series indexed by date; model_window=True drops the startup-censored days (before MODEL_START)."""
    df = pd.read_csv(OUT_D / "daily_series.csv", parse_dates=["date"], index_col="date")
    df["holiday_name"] = df.holiday_name.fillna("")
    df = df.asfreq("D")
    if model_window:
        df = df.loc[MODEL_START:]
        assert df.startup_censored.sum() == 0
    return df


def load_survival():
    return pd.read_csv(OUT_D / "theater_survival.csv", parse_dates=["first_booking_date", "last_booking_date"])
