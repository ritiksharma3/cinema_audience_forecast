"""Project paths and analysis constants shared by all phases."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
OUT_X = OUT / "excel"
OUT_D = OUT / "data"
OUT_FIG = OUT / "figures"
OUT_REP = OUT / "reports"
for _p in (OUT_X, OUT_D, OUT_FIG, OUT_REP):
    _p.mkdir(parents=True, exist_ok=True)

TARGET = "total_demand"          # dedup target (confirmed Phase 1 decision)
RAMP_UP_DAYS = 28                # daily series: first N days under-counted (bookings before data start not in files)
MODEL_START = "2023-01-29"       # first day after the startup-censored window; used for EDA and modelling
INACTIVE_AFTER_DAYS = 28         # survival: no booking in the last N days of data => event (closed)
LEFT_TRUNC_DAYS = 7              # survival: first booking within N days of data start => likely active earlier
SEASONAL_PERIOD = 7              # weekly cycle; confirmed/overridden by Phase 2 periodogram
