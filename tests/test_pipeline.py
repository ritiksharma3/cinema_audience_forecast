"""Regression tests for the invariants that broke (or nearly broke) during development.

Unit tests run on synthetic data. Output tests check the files written by `run_pipeline.py --phase all`
and are skipped if those outputs do not exist yet.
    .venv\\Scripts\\python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import combination as cb
from src import models as mdl
from src.config import OUT_D, OUT_X
from src.diagnostics import accuracy
from src.loading import parse_dt
from src.phase3_models import SPLIT
from src.quality_log import QualityLog


# ------------------------------------------------------------------ unit tests
def test_parse_dt_reads_iso_as_year_month_day():
    # dayfirst=True would turn 2023-01-02 into 1 Feb 2023 - the bug the spec's instruction would have caused
    df = pd.DataFrame({"d": ["2023-01-02 19:00:00", "2023-12-25 10:00:00"]})
    q = QualityLog(verbose=False)
    out = parse_dt(q, df, "d", "%Y-%m-%d %H:%M:%S", "x.csv")
    assert list(out.d) == [pd.Timestamp("2023-01-02 19:00"), pd.Timestamp("2023-12-25 10:00")]
    # 2023-01-02 would be silently swapped to 1 Feb; 2023-12-25 (month 25) would become NaT
    assert "would corrupt 2 of 2 rows (1 silently month/day-swapped, 1 -> NaT)" in q.logs[0]["issue"]


def test_parse_dt_fails_loudly_on_bad_values():
    df = pd.DataFrame({"d": ["2023-01-02 19:00:00", "02-01-2023 19:00"]})
    with pytest.raises(ValueError):
        parse_dt(QualityLog(verbose=False), df, "d", "%Y-%m-%d %H:%M:%S", "x.csv")


def test_accuracy_metrics():
    m = accuracy(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert m["MAE"] == pytest.approx(15.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["MAPE"] == pytest.approx(10.0)


def test_exog_frame_weekday_dummies_monday_base():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")          # starts on a Monday
    df = pd.DataFrame({"is_holiday": 0, "booknow_missing": 0, "mean_lead_hrs_combined": 100.0}, index=idx)
    X = mdl.exog_frame(df)
    dows = [c for c in X if c.startswith("dow_")]
    assert len(dows) == 6
    assert X.loc[idx[0], dows].sum() == 0                            # Monday = all zeros
    assert (X[dows].sum(axis=1) <= 1).all()


def _forecasts(seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.uniform(10_000, 40_000, 40))
    F = pd.DataFrame({"good": y + rng.normal(0, 1_000, 40), "bad": y + rng.normal(0, 8_000, 40)})
    return F, y


def test_bates_granger_weights_sum_to_one_and_favour_accurate_model():
    F, y = _forecasts()
    w = cb.bates_granger(F, y)["weights"]
    assert w.sum() == pytest.approx(1.0)
    assert w["good"] > 0.9


def test_granger_ramanathan_constrained_sums_to_one():
    F, y = _forecasts()
    assert cb.granger_ramanathan_constrained(F, y)["weights"].sum() == pytest.approx(1.0)


def test_diebold_mariano_sign():
    F, y = _forecasts()
    stat, p = cb.diebold_mariano(y - F["good"], y - F["bad"])
    assert stat < 0 and p < 0.05                                     # first forecast has lower loss


def test_dec_multiplier_is_one_without_december():
    idx = pd.date_range("2023-02-01", periods=60, freq="D")
    df = pd.DataFrame({"total_demand": 1000.0 + np.arange(60), "is_holiday": 0, "booknow_missing": 0}, index=idx)
    assert mdl.dec_multiplier(df) == (1.0, pytest.approx(np.nan, nan_ok=True))


def test_splits_are_contiguous():
    ends = [pd.Timestamp(SPLIT[k][1]) for k in ("train", "val")]
    starts = [pd.Timestamp(SPLIT[k][0]) for k in ("val", "test")]
    for e, s in zip(ends, starts):
        assert s - e == pd.Timedelta(days=1)


# ------------------------------------------------------------------ output tests
needs = lambda f: pytest.mark.skipif(not f.exists(), reason=f"{f.name} not generated yet")


@needs(OUT_D / "daily_series.csv")
def test_daily_series_complete_and_dedup_identity():
    d = pd.read_csv(OUT_D / "daily_series.csv", parse_dates=["date"])
    assert len(d) == 424 and d.date.is_monotonic_increasing
    assert (d.date.diff().dropna() == pd.Timedelta(days=1)).all()   # no silent gaps
    assert d.total_demand.notna().all()
    assert (d.total_demand == d.daily_tickets_booked + d.daily_tickets_sold - d.overlap_tickets).all()
    assert d.startup_censored.sum() == 28


@needs(OUT_D / "phase3_forecasts_test.csv")
def test_phase3_forecasts_cover_all_models_and_split_dates():
    for split in ("val", "test"):
        f = pd.read_csv(OUT_D / f"phase3_forecasts_{split}.csv", parse_dates=["date"])
        assert f.model.nunique() == len(mdl.MODELS)
        a, b = SPLIT[split]
        assert f.date.min() == pd.Timestamp(a) and f.date.max() == pd.Timestamp(b)
        assert (f["mean"] > 0).all()


@needs(OUT_X / "forecast_combination.xlsx")
def test_forward_forecast_starts_on_leap_day_with_weekend_peak():
    f = pd.read_excel(OUT_X / "forecast_combination.xlsx", sheet_name="forward_forecast", parse_dates=["date"])
    assert f.date.iloc[0] == pd.Timestamp("2024-02-29")                # day after the last observation
    by_day = f.groupby(f.date.dt.dayofweek).combined_mean.mean()
    assert set(by_day.nlargest(2).index) == {5, 6}                    # Saturday and Sunday peak


@needs(OUT_X / "forecast_combination.xlsx")
def test_combination_weights_sum_to_one():
    w = pd.read_excel(OUT_X / "forecast_combination.xlsx", sheet_name="weights_all_methods").set_index("method")
    for m in ("Simple average", "Bates-Granger (inverse MSE)", "Granger-Ramanathan (sum-to-one, no intercept)"):
        assert w.loc[m, "sum_of_weights"] == pytest.approx(1.0)


@needs(OUT_D / "theater_survival.csv")
def test_survival_extract_flags_consistent():
    s = pd.read_csv(OUT_D / "theater_survival.csv")
    assert (s.event_inactive + s.censored == 1).all()
    assert (s.duration_days >= 1).all()
    assert s.groupby("system").theater_id.apply(lambda x: x.is_unique).all()
