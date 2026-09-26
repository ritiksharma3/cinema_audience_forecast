"""Phase 1 - merge raw booking systems into one unified daily demand series.

Outputs
  outputs/excel/combined_data.xlsx   daily_series, data_quality_log, merge_row_counts,
                                     theater_dim, overlap_detail
  outputs/excel/theater_survival.xlsx
  outputs/data/daily_series.csv, outputs/data/theater_survival.csv  (for later phases)
"""
import warnings

import pandas as pd

from .bookings import clean_bookings, daily_agg, overlap
from .calendar_features import add_calendar
from .config import OUT_D, OUT_X, RAMP_UP_DAYS, MODEL_START
from .loading import load
from .quality_log import QualityLog
from .survival_extract import survival
from .theaters import theater_dim

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_phase1(q=None):
    """Run the full Phase 1 merge; returns (daily, survival) and writes all outputs."""
    q = q or QualityLog()
    d = load(q)
    bb, cb, vi, rel = d["bb"], d["cb"], d["vi"], d["rel"]

    print("Cleaning bookings")
    bb = clean_bookings(q, bb, "book_theater_id", "tickets_booked", "booknow_booking")
    cb = clean_bookings(q, cb, "cine_theater_id", "tickets_sold", "cinePOS_booking")

    vdup = vi.duplicated(["book_theater_id", "show_date"], keep=False)
    q.log("visits", f"booknow_visits: {vdup.sum()} rows share (theater, date) with another row - all on "
        f"{', '.join(vi[vdup].show_date.dt.strftime('%Y-%m-%d').unique())}",
        "SUMMED per theater-date (looks like a split load); only used as Phase 5 covariate, not in target",
        vdup.sum())
    n0 = len(vi)
    vi = vi.groupby(["book_theater_id", "show_date"], as_index=False).audience_count.sum()
    q.count("booknow_visits: collapse duplicate theater-dates", n0, len(vi))
    q.log("visits", "booknow_visits covers 826 theaters vs 301 in booknow_booking; audience_count is NOT part of "
        "the target per your spec", "not merged into total_demand", None)

    print("Daily aggregation")
    ab = daily_agg(bb, "tickets_booked", "book_theater_id", "booked")
    ac = daily_agg(cb, "tickets_sold", "cine_theater_id", "sold")
    q.count("booknow_booking -> daily", len(bb), len(ab))
    q.count("cinePOS_booking -> daily", len(cb), len(ac))
    ab = ab.rename(columns={"n_bookings_booked": "n_bookings_booknow", "n_active_theaters_booked": "n_active_theaters_booknow",
                            "mean_lead_hrs_booked": "mean_lead_hrs_booknow", "median_lead_hrs_booked": "median_lead_hrs_booknow"})
    ac = ac.rename(columns={"n_bookings_sold": "n_bookings_cinepos", "n_active_theaters_sold": "n_active_theaters_cinepos",
                            "mean_lead_hrs_sold": "mean_lead_hrs_cinepos", "median_lead_hrs_sold": "median_lead_hrs_cinepos"})

    start = min(ab.index.min(), ac.index.min()); end = max(ab.index.max(), ac.index.max())
    idx = pd.date_range(start, end, freq="D", name="date")
    daily = pd.DataFrame(index=idx)
    q.count("complete daily date index", None, len(idx), f"{start:%Y-%m-%d} .. {end:%Y-%m-%d}")

    n = len(daily); daily = daily.join(ac); q.count("join cinePOS daily", n, len(daily))
    n = len(daily); daily = daily.join(ab); q.count("join booknow daily", n, len(daily))

    miss_c = daily.daily_tickets_sold.isna()
    miss_b = daily.daily_tickets_booked.isna()
    q.log("date_index", f"cinePOS has bookings on {(~miss_c).sum()} of {len(idx)} days",
        f"{miss_c.sum()} missing days", miss_c.sum())
    months = daily[miss_b].index.to_period("M").value_counts().sort_index()
    q.log("date_index", f"BOOKNOW OUTAGE: {miss_b.sum()} days with zero booknow records "
        f"(by month: {', '.join(f'{p}:{c}' for p, c in months.items())}); Aug 2023 has 2 tickets, Sep 12 total",
        "daily_tickets_booked filled with 0 so total_demand stays defined, BUT booknow_missing=1 flag added. "
        "This is almost certainly a data/system gap, not zero demand. Booknow is ~3.6% of total so impact is "
        "small (~100-400 tickets/day understated). CONFIRMED by user: keep 0 + flag (alternatives rejected: impute, or model "
        "cinePOS only).", miss_b.sum(),
        f"missing dates: {', '.join(daily[miss_b].index.strftime('%Y-%m-%d'))}")
    daily["booknow_missing"] = miss_b.astype(int)
    daily["cinepos_missing"] = miss_c.astype(int)
    daily["startup_censored"] = (daily.index < start + pd.Timedelta(days=RAMP_UP_DAYS)).astype(int)
    q.log("date_index", f"DATA-START CENSORING: earliest booking_datetime in both files is {start:%Y-%m-%d}, so shows "
        f"in the first weeks are missing all bookings made before that date. Weekly mean demand ramps "
        f"510 -> 4.7k -> 10.4k -> 12.0k -> 14.1k over the first 5 weeks, and lead time ramps 5h -> 176h.",
        f"flagged startup_censored=1 for the first {RAMP_UP_DAYS} days, NOT dropped. Recommend excluding them from "
        f"model training. CONFIRMED by user: EDA and modelling start {MODEL_START}.", RAMP_UP_DAYS)
    for c in ["daily_tickets_booked", "n_bookings_booknow", "n_active_theaters_booknow", "_lead_x_tk_booked", "_tk_valid_booked",
              "daily_tickets_sold", "n_bookings_cinepos", "n_active_theaters_cinepos", "_lead_x_tk_sold", "_tk_valid_sold"]:
        daily[c] = daily[c].fillna(0)
    q.log("date_index", "booknow mean/median lead time on missing days", "left as NaN (no bookings -> undefined); "
        "use mean_lead_hrs_combined (always defined) as the exogenous feature", miss_b.sum())

    daily["mean_lead_hrs_combined"] = ((daily._lead_x_tk_booked + daily._lead_x_tk_sold)
                                       / (daily._tk_valid_booked + daily._tk_valid_sold))
    daily = daily.drop(columns=[c for c in daily.columns if c.startswith("_")])

    print("Overlap check")
    ov_daily, ov_pairs = overlap(q, bb, cb, rel)
    n = len(daily); daily = daily.join(ov_daily); q.count("join overlap daily", n, len(daily))
    daily["overlap_tickets"] = daily.overlap_tickets.fillna(0)

    daily["total_demand_naive"] = daily.daily_tickets_booked + daily.daily_tickets_sold
    daily["total_demand"] = daily.total_demand_naive - daily.overlap_tickets
    daily["booknow_share"] = daily.daily_tickets_booked / daily.total_demand_naive

    print("Theater metadata")
    dim = theater_dim(q, d["bt"], d["ct"], rel, bb, cb)
    q.count("theater_dim (both systems, incl. Unknown)", None, len(dim))
    # daily ticket share by theater_type (combined systems)
    t = pd.concat([
        bb.merge(dim[dim.system == "booknow"], left_on="book_theater_id", right_on="theater_id")[["show_date", "theater_type", "tickets_booked"]]
          .rename(columns={"tickets_booked": "tk"}),
        cb.merge(dim[dim.system == "cinePOS"], left_on="cine_theater_id", right_on="theater_id")[["show_date", "theater_type", "tickets_sold"]]
          .rename(columns={"tickets_sold": "tk"}),
    ])
    q.count("bookings joined to theater_dim (both systems)", len(bb) + len(cb), len(t), "inner join; every booking matched a dim row (Unknown included)")
    share = t.pivot_table(index="show_date", columns="theater_type", values="tk", aggfunc="sum").fillna(0)
    share = share.div(share.sum(axis=1), axis=0).add_prefix("share_tickets_type_")
    n = len(daily); daily = daily.join(share); q.count("join theater_type shares", n, len(daily))

    print("Calendar")
    daily = add_calendar(q, daily)

    assert daily.index.is_unique and len(daily) == len(idx)
    assert daily.total_demand.notna().all()
    q.log("final", f"daily_series: {len(daily)} rows, {daily.index.min():%Y-%m-%d}..{daily.index.max():%Y-%m-%d}, "
        f"no missing dates, total_demand non-null on every day", "OK", len(daily))

    first_cols = ["total_demand", "total_demand_naive", "overlap_tickets", "daily_tickets_booked", "daily_tickets_sold",
                  "booknow_missing", "cinepos_missing", "startup_censored", "booknow_share", "day_of_week", "day_name", "is_weekend",
                  "month", "year", "is_holiday", "is_major_festival", "holiday_name",
                  "mean_lead_hrs_combined", "mean_lead_hrs_booknow", "median_lead_hrs_booknow",
                  "mean_lead_hrs_cinepos", "median_lead_hrs_cinepos"]
    daily = daily[first_cols + [c for c in daily.columns if c not in first_cols]]

    print("Survival extract")
    surv = survival(q, bb, cb, rel, dim, vi, end, start)
    q.count("theater_survival rows (one per theater per system)", None, len(surv))

    # write
    daily_out = daily.reset_index()
    daily_out["date"] = daily_out.date.dt.date
    with pd.ExcelWriter(OUT_X / "combined_data.xlsx", engine="openpyxl") as w:
        daily_out.to_excel(w, sheet_name="daily_series", index=False)
        q.log_frame().to_excel(w, sheet_name="data_quality_log", index=False)
        q.count_frame().to_excel(w, sheet_name="merge_row_counts", index=False)
        dim.to_excel(w, sheet_name="theater_dim", index=False)
        ov_pairs.to_excel(w, sheet_name="overlap_detail", index=False)
    so = surv.copy()
    for c in ("first_booking_date", "last_booking_date"):
        so[c] = so[c].dt.date
    so.to_excel(OUT_X / "theater_survival.xlsx", sheet_name="theater_survival", index=False)
    daily.to_csv(OUT_D / "daily_series.csv")
    surv.to_csv(OUT_D / "theater_survival.csv", index=False)
    print("Wrote", OUT_X / "combined_data.xlsx", "and", OUT_X / "theater_survival.xlsx")
    return daily, surv


if __name__ == "__main__":
    run_phase1()
