"""Per-theater survival extract (first/last booking, event/censored flags)."""
import warnings

import numpy as np
import pandas as pd

from .config import INACTIVE_AFTER_DAYS, LEFT_TRUNC_DAYS


def survival(q, bb, cb, rel, dim, vi, end_date, start_date):
    b = bb.groupby("book_theater_id").agg(
        first_booking_date=("show_date", "min"), last_booking_date=("show_date", "max"),
        total_tickets=("tickets_booked", "sum"), active_days=("show_date", "nunique"),
        mean_lead_hrs=("lead_hrs", "mean")).reset_index().rename(columns={"book_theater_id": "theater_id"})
    b["system"] = "booknow"
    c = cb.groupby("cine_theater_id").agg(
        first_booking_date=("show_date", "min"), last_booking_date=("show_date", "max"),
        total_tickets=("tickets_sold", "sum"), active_days=("show_date", "nunique"),
        mean_lead_hrs=("lead_hrs", "mean")).reset_index().rename(columns={"cine_theater_id": "theater_id"})
    c["system"] = "cinePOS"
    s = pd.concat([b, c], ignore_index=True)
    s = s.merge(dim[["system", "theater_id", "theater_type", "theater_area"]], on=["system", "theater_id"], how="left")

    # mapped pairs: keep separate rows (areas disagree across systems) but tag the physical-theater link
    s["linked_theater_id"] = s.theater_id.map(
        {**dict(zip(rel.book_theater_id, rel.cine_theater_id)), **dict(zip(rel.cine_theater_id, rel.book_theater_id))})

    aud = vi.groupby("book_theater_id").audience_count.mean().rename("mean_daily_audience")
    s = s.merge(aud, left_on="theater_id", right_index=True, how="left")

    s["duration_days"] = (s.last_booking_date - s.first_booking_date).dt.days + 1
    s["days_since_last_booking"] = (end_date - s.last_booking_date).dt.days
    s["event_inactive"] = (s.days_since_last_booking > INACTIVE_AFTER_DAYS).astype(int)
    s["censored"] = 1 - s.event_inactive
    s["left_truncated"] = ((s.first_booking_date - start_date).dt.days < LEFT_TRUNC_DAYS).astype(int)
    s["mean_daily_tickets"] = s.total_tickets / s.active_days

    q.log("survival", f"event definition: theater 'inactive' (event=1) if no booking in final "
        f"{INACTIVE_AFTER_DAYS} days of data (after {end_date - pd.Timedelta(days=INACTIVE_AFTER_DAYS):%Y-%m-%d}); "
        "else censored", f"{s.event_inactive.sum()} events / {s.censored.sum()} censored of {len(s)} theaters",
        len(s), "CONFIRMED by user: 28-day window, mapped pairs kept as separate rows")
    q.log("survival", f"{s.left_truncated.sum()} theaters first appear within {LEFT_TRUNC_DAYS} days of data start "
        "- their true start date is unknown (left truncation)",
        "flagged left_truncated=1; Phase 5 should account for this", s.left_truncated.sum())
    bn_out = s[(s.system == "booknow") & (s.last_booking_date.between("2023-07-20", "2023-10-15"))]
    q.log("survival", f"{len(bn_out)} booknow theaters have their last booking inside/near the Aug-Sep 2023 booknow "
        "outage window", "flagged; these 'deaths' may be system artefacts rather than closures", len(bn_out))
    s["single_day_theater"] = (s.active_days == 1).astype(int)
    q.log("survival", f"{s.single_day_theater.sum()} theaters have bookings on only 1 day",
        "kept + flagged single_day_theater=1 (duration=1)", s.single_day_theater.sum())
    q.log("survival", "mean_daily_audience (booknow_visits) only exists for booknow theaters",
        f"NaN for all cinePOS theaters ({(s.system=='cinePOS').sum()} rows) - Phase 5 Cox must use "
        "mean_daily_tickets as the demand covariate or be restricted to booknow", None)
    cols = ["system", "theater_id", "linked_theater_id", "theater_type", "theater_area", "first_booking_date",
            "last_booking_date", "duration_days", "days_since_last_booking", "event_inactive", "censored",
            "left_truncated", "single_day_theater", "active_days", "total_tickets", "mean_daily_tickets",
            "mean_lead_hrs", "mean_daily_audience"]
    return s[cols]
