"""Booking cleaning, daily aggregation, and cross-system overlap detection."""
import warnings

import numpy as np
import pandas as pd


def clean_bookings(q, df, id_col, tk_col, name):
    dups = df.duplicated().sum()
    q.log("bookings", f"{name}: {dups} fully identical rows (same theater, show hour, booking hour, tickets)",
        "KEPT - timestamps are hour-granular so identical rows can be separate genuine bookings; "
        "duplicate volume scales with monthly volume (not a one-off load error). Revisit if you disagree.",
        dups)
    df = df.copy()
    df["lead_hrs"] = (df.show_datetime - df.booking_datetime).dt.total_seconds() / 3600
    neg = (df.lead_hrs < 0).sum()
    q.log("bookings", f"{name}: {neg} rows booked AFTER show start (lead time < 0, min {df.lead_hrs.min():.0f}h)",
        "tickets KEPT in demand; lead_hrs set to NaN for these rows so they don't bias lead-time averages", neg)
    df.loc[df.lead_hrs < 0, "lead_hrs"] = np.nan
    long = (df.lead_hrs > 24 * 60).sum()
    q.log("bookings", f"{name}: {long} rows with lead time > 60 days (max {df.lead_hrs.max()/24:.0f} days)",
        "KEPT - reported mean AND median lead time; median is robust to these", long)
    df["show_date"] = df.show_datetime.dt.normalize()
    return df


def daily_agg(df, tk_col, id_col, sfx):
    df = df.assign(_lead_x_tk=df.lead_hrs * df[tk_col])
    g = df.groupby("show_date")
    out = pd.DataFrame({
        f"daily_tickets_{sfx}": g[tk_col].sum(),
        f"n_bookings_{sfx}": g.size(),
        f"n_active_theaters_{sfx}": g[id_col].nunique(),
        f"mean_lead_hrs_{sfx}": g.lead_hrs.mean(),
        f"median_lead_hrs_{sfx}": g.lead_hrs.median(),
        f"_lead_x_tk_{sfx}": g._lead_x_tk.sum(),
        f"_tk_valid_{sfx}": df[df.lead_hrs.notna()].groupby("show_date")[tk_col].sum(),
    })
    return out


def overlap(q, bb, cb, rel):
    """Record-level match of bookings for theaters that the relation file says are the same physical theater."""
    k = ["cine_theater_id", "show_datetime", "booking_datetime", "tk"]
    bm = bb.merge(rel, on="book_theater_id").assign(tk=lambda x: x.tickets_booked)
    cm = cb[cb.cine_theater_id.isin(rel.cine_theater_id)].assign(tk=lambda x: x.tickets_sold)
    # occurrence index so N identical rows on one side match at most N on the other
    bm["occ"] = bm.groupby(k).cumcount()
    cm["occ"] = cm.groupby(k).cumcount()
    m = bm.merge(cm[k + ["occ"]], on=k + ["occ"])
    q.count("overlap: booknow rows at mapped theaters", len(bb), len(bm))
    q.count("overlap: cinePOS rows at mapped theaters", len(cb), len(cm))
    q.count("overlap: exact record matches (both systems)", len(bm), len(m),
          "same theater pair + show hour + booking hour + tickets")

    q.log("overlap", f"relation file maps {len(rel)} booknow<->cinePOS theater pairs "
        f"({rel.book_theater_id.isin(bb.book_theater_id).sum()} of them have booknow bookings, "
        f"{rel.cine_theater_id.isin(cb.cine_theater_id).sum()} have cinePOS bookings)",
        "checked for double counting at record level", len(rel))
    q.log("overlap", f"DOUBLE COUNTING FOUND: {len(m)} booking records / {m.tk.sum()} tickets appear "
        f"identically in both systems (booknow share at mapped theaters: {bm.tk.sum()} tickets; "
        f"cinePOS: {cm.tk.sum()} tickets)",
        "total_demand = booked + sold - overlap_tickets (dedup). Naive additive sum kept as "
        "total_demand_naive for comparison. CONFIRMED by user (dedup target).", len(m))
    q.log("overlap", f"mapped theaters are only PARTIALLY overlapping: {len(bm) - len(m)} booknow and "
        f"{len(cm) - len(m)} cinePOS records at mapped theaters have no counterpart",
        "treated as distinct bookings (additive) - only exact matches removed", None)

    daily = m.groupby(m.show_datetime.dt.normalize()).tk.sum().rename("overlap_tickets")
    detail = m.groupby(["book_theater_id", "cine_theater_id"]).agg(
        matched_records=("tk", "size"), matched_tickets=("tk", "sum")).reset_index()
    per_pair = (bm.groupby(["book_theater_id", "cine_theater_id"]).tk.sum().rename("booknow_tickets").reset_index()
                .merge(cm.merge(rel, on="cine_theater_id").groupby(["book_theater_id", "cine_theater_id"])
                       .tk.sum().rename("cinepos_tickets").reset_index(), how="outer")
                .merge(detail, how="left").fillna(0))
    return daily, per_pair
