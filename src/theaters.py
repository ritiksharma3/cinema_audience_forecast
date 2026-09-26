"""Unified theater dimension across booknow and cinePOS."""
import warnings

import numpy as np
import pandas as pd


def theater_dim(q, bt, ct, rel, bb, cb):
    n0 = len(bt)
    blank = bt.book_theater_id.isna() | (bt.book_theater_id.astype(str).str.strip() == "")
    q.log("theaters", f"booknow_theaters: {blank.sum()} of {n0} rows have blank book_theater_id "
        f"({bt[blank].drop(columns='book_theater_id').drop_duplicates().shape[0]} distinct type/area/latlong combos)",
        "DROPPED - cannot be joined to any booking or visit record", blank.sum())
    bt = bt[~blank]
    q.count("booknow_theaters: drop blank IDs", n0, len(bt))
    n1 = len(bt)
    bt = bt.drop_duplicates("book_theater_id")
    q.count("booknow_theaters: dedupe IDs", n1, len(bt))

    miss_ll = ct.latitude.isna().sum()
    q.log("theaters", f"cinePOS_theaters: {miss_ll} of {len(ct)} rows missing latitude/longitude",
        "KEPT with NaN lat/long (type & area still usable)", miss_ll)
    ll = pd.concat([bt, ct.dropna(subset=["latitude"])])
    q.log("theaters", "latitude/longitude look like area-level centroids, not per-theater "
        f"(only {ll[['latitude','longitude']].drop_duplicates().shape[0]} distinct points across {len(ll)} theaters)",
        "kept as-is; treat as area-level location, not theater location", None)

    b = bt.assign(system="booknow", theater_id=bt.book_theater_id).drop(columns="book_theater_id")
    c = ct.assign(system="cinePOS", theater_id=ct.cine_theater_id).drop(columns="cine_theater_id")
    dim = pd.concat([b, c], ignore_index=True)

    for sys_, ids, known in [("booknow", bb.book_theater_id.unique(), set(bt.book_theater_id)),
                             ("cinePOS", cb.cine_theater_id.unique(), set(ct.cine_theater_id))]:
        miss = [i for i in ids if i not in known]
        q.log("theaters", f"{sys_}: {len(miss)} of {len(ids)} theaters with bookings have NO metadata row",
            "theater_type/area = 'Unknown' for these (not dropped)", len(miss))
        dim = pd.concat([dim, pd.DataFrame({"system": sys_, "theater_id": miss,
                                            "theater_type": "Unknown", "theater_area": "Unknown"})],
                        ignore_index=True)

    x = (rel.merge(bt, on="book_theater_id").merge(ct, on="cine_theater_id", suffixes=("_bn", "_cp")))
    q.log("theaters", f"relation-file pairs with metadata in both systems: {len(x)}; "
        f"theater_type agrees {(x.theater_type_bn == x.theater_type_cp).mean():.0%}, "
        f"theater_area agrees {(x.theater_area_bn == x.theater_area_cp).mean():.0%}",
        "FLAG: area codes are probably system-specific (Area_xxx not a shared namespace), or the mapping "
        "is unreliable. Areas NOT merged across systems.", len(x))
    return dim
