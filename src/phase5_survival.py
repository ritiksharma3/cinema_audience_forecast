"""Phase 5 - survival analysis orchestrator.

Unit: one theater per booking system (mapped pairs kept separate - confirmed Phase 1 decision).
Time origin: first booking in the data. Event: no booking in the final 28 days (confirmed definition).
Outputs
  outputs/excel/survival_analysis.xlsx
  outputs/figures/phase5_*.png
"""
import warnings

import numpy as np
import pandas as pd

from . import survival_analysis as sa
from .config import INACTIVE_AFTER_DAYS, OUT_X
from .data_access import load_survival

warnings.filterwarnings("ignore")


def run_phase5():
    s, info = sa.prepare(load_survival(), sa.booking_gaps())
    print(f"theaters {len(s)}, events {s.event.sum()} (gap-adjusted {s.event_gap_adjusted.sum()}), "
          f"big areas {len(info['big_areas'])}")

    # ---------------- Kaplan-Meier + log-rank
    km_all, f_all = sa.km_table(s)
    km_sys, f_sys = sa.km_table(s, "system")
    km_type, f_type = sa.km_table(s, "type_group")
    km_area, f_area = sa.km_table(s, "area_km")
    km_gap, _ = sa.km_table(s, "system", event_col="event_gap_adjusted")
    km_gap["group"] = km_gap.group + " (gap-adjusted event)"
    lr = pd.DataFrame([sa.logrank(s, c) for c in ("system", "type_group", "area_km")] +
                      [dict(sa.logrank(s[s.metadata_missing == 0], "type_group"), stratified_by="type_group (metadata known only)"),
                       dict(sa.logrank(s[s.metadata_missing == 0], "area_km"), stratified_by="area_km (metadata known only)")])
    sa.km_plot({**f_all, **f_sys}, "Kaplan-Meier: probability a theater is still active", "phase5_01_km_overall_system.png")
    sa.km_plot(f_type, "Kaplan-Meier by theater_type (Unknown = no metadata row)", "phase5_02_km_type.png")
    area_fits = dict(sorted(f_area.items(), key=lambda kv: -kv[1].event_observed.size)[:8])
    sa.km_plot(area_fits, "Kaplan-Meier by theater_area (8 largest groups)", "phase5_03_km_area.png")

    # ---------------- Cox PH (primary)
    X = sa.cox_design(s)
    res = sa.fit_cox(s, X)
    ph = sa.ph_test(res)
    notes = (f"Primary model: all {len(s)} theaters, event = no booking in final {INACTIVE_AFTER_DAYS} days. "
             "References: type Other, area = pooled areas, cinePOS. Unknown metadata carried by metadata_missing. "
             f"lead_100h winsorised at p99 ({info['lead_winsor_p99_hrs']:.0f}h). Demand covariate = log2 mean daily tickets "
             "(HR per doubling); audience_count only exists for booknow - see sensitivity C. "
             "Estimated with statsmodels PHReg (Efron ties); lifelines CoxPHFitter failed to converge on this data.")
    cox_main = sa.cox_inference_table("Cox primary", res, ph, notes)
    fit_rows = [sa.model_fit_row("Cox primary", res)]
    sa.forest_plot(cox_main, "phase5_04_cox_forest.png", "Cox PH hazard ratios (primary)")
    sa.schoenfeld_plot(res, ["is_booknow", "metadata_missing", "lead_100h", "log2_mean_daily_tickets"], "phase5_05_schoenfeld.png")

    # ---------------- sensitivity models
    sens = []
    rA = sa.fit_cox(s, X, event_col="event_gap_adjusted")
    sens.append(sa.cox_inference_table("Sensitivity A: gap-adjusted event", rA, None,
                                       "Event only if the final silence also exceeds the theater's own longest earlier booking gap."))
    fit_rows.append(sa.model_fit_row("Sensitivity A: gap-adjusted event", rA))
    keep = s.left_truncated == 0
    rB = sa.fit_cox(s[keep], sa.cox_design(s[keep]))
    sens.append(sa.cox_inference_table("Sensitivity B: excl. left-truncated", rB, None,
                                       "Drops theaters first seen in the first 7 days of data (true start unknown)."))
    fit_rows.append(sa.model_fit_row("Sensitivity B: excl. left-truncated", rB))
    bk = s[(s.system == "booknow") & s.mean_daily_audience.notna()].copy()
    bk["log2_mean_daily_audience"] = np.log2(bk.mean_daily_audience)
    Xc = bk[["lead_100h", "log2_mean_daily_tickets", "log2_mean_daily_audience", "metadata_missing"]]
    rC = sa.fit_cox(bk, Xc.loc[:, Xc.nunique() > 1])
    sens.append(sa.cox_inference_table("Sensitivity C: booknow + audience_count", rC, None,
                                       f"booknow theaters with visits data only (n={len(bk)}, events={int(bk.event.sum())}); "
                                       "type/area dummies omitted (too few events per level)."))
    fit_rows.append(sa.model_fit_row("Sensitivity C: booknow + audience_count", rC))
    viol = [c for c in ph[ph.PH_p < 0.05].covariate if c in ("metadata_missing", "is_booknow")]
    if viol:
        rD = sa.fit_cox(s, X.drop(columns=viol), strata=viol)
        sens.append(sa.cox_inference_table(f"Sensitivity D: stratified by {', '.join(viol)}", rD, sa.ph_test(rD),
                                           "Remedy for PH violation: violating binary covariates moved to strata "
                                           "(separate baseline hazard per stratum; no HR estimated for them)."))
        fit_rows.append(sa.model_fit_row(f"Sensitivity D: stratified by {', '.join(viol)}", rD))

    event_check = pd.DataFrame([
        dict(check="events (28-day rule)", value=int(s.event.sum())),
        dict(check="events whose final silence <= own longest earlier gap", value=int(((s.event == 1) & (s.event_gap_adjusted == 0)).sum())),
        dict(check="events (gap-adjusted)", value=int(s.event_gap_adjusted.sum())),
        dict(check="median longest earlier gap, event theaters (days)", value=float(s[s.event == 1].max_gap_days.median())),
        dict(check="median longest earlier gap, censored theaters (days)", value=float(s[s.event == 0].max_gap_days.median())),
        dict(check="events with last booking in Dec 2023-Jan 2024", value=int(((s.event == 1) & (s.last_booking_date >= "2023-12-01")).sum())),
        dict(check="event rate: metadata known / missing (%)",
             value=f"{100 * s[s.metadata_missing == 0].event.mean():.1f} / {100 * s[s.metadata_missing == 1].event.mean():.1f}"),
    ])

    with pd.ExcelWriter(OUT_X / "survival_analysis.xlsx", engine="openpyxl") as w:
        pd.DataFrame(dict(item=["unit", "time origin", "survival time T", "event", "censoring", "data end",
                                "areas with own KM curve (>=100 theaters)", "areas with own Cox dummy (>=100 theaters and >=5 events)",
                                "large areas pooled in Cox (too few events)", "reference levels"],
                          value=["theater x booking system (mapped pairs separate)", "first booking in data",
                                 "event: last - first booking + 1; censored: data end - first booking + 1",
                                 f"no booking in final {INACTIVE_AFTER_DAYS} days", "still booking in final 28 days",
                                 str(info["end"].date()), ", ".join(info["big_areas"]), ", ".join(info["cox_areas"]),
                                 "; ".join(info["pooled_for_cox"]), "type=Other, area=pooled areas, system=cinePOS"]
                          )).to_excel(w, sheet_name="setup", index=False)
        pd.concat([km_all, km_sys, km_gap]).to_excel(w, sheet_name="km_overall_system", index=False)
        km_type.to_excel(w, sheet_name="km_by_type", index=False)
        km_area.sort_values("n", ascending=False).to_excel(w, sheet_name="km_by_area", index=False)
        lr.to_excel(w, sheet_name="logrank_tests", index=False)
        cox_main.to_excel(w, sheet_name="cox_inference", index=False)
        pd.DataFrame(fit_rows).to_excel(w, sheet_name="cox_model_fit", index=False)
        ph.to_excel(w, sheet_name="ph_test_schoenfeld", index=False)
        pd.concat(sens, ignore_index=True).to_excel(w, sheet_name="cox_sensitivity", index=False)
        event_check.to_excel(w, sheet_name="event_definition_check", index=False)
    print("Wrote", OUT_X / "survival_analysis.xlsx")
    return dict(s=s, km=(km_all, km_sys, km_type, km_area, km_gap), lr=lr, cox=cox_main, ph=ph, sens=sens,
                fit=pd.DataFrame(fit_rows), event_check=event_check)


if __name__ == "__main__":
    run_phase5()
