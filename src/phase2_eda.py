"""Phase 2 - EDA & decomposition orchestrator.

Outputs
  outputs/figures/phase2_*.png
  outputs/excel/eda_results.xlsx
  outputs/reports/phase2_eda_summary.md
  outputs/data/phase2_outlier_dates.csv   (for Phase 3 dummies)
"""
import json
import warnings

import numpy as np
import pandas as pd

from . import eda
from .config import OUT_D, OUT_REP, OUT_X, SEASONAL_PERIOD, TARGET
from .data_access import load_daily

warnings.filterwarnings("ignore")


def _break_candidates(full):
    """Pre-specified candidate dates, derived from the data rather than hardcoded."""
    m = full.booknow_missing.astype(bool)
    return {
        "booknow outage start": m[m].index.min(),
        "booknow outage end (first day back)": m[m].index.max() + pd.Timedelta(days=1),
        "December 2023 surge start": pd.Timestamp("2023-12-01"),
    }


def run_phase2():
    df = load_daily(model_window=True)
    y = df[TARGET].astype(float)
    y_log = np.log(y)
    print(f"Phase 2 on {len(y)} days {y.index.min():%Y-%m-%d}..{y.index.max():%Y-%m-%d}")

    figs = {"series": eda.plot_series(df, TARGET)}
    figs["weekday"], weekday = eda.plot_weekday_profile(df, TARGET)
    peaks, figs["periodogram"] = eda.detect_periods(y)
    decomp, p = eda.classical_decomp(y, SEASONAL_PERIOD); figs.update({f"classical_{k}": v for k, v in p.items()})
    stl, strength, figs["stl"] = eda.stl_decomp(y, SEASONAL_PERIOD)

    raw_set = {"raw": y, "Δ1": y.diff(), "Δ7": y.diff(7), "Δ1Δ7": y.diff(7).diff()}
    log_set = {"log": y_log, "Δ1 log": y_log.diff(), "Δ7 log": y_log.diff(7), "Δ1Δ7 log": y_log.diff(7).diff()}
    stat = eda.stationarity_tests({**raw_set, **log_set})
    diffs = eda.suggest_differencing(y_log, SEASONAL_PERIOD)
    sig_raw, figs["acf_raw"] = eda.acf_pacf_plots(raw_set, "raw")
    sig_log, figs["acf_log"] = eda.acf_pacf_plots(log_set, "log")
    sig = pd.concat([sig_raw, sig_log], ignore_index=True)

    outliers = eda.flag_outliers(df, TARGET, stl["log"])
    breaks, scan, figs["breaks"] = eda.structural_breaks(y_log, _break_candidates(df))

    # ---- trend summary
    tr = stl["raw"].trend
    monthly = pd.DataFrame({"mean_demand": y.resample("MS").mean().round(0),
                            "stl_trend_mean": tr.resample("MS").mean().round(0),
                            "days": y.resample("MS").size()})
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.index.name = "month"
    X = eda._design(y_log.index)
    from statsmodels.regression.linear_model import OLS
    slope = OLS(y_log, X).fit().params["trend"]                 # log change per 30 days
    trend_facts = dict(start_28d=tr.iloc[:28].mean(), end_28d=tr.iloc[-28:].mean(),
                       pct_change=(tr.iloc[-28:].mean() / tr.iloc[:28].mean() - 1) * 100,
                       pct_per_month=(np.exp(slope) - 1) * 100)

    # ---- write excel
    sig_x = sig.copy()
    for c in ("acf_sig_lags", "pacf_sig_lags"):
        sig_x[c] = sig_x[c].apply(lambda v: ", ".join(map(str, v)))
    with pd.ExcelWriter(OUT_X / "eda_results.xlsx", engine="openpyxl") as w:
        stat.to_excel(w, sheet_name="stationarity_tests", index=False)
        diffs.to_excel(w, sheet_name="differencing_suggestion", index=False)
        decomp.to_excel(w, sheet_name="decomposition_comparison", index=False)
        strength.to_excel(w, sheet_name="seasonality_strength", index=False)
        peaks.to_excel(w, sheet_name="periodogram_peaks", index=False)
        weekday.to_excel(w, sheet_name="weekday_profile", index=False)
        sig_x.to_excel(w, sheet_name="acf_pacf_significant_lags", index=False)
        outliers.assign(date=outliers.date.dt.date).to_excel(w, sheet_name="outlier_dates", index=False)
        breaks.to_excel(w, sheet_name="structural_breaks", index=False)
        monthly.to_excel(w, sheet_name="monthly_trend")
    outliers.to_csv(OUT_D / "phase2_outlier_dates.csv", index=False)

    results = dict(df=df, y=y, peaks=peaks, decomp=decomp, strength=strength, stat=stat, diffs=diffs,
                   sig=sig, outliers=outliers, breaks=breaks, monthly=monthly, trend=trend_facts,
                   weekday=weekday, figs=figs)
    write_summary(results)
    # machine-readable hand-off for Phase 3
    handoff = dict(seasonal_period=SEASONAL_PERIOD, transform="log",
                   d_recommended=0, D_recommended=1,          # from ACF of Δ1 log (seasonal unit root), see summary §8
                   starting_order=[1, 0, 0], starting_seasonal_order=[0, 1, 1, SEASONAL_PERIOD],
                   pmdarima_tests={f"{q} {t}": int(v) for q, t, v in diffs.itertuples(index=False)},
                   outlier_dates={str(d.date()): sev for d, sev in zip(outliers.date, outliers.severity)})
    (OUT_D / "phase2_handoff.json").write_text(json.dumps(handoff, indent=2))
    print("Wrote", OUT_X / "eda_results.xlsx", "and", OUT_REP / "phase2_eda_summary.md")
    return results


def _md(df, floatfmt=".3g"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_summary(r):
    y, stat, decomp, strength, peaks = r["y"], r["stat"], r["decomp"], r["strength"], r["peaks"]
    t = r["trend"]
    add, mul = decomp.set_index("model").loc["additive"], decomp.set_index("model").loc["multiplicative"]
    better = "multiplicative" if mul.resid_var_tickets < add.resid_var_tickets else "additive"
    st_log = strength.set_index("scale").loc["log"]
    lvl = stat[stat.kpss_regression == "level (c)"].set_index("series")
    d_s = r["diffs"]
    lines = [
        "# Phase 2 — EDA & decomposition summary",
        "",
        f"**Series:** `total_demand` (dedup target), {len(y)} days, {y.index.min():%d %b %Y} – "
        f"{y.index.max():%d %b %Y}. The first 28 startup-censored days are excluded (confirmed Phase 1 decision). "
        f"Mean {y.mean():,.0f}/day, median {y.median():,.0f}, min {y.min():,.0f}, max {y.max():,.0f}.",
        "",
        "## 1. Trend",
        f"- STL trend goes from **{t['start_28d']:,.0f}/day** (first 28 days) to **{t['end_28d']:,.0f}/day** "
        f"(last 28 days), **{t['pct_change']:+.0f}%**. A log-linear fit with weekday dummies gives "
        f"**{t['pct_per_month']:+.1f}% per month**.",
        "- The trend isn't a straight line: it's flat from Feb to Oct 2023, then rises from Nov 2023, peaks in December, "
        "and stays above its 2023 level through Jan–Feb 2024. Monthly means:",
        "",
        _md(r["monthly"].reset_index(), ",.0f"),
        "",
        "## 2. Seasonality",
        f"- Periodogram peaks (detrended log demand): "
        + ", ".join(f"**{p:g} days** ({s:.0%} of power)" for p, s in zip(peaks.period_days, peaks.share_of_total_power))
        + ".",
        f"- STL on log demand: seasonal strength **F_s = {st_log.seasonal_strength:.2f}**, trend strength "
        f"**F_t = {st_log.trend_strength:.2f}** (Hyndman scale 0–1; F_s > 0.64 is the usual cut-off for seasonal differencing).",
        "- Day-of-week profile (demand ÷ centred 7-day mean, median): "
        + ", ".join(f"{d[:3]} {v:.2f}×" for d, v in zip(r["weekday"].day, r["weekday"].median_ratio)) + ".",
        "- **Annual seasonality can't be estimated**: only 13 months of data, so each calendar month appears about once. "
        "The December peak can't be separated into a 'yearly pattern' and a 'one-off event' from this data alone.",
        "",
        "## 3. Additive vs multiplicative",
        _md(decomp[["model", "resid_var_tickets", "resid_sd_tickets", "resid_sd_pct_of_trend",
                    "corr_abs_resid_vs_level", "corr_abs_rel_resid_vs_level"]], ",.3g"),
        "",
        f"- Residual variance in ticket units is lower for the **{better}** model "
        f"({min(add.resid_var_tickets, mul.resid_var_tickets):,.0f} vs {max(add.resid_var_tickets, mul.resid_var_tickets):,.0f}).",
        f"- Additive |residual| correlates with the trend level (r = {add.corr_abs_resid_vs_level:.2f}): the size of the "
        "weekly swing grows with demand, which points to multiplicative seasonality, i.e. modelling log demand.",
        "",
        "## 4. Stationarity (ADF + KPSS level; α = 0.05)",
        _md(lvl.reset_index()[["series", "adf_stat", "adf_p", "kpss_stat", "kpss_p", "kpss_p_note", "conclusion"]], ".3g"),
        "",
        "Trend-KPSS variant and critical values are in `eda_results.xlsx → stationarity_tests`.",
        "",
        "pmdarima differencing estimators on log demand: "
        + "; ".join(f"{q} via {tst} = **{v}**" for q, tst, v in d_s.itertuples(index=False)) + ".",
        "",
        "## 5. ACF / PACF (significant lags, 95% band, up to lag 42)",
        _md(r["sig"].assign(acf_sig_lags=r["sig"].acf_sig_lags.apply(lambda v: ", ".join(map(str, v[:15])) + (" …" if len(v) > 15 else "")),
                            pacf_sig_lags=r["sig"].pacf_sig_lags.apply(lambda v: ", ".join(map(str, v[:15])) + (" …" if len(v) > 15 else "")))),
        "",
        "## 6. Outliers (flagged, NOT removed)",
        f"Robust z-score (median/MAD) of the log-scale STL remainder: **{len(r['outliers'])} days** with |z| > 3.5, "
        f"of which **{(r['outliers'].severity == 'extreme').sum()} extreme** (|z| > 7). "
        f"Spikes: {(r['outliers'].direction == 'spike').sum()}, dips: {(r['outliers'].direction == 'dip').sum()}.",
        "",
        "- About 15% of days are flagged at any threshold or STL window setting (checked 7/13/21/35). The irregular part of "
        "the series is **heavy-tailed and event-driven**, not a handful of glitches. Spikes cluster on Thu/Fri (typical Indian "
        "film-release days) and on some Mondays; the files have no release calendar, so this can't be confirmed.",
        "- Checked as a possible data error: the 27–28 Feb 2023 spike (same date as the booknow_visits duplicates) "
        "has a normal theater count, ticket size, duplicate rate and booking-date spread, so it looks like **real demand**, not a load artefact.",
        "- 31 Dec 2023 (−73%) and 1 Jan 2024 (−41%) are sharp New Year's Eve/Day dips inside the December surge.",
        "",
        _md(r["outliers"].assign(date=r["outliers"].date.dt.date)[
            ["date", "day_name", "actual", "expected_trend_x_season", "pct_vs_expected", "robust_z", "severity", "holiday_name"]], ",.3g"),
        "",
        "## 7. Structural breaks",
        _md(r["breaks"], ".3g"),
        "",
        "- CUSUM rejects parameter stability. The biggest break is around **Nov–Dec 2023** (sup-F peaks 16 Nov and 7 Dec; Chow at 1 Dec is the largest "
        "pre-specified test). The booknow outage boundaries show smaller but significant shifts. The spring peak (early Apr 2023) is a third, weaker candidate.",
        "- The December 2023 surge is best treated as a **temporary regime/event** (demand ~2.5× for about 4 weeks, then settling ~70% above "
        "the 2023 level), not a permanent level shift. It can't be classified with certainty without a second December.",
        "",
        "## 8. Implications for Phase 3",
        f"- **Transform:** model `log(total_demand)` (multiplicative seasonality; variance grows with level).",
        f"- **Seasonal period m = 7.** Weekly pattern: Sat ≈ 2.1× and Sun ≈ 1.9× the weekly mean, Tue lowest ≈ 0.6×.",
        f"- **Differencing:** log level is non-stationary. Δ7 log and Δ1Δ7 log are stationary by both tests; Δ1 log is borderline "
        f"(KPSS p ≈ 0.047). pmdarima's test-based estimators give d = {int(d_s[d_s.quantity.str.startswith('d')].value.mode()[0])}, "
        f"D = {int(d_s[d_s.quantity.str.startswith('D')].value.max())}, **but the ACF disagrees on D**: Δ1 log keeps ACF ≈ 0.6–0.7 at lags "
        "7, 14, …, 42 with almost no decay, which is the signature of a seasonal unit root. Recommendation: **D = 1, d = 0** "
        "(Δ7 log is stationary on its own). Keep d = 1, D = 0 as the comparison model auto_arima will propose.",
        "- **ACF/PACF of Δ7 log:** ACF lag 1 ≈ 0.43 decaying; PACF cuts off after lag 1 → **non-seasonal AR(1)**. Negative ACF spike at lag 7 "
        "with PACF decaying across 7/14/21 → **seasonal MA(1)**. Starting candidate: **SARIMA(1,0,0)(0,1,1)7 on log demand**. "
        "Δ1Δ7 log shows ACF −0.25 at lag 1 and −0.37 at lag 7 (over-differencing signs), which argues against also taking d = 1.",
        "- **Exogenous needs:** holiday flag, weekday dummies, a December-2023 regime dummy (or intervention) and `booknow_missing`. "
        "Extreme outlier dates are passed to Phase 3 (`phase2_handoff.json`) as candidate pulse dummies.",
        "- **Split warning:** a chronological train/val/test split puts December 2023 in either validation or test. Whichever it lands in will "
        "dominate that window's error metrics (resolved in Phase 3: train to 31 Dec 2023, validate Jan 2024, test Feb 2024).",
        "",
        "## Figures (outputs/figures/)",
        *[f"- `{p.name}`" for p in r["figs"].values()],
        "",
    ]
    (OUT_REP / "phase2_eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_phase2()
