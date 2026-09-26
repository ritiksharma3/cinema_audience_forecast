"""Phase 4 - forecast combination.

Members (user-confirmed): SARIMAX_no_lead, SARIMA, HoltWinters_multiplicative, MLR_no_lead.
Weights are estimated on the validation window (Jan 2024: forecasts from train-fitted models) and applied to the
sealed test window (Feb 2024: forecasts from train+val refits). The designated final method (Bates-Granger) is
fixed BEFORE looking at test results. A forward forecast (29 Feb - 31 Mar 2024) uses members refit on all data and weights
re-estimated on val+test (59 days of out-of-sample errors).

Outputs
  outputs/excel/forecast_combination.xlsx
  outputs/figures/phase4_*.png
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import combination as cb
from . import models as mdl
from .calendar_features import indian_holidays
from .config import OUT_FIG, OUT_X, TARGET
from .data_access import load_daily
from .diagnostics import accuracy
from .eda import C, _thousands

warnings.filterwarnings("ignore")

MEMBERS = ["SARIMAX_no_lead", "SARIMA", "HoltWinters_multiplicative", "MLR_no_lead"]
FINAL_METHOD = "Bates-Granger (inverse MSE)"        # fixed a priori
FORWARD_END = "2024-03-31"                           # forward horizon starts the day after the last observation
PAL = dict(zip(MEMBERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
COMB_COLOR = "#4a3aa7"


def _wide(split):
    from .config import OUT_D
    f = pd.read_csv(OUT_D / f"phase3_forecasts_{split}.csv", parse_dates=["date"])
    f = f[f.model.isin(MEMBERS)]
    F = f.pivot(index="date", columns="model", values="mean")[MEMBERS]
    lo = f.pivot(index="date", columns="model", values="lower")[MEMBERS]
    hi = f.pivot(index="date", columns="model", values="upper")[MEMBERS]
    y = f.drop_duplicates("date").set_index("date").actual
    return F, lo, hi, y


def _row(name, kind, y, pred, lo=None, hi=None):
    m = accuracy(y.values, pred.values)
    cov = float(((y >= lo) & (y <= hi)).mean() * 100) if lo is not None else np.nan
    return dict(forecast=name, type=kind, test_MAE=m["MAE"], test_RMSE=m["RMSE"], test_MAPE=m["MAPE"],
                test_95pct_interval_coverage=cov)


def forward_forecast(df, weights_spec, sd):
    """Refit members on all data and forecast FORWARD; combine with the given weights."""
    # step 1 of every model's forecast is the day after the last observation (2024-02-29: leap day, not in the data)
    idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), FORWARD_END, freq="D", name="date")
    assert idx[0] - df.index[-1] == pd.Timedelta(days=1) and df.index.freq == "D"
    fut = pd.DataFrame(index=idx)
    fut[TARGET] = np.nan
    hol = indian_holidays()
    fut["is_holiday"] = fut.index.isin(hol.index).astype(int)
    fut["holiday_name"] = hol.holiday_name.reindex(idx).fillna("").values
    fut["booknow_missing"] = 0
    fut["mean_lead_hrs_combined"] = np.nan
    fits = {f.name: f for f in (fn(df, fut) for fn in mdl.MODELS if fn.__name__.replace("fit_", "") in
                                 {"sarimax_no_lead", "sarima", "hw_mul", "mlr_no_lead"})}
    F = pd.DataFrame({m: fits[m].forecast["mean"] for m in MEMBERS})
    out = cb.interval(cb.combine(F, weights_spec), sd)
    return pd.concat([F, out], axis=1), fits


def run_phase4():
    F_val, lo_val, hi_val, y_val = _wide("val")
    F_test, lo_test, hi_test, y_test = _wide("test")
    print(f"members: {MEMBERS}; val {len(y_val)} days, test {len(y_test)} days")

    # ---------------- individual members
    rows = [_row(m, "individual", y_test, F_test[m], lo_test[m], hi_test[m]) for m in MEMBERS]
    val_rmse = {m: accuracy(y_val.values, F_val[m].values)["RMSE"] for m in MEMBERS}
    val_mape = {m: accuracy(y_val.values, F_val[m].values)["MAPE"] for m in MEMBERS}
    ex_ante = min(val_rmse, key=val_rmse.get)

    # ---------------- combinations
    specs, comb_test, comb_val, sds, weight_rows = {}, {}, {}, {}, []
    for name, fn in cb.METHODS.items():
        sp = fn(F_val, y_val)
        specs[name] = sp
        comb_val[name] = cb.combine(F_val, sp)
        comb_test[name] = cb.combine(F_test, sp)
        sds[name] = cb.log_error_sd(comb_val[name], y_val)
        ci = cb.interval(comb_test[name], sds[name])
        rows.append(_row(name, "combination", y_test, comb_test[name], ci.lower, ci.upper))
        weight_rows.append(dict(method=name, intercept=sp["intercept"], **sp["weights"].to_dict(),
                                sum_of_weights=float(sp["weights"].sum()), log_error_sd_val=sds[name]))
    comp = pd.DataFrame(rows)
    comp["val_RMSE"] = comp.forecast.map({**val_rmse, **{k: accuracy(y_val.values, v.values)["RMSE"] for k, v in comb_val.items()}})
    comp["val_MAPE"] = comp.forecast.map({**val_mape, **{k: accuracy(y_val.values, v.values)["MAPE"] for k, v in comb_val.items()}})
    comp["val_note"] = np.where(comp.type == "combination", "in-sample for the weights (optimistic)", "out-of-sample")
    ex_post = comp[comp.type == "individual"].sort_values("test_RMSE").forecast.iloc[0]
    ex_post_mape = comp[comp.type == "individual"].sort_values("test_MAPE").forecast.iloc[0]
    for k in ("RMSE", "MAPE", "MAE"):
        comp[f"rank_test_{k}"] = comp[f"test_{k}"].rank(method="min").astype(int)
    best = comp.set_index("forecast")
    comp["beats_ex_ante_best_RMSE"] = comp.test_RMSE < best.loc[ex_ante, "test_RMSE"]
    comp["beats_ex_post_best_RMSE"] = comp.test_RMSE < best.loc[ex_post, "test_RMSE"]
    comp["beats_best_single_MAPE"] = comp.test_MAPE < best.loc[ex_post_mape, "test_MAPE"]
    comp["is_final_method"] = comp.forecast == FINAL_METHOD
    comp = comp.sort_values("test_RMSE").reset_index(drop=True)

    # ---------------- Diebold-Mariano: each combination vs single-model benchmarks
    dm = []
    for name in cb.METHODS:
        for bench, lab in ((ex_ante, "ex-ante best single (lowest val RMSE)"), (ex_post, "ex-post best single (lowest test RMSE)")):
            s, p = cb.diebold_mariano(y_test - comb_test[name], y_test - F_test[bench])
            dm.append(dict(combination=name, benchmark=bench, benchmark_role=lab, DM_HLN_stat=s, p_value=p,
                           verdict=("combination significantly better" if (s < 0 and p < 0.05) else
                                    "benchmark significantly better" if (s > 0 and p < 0.05) else
                                    "no significant difference (5%)")))
    dm = pd.DataFrame(dm)

    # ---------------- final test forecast table
    fin = cb.interval(comb_test[FINAL_METHOD], sds[FINAL_METHOD])
    final_test = pd.concat([y_test.rename("actual"), F_test.add_prefix("fc_"),
                            pd.DataFrame(comb_test).add_prefix("comb_"),
                            fin.add_prefix("final_")], axis=1).rename_axis("date").reset_index()

    # ---------------- forward forecast (Mar 2024)
    F_all = pd.concat([F_val, F_test]); y_all = pd.concat([y_val, y_test])
    fwd_spec = cb.METHODS[FINAL_METHOD](F_all, y_all)
    fwd_sd = cb.log_error_sd(cb.combine(F_all, fwd_spec), y_all)
    df = load_daily(model_window=True)
    fwd, fwd_fits = forward_forecast(df, fwd_spec, fwd_sd)
    fwd = fwd.rename(columns={m: f"fc_{m}" for m in MEMBERS}).rename(columns={"mean": "combined_mean", "lower": "combined_lower_95",
                                                                             "upper": "combined_upper_95"})
    fwd.insert(0, "day_name", fwd.index.day_name())
    fwd.insert(1, "holiday", indian_holidays().holiday_name.reindex(fwd.index).fillna("").values)

    _write_excel(comp, dm, specs, weight_rows, final_test, fwd, fwd_spec, fwd_sd, ex_ante, ex_post)
    _plots(df, y_test, F_test, comb_test, fin, comp, fwd)
    print(comp[["forecast", "test_RMSE", "test_MAPE", "test_MAE", "test_95pct_interval_coverage"]].round(1).to_string(index=False))
    return dict(comp=comp, dm=dm, specs=specs, fwd=fwd, ex_ante=ex_ante, ex_post=ex_post, fwd_spec=fwd_spec)


def _write_excel(comp, dm, specs, weight_rows, final_test, fwd, fwd_spec, fwd_sd, ex_ante, ex_post):
    with pd.ExcelWriter(OUT_X / "forecast_combination.xlsx", engine="openpyxl") as w:
        pd.DataFrame(dict(
            item=["members (user-confirmed)", "weight estimation window", "evaluation window", "final method (fixed a priori)",
                  "ex-ante best single model", "ex-post best single model", "combined interval method"],
            value=[", ".join(MEMBERS), "validation Jan 2024 (31 days, forecasts from train-fitted models)",
                   "test Feb 2024 (28 days, forecasts from train+val refits)", FINAL_METHOD,
                   f"{ex_ante} (lowest validation RMSE)", f"{ex_post} (lowest test RMSE - known only afterwards)",
                   "combined × exp(±1.96·sd), sd = SD of log(actual/combined) on the weight window"])).to_excel(w, sheet_name="setup", index=False)
        comp.to_excel(w, sheet_name="test_comparison", index=False)
        dm.to_excel(w, sheet_name="diebold_mariano", index=False)
        pd.DataFrame(weight_rows).to_excel(w, sheet_name="weights_all_methods", index=False)
        specs["Bates-Granger (inverse MSE)"]["table"].to_excel(w, sheet_name="bates_granger", index=False)
        gu = specs["Granger-Ramanathan (unconstrained, intercept)"]
        gu["table"].to_excel(w, sheet_name="GR_unconstrained", index=False)
        pd.DataFrame([gu["fit"]]).to_excel(w, sheet_name="GR_unconstrained", index=False, startrow=len(gu["table"]) + 3)
        specs["Granger-Ramanathan (sum-to-one, no intercept)"]["table"].to_excel(w, sheet_name="GR_sum_to_one", index=False)
        final_test.to_excel(w, sheet_name="final_test_forecast", index=False)
        f = fwd.reset_index(); f["date"] = f.date.dt.date
        f.to_excel(w, sheet_name="forward_forecast", index=False)
        pd.DataFrame({"model": fwd_spec["weights"].index, "weight": fwd_spec["weights"].values}).assign(
            log_error_sd=fwd_sd, note="Bates-Granger weights re-estimated on val+test (59 out-of-sample days)").to_excel(
            w, sheet_name="forward_forecast", index=False, startcol=f.shape[1] + 2)


def _plots(df, y_test, F_test, comb_test, fin, comp, fwd):
    # 1. overlay on the test window
    fig, ax = plt.subplots(figsize=(12, 5))
    ctx = df.loc["2024-01-01":"2024-01-31", TARGET]
    ax.plot(ctx.index, ctx, color=C["muted"], lw=1, label="actual (validation, context)")
    for m in MEMBERS:
        ax.plot(F_test.index, F_test[m], color=PAL[m], lw=1.1, alpha=0.8, label=m)
    ax.fill_between(fin.index, fin.lower, fin.upper, color=COMB_COLOR, alpha=0.12, lw=0, label="95% interval (final)")
    ax.plot(comb_test["Simple average"].index, comb_test["Simple average"], color=COMB_COLOR, lw=1.4, ls="--",
            label="combined: simple average")
    ax.plot(fin.index, fin["mean"], color=COMB_COLOR, lw=2.6, label=f"combined: {FINAL_METHOD} (final)")
    ax.plot(y_test.index, y_test, color=C["ink"], lw=2, marker="o", ms=3, label="actual (test)")
    ax.axvline(pd.Timestamp("2024-02-01"), color=C["ink2"], lw=1, ls=":")
    _thousands(ax)
    ax.set_title("Test window (Feb 2024): actual vs individual members vs combined forecast")
    ax.legend(loc="upper left", ncol=3, fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT_FIG / "phase4_01_test_overlay.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # 2. error comparison bars
    c = comp.sort_values("test_RMSE", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, col, lab in ((axes[0], "test_RMSE", "RMSE (tickets)"), (axes[1], "test_MAPE", "MAPE (%)")):
        colors = [COMB_COLOR if t == "combination" else C["s1"] for t in c.type]
        ax.barh(c.forecast, c[col], color=colors, height=0.6)
        for yv, v in enumerate(c[col]):
            ax.text(v, yv, f" {v:,.0f}" if col == "test_RMSE" else f" {v:.1f}%", va="center", fontsize=8, color=C["ink2"])
        best_single = comp[comp.type == "individual"][col].min()
        ax.axvline(best_single, color=C["ink2"], ls="--", lw=1)
        ax.set_title(f"Test {lab} — dashed = best single model", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].text(0.99, 0.02, "blue = individual   violet = combination", transform=axes[0].transAxes, ha="right", fontsize=8, color=C["ink2"])
    fig.tight_layout(); fig.savefig(OUT_FIG / "phase4_02_test_errors.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # 3. forward forecast
    fig, ax = plt.subplots(figsize=(12, 4.5))
    h = df.loc["2024-01-01":, TARGET]
    ax.plot(h.index, h, color=C["ink"], lw=1.4, label="actual")
    ax.fill_between(fwd.index, fwd.combined_lower_95, fwd.combined_upper_95, color=COMB_COLOR, alpha=0.15, lw=0, label="95% interval")
    ax.plot(fwd.index, fwd.combined_mean, color=COMB_COLOR, lw=2.4, label="combined forecast (Bates-Granger)")
    for d, r in fwd[fwd.holiday != ""].iterrows():
        ax.annotate(r.holiday, (d, r.combined_upper_95), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=C["ink2"])
    _thousands(ax)
    ax.set_title(f"Forward forecast, {fwd.index[0]:%d %b} – {fwd.index[-1]:%d %b %Y} (members refit on all 396 days)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(OUT_FIG / "phase4_03_forward_forecast.png", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    run_phase4()
