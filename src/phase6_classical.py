"""Phase 6 - classical case-study view of the same series (mirrors the L&T spare-parts teaching case).

Adds the textbook steps the main pipeline does not report on their own:
  1. seasonal index (simple-average method; weekday is the season, m = 7)
  2. one-way ANOVA of demand across weekdays, plus the two-way (weekday + month) version that removes the level
  3. trend + seasonal-dummy regression: full model and reduced model (backward elimination, p < 0.05), with ANOVA table
  4. SPSS-style fit statistics for every Phase 3 model on the training window: R-squared, stationary R-squared,
     RMSE, MAPE, MaxAPE, MAE, MaxAE, normalised BIC, Ljung-Box Q(18)
  5. ARIMA candidate orders considered (auto_arima vs the Phase 2 identified SARIMA)
Everything is estimated on train (29 Jan - 31 Dec 2023) and scored on validation (Jan 2024), as in Phase 3.
Outputs: outputs/excel/classical_analysis.xlsx, outputs/figures/phase6_*.png
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from . import models as mdl
from .config import OUT_X, TARGET
from .data_access import load_daily
from .diagnostics import accuracy
from .eda import C as PAL, _fmt_dates, _save, _thousands
from .phase3_models import split

warnings.filterwarnings("ignore")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------------------------------------------------ 1. seasonal index
def seasonal_index(train):
    """Weekday x month table of mean demand; index = weekday average / grand average (simple-average method).
    Also the ratio-to-centred-moving-average index, which removes the trend before averaging."""
    d = train.loc["2023-02-01":, [TARGET]].copy()           # full months only
    d["weekday"] = pd.Categorical(d.index.dayofweek.map(dict(enumerate(DAYS))), DAYS, ordered=True)
    d["month"] = d.index.strftime("%b")
    tab = d.pivot_table(index="weekday", columns="month", values=TARGET, aggfunc="mean", observed=False)
    tab = tab[pd.unique(d.month)]
    tab["Average"] = tab.mean(axis=1)
    tab["Index_simple_avg"] = tab.Average / tab.Average.mean() * 100
    no_dec = tab.drop(columns=["Dec", "Average", "Index_simple_avg"]).mean(axis=1)
    tab["Index_excl_Dec"] = no_dec / no_dec.mean() * 100
    cma = train[TARGET].rolling(7, center=True).mean()
    ratio = (train[TARGET] / cma).dropna()
    r = ratio.groupby(ratio.index.dayofweek).mean()
    tab["Index_ratio_to_MA"] = (r / r.mean() * 100).values
    return tab


# ------------------------------------------------------------------ 2. ANOVA
def _anova_rows(model, label):
    a = sm.stats.anova_lm(model, typ=2)
    out = []
    for src, row in a.iterrows():
        ms = row.sum_sq / row.df
        fcrit = stats.f.ppf(0.95, row.df, a.loc["Residual", "df"]) if src != "Residual" else np.nan
        out.append(dict(analysis=label, source=src.replace("C(weekday)", "Between weekdays").replace("C(month)", "Between months")
                        .replace("Residual", "Within (residual)"),
                        SS=row.sum_sq, df=int(row.df), MS=ms, F=row.F, p_value=row["PR(>F)"], F_crit_5pct=fcrit))
    out.append(dict(analysis=label, source="Total", SS=a.sum_sq.sum(), df=int(a.df.sum()), MS=np.nan, F=np.nan,
                    p_value=np.nan, F_crit_5pct=np.nan))
    return out


def weekday_anova(train):
    d = pd.DataFrame({"y": train[TARGET].astype(float), "ly": np.log(train[TARGET].astype(float)),
                      "weekday": train.index.dayofweek, "month": train.index.strftime("%Y-%m")})
    rows = _anova_rows(smf.ols("y ~ C(weekday)", d).fit(), "one-way, tickets")
    rows += _anova_rows(smf.ols("ly ~ C(weekday)", d).fit(), "one-way, log tickets")
    rows += _anova_rows(smf.ols("ly ~ C(weekday) + C(month)", d).fit(), "two-way (weekday + month), log tickets")
    kw = stats.kruskal(*[g.values for _, g in d.groupby("weekday").y])
    rows.append(dict(analysis="Kruskal-Wallis (rank-based, robust to outliers)", source="Between weekdays",
                     SS=np.nan, df=6, MS=np.nan, F=kw.statistic, p_value=kw.pvalue, F_crit_5pct=stats.chi2.ppf(0.95, 6)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. trend + dummy regression
def _backward(hist, fut, alpha=0.05):
    keep = list(mdl.trend_dummy_design(hist.index, hist.index[0]).columns)
    while True:
        f = mdl.fit_trend_dummy_reg(hist, fut, keep=keep)
        p = f.extra["ols"].pvalues.drop("const")
        if p.max() < alpha:
            return f
        keep.remove(p.idxmax())


def _reg_tables(f, label):
    r = f.extra["ols"]
    coef = pd.DataFrame({"model": label, "term": r.params.index, "coefficient": r.params.values, "std_error": r.bse.values,
                         "t_or_z": r.tvalues.values, "p_value": r.pvalues.values, "multiplier_exp_coef": np.exp(r.params.values)})
    ssr, sse = r.ess, r.ssr
    F = (ssr / r.df_model) / (sse / r.df_resid)           # classical F (r.fvalue is the HAC Wald test)
    anova = pd.DataFrame([dict(model=label, source="Regression", df=r.df_model, SS=ssr, MS=ssr / r.df_model, F=F,
                               significance_F=stats.f.sf(F, r.df_model, r.df_resid)),
                          dict(model=label, source="Residual", df=r.df_resid, SS=sse, MS=sse / r.df_resid),
                          dict(model=label, source="Total", df=r.df_model + r.df_resid, SS=ssr + sse)])
    stats_ = pd.DataFrame([dict(model=label, multiple_R=np.sqrt(r.rsquared), R_square=r.rsquared, adj_R_square=r.rsquared_adj,
                                std_error_log=np.sqrt(r.mse_resid), observations=int(r.nobs))])
    return coef, anova, stats_


def trend_dummy_regression(s):
    full = mdl.fit_trend_dummy_reg(s["train"], s["val"])
    red = _backward(s["train"], s["val"])
    parts = [_reg_tables(full, "full"), _reg_tables(red, "reduced (p<0.05)")]
    coef, anova, st = (pd.concat(x, ignore_index=True) for x in zip(*parts))
    for f, lab in ((full, "full"), (red, "reduced (p<0.05)")):
        m = accuracy(s["val"][TARGET].values, f.forecast["mean"].values)
        st.loc[st.model == lab, ["val_RMSE", "val_MAPE", "val_MAE"]] = [m["RMSE"], m["MAPE"], m["MAE"]]
    return full, red, coef, anova, st


def plot_regression_fit(df, s, full):
    y_adj, _ = mdl.univariate_target(s["train"])
    fitted = np.exp(np.log(y_adj) - full.resid) * (s["train"][TARGET] / y_adj)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[TARGET], color=PAL["muted"], lw=0.9, label="actual demand")
    ax.plot(fitted.index, fitted, color=PAL["s1"], lw=1.3, label="fitted (train)")
    ax.plot(full.forecast.index, full.forecast["mean"], color=PAL["s2"], lw=1.6, label="forecast (validation, Jan 2024)")
    ax.set_yscale("log"); _thousands(ax)
    ax.set_title("Trend + weekday-dummy regression: fit and validation forecast (log axis)")
    ax.legend(loc="upper left")
    _fmt_dates(ax)
    return _save(fig, "phase6_02_trend_dummy_fit.png")


def plot_seasonal_index(si):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(7)
    for off, col, lab, c in ((-0.27, "Index_simple_avg", "simple average (all months)", PAL["muted"]),
                             (0, "Index_excl_Dec", "simple average, excl. December", PAL["s1"]),
                             (0.27, "Index_ratio_to_MA", "ratio to centred 7-day MA", PAL["s2"])):
        ax.bar(x + off, si[col], width=0.26, color=c, label=lab)
    ax.axhline(100, color=PAL["ink2"], lw=0.8, ls="--")
    ax.set_xticks(x, DAYS)
    ax.set_ylabel("seasonal index (%)")
    ax.set_title("Weekday seasonal index (train, Feb-Dec 2023)")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, "phase6_01_seasonal_index.png")


# ------------------------------------------------------------------ 4. SPSS-style fit statistics
def _fitted_tickets(fn, f, train):
    """In-sample fitted values on the ticket scale, aligned with the real (unadjusted) training demand."""
    y = train[TARGET].astype(float)
    if fn in mdl.UNIVARIATE:
        y_fit, _ = mdl.univariate_target(train)
        back = y / y_fit                                     # Dec-surge multiplier (1 outside December)
    else:
        y_fit, back = y, pd.Series(1.0, index=y.index)
    idx = f.resid.index
    if f.scale == "log":
        fit = np.exp(np.log(y_fit.loc[idx]) - f.resid)
    elif "error=M" in f.spec:                                # multiplicative-error ETS: resid = (y - fit) / fit
        fit = y_fit.loc[idx] / (1 + f.resid)
    else:
        fit = y_fit.loc[idx] - f.resid
    return y.loc[idx], fit * back.loc[idx]


def fit_statistics(s):
    rows, fits = [], {}
    y_all = s["train"][TARGET].astype(float)
    d7 = (y_all - y_all.shift(7)).dropna()
    for fn in mdl.MODELS:
        f = fn(s["train"], s["val"])
        fits[f.name] = f
        act, fit = _fitted_tickets(fn, f, s["train"])
        e = act - fit
        n, k = len(e), len(f.params)
        mse = float((e ** 2).mean())
        ape = (e.abs() / act * 100)
        d7i = d7.reindex(e.index).dropna()
        seas_naive_sse = float(((d7i - d7i.mean()) ** 2).sum())
        lb = acorr_ljungbox(f.resid.dropna(), lags=[18], model_df=f.n_arma_params)
        v = accuracy(s["val"][TARGET].values, f.forecast["mean"].values)
        rows.append(dict(model=f.name, n_predictors=k,
                         stationary_R2=1 - float((e.reindex(d7i.index) ** 2).sum()) / seas_naive_sse,
                         R2=1 - float((e ** 2).sum()) / float(((act - act.mean()) ** 2).sum()),
                         RMSE=np.sqrt(mse), MAPE=float(ape.mean()), MaxAPE=float(ape.max()),
                         MAE=float(e.abs().mean()), MaxAE=float(e.abs().max()),
                         normalized_BIC=np.log(mse) + k * np.log(n) / n,
                         LjungBox_Q18=float(lb.lb_stat.iloc[0]), LB_df=18 - f.n_arma_params, LB_sig=float(lb.lb_pvalue.iloc[0]),
                         val_RMSE=v["RMSE"], val_MAPE=v["MAPE"], val_MAE=v["MAE"], specification=f.spec))
        print(f"  {f.name:28s} in-sample MAPE {ape.mean():5.1f}%  val MAPE {v['MAPE']:5.1f}%")
    return pd.DataFrame(rows).sort_values("val_MAPE").reset_index(drop=True), fits


def arima_candidates(fits):
    out = []
    for name in ("ARIMA", "SARIMA", "SARIMAX_no_lead"):
        sel = fits[name].extra.get("selection")
        if sel is not None:
            t = sel.head(5).copy().astype({c: str for c in sel.columns if sel[c].dtype == object})
            t.insert(0, "model_family", name)
            t.insert(1, "rank_by_AIC", range(1, len(t) + 1))
            out.append(t)
    return pd.concat(out, ignore_index=True)


def run_phase6():
    df = load_daily(model_window=True)
    s = split(df)
    si = seasonal_index(s["train"])
    an = weekday_anova(s["train"])
    full, red, coef, anova, st = trend_dummy_regression(s)
    plot_seasonal_index(si)
    plot_regression_fit(df.loc[:"2024-01-31"], s, full)
    fs, fits = fit_statistics(s)
    cand = arima_candidates(fits)
    path = OUT_X / "classical_analysis.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        si.round(1).to_excel(w, sheet_name="seasonal_index")
        an.to_excel(w, sheet_name="anova_weekday", index=False)
        coef.to_excel(w, sheet_name="trend_dummy_coefficients", index=False)
        anova.to_excel(w, sheet_name="trend_dummy_anova", index=False)
        st.to_excel(w, sheet_name="trend_dummy_fit", index=False)
        fs.to_excel(w, sheet_name="fit_statistics_all_models", index=False)
        cand.to_excel(w, sheet_name="arima_candidates", index=False)
    print("Wrote", path)
    return si, an, coef, anova, st, fs, cand


if __name__ == "__main__":
    run_phase6()
