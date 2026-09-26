"""Phase 2 - EDA, decomposition, stationarity, outliers and structural breaks (reusable functions)."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .config import OUT_FIG

# reference palette (dataviz skill, light mode)
C = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", ink="#0b0b0b", ink2="#52514e",
         muted="#8a8984", grid="#e6e5e0", surface="#fcfcfb", band="#efeee9")

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 8,
    "font.size": 9,
})


def _save(fig, name):
    path = OUT_FIG / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def _thousands(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))


def _shade_outage(ax, df):
    m = df.booknow_missing.astype(bool)
    runs = (m != m.shift()).cumsum()[m]
    for i, (_, g) in enumerate(runs.groupby(runs)):
        ax.axvspan(g.index[0], g.index[-1] + pd.Timedelta(days=1), color=C["band"], zorder=0,
                   label="booknow outage (booknow=0)" if i == 0 else None)


# ------------------------------------------------------------------ 1. series plot
def plot_series(df, target):
    y = df[target]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for ax, log in zip(axes, (False, True)):
        _shade_outage(ax, df)
        ax.plot(y.index, y, color=C["s1"], lw=1.0, alpha=0.55, label="daily total_demand")
        ax.plot(y.index, y.rolling(7, center=True).mean(), color=C["s1"], lw=2, label="7-day centred mean")
        ax.plot(y.index, y.rolling(28, center=True).mean(), color=C["s2"], lw=2, label="28-day centred mean")
        maj = df[df.is_major_festival == 1]
        ax.scatter(maj.index, y[maj.index], s=36, color=C["ink"], zorder=5, marker="D",
                   label="major festival" if not log else None)
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("tickets (log scale)")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))
        else:
            _thousands(ax)
            ax.set_ylabel("tickets / day")
            for d, r in maj.iterrows():
                ax.annotate(r.holiday_name, (d, y[d]), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=7, color=C["ink2"])
    axes[0].set_title(f"Daily total demand, {y.index.min():%d %b %Y} – {y.index.max():%d %b %Y} "
                      f"(n={len(y)}, startup-censored days excluded)")
    axes[1].set_title("Same series on a log scale — weekly swing is roughly proportional to level")
    axes[0].legend(loc="upper left", ncol=3)
    _fmt_dates(axes[1])
    fig.tight_layout()
    return _save(fig, "phase2_01_series.png")


def plot_weekday_profile(df, target):
    y = np.log(df[target])
    rel = y - y.rolling(7, center=True).mean()           # deviation from local weekly level
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data = [np.exp(rel[df.day_name == d].dropna()) for d in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, labels=[d[:3] for d in order], patch_artist=True, widths=0.55,
                    medianprops=dict(color=C["ink"], lw=1.6), flierprops=dict(marker="o", ms=3, mec=C["muted"]))
    for b in bp["boxes"]:
        b.set(facecolor=C["s1"], alpha=0.35, edgecolor=C["s1"])
    ax.axhline(1, color=C["muted"], lw=1, ls="--")
    ax.set_ylabel("demand ÷ local 7-day mean")
    ax.set_title("Day-of-week profile (ratio to centred 7-day mean)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    fig.tight_layout()
    prof = pd.DataFrame({"day": order, "median_ratio": [float(np.median(x)) for x in data],
                         "iqr_low": [float(np.percentile(x, 25)) for x in data],
                         "iqr_high": [float(np.percentile(x, 75)) for x in data]})
    return _save(fig, "phase2_07_weekday_profile.png"), prof


# ------------------------------------------------------------------ 2. periodogram
def detect_periods(y, top=5, max_period=120):
    """Periodogram of the linearly detrended log series; returns strongest periods (days)."""
    f, p = periodogram(np.log(y.values), detrend="linear", scaling="spectrum")
    keep = (f > 0) & (1 / np.where(f > 0, f, np.nan) <= max_period)
    pk = pd.DataFrame({"period_days": 1 / f[keep], "power": p[keep]})
    # local maxima only, so neighbouring bins of one peak are not reported twice
    pw = pk.power.values
    is_peak = np.r_[pw[0] > pw[1], (pw[1:-1] > pw[:-2]) & (pw[1:-1] > pw[2:]), pw[-1] > pw[-2]]
    pk = pk[is_peak].sort_values("power", ascending=False).head(top).reset_index(drop=True)
    pk["share_of_total_power"] = pk.power / p[f > 0].sum()
    pk["period_days"] = pk.period_days.round(2)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    per = 1 / f[keep]
    ax.plot(per, p[keep], color=C["s1"], lw=1.4)
    ax.set_xscale("log")
    ax.set_xticks([2, 3.5, 7, 14, 30, 60, 120])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for _, r in pk.head(3).iterrows():
        ax.annotate(f"{r.period_days:g} d", (r.period_days, r.power), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=C["ink"])
    ax.set_xlabel("period (days, log scale)")
    ax.set_ylabel("spectral power")
    ax.set_title("Periodogram of detrended log demand")
    fig.tight_layout()
    return pk, _save(fig, "phase2_08_periodogram.png")


# ------------------------------------------------------------------ 3. classical decomposition
def classical_decomp(y, period=7):
    res, paths = {}, {}
    for model in ("additive", "multiplicative"):
        dec = seasonal_decompose(y, model=model, period=period)
        fitted = dec.trend + dec.seasonal if model == "additive" else dec.trend * dec.seasonal
        resid_tickets = (y - fitted).dropna()
        rel = (resid_tickets / dec.trend.dropna()).dropna()
        # heteroskedasticity: does |residual| grow with the trend level?
        lvl = dec.trend.reindex(resid_tickets.index)
        res[model] = dict(
            model=model,
            resid_var_tickets=resid_tickets.var(),
            resid_sd_tickets=resid_tickets.std(),
            resid_sd_pct_of_trend=rel.std() * 100,
            resid_var_share_of_series=resid_tickets.var() / y.loc[resid_tickets.index].var(),
            corr_abs_resid_vs_level=np.corrcoef(resid_tickets.abs(), lvl)[0, 1],
            corr_abs_rel_resid_vs_level=np.corrcoef(rel.abs(), lvl.reindex(rel.index))[0, 1],
        )
        fig = dec.plot()
        fig.set_size_inches(12, 8)
        for ax in fig.axes:
            for ln in ax.get_lines():
                ln.set_color(C["s1"]); ln.set_linewidth(1.1); ln.set_markersize(2)
            _fmt_dates(ax)
        fig.axes[0].set_title(f"Classical {model} decomposition (period={period})")
        fig.tight_layout()
        paths[model] = _save(fig, f"phase2_0{2 if model == 'additive' else 3}_classical_{model[:4]}.png")
    tab = pd.DataFrame(res.values())
    tab["lower_resid_var"] = tab.resid_var_tickets == tab.resid_var_tickets.min()
    return tab, paths


# ------------------------------------------------------------------ 4. STL
def _strengths(r):
    ft = max(0.0, 1 - np.var(r.resid) / np.var(r.trend + r.resid))
    fs = max(0.0, 1 - np.var(r.resid) / np.var(r.seasonal + r.resid))
    return ft, fs


def stl_decomp(y, period=7, seasonal=13):
    """Robust STL on raw and log scale. seasonal=13 (vs minimum 7) keeps the weekly pattern stable enough
    that one event day does not bend it and create artificial dips on neighbouring days."""
    out, rows = {}, []
    for scale, series in (("raw", y), ("log", np.log(y))):
        r = STL(series, period=period, seasonal=seasonal, robust=True).fit()
        out[scale] = r
        ft, fs = _strengths(r)
        rows.append(dict(scale=scale, period=period, stl_seasonal_window=seasonal, trend_strength=ft, seasonal_strength=fs,
                         resid_sd=r.resid.std(),
                         seasonal_amplitude=r.seasonal.max() - r.seasonal.min()))
    r = out["raw"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, comp, lab in zip(axes, (y, r.trend, r.seasonal, r.resid),
                             ("observed", "trend", "seasonal (weekly)", "remainder")):
        if lab == "remainder":
            ax.vlines(comp.index, 0, comp, color=C["s1"], lw=0.9)
            ax.axhline(0, color=C["muted"], lw=0.8)
        else:
            ax.plot(comp.index, comp, color=C["s1"], lw=1.1)
        ax.set_ylabel(lab)
        _thousands(ax)
    axes[0].set_title(f"STL decomposition (robust, period={period}, seasonal window={seasonal}) — raw scale")
    _fmt_dates(axes[-1])
    fig.tight_layout()
    return out, pd.DataFrame(rows), _save(fig, "phase2_04_stl.png")


# ------------------------------------------------------------------ 5. stationarity
def _conclude(adf_p, kpss_p, alpha=0.05):
    adf_st, kpss_st = adf_p < alpha, kpss_p > alpha
    if adf_st and kpss_st:
        return "stationary (both tests agree)"
    if not adf_st and not kpss_st:
        return "non-stationary (both tests agree)"
    if adf_st and not kpss_st:
        return "conflict: ADF rejects unit root, KPSS rejects stationarity (breaks/heteroskedasticity or near-integrated)"
    return "conflict: neither test rejects (low power) - inconclusive"


def stationarity_tests(series):
    rows = []
    for name, s in series.items():
        s = s.dropna()
        adf_stat, adf_p, adf_lag, _, adf_cv, _ = adfuller(s, autolag="AIC")
        for reg in ("c", "ct"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                k_stat, k_p, k_lag, _ = kpss(s, regression=reg, nlags="auto")
            bounded = any("InterpolationWarning" in str(x.category) for x in w)
            rows.append(dict(
                series=name, n=len(s), kpss_regression="level (c)" if reg == "c" else "trend (ct)",
                adf_stat=adf_stat, adf_p=adf_p, adf_lags=adf_lag, adf_cv_5pct=adf_cv["5%"],
                kpss_stat=k_stat, kpss_p=k_p,
                kpss_p_note=("p is outside table range: " + ("<=0.01" if k_p <= 0.01 else ">=0.10")) if bounded else "",
                kpss_lags=k_lag, conclusion=_conclude(adf_p, k_p)))
    return pd.DataFrame(rows)


def suggest_differencing(y_log, m=7):
    """Cross-check d and D with pmdarima's test-based estimators on log demand."""
    from pmdarima.arima import ndiffs, nsdiffs
    return pd.DataFrame([
        dict(quantity="d (non-seasonal)", test="KPSS", value=ndiffs(y_log, test="kpss", max_d=2)),
        dict(quantity="d (non-seasonal)", test="ADF", value=ndiffs(y_log, test="adf", max_d=2)),
        dict(quantity="d (non-seasonal)", test="PP", value=ndiffs(y_log, test="pp", max_d=2)),
        dict(quantity=f"D (seasonal, m={m})", test="OCSB", value=nsdiffs(y_log, m=m, test="ocsb", max_D=1)),
        dict(quantity=f"D (seasonal, m={m})", test="Canova-Hansen", value=nsdiffs(y_log, m=m, test="ch", max_D=1)),
    ])


# ------------------------------------------------------------------ 6. ACF / PACF
def _sig_lags(x, fn, nlags):
    vals, ci = fn(x, nlags=nlags, alpha=0.05)
    half = ci[:, 1] - vals
    return [int(k) for k in range(1, nlags + 1) if abs(vals[k]) > half[k]]


def acf_pacf_plots(series, name, lags=42):
    fig, axes = plt.subplots(len(series), 2, figsize=(12, 2.4 * len(series)))
    rows = []
    n_ref = len(next(iter(series.values())))
    for i, (lab, s) in enumerate(series.items()):
        s = s.dropna()
        band = 1.96 / np.sqrt(len(s))
        for j, (fn, title) in enumerate(((acf, "ACF"), (pacf, "PACF"))):
            v = fn(s, nlags=lags) if fn is acf else fn(s, nlags=lags, method="ywm")
            ax = axes[i, j]
            k = np.arange(1, lags + 1)
            weekly = (k % 7 == 0)
            ax.vlines(k[~weekly], 0, v[1:][~weekly], color=C["s1"], lw=1.6)
            ax.vlines(k[weekly], 0, v[1:][weekly], color=C["s2"], lw=2.2)
            ax.axhspan(-band, band, color=C["band"], zorder=0)
            ax.axhline(0, color=C["muted"], lw=0.8)
            ax.set_xlim(0, lags + 1)
            ax.set_xticks(range(0, lags + 1, 7))
            ax.set_ylim(-1, 1)
            ax.set_title(f"{title} — {lab}", fontsize=9)
        rows.append(dict(series=lab, n=len(s),
                         acf_sig_lags=_sig_lags(s, acf, lags),
                         pacf_sig_lags=_sig_lags(s, lambda x, nlags, alpha: pacf(x, nlags=nlags, alpha=alpha, method="ywm"), lags)))
    fig.suptitle(f"ACF / PACF ({name}); orange = weekly lags 7, 14, 21…; shaded = ±1.96/√n",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return pd.DataFrame(rows), _save(fig, f"phase2_05_acf_pacf_{name}.png")


# ------------------------------------------------------------------ 7. outliers
def flag_outliers(df, target, stl_log, z_thresh=3.5, z_extreme=7.0):
    """Robust z of the log-scale STL remainder (median/MAD). Flags only - nothing is removed."""
    r = stl_log.resid
    mad = np.median(np.abs(r - np.median(r)))
    z = 0.6745 * (r - np.median(r)) / mad
    expected = np.exp(stl_log.trend + stl_log.seasonal)
    out = pd.DataFrame({
        "date": r.index, "day_name": df.day_name.values, "actual": df[target].values,
        "expected_trend_x_season": expected.round(0).values,
        "pct_vs_expected": ((df[target].values / expected.values) - 1) * 100,
        "robust_z": z.values, "holiday_name": df.holiday_name.values,
        "booknow_missing": df.booknow_missing.values})
    out = out[np.abs(out.robust_z) > z_thresh].sort_values("date").reset_index(drop=True)
    out["direction"] = np.where(out.robust_z > 0, "spike", "dip")
    out["severity"] = np.where(out.robust_z.abs() > z_extreme, "extreme", "moderate")
    return out


# ------------------------------------------------------------------ 8. structural breaks
def _design(idx):
    X = pd.get_dummies(idx.dayofweek, prefix="dow", drop_first=True).astype(float)
    X.index = idx
    X.insert(0, "trend", np.arange(len(idx)) / 30.0)
    X.insert(0, "const", 1.0)
    return X


def _chow(y, X, i):
    k = X.shape[1]
    ssr = OLS(y, X).fit().ssr
    s1 = OLS(y.iloc[:i], X.iloc[:i]).fit().ssr
    s2 = OLS(y.iloc[i:], X.iloc[i:]).fit().ssr
    f = ((ssr - s1 - s2) / k) / ((s1 + s2) / (len(y) - 2 * k))
    return f


def structural_breaks(y_log, candidates, trim=0.15):
    """OLS-CUSUM + sup-F (Quandt-Andrews style) scan + Chow tests at pre-specified dates.
    Model: log demand ~ const + linear trend + day-of-week dummies."""
    from scipy.stats import f as fdist
    X = _design(y_log.index)
    fit = OLS(y_log, X).fit()
    cus_stat, cus_p, _ = breaks_cusumolsresid(fit.resid.values, ddof=X.shape[1])
    n, k = len(y_log), X.shape[1]
    lo, hi = int(n * trim), int(n * (1 - trim))
    scan = pd.Series({y_log.index[i]: _chow(y_log, X, i) for i in range(lo, hi)}, name="chow_F")
    # top local maxima at least 21 days apart
    top = []
    for d, v in scan.sort_values(ascending=False).items():
        if all(abs((d - t).days) >= 21 for t, _ in top):
            top.append((d, v))
        if len(top) == 3:
            break
    rows = [dict(test="OLS-CUSUM (Ploberger-Kramer)", date="", statistic=cus_stat, p_value=cus_p,
                 note="H0: parameter stability over the whole sample")]
    for d, v in top:
        rows.append(dict(test="sup-F scan: local max", date=d.date(), statistic=v, p_value=np.nan,
                         note="date chosen by search - pointwise F p-value not valid; compare size across dates"))
    for lab, d in candidates.items():
        i = y_log.index.get_loc(pd.Timestamp(d))
        v = _chow(y_log, X, i)
        rows.append(dict(test=f"Chow at pre-specified date: {lab}", date=pd.Timestamp(d).date(), statistic=v,
                         p_value=1 - fdist.cdf(v, k, n - 2 * k),
                         note="residuals are autocorrelated, so p-value is optimistic"))
    tab = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    axes[0].plot(y_log.index, np.exp(y_log), color=C["s1"], lw=1, alpha=0.6, label="daily demand")
    axes[0].plot(y_log.index, np.exp(fit.fittedvalues), color=C["s2"], lw=1.4, label="trend + weekday fit")
    axes[0].set_yscale("log"); _thousands(axes[0])
    axes[0].legend(loc="upper left")
    axes[0].set_title("Structural-break scan: log demand ~ trend + day-of-week")
    axes[1].plot(scan.index, scan.values, color=C["s1"], lw=1.6)
    axes[1].set_ylabel("Chow F at split date")
    for d, v in top:
        for ax in axes:
            ax.axvline(d, color=C["ink2"], lw=1, ls="--")
        axes[1].annotate(f"{d:%d %b %Y}", (d, v), xytext=(4, -2), textcoords="offset points", fontsize=8)
    _fmt_dates(axes[1])
    fig.tight_layout()
    return tab, scan, _save(fig, "phase2_06_breaks.png")
