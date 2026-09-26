"""Phase 5 - survival analysis functions: Kaplan-Meier, log-rank, Cox PH, Schoenfeld PH checks."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

from .config import INACTIVE_AFTER_DAYS, OUT_FIG, RAW
from .eda import C

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BIG_AREA_MIN = 100            # areas with >= this many theaters get their own KM curve
COX_AREA_MIN_EVENTS = 5       # ...and their own Cox dummy only if they also have >= this many events
POOLED = "other areas (pooled)"
KM_TIMES = [30, 90, 180, 365]


# ------------------------------------------------------------------ preparation
def booking_gaps():
    """Per-theater longest gap (days) between consecutive booking show-dates, BEFORE the final silence."""
    out = []
    for f, idc, sysname in (("booknow_booking.csv", "book_theater_id", "booknow"),
                            ("cinePOS_booking.csv", "cine_theater_id", "cinePOS")):
        b = pd.read_csv(RAW / f, usecols=[idc, "show_datetime"])
        b["d"] = pd.to_datetime(b.show_datetime.str[:10], format="%Y-%m-%d")
        b = b[[idc, "d"]].drop_duplicates().sort_values([idc, "d"])
        b["gap"] = b.groupby(idc).d.diff().dt.days
        g = b.groupby(idc).gap.agg(max_gap_days="max", p90_gap_days=lambda x: x.quantile(0.9)).reset_index()
        out.append(g.rename(columns={idc: "theater_id"}).assign(system=sysname))
    return pd.concat(out, ignore_index=True)


def prepare(s, gaps):
    s = s.merge(gaps, on=["system", "theater_id"], how="left")
    end = s.last_booking_date.max()
    s["event"] = s.event_inactive
    # time on study: to last booking for events, to end of data for censored theaters
    s["T"] = np.where(s.event == 1, (s.last_booking_date - s.first_booking_date).dt.days + 1,
                      (end - s.first_booking_date).dt.days + 1)
    # gap-adjusted event: the final silence must also exceed the theater's own longest earlier gap
    s["event_gap_adjusted"] = ((s.event == 1) & (s.days_since_last_booking > s.max_gap_days.fillna(0))).astype(int)
    s["metadata_missing"] = (s.theater_type == "Unknown").astype(int)
    s["type_group"] = np.where(s.metadata_missing == 1, "Unknown", s.theater_type)
    s["area_label"] = np.where(s.metadata_missing == 1, "Unknown", s.system + ":" + s.theater_area.astype(str))
    known = s[s.metadata_missing == 0]
    n_area = known.area_label.value_counts()
    ev_area = known.groupby("area_label").event.sum()
    big = n_area[n_area >= BIG_AREA_MIN].index
    cox_big = [a for a in big if ev_area[a] >= COX_AREA_MIN_EVENTS]
    s["area_km"] = np.where(s.metadata_missing == 1, "Unknown", np.where(s.area_label.isin(big), s.area_label, POOLED))
    # zero/very-few-event areas make the Cox likelihood monotone (HR -> 0, no convergence): pool them
    s["area_group"] = np.where(s.metadata_missing == 1, "Unknown", np.where(s.area_label.isin(cox_big), s.area_label, POOLED))
    s["log2_mean_daily_tickets"] = np.log2(s.mean_daily_tickets)
    p99 = s.mean_lead_hrs.quantile(0.99)
    s["lead_100h"] = s.mean_lead_hrs.clip(upper=p99) / 100.0
    s["is_booknow"] = (s.system == "booknow").astype(int)
    return s, dict(end=end, lead_winsor_p99_hrs=p99, big_areas=list(big), cox_areas=cox_big,
                   pooled_for_cox=[f"{a} (n={n_area[a]}, events={ev_area[a]})" for a in big if a not in cox_big])


def cox_design(s, extra=()):
    """Covariate matrix. Type/area dummies are only defined for theaters with metadata; metadata_missing carries
    the Unknown group (Unknown type and Unknown area are the same theaters, so they can't both enter)."""
    X = pd.DataFrame(index=s.index)
    for t in ("Action", "Comedy", "Drama"):                              # reference: Other
        X[f"type_{t}"] = (s.type_group == t).astype(int)
    for a in sorted(s.area_group.unique()):
        if a not in ("Unknown", POOLED):                                 # reference: pooled areas
            X[f"area_{a.split(':')[1]}"] = (s.area_group == a).astype(int)
    X["metadata_missing"] = s.metadata_missing
    X["is_booknow"] = s.is_booknow
    X["lead_100h"] = s.lead_100h
    X["log2_mean_daily_tickets"] = s.log2_mean_daily_tickets
    for c in extra:
        X[c] = s[c]
    X = X.loc[:, X.nunique() > 1]
    return X


# ------------------------------------------------------------------ Kaplan-Meier
def km_table(s, group_col=None, event_col="event", times=KM_TIMES, min_n=1):
    rows, fits = [], {}
    groups = [("all theaters", s)] if group_col is None else list(s.groupby(group_col))
    for g, d in groups:
        if len(d) < min_n:
            continue
        k = KaplanMeierFitter().fit(d["T"], d[event_col], label=str(g))
        fits[str(g)] = k
        ci = k.confidence_interval_survival_function_
        row = dict(group=str(g), n=len(d), events=int(d[event_col].sum()),
                   event_rate_pct=100 * d[event_col].mean(),
                   median_survival_days=k.median_survival_time_)
        for t in times:
            row[f"S({t}d)"] = float(k.survival_function_at_times(t).iloc[0])
            lo = ci.iloc[:, 0].reindex(ci.index.union([t])).ffill().loc[t]
            hi = ci.iloc[:, 1].reindex(ci.index.union([t])).ffill().loc[t]
            row[f"S({t}d)_95CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows), fits


def logrank(s, group_col, event_col="event"):
    r = multivariate_logrank_test(s["T"], s[group_col], s[event_col])
    return dict(stratified_by=group_col, groups=s[group_col].nunique(), test_statistic=r.test_statistic,
                df=r.degrees_of_freedom, p_value=r.p_value)


def km_plot(fits, title, fname, ylim=(0.75, 1.0)):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (g, k) in enumerate(fits.items()):
        col = CAT[i % len(CAT)] if len(fits) > 1 else C["s1"]
        k.plot_survival_function(ax=ax, ci_show=True, color=col, lw=1.8, ci_alpha=0.12)
    ax.set_ylim(*ylim)
    ax.set_xlabel("days since first booking")
    ax.set_ylabel("P(theater still active)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2 if len(fits) > 6 else 1)
    fig.tight_layout()
    fig.savefig(OUT_FIG / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Cox PH
# Estimated with statsmodels PHReg (Efron ties, BFGS). lifelines' Newton-Raphson CoxPHFitter fails on this data
# (ConvergenceError / absurd coefficients even with penalisation) although the partial likelihood is well-behaved:
# the profile likelihood of is_booknow is concave with its maximum at ~2.2, which PHReg reproduces (score ~ 1e-5).
@dataclass
class CoxResult:
    params: pd.Series
    bse: pd.Series
    cov: pd.DataFrame
    llf: float
    ll0: float
    n: int
    events: int
    event_col: str
    X: pd.DataFrame
    T: pd.Series
    E: pd.Series
    schoenfeld: np.ndarray
    strata: tuple = ()


def fit_cox(s, X, event_col="event", strata=None):
    X = X.astype(float)
    T, E = s["T"].astype(float), s[event_col].astype(float)
    st = None if strata is None else s[list(strata)].astype(str).agg("|".join, axis=1)
    m = PHReg(T, X, status=E, ties="efron", strata=st)
    r = m.fit(method="bfgs", maxiter=5000, disp=0, gtol=1e-8)
    if np.abs(m.score(r.params)).max() > 1e-3:
        raise RuntimeError("Cox fit did not converge (score not ~0)")
    cov = pd.DataFrame(r.cov_params(), index=X.columns, columns=X.columns)
    return CoxResult(pd.Series(r.params, X.columns), pd.Series(r.bse, X.columns), cov, float(r.llf),
                     float(m.loglike(np.zeros(X.shape[1]))), len(X), int(E.sum()), event_col, X, T, E,
                     np.asarray(r.schoenfeld_residuals), tuple(strata or ()))


def cox_inference_table(name, res, ph=None, notes=""):
    z = res.params / res.bse
    t = pd.DataFrame({"model": name, "covariate": res.params.index, "coef": res.params.values,
                      "hazard_ratio": np.exp(res.params.values), "std_error": res.bse.values, "z": z.values,
                      "p_value": 2 * stats.norm.sf(np.abs(z.values)),
                      "HR_lower_95": np.exp(res.params.values - 1.96 * res.bse.values),
                      "HR_upper_95": np.exp(res.params.values + 1.96 * res.bse.values)})
    t["significance"] = t.p_value.map(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else "ns")
    binary = [set(res.X[c].unique()) <= {0, 1} for c in res.params.index]
    t["n_in_group"] = [int(res.X[c].sum()) if bi else np.nan for c, bi in zip(res.params.index, binary)]
    t["events_in_group"] = [int(res.E[res.X[c] == 1].sum()) if bi else np.nan for c, bi in zip(res.params.index, binary)]
    if ph is not None:
        t = t.merge(ph, on="covariate", how="left")
    t["notes"] = ""
    t.loc[0, "notes"] = notes
    return t


def model_fit_row(name, res):
    k = len(res.params)
    lr = 2 * (res.llf - res.ll0)
    risk = res.X.values @ res.params.values
    return dict(model=name, n=res.n, events=res.events, strata=", ".join(res.strata),
                concordance=concordance_index(res.T, -risk, res.E), partial_log_likelihood=res.llf,
                partial_AIC=-2 * res.llf + 2 * k, LR_test_stat=lr, LR_df=k, LR_p=stats.chi2.sf(lr, k))


def _scaled_schoenfeld(res):
    ev = ~np.isnan(res.schoenfeld).all(axis=1)
    r = res.schoenfeld[ev]
    scaled = res.events * r @ res.cov.values          # Grambsch & Therneau (1994) approximation
    return scaled, res.T.values[ev]


def ph_test(res):
    """Grambsch-Therneau test on scaled Schoenfeld residuals vs rank(time), per covariate (chi2, 1 df)."""
    scaled, t = _scaled_schoenfeld(res)
    g = stats.rankdata(t)
    g = g - g.mean()
    num = (g @ scaled) ** 2
    den = res.events * np.diag(res.cov.values) * (g ** 2).sum()
    stat = num / den
    out = pd.DataFrame({"covariate": res.params.index, "PH_test_stat": stat, "PH_p": stats.chi2.sf(stat, 1)})
    out["PH_verdict"] = np.where(out.PH_p < 0.05, "PH violated (p<0.05)", "PH not rejected")
    return out


def schoenfeld_plot(res, covariates, fname):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    scaled, times = _scaled_schoenfeld(res)
    n = len(covariates)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, c in zip(axes[0], covariates):
        j = list(res.params.index).index(c)
        y = scaled[:, j] + res.params[c]
        ax.scatter(times, y, s=5, color=C["s1"], alpha=0.35)
        lw = lowess(y, times, frac=0.4, it=0)       # it=0: robust reweighting would drop the minority band of binary covariates
        ax.plot(lw[:, 0], lw[:, 1], color=C["s2"], lw=2)
        ax.axhline(res.params[c], color=C["ink2"], ls="--", lw=1)
        lo, hi = np.percentile(y, [2, 98])
        ax.set_ylim(lo, hi)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("days since first booking")
    axes[0][0].set_ylabel("scaled Schoenfeld residual + coef")
    fig.suptitle("PH check: flat LOWESS line (orange) around the Cox coefficient (dashed) = proportional hazards",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def forest_plot(t, fname, title):
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(t) + 1.5))
    y = np.arange(len(t))
    sig = t.p_value < 0.05
    ax.hlines(y, t.HR_lower_95, t.HR_upper_95, color=np.where(sig, C["s1"], C["muted"]), lw=2)
    ax.scatter(t.hazard_ratio, y, color=np.where(sig, C["s1"], C["muted"]), s=30, zorder=3)
    ax.axvline(1, color=C["ink2"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(t.covariate, fontsize=8)
    ax.set_xlabel("hazard ratio of becoming inactive (log scale, 95% CI)")
    ax.set_title(title + "  (blue = p<0.05)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
