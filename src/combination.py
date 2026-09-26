"""Phase 4 - forecast combination methods (reusable).

All weights are estimated on one window of out-of-sample forecasts (F_est, y_est) and applied to another (F_new).
F is a DataFrame: rows = dates, columns = member models, values = forecasts on the ticket scale.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ------------------------------------------------------------------ weights
def simple_average(F_est, y_est):
    k = F_est.shape[1]
    return dict(weights=pd.Series(1 / k, index=F_est.columns), intercept=0.0, table=None)


def bates_granger(F_est, y_est):
    """Inverse error-variance (MSE) weights, ignoring error covariances (Bates & Granger 1969)."""
    mse = F_est.sub(y_est, axis=0).pow(2).mean()
    w = (1 / mse) / (1 / mse).sum()
    tab = pd.DataFrame({"model": w.index, "val_MSE": mse.values, "inverse_MSE": (1 / mse).values, "weight": w.values})
    return dict(weights=w, intercept=0.0, table=tab)


def _vif(F):
    X = sm.add_constant(F)
    return pd.Series([1 / (1 - sm.OLS(X[c], X.drop(columns=c)).fit().rsquared) for c in F.columns], index=F.columns)


def granger_ramanathan_unconstrained(F_est, y_est):
    """GR 'regression 3' form: y = a + sum(b_i f_i) + e, OLS, weights unrestricted."""
    r = sm.OLS(y_est, sm.add_constant(F_est)).fit()
    tab = pd.DataFrame({"term": r.params.index, "estimate": r.params.values, "std_error": r.bse.values,
                        "p_value": r.pvalues.values})
    tab["VIF"] = tab.term.map(_vif(F_est)).astype(float)
    return dict(weights=r.params.drop("const"), intercept=float(r.params["const"]), table=tab,
                fit=dict(R2=r.rsquared, condition_number=float(np.linalg.cond(sm.add_constant(F_est).values)),
                         sum_of_weights=float(r.params.drop("const").sum())))


def granger_ramanathan_constrained(F_est, y_est):
    """GR 'regression 2' form: no intercept, weights sum to one. Estimated by regressing (y - f_K) on (f_i - f_K)."""
    base = F_est.columns[-1]
    Z = F_est.drop(columns=base).sub(F_est[base], axis=0)
    r = sm.OLS(y_est - F_est[base], Z).fit()
    w = r.params.copy()
    w[base] = 1 - w.sum()
    # standard error of the implied last weight: sqrt(1' V 1)
    se = r.bse.copy()
    se[base] = float(np.sqrt(np.ones(len(r.params)) @ r.cov_params().values @ np.ones(len(r.params))))
    tab = pd.DataFrame({"term": w.index, "estimate": w.values, "std_error": se.reindex(w.index).values})
    tab["t_stat"] = tab.estimate / tab.std_error
    tab["p_value"] = 2 * stats.t.sf(tab.t_stat.abs(), df=len(y_est) - len(r.params))
    return dict(weights=w.reindex(F_est.columns), intercept=0.0, table=tab,
                fit=dict(any_negative_weight=bool((w < 0).any())))


METHODS = {
    "Simple average": simple_average,
    "Bates-Granger (inverse MSE)": bates_granger,
    "Granger-Ramanathan (unconstrained, intercept)": granger_ramanathan_unconstrained,
    "Granger-Ramanathan (sum-to-one, no intercept)": granger_ramanathan_constrained,
}


def combine(F, spec):
    return spec["intercept"] + F[spec["weights"].index].mul(spec["weights"], axis=1).sum(axis=1)


# ------------------------------------------------------------------ intervals
def log_error_sd(combined_est, y_est):
    """SD of log(actual / combined) on the estimation window - drives the combined prediction interval."""
    return float(np.log(y_est / combined_est).std(ddof=1))


def interval(combined, sd, z=1.96):
    return pd.DataFrame({"mean": combined, "lower": combined * np.exp(-z * sd), "upper": combined * np.exp(z * sd)})


# ------------------------------------------------------------------ evaluation
def diebold_mariano(e1, e2, h=1):
    """DM test of equal squared-error loss with the Harvey-Leybourne-Newbold small-sample correction.
    Negative statistic => forecast 1 has lower loss. Returns (statistic, two-sided p-value)."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    lag = max(h - 1, int(np.floor(n ** (1 / 3))))
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(lag + 1)]
    lrv = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    dm = d.mean() / np.sqrt(lrv / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(hln), float(2 * stats.t.sf(abs(hln), df=n - 1))
