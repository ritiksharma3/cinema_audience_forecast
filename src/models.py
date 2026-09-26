"""Phase 3 - model fitting functions with one common interface.

Every fit_* function takes (hist, future) - two slices of the daily frame - and returns a ModelFit:
the model is estimated on `hist` and forecasts every date in `future` (multi-step, from the end of hist).
Forecasts are always returned on the ticket scale; log-scale models are back-transformed with exp()
(i.e. the forecast median, no bias correction).
"""
import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from .config import SEASONAL_PERIOD as M
from .config import TARGET

warnings.filterwarnings("ignore")

DEC_SURGE = ("2023-12-01", "2023-12-31")
ADJUST_DEC_FOR_UNIVARIATE = True   # False only for the sensitivity run


@dataclass
class ModelFit:
    name: str
    spec: str                      # order / specification string
    params: pd.DataFrame           # coefficient_or_component, estimate, std_error, p_value
    aic: float
    bic: float
    resid: pd.Series               # in-sample residuals (burn-in removed), on the model's fitting scale
    forecast: pd.DataFrame         # index=future dates; mean, lower, upper (tickets, 95%)
    scale: str                     # "log" or "raw" - scale the residuals / AIC refer to
    n_arma_params: int = 0         # degrees of freedom for Ljung-Box
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ features
def exog_frame(df, use_lead=True):
    """Deterministic/known regressors. lead_100h is the realised mean booking lead time (hundreds of hours)."""
    X = pd.get_dummies(df.index.dayofweek, prefix="dow", drop_first=True).astype(float)   # Monday = base
    X.columns = ["dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat", "dow_Sun"]
    X.index = df.index
    X["is_holiday"] = df.is_holiday.astype(float)
    X["dec_surge_2023"] = ((df.index >= DEC_SURGE[0]) & (df.index <= DEC_SURGE[1])).astype(float)
    X["booknow_missing"] = df.booknow_missing.astype(float)
    if use_lead:
        X["lead_100h"] = df.mean_lead_hrs_combined / 100.0
    return X


def _drop_constant_cols(Xh, Xf):
    """Drop regressors with no variation in the estimation window (e.g. booknow_missing=0 in train+val refits)."""
    keep = [c for c in Xh.columns if Xh[c].nunique() > 1]
    return Xh[keep], Xf[keep]


def dec_multiplier(hist):
    """exp(beta) of the Dec-2023 dummy in  log y ~ const + trend + weekday + holiday + dec_surge  (OLS, HAC SE),
    estimated on `hist` only. Returns (multiplier, p_value); (1.0, nan) if hist contains no December days."""
    X = exog_frame(hist, use_lead=False).drop(columns="booknow_missing")
    X = X.loc[:, X.nunique() > 1]
    if "dec_surge_2023" not in X:
        return 1.0, np.nan
    X.insert(0, "trend", np.arange(len(X)) / 30.0)
    r = sm.OLS(np.log(hist[TARGET].astype(float)), sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": M})
    return float(np.exp(r.params["dec_surge_2023"])), float(r.pvalues["dec_surge_2023"])


def univariate_target(hist):
    """Target for models that cannot take regressors: December 2023 divided by the estimated surge multiplier
    (intervention adjustment), so the temporary regime is not extrapolated past the end of the sample."""
    y = hist[TARGET].astype(float)
    if not ADJUST_DEC_FOR_UNIVARIATE:
        return y, "No December adjustment (sensitivity run)."
    mult, p = dec_multiplier(hist)
    dec = (y.index >= DEC_SURGE[0]) & (y.index <= DEC_SURGE[1])
    return y.where(~dec, y / mult), (f"Fitted on December-adjusted demand: Dec 2023 divided by the surge multiplier "
                                     f"{mult:.2f}x (OLS dummy on train, p={p:.1g}); forecasts are for normal (non-surge) conditions.")


def _param_table(names, est, se, p):
    return pd.DataFrame({"coefficient_or_component": list(names), "estimate": np.asarray(est, float),
                         "std_error": np.asarray(se, float), "p_value": np.asarray(p, float)})


def _log_fc(frame):
    """Back-transform a statsmodels summary_frame on the log scale to tickets."""
    return pd.DataFrame({"mean": np.exp(frame["mean"]), "lower": np.exp(frame["mean_ci_lower"]),
                         "upper": np.exp(frame["mean_ci_upper"])})


# ------------------------------------------------------------------ 1-3. exponential smoothing (ETS state space)
def _fit_ets(name, hist, future, error, trend, seasonal, notes):
    y, adj = univariate_target(hist)
    notes = f"{notes} {adj}"
    kw = dict(error=error, trend=trend, seasonal=seasonal)
    if seasonal:
        kw["seasonal_periods"] = M
    r = ETSModel(y, **kw).fit(disp=False, maxiter=2000)
    pr = r.get_prediction(start=len(y), end=len(y) + len(future) - 1).summary_frame(alpha=0.05)
    pr.index = future.index
    fc = pd.DataFrame({"mean": pr["mean"], "lower": pr["pi_lower"], "upper": pr["pi_upper"]})
    params = _param_table(r.param_names, r.params, r.bse, r.pvalues)
    spec = f"ETS(error={error[0].upper()}, trend={'A' if trend else 'N'}, seasonal={seasonal[0].upper() if seasonal else 'N'}{f', m={M}' if seasonal else ''})"
    return ModelFit(name, spec, params, r.aic, r.bic, pd.Series(r.resid, index=y.index), fc, "raw",
                    notes=notes + " AIC/BIC are on the raw ticket likelihood - not comparable with log-scale models.")


def fit_ses(hist, future):
    return _fit_ets("SES", hist, future, "add", None, None,
                    "Baseline: flat forecast at the last smoothed level; ignores trend and weekly seasonality.")


def fit_holt(hist, future):
    return _fit_ets("Holt_linear", hist, future, "add", "add", None,
                    "Level + linear trend; no seasonality. Trend is estimated through the December surge/decline.")


def fit_hw_add(hist, future):
    return _fit_ets("HoltWinters_additive", hist, future, "add", "add", "add",
                    "Additive trend and additive weekly seasonality (Phase 2 favours multiplicative).")


def fit_hw_mul(hist, future):
    return _fit_ets("HoltWinters_multiplicative", hist, future, "mul", "add", "mul",
                    "Additive trend, multiplicative weekly seasonality and multiplicative errors.")


# ------------------------------------------------------------------ 4a. trend + seasonal-dummy regression
def trend_dummy_design(idx, start):
    """const + linear trend (months since `start`) + weekday dummies (Monday = base)."""
    X = pd.get_dummies(idx.dayofweek, prefix="dow", drop_first=True).astype(float)
    X.columns = ["dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat", "dow_Sun"]
    X.index = idx
    X.insert(0, "trend_month", (idx - pd.Timestamp(start)).days / 30.0)
    return sm.add_constant(X)


def fit_trend_dummy_reg(hist, future, keep=None):
    """Classical decomposition regression: log demand ~ const + linear trend + weekday dummies (OLS, HAC(7) SE).
    `keep` restricts the regressors (used for the reduced, significant-terms-only model)."""
    y, adj = univariate_target(hist)
    ly = np.log(y)
    Xh, Xf = trend_dummy_design(hist.index, hist.index[0]), trend_dummy_design(future.index, hist.index[0])
    if keep is not None:
        Xh, Xf = Xh[keep], Xf[keep]
    r = sm.OLS(ly, Xh).fit(cov_type="HAC", cov_kwds={"maxlags": M})
    pr = r.get_prediction(Xf).summary_frame(alpha=0.05)
    pr.index = future.index
    fc = pd.DataFrame({"mean": np.exp(pr["mean"]), "lower": np.exp(pr["obs_ci_lower"]), "upper": np.exp(pr["obs_ci_upper"])})
    params = _param_table(r.params.index, r.params, r.bse, r.pvalues)
    notes = ("Trend + seasonal-dummy regression on log demand (exp(coef) = multiplier vs Monday). "
             "No lags, so residual autocorrelation is expected; HAC(7) standard errors. " + adj)
    name = "Trend_Dummy_Reg" if keep is None else "Trend_Dummy_Reg_reduced"
    return ModelFit(name, "OLS: const + linear trend + weekday dummies" + ("" if keep is None else f" (kept: {', '.join(keep[1:])})"),
                    params, r.aic, r.bic, r.resid, fc, "log", notes=notes,
                    extra=dict(r2=r.rsquared, r2_adj=r.rsquared_adj, ols=r))


# ------------------------------------------------------------------ 4b. multiple linear regression
def _fit_mlr(name, hist, future, use_lead):
    """log demand ~ lag1 + lag7 of log demand + weekday dummies + holiday + Dec-surge + booknow_missing (+ lead).
    Multi-step forecasts are recursive: forecasted log demand is fed back as the lag."""
    ly = np.log(pd.concat([hist[TARGET], future[TARGET]]).astype(float))
    X_all = exog_frame(pd.concat([hist, future]), use_lead)
    Xh, Xf = _drop_constant_cols(X_all.loc[hist.index], X_all.loc[future.index])
    lh = ly.loc[hist.index]
    D = pd.concat([lh.shift(1).rename("lag1_log_demand"), lh.shift(7).rename("lag7_log_demand"), Xh], axis=1).dropna()
    D = sm.add_constant(D)
    yy = lh.loc[D.index]
    r = sm.OLS(yy, D).fit(cov_type="HAC", cov_kwds={"maxlags": M})
    path = lh.copy()
    for d in future.index:
        row = pd.concat([pd.Series({"const": 1.0, "lag1_log_demand": path.iloc[-1],
                                    "lag7_log_demand": path.iloc[-7]}), Xf.loc[d]])
        path.loc[d] = float(r.params @ row[r.params.index])
    mean = np.exp(path.loc[future.index])
    # approximate 95% interval: in-sample residual sd scaled by sqrt(h), capped - recursive OLS has no closed form
    h = np.arange(1, len(future) + 1)
    sd = r.resid.std() * np.sqrt(np.minimum(h, M))
    fc = pd.DataFrame({"mean": mean, "lower": mean * np.exp(-1.96 * sd), "upper": mean * np.exp(1.96 * sd)})
    params = _param_table(r.params.index, r.params, r.bse, r.pvalues)
    notes = ("OLS on log demand, HAC(7) standard errors. Recursive multi-step forecast. "
             "Interval is an approximation (residual sd × sqrt(min(h,7))).")
    if use_lead:
        notes += (" USES REALISED LEAD TIME for forecast dates (information not available in advance) - "
                  "optimistic/oracle scenario.")
    return ModelFit(name, "OLS: lag1, lag7, weekday dummies, holiday, dec_surge, booknow_missing" + (", lead" if use_lead else ""),
                    params, r.aic, r.bic, r.resid, fc, "log", n_arma_params=2, notes=notes,
                    extra=dict(r2=r.rsquared, r2_adj=r.rsquared_adj))


def fit_mlr(hist, future):
    return _fit_mlr("MLR_with_lead", hist, future, use_lead=True)


def fit_mlr_no_lead(hist, future):
    return _fit_mlr("MLR_no_lead", hist, future, use_lead=False)


# ------------------------------------------------------------------ ARIMA-family helpers
def _arima(ly, order, seasonal_order=(0, 0, 0, 0), trend="n", exog=None):
    return ARIMA(ly, order=order, seasonal_order=seasonal_order, trend=trend, exog=exog).fit()


def _arima_fit(name, spec, r, future, exog_f, n_arma, burn, notes, extra=None):
    fr = r.get_forecast(len(future), exog=exog_f).summary_frame(alpha=0.05)
    fr.index = future.index
    names = [("drift (log/day)" if n == "x1" else n) for n in r.params.index]
    params = _param_table(names, r.params, r.bse, r.pvalues)
    resid = r.resid.iloc[burn:]
    return ModelFit(name, spec, params, r.aic, r.bic, resid, _log_fc(fr), "log", n_arma_params=n_arma,
                    notes=notes, extra=extra or {})


def _grid_one(ly, o, so, t, exog):
    try:
        r = _arima(ly, o, so, t, exog)
        return dict(order=o, seasonal_order=so, trend=t, aic=r.aic, bic=r.bic,
                    converged=r.mle_retvals.get("converged", True))
    except Exception as e:  # invalid trend/differencing combinations etc.
        return dict(order=o, seasonal_order=so, trend=t, aic=np.nan, bic=np.nan, converged=False, error=str(e)[:80])


def _grid(ly, orders, seasonal_orders, trends, exog=None):
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=-1)(delayed(_grid_one)(ly, o, so, t, exog)
                               for o, so, t in itertools.product(orders, seasonal_orders, trends))
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


# ------------------------------------------------------------------ 5. AR / ARMA on the stationary series
def _ar_arma(name, hist, future, max_q):
    """ARMA on Δ7 log demand (stationary per Phase 2). Implemented as ARIMA(p,0,q)(0,1,0)7 with trend='t',
    which is exactly an ARMA(p,q) with constant on Δ7 log - so forecasts/intervals integrate back automatically."""
    y, adj = univariate_target(hist)
    ly = np.log(y)
    orders = [(p, 0, q) for p in range(0, 8) for q in range(0, max_q + 1) if p + q > 0]
    g = _grid(ly, orders, [(0, 1, 0, M)], ["t"])
    best = g.iloc[0]
    r = _arima(ly, best.order, (0, 1, 0, M), "t")
    p, _, q = best.order
    spec = f"ARMA({p},{q}) + const on Δ7 log demand  [= ARIMA({p},0,{q})(0,1,0){M}]"
    notes = (f"Stationary series = seasonal difference of log demand (ADF & KPSS agree, Phase 2). "
             f"Order chosen by AIC over p≤7{', q≤' + str(max_q) if max_q else ', q=0'} ({len(g)} candidates). "
             "'drift' is the constant of the Δ7 equation expressed per day. " + adj)
    return _arima_fit(name, spec, r, future, None, p + q, M + 1, notes, dict(selection=g.head(10)))


def fit_ar(hist, future):
    return _ar_arma("AR", hist, future, max_q=0)


def fit_arma(hist, future):
    return _ar_arma("ARMA", hist, future, max_q=3)


# ------------------------------------------------------------------ 6. ARIMA via auto_arima
def fit_arima(hist, future):
    import pmdarima as pm
    y, adj = univariate_target(hist)
    ly = np.log(y)
    auto = pm.auto_arima(ly, seasonal=False, d=None, test="kpss", start_p=0, start_q=0, max_p=7, max_q=3, max_order=None,
                         information_criterion="aic", stepwise=False, n_jobs=-1, suppress_warnings=True,
                         error_action="ignore", return_valid_fits=True)
    fits = auto if isinstance(auto, (list, tuple)) else [auto]
    sel = pd.DataFrame([dict(order=f.order, with_intercept=f.with_intercept, aic=f.aic(), bic=f.bic()) for f in fits]) \
        .sort_values("aic").reset_index(drop=True)
    best = fits[int(np.argmin([f.aic() for f in fits]))]
    order = best.order
    trend = ("c" if order[1] == 0 else "t" if order[1] == 1 else "n") if best.with_intercept else "n"
    r = _arima(ly, order, trend=trend)
    notes = (f"auto_arima (exhaustive grid, AIC, KPSS unit-root test chose d={order[1]}, non-seasonal, p≤7, q≤3) evaluated "
             f"{len(sel)} candidates; best {order} with intercept={best.with_intercept}. Refit in statsmodels ARIMA "
             f"(trend='{trend}'). Non-seasonal by construction: the weekly cycle can only be absorbed through high-order AR lags. " + adj)
    return _arima_fit("ARIMA", f"ARIMA{order} trend={trend} on log demand", r, future, None,
                      order[0] + order[2], order[1] + 1, notes, dict(selection=sel.head(10)))


# ------------------------------------------------------------------ 7. SARIMA
def fit_sarima(hist, future):
    """Grid around the Phase 2 starting point SARIMA(1,0,0)(0,1,1)7 on log demand (d=0, D=1 fixed by Phase 2)."""
    y, adj = univariate_target(hist)
    ly = np.log(y)
    orders = [(p, 0, q) for p in range(0, 3) for q in range(0, 3)]
    sorders = [(P, 1, Q, M) for P in range(0, 2) for Q in range(0, 2)]
    g = _grid(ly, orders, sorders, ["n", "t"])
    best = g.iloc[0]
    r = _arima(ly, best.order, best.seasonal_order, best.trend)
    (p, _, q), (P, _, Q, _) = best.order, best.seasonal_order
    start = g[(g.order == (1, 0, 0)) & (g.seasonal_order == (0, 1, 1, M))].aic.min()
    notes = (f"d=0, D=1 fixed from Phase 2 (seasonal unit root in ACF). Grid p,q≤2, P,Q≤1, trend n/t "
             f"({len(g)} fits) by AIC. Phase 2 starting model (1,0,0)(0,1,1)7 AIC={start:.1f}; selected AIC={best.aic:.1f}. " + adj)
    return _arima_fit("SARIMA", f"SARIMA{best.order}{best.seasonal_order} trend={best.trend} on log demand", r, future,
                      None, p + q + P + Q, M + 1, notes, dict(selection=g.head(10)))


# ------------------------------------------------------------------ 8. SARIMAX
def _sarimax(name, hist, future, use_lead):
    """Regression with seasonal ARIMA errors on log demand. d=1, D=0: with D=1 the weekday dummies would be
    differenced to zero, so the weekly pattern is carried by the dummies and the seasonal AR/MA terms instead."""
    ly = np.log(hist[TARGET].astype(float))
    X_all = exog_frame(pd.concat([hist, future]), use_lead)
    Xh, Xf = _drop_constant_cols(X_all.loc[hist.index], X_all.loc[future.index])
    orders = [(p, 1, q) for p in range(0, 3) for q in range(0, 3)]
    sorders = [(P, 0, Q, M) for P in range(0, 2) for Q in range(0, 2)]
    g = _grid(ly, orders, sorders, ["n", "t"], exog=Xh)
    best = g.iloc[0]
    r = _arima(ly, best.order, best.seasonal_order, best.trend, exog=Xh)
    (p, _, q), (P, _, Q, _) = best.order, best.seasonal_order
    notes = (f"Exog: weekday dummies (Mon base), holiday, dec_surge_2023, booknow_missing{', lead_100h' if use_lead else ''}. "
             f"d=1, D=0 because seasonal differencing would annihilate the weekday dummies. Grid p,q≤2, P,Q≤1, "
             f"trend n/t ({len(g)} fits) by AIC. theater_area not used: this is a single aggregate series, not panel data.")
    if use_lead:
        notes += " USES REALISED LEAD TIME for forecast dates - optimistic/oracle scenario."
    return _arima_fit(name, f"SARIMAX{best.order}{best.seasonal_order} trend={best.trend} + exog on log demand", r, future,
                      Xf, p + q + P + Q, 2, notes, dict(selection=g.head(10)))


def fit_sarimax(hist, future):
    return _sarimax("SARIMAX_with_lead", hist, future, use_lead=True)


def fit_sarimax_no_lead(hist, future):
    return _sarimax("SARIMAX_no_lead", hist, future, use_lead=False)


UNIVARIATE = [fit_ses, fit_holt, fit_hw_add, fit_hw_mul, fit_trend_dummy_reg, fit_ar, fit_arma, fit_arima, fit_sarima]
MODELS = [fit_ses, fit_holt, fit_hw_add, fit_hw_mul, fit_trend_dummy_reg, fit_mlr, fit_mlr_no_lead, fit_ar, fit_arma,
          fit_arima, fit_sarima, fit_sarimax, fit_sarimax_no_lead]
