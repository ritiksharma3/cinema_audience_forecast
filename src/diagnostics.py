"""Forecast accuracy metrics and residual diagnostics shared by Phases 3-4."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from .config import OUT_FIG
from .eda import C  # shared palette + rcParams


def accuracy(actual, pred):
    """MAE, RMSE (tickets) and MAPE (%) via scikit-learn."""
    return dict(MAE=float(mean_absolute_error(actual, pred)),
                RMSE=float(np.sqrt(mean_squared_error(actual, pred))),
                MAPE=float(mean_absolute_percentage_error(actual, pred) * 100))


def residual_diagnostics(resid, n_arma_params=0, lags=(7, 14, 21)):
    r = pd.Series(resid).dropna()
    rows = []
    for L in lags:
        lb = acorr_ljungbox(r, lags=[L], model_df=n_arma_params if L > n_arma_params else 0)
        rows.append(dict(test=f"Ljung-Box lag {L}", statistic=float(lb.lb_stat.iloc[0]), p_value=float(lb.lb_pvalue.iloc[0]),
                         df=L - (n_arma_params if L > n_arma_params else 0),
                         verdict="residuals white noise" if lb.lb_pvalue.iloc[0] > 0.05 else "autocorrelation left"))
    jb = stats.jarque_bera(r)
    rows.append(dict(test="Jarque-Bera normality", statistic=float(jb.statistic), p_value=float(jb.pvalue), df=2,
                     verdict="normal" if jb.pvalue > 0.05 else "non-normal"))
    sw = stats.shapiro(r)
    rows.append(dict(test="Shapiro-Wilk normality", statistic=float(sw.statistic), p_value=float(sw.pvalue), df=np.nan,
                     verdict="normal" if sw.pvalue > 0.05 else "non-normal"))
    rows.append(dict(test="skewness", statistic=float(stats.skew(r)), p_value=np.nan, df=np.nan, verdict=""))
    rows.append(dict(test="excess kurtosis", statistic=float(stats.kurtosis(r)), p_value=np.nan, df=np.nan, verdict=""))
    return pd.DataFrame(rows)


def residual_plot(name, resid, scale, lags=28):
    r = pd.Series(resid).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw={"width_ratios": [2.2, 1.4, 1]})
    axes[0].vlines(r.index, 0, r.values, color=C["s1"], lw=0.8)
    axes[0].axhline(0, color=C["muted"], lw=0.8)
    axes[0].set_title(f"{name}: residuals ({scale} scale)", fontsize=9)
    axes[0].tick_params(axis="x", labelsize=7)
    v = acf(r, nlags=lags)
    k = np.arange(1, lags + 1)
    wk = k % 7 == 0
    band = 1.96 / np.sqrt(len(r))
    axes[1].vlines(k[~wk], 0, v[1:][~wk], color=C["s1"], lw=1.6)
    axes[1].vlines(k[wk], 0, v[1:][wk], color=C["s2"], lw=2.2)
    axes[1].axhspan(-band, band, color=C["band"], zorder=0)
    axes[1].axhline(0, color=C["muted"], lw=0.8)
    axes[1].set_xticks(range(0, lags + 1, 7))
    axes[1].set_ylim(-0.6, 0.6)
    axes[1].set_title("Residual ACF (orange = weekly lags)", fontsize=9)
    stats.probplot(r, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set(color=C["s1"], markersize=2.5)
    axes[2].get_lines()[1].set(color=C["s2"])
    axes[2].set_title("Normal Q-Q", fontsize=9)
    axes[2].set_xlabel(""); axes[2].set_ylabel("")
    fig.tight_layout()
    path = OUT_FIG / f"phase3_resid_{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
