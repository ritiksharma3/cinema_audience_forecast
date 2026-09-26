# Phase 2 — EDA & decomposition summary

**Series:** `total_demand` (dedup target), 396 days, 29 Jan 2023 – 28 Feb 2024. The first 28 startup-censored days are excluded (confirmed Phase 1 decision). Mean 20,759/day, median 15,442, min 5,487, max 167,804.

## 1. Trend
- STL trend goes from **12,195/day** (first 28 days) to **26,822/day** (last 28 days), **+120%**. A log-linear fit with weekday dummies gives **+6.7% per month**.
- The trend isn't a straight line: it's flat from Feb to Oct 2023, then rises from Nov 2023, peaks in December, and stays above its 2023 level through Jan–Feb 2024. Monthly means:

| month   |   mean_demand |   stl_trend_mean |   days |
|:--------|--------------:|-----------------:|-------:|
| 2023-01 |        21,311 |           11,658 |      3 |
| 2023-02 |        13,247 |           12,341 |     28 |
| 2023-03 |        18,164 |           18,139 |     31 |
| 2023-04 |        16,583 |           16,202 |     30 |
| 2023-05 |        13,791 |           14,316 |     31 |
| 2023-06 |        15,037 |           15,219 |     30 |
| 2023-07 |        16,438 |           15,548 |     31 |
| 2023-08 |        15,093 |           15,795 |     31 |
| 2023-09 |        16,191 |           15,475 |     30 |
| 2023-10 |        17,350 |           17,604 |     31 |
| 2023-11 |        21,002 |           21,944 |     30 |
| 2023-12 |        55,106 |           45,071 |     31 |
| 2024-01 |        25,066 |           26,168 |     31 |
| 2024-02 |        26,079 |           26,822 |     28 |

## 2. Seasonality
- Periodogram peaks (detrended log demand): **6.95 days** (27% of power), **3.5 days** (8% of power), **30.46 days** (3% of power), **6.6 days** (2% of power), **66 days** (2% of power).
- STL on log demand: seasonal strength **F_s = 0.76**, trend strength **F_t = 0.72** (Hyndman scale 0–1; F_s > 0.64 is the usual cut-off for seasonal differencing).
- Day-of-week profile (demand ÷ centred 7-day mean, median): Mon 0.82×, Tue 0.60×, Wed 0.68×, Thu 0.88×, Fri 0.85×, Sat 2.07×, Sun 1.87×.
- **Annual seasonality can't be estimated**: only 13 months of data, so each calendar month appears about once. The December peak can't be separated into a 'yearly pattern' and a 'one-off event' from this data alone.

## 3. Additive vs multiplicative
| model          |   resid_var_tickets |   resid_sd_tickets |   resid_sd_pct_of_trend |   corr_abs_resid_vs_level |   corr_abs_rel_resid_vs_level |
|:---------------|--------------------:|-------------------:|------------------------:|--------------------------:|------------------------------:|
| additive       |            1.14e+08 |           1.07e+04 |                    34.2 |                      0.68 |                         0.223 |
| multiplicative |            9.38e+07 |           9.68e+03 |                    28.3 |                      0.61 |                         0.25  |

- Residual variance in ticket units is lower for the **multiplicative** model (93,791,512 vs 114,075,226).
- Additive |residual| correlates with the trend level (r = 0.68): the size of the weekly swing grows with demand, which points to multiplicative seasonality, i.e. modelling log demand.

## 4. Stationarity (ADF + KPSS level; α = 0.05)
| series   |   adf_stat |    adf_p |   kpss_stat |   kpss_p | kpss_p_note                      | conclusion                                                                                                |
|:---------|-----------:|---------:|------------:|---------:|:---------------------------------|:----------------------------------------------------------------------------------------------------------|
| raw      |      -3.23 | 0.0184   |      1.46   |    0.01  | p is outside table range: <=0.01 | conflict: ADF rejects unit root, KPSS rejects stationarity (breaks/heteroskedasticity or near-integrated) |
| Δ1       |      -5.76 | 5.61e-07 |      0.124  |    0.1   | p is outside table range: >=0.10 | stationary (both tests agree)                                                                             |
| Δ7       |      -6.54 | 9.48e-09 |      0.0414 |    0.1   | p is outside table range: >=0.10 | stationary (both tests agree)                                                                             |
| Δ1Δ7     |      -6.26 | 4.22e-08 |      0.0927 |    0.1   | p is outside table range: >=0.10 | stationary (both tests agree)                                                                             |
| log      |      -2.07 | 0.257    |      2.42   |    0.01  | p is outside table range: <=0.01 | non-stationary (both tests agree)                                                                         |
| Δ1 log   |      -6.37 | 2.31e-08 |      0.476  |    0.047 |                                  | conflict: ADF rejects unit root, KPSS rejects stationarity (breaks/heteroskedasticity or near-integrated) |
| Δ7 log   |      -6.32 | 3.05e-08 |      0.0318 |    0.1   | p is outside table range: >=0.10 | stationary (both tests agree)                                                                             |
| Δ1Δ7 log |      -7.66 | 1.72e-11 |      0.117  |    0.1   | p is outside table range: >=0.10 | stationary (both tests agree)                                                                             |

Trend-KPSS variant and critical values are in `eda_results.xlsx → stationarity_tests`.

pmdarima differencing estimators on log demand: d (non-seasonal) via KPSS = **1**; d (non-seasonal) via ADF = **1**; d (non-seasonal) via PP = **0**; D (seasonal, m=7) via OCSB = **0**; D (seasonal, m=7) via Canova-Hansen = **0**.

## 5. ACF / PACF (significant lags, 95% band, up to lag 42)
| series   |   n | acf_sig_lags                                           | pacf_sig_lags                                          |
|:---------|----:|:-------------------------------------------------------|:-------------------------------------------------------|
| raw      | 396 | 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 20, 21, 28 …   | 1, 2, 3, 5, 6, 7, 9, 11, 12, 18, 21, 33, 34, 42        |
| Δ1       | 395 | 2, 3, 4, 5, 6, 7, 9, 13, 14, 16, 21, 28, 35, 42        | 2, 3, 4, 5, 6, 8, 10, 11, 15, 20, 21, 27, 29, 32, 41   |
| Δ7       | 389 | 1, 5, 6, 7, 12, 14, 15, 20, 21                         | 1, 5, 6, 7, 8, 9, 12, 14, 16, 18, 20, 21, 23, 26, 27 … |
| Δ1Δ7     | 388 | 1, 2, 6, 7, 8, 15                                      | 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 18, 19 …   |
| log      | 396 | 1, 2, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29 … | 1, 2, 3, 5, 6, 7, 8, 14, 15, 21, 28, 36, 42            |
| Δ1 log   | 395 | 2, 3, 4, 5, 7, 9, 12, 14, 16, 19, 21, 23, 26, 28, 35 … | 2, 3, 4, 5, 6, 7, 12, 13, 19, 20, 27, 35, 41           |
| Δ7 log   | 389 | 1, 2, 4, 5, 7, 14, 15, 21, 29, 30                      | 1, 5, 7, 8, 9, 13, 14, 21, 23, 29, 35                  |
| Δ1Δ7 log | 388 | 1, 2, 5, 6, 7, 30                                      | 1, 2, 3, 4, 6, 7, 9, 10, 13, 16, 17, 20, 23, 25, 26 …  |

## 6. Outliers (flagged, NOT removed)
Robust z-score (median/MAD) of the log-scale STL remainder: **61 days** with |z| > 3.5, of which **25 extreme** (|z| > 7). Spikes: 31, dips: 30.

- About 15% of days are flagged at any threshold or STL window setting (checked 7/13/21/35). The irregular part of the series is **heavy-tailed and event-driven**, not a handful of glitches. Spikes cluster on Thu/Fri (typical Indian film-release days) and on some Mondays; the files have no release calendar, so this can't be confirmed.
- Checked as a possible data error: the 27–28 Feb 2023 spike (same date as the booknow_visits duplicates) has a normal theater count, ticket size, duplicate rate and booking-date spread, so it looks like **real demand**, not a load artefact.
- 31 Dec 2023 (−73%) and 1 Jan 2024 (−41%) are sharp New Year's Eve/Day dips inside the December surge.

| date       | day_name   |   actual |   expected_trend_x_season |   pct_vs_expected |   robust_z | severity   | holiday_name     |
|:-----------|:-----------|---------:|--------------------------:|------------------:|-----------:|:-----------|:-----------------|
| 2023-01-29 | Sunday     | 2.88e+04 |                  2.07e+04 |              39.1 |       4.81 | moderate   |                  |
| 2023-02-10 | Friday     | 2.19e+04 |                  7.64e+03 |             187   |      15.3  | extreme    |                  |
| 2023-02-18 | Saturday   | 9.31e+03 |                  1.27e+04 |             -26.7 |      -4.49 | moderate   | Maha Shivaratri  |
| 2023-02-25 | Saturday   | 1.09e+04 |                  1.5e+04  |             -27.7 |      -4.68 | moderate   |                  |
| 2023-02-27 | Monday     | 2.76e+04 |                  1.87e+04 |              47.4 |       5.64 | moderate   |                  |
| 2023-02-28 | Tuesday    | 1.93e+04 |                  8.12e+03 |             138   |      12.6  | extreme    |                  |
| 2023-03-04 | Saturday   | 2.15e+04 |                  1.56e+04 |              37.7 |       4.66 | moderate   |                  |
| 2023-03-06 | Monday     | 9.62e+03 |                  1.66e+04 |             -42   |      -7.88 | extreme    |                  |
| 2023-03-11 | Saturday   | 2.92e+04 |                  2.3e+04  |              27.2 |       3.51 | moderate   |                  |
| 2023-03-13 | Monday     | 1.16e+04 |                  1.88e+04 |             -38   |      -6.91 | moderate   |                  |
| 2023-03-28 | Tuesday    | 1.62e+04 |                  1.07e+04 |              51.7 |       6.06 | moderate   |                  |
| 2023-03-29 | Wednesday  | 1.82e+04 |                  1.29e+04 |              41.3 |       5.03 | moderate   |                  |
| 2023-03-30 | Thursday   | 2.34e+04 |                  1.56e+04 |              50.1 |       5.91 | moderate   | Ram Navami       |
| 2023-04-01 | Saturday   | 2.41e+04 |                  3.9e+04  |             -38.3 |      -6.97 | moderate   |                  |
| 2023-04-02 | Sunday     | 1.94e+04 |                  2.79e+04 |             -30.5 |      -5.26 | moderate   |                  |
| 2023-04-03 | Monday     | 9.22e+03 |                  1.43e+04 |             -35.7 |      -6.38 | moderate   |                  |
| 2023-04-10 | Monday     | 8.55e+03 |                  1.19e+04 |             -28.4 |      -4.82 | moderate   |                  |
| 2023-04-17 | Monday     | 9.57e+03 |                  1.33e+04 |             -28.1 |      -4.77 | moderate   |                  |
| 2023-04-24 | Monday     | 1e+04    |                  1.48e+04 |             -32.3 |      -5.63 | moderate   |                  |
| 2023-04-28 | Friday     | 3.24e+04 |                  1.35e+04 |             140   |      12.7  | extreme    |                  |
| 2023-04-29 | Saturday   | 1.94e+04 |                  3.52e+04 |             -45   |      -8.63 | extreme    |                  |
| 2023-05-02 | Tuesday    | 1.45e+04 |                  7.2e+03  |             101   |      10.2  | extreme    |                  |
| 2023-05-03 | Wednesday  | 1.79e+04 |                  7.99e+03 |             125   |      11.8  | extreme    |                  |
| 2023-05-04 | Thursday   | 1.63e+04 |                  1.04e+04 |              55.6 |       6.43 | moderate   |                  |
| 2023-05-06 | Saturday   | 1.38e+04 |                  2.52e+04 |             -45.4 |      -8.75 | extreme    |                  |
| 2023-06-01 | Thursday   | 8.24e+03 |                  1.16e+04 |             -28.7 |      -4.88 | moderate   |                  |
| 2023-06-02 | Friday     | 7.99e+03 |                  1.09e+04 |             -26.8 |      -4.5  | moderate   |                  |
| 2023-07-17 | Monday     | 1.78e+04 |                  1.05e+04 |              70   |       7.71 | extreme    |                  |
| 2023-08-10 | Thursday   | 2.53e+04 |                  1.47e+04 |              72.6 |       7.94 | extreme    |                  |
| 2023-08-12 | Saturday   | 2.01e+04 |                  3.19e+04 |             -36.8 |      -6.63 | moderate   |                  |
| 2023-08-13 | Sunday     | 2.07e+04 |                  2.65e+04 |             -22   |      -3.58 | moderate   |                  |
| 2023-08-14 | Monday     | 1.6e+04  |                  1.13e+04 |              41.2 |       5.03 | moderate   |                  |
| 2023-08-15 | Tuesday    | 1.27e+04 |                  8.81e+03 |              44.4 |       5.35 | moderate   | Independence Day |
| 2023-09-01 | Friday     | 8.07e+03 |                  1.17e+04 |             -31.3 |      -5.41 | moderate   |                  |
| 2023-09-18 | Monday     | 1.94e+04 |                  1.11e+04 |              74.3 |       8.08 | extreme    |                  |
| 2023-09-21 | Thursday   | 2.36e+04 |                  1.34e+04 |              76.5 |       8.26 | extreme    |                  |
| 2023-09-23 | Saturday   | 2.37e+04 |                  3.09e+04 |             -23.5 |      -3.86 | moderate   |                  |
| 2023-10-09 | Monday     | 1.95e+04 |                  1.11e+04 |              75.3 |       8.16 | extreme    |                  |
| 2023-11-01 | Wednesday  | 9.08e+03 |                  1.21e+04 |             -24.8 |      -4.11 | moderate   |                  |
| 2023-11-02 | Thursday   | 2.42e+04 |                  1.48e+04 |              63.6 |       7.16 | extreme    |                  |
| 2023-11-04 | Saturday   | 2.32e+04 |                  3.22e+04 |             -28   |      -4.74 | moderate   |                  |
| 2023-11-22 | Wednesday  | 3.34e+04 |                  1.37e+04 |             143   |      12.9  | extreme    |                  |
| 2023-12-01 | Friday     | 1.5e+04  |                  2.09e+04 |             -28.4 |      -4.83 | moderate   |                  |
| 2023-12-09 | Saturday   | 1.07e+05 |                  6.39e+04 |              68   |       7.54 | extreme    |                  |
| 2023-12-16 | Saturday   | 1.68e+05 |                  1.03e+05 |              62.2 |       7.03 | extreme    |                  |
| 2023-12-18 | Monday     | 3.32e+04 |                  4.47e+04 |             -25.7 |      -4.29 | moderate   |                  |
| 2023-12-22 | Friday     | 1.57e+05 |                  6.09e+04 |             157   |      13.7  | extreme    |                  |
| 2023-12-23 | Saturday   | 5.58e+04 |                  1.48e+05 |             -62.2 |     -14.1  | extreme    |                  |
| 2023-12-24 | Sunday     | 5.27e+04 |                  1.65e+05 |             -68.1 |     -16.5  | extreme    |                  |
| 2023-12-25 | Monday     | 3.14e+04 |                  5.75e+04 |             -45.3 |      -8.73 | extreme    | Christmas        |
| 2023-12-28 | Thursday   | 7.5e+04  |                  5.07e+04 |              47.9 |       5.7  | moderate   |                  |
| 2023-12-29 | Friday     | 7.88e+04 |                  3.9e+04  |             102   |      10.2  | extreme    |                  |
| 2023-12-31 | Sunday     | 1.6e+04  |                  7.23e+04 |             -77.9 |     -21.9  | extreme    |                  |
| 2024-01-01 | Monday     | 1.33e+04 |                  2.12e+04 |             -37.3 |      -6.75 | moderate   | New Year's Day   |
| 2024-01-02 | Tuesday    | 3.15e+04 |                  1.39e+04 |             127   |      11.9  | extreme    |                  |
| 2024-01-03 | Wednesday  | 2.37e+04 |                  1.52e+04 |              56.6 |       6.53 | moderate   |                  |
| 2024-01-08 | Monday     | 2.79e+04 |                  1.32e+04 |             110   |      10.8  | extreme    |                  |
| 2024-01-09 | Tuesday    | 1.59e+04 |                  1.01e+04 |              56.6 |       6.53 | moderate   |                  |
| 2024-02-01 | Thursday   | 1.53e+04 |                  2.13e+04 |             -28.1 |      -4.76 | moderate   |                  |
| 2024-02-02 | Friday     | 1.43e+04 |                  2.02e+04 |             -29   |      -4.94 | moderate   |                  |
| 2024-02-03 | Saturday   | 3.7e+04  |                  4.8e+04  |             -23   |      -3.78 | moderate   |                  |

## 7. Structural breaks
| test                                                            | date       |   statistic |    p_value | note                                                                             |
|:----------------------------------------------------------------|:-----------|------------:|-----------:|:---------------------------------------------------------------------------------|
| OLS-CUSUM (Ploberger-Kramer)                                    |            |        2.44 |   1.37e-05 | H0: parameter stability over the whole sample                                    |
| sup-F scan: local max                                           | 2023-12-07 |       16.8  | nan        | date chosen by search - pointwise F p-value not valid; compare size across dates |
| sup-F scan: local max                                           | 2023-11-16 |       12.8  | nan        | date chosen by search - pointwise F p-value not valid; compare size across dates |
| sup-F scan: local max                                           | 2023-04-02 |       10.1  | nan        | date chosen by search - pointwise F p-value not valid; compare size across dates |
| Chow at pre-specified date: booknow outage start                | 2023-07-27 |        5.19 |   3.62e-06 | residuals are autocorrelated, so p-value is optimistic                           |
| Chow at pre-specified date: booknow outage end (first day back) | 2023-10-26 |        7.65 |   1.67e-09 | residuals are autocorrelated, so p-value is optimistic                           |
| Chow at pre-specified date: December 2023 surge start           | 2023-12-01 |       15.5  |   1.11e-16 | residuals are autocorrelated, so p-value is optimistic                           |

- CUSUM rejects parameter stability. The biggest break is around **Nov–Dec 2023** (sup-F peaks 16 Nov and 7 Dec; Chow at 1 Dec is the largest pre-specified test). The booknow outage boundaries show smaller but significant shifts. The spring peak (early Apr 2023) is a third, weaker candidate.
- The December 2023 surge is best treated as a **temporary regime/event** (demand ~2.5× for about 4 weeks, then settling ~70% above the 2023 level), not a permanent level shift. It can't be classified with certainty without a second December.

## 8. Implications for Phase 3
- **Transform:** model `log(total_demand)` (multiplicative seasonality; variance grows with level).
- **Seasonal period m = 7.** Weekly pattern: Sat ≈ 2.1× and Sun ≈ 1.9× the weekly mean, Tue lowest ≈ 0.6×.
- **Differencing:** log level is non-stationary. Δ7 log and Δ1Δ7 log are stationary by both tests; Δ1 log is borderline (KPSS p ≈ 0.047). pmdarima's test-based estimators give d = 1, D = 0, **but the ACF disagrees on D**: Δ1 log keeps ACF ≈ 0.6–0.7 at lags 7, 14, …, 42 with almost no decay, which is the signature of a seasonal unit root. Recommendation: **D = 1, d = 0** (Δ7 log is stationary on its own). Keep d = 1, D = 0 as the comparison model auto_arima will propose.
- **ACF/PACF of Δ7 log:** ACF lag 1 ≈ 0.43 decaying; PACF cuts off after lag 1 → **non-seasonal AR(1)**. Negative ACF spike at lag 7 with PACF decaying across 7/14/21 → **seasonal MA(1)**. Starting candidate: **SARIMA(1,0,0)(0,1,1)7 on log demand**. Δ1Δ7 log shows ACF −0.25 at lag 1 and −0.37 at lag 7 (over-differencing signs), which argues against also taking d = 1.
- **Exogenous needs:** holiday flag, weekday dummies, a December-2023 regime dummy (or intervention) and `booknow_missing`. Extreme outlier dates are passed to Phase 3 (`phase2_handoff.json`) as candidate pulse dummies.
- **Split warning:** a chronological train/val/test split puts December 2023 in either validation or test. Whichever it lands in will dominate that window's error metrics (resolved in Phase 3: train to 31 Dec 2023, validate Jan 2024, test Feb 2024).

## Figures (outputs/figures/)
- `phase2_01_series.png`
- `phase2_07_weekday_profile.png`
- `phase2_08_periodogram.png`
- `phase2_02_classical_addi.png`
- `phase2_03_classical_mult.png`
- `phase2_04_stl.png`
- `phase2_05_acf_pacf_raw.png`
- `phase2_05_acf_pacf_log.png`
- `phase2_06_breaks.png`
