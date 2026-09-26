# Cinema audience demand forecasting

Daily total ticket demand (booknow + cinePOS, overlap removed) for 29 Jan 2023 – 28 Feb 2024:
data merge, EDA, 13 forecasting models, forecast combination, theater survival analysis, and a classical
case-study view (seasonal index, ANOVA, trend + dummy regression, SPSS-style fit statistics).
Write-up: `outputs/reports/project_report.md` (structured like the L&T spare-parts case in `ppt/`).

## Setup and run
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_pipeline.py --phase all      # or --phase 1 | 2 | 3 | 4 | 5 | 6   (~4 min for all)
.venv\Scripts\python -m pytest -q                     # 14 regression tests
```
Raw CSVs are read from `./raw/`.

## Structure
| Phase | Code (`src/`) | Main outputs (`outputs/`) |
|---|---|---|
| 1 Merge | `loading`, `bookings`, `theaters`, `calendar_features`, `survival_extract`, `phase1_merge` | `excel/combined_data.xlsx`, `excel/theater_survival.xlsx` |
| 2 EDA | `eda`, `phase2_eda` | `reports/phase2_eda_summary.md`, `excel/eda_results.xlsx`, `figures/phase2_*` |
| 3 Models | `models`, `diagnostics`, `phase3_models` | `excel/model_inference_tables.xlsx`, `figures/phase3_*` |
| 4 Combination | `combination`, `phase4_combination` | `excel/forecast_combination.xlsx`, `figures/phase4_*` |
| 5 Survival | `survival_analysis`, `phase5_survival` | `excel/survival_analysis.xlsx`, `figures/phase5_*` |
| 6 Classical view | `phase6_classical` | `excel/classical_analysis.xlsx`, `figures/phase6_*` |

Shared code: `config.py` (paths and constants), `quality_log.py`, `data_access.py`.

## Key decisions (all logged in `combined_data.xlsx → data_quality_log`)
- Dates are ISO format, not DD-MM-YYYY. They're parsed with an explicit format; `dayfirst=True` would corrupt most rows.
- Target = booked + sold − 11,335 tickets that appear identically in both systems.
- booknow outage (84 days, Jul–Oct 2023): booknow set to 0 and flagged with `booknow_missing`.
- The first 28 days are excluded from modelling, because bookings made before the data start are missing.
- Split: train to 31 Dec 2023, validate Jan 2024, test Feb 2024. Univariate models are fitted on December-adjusted demand (surge multiplier 2.50×).
- Combination members: SARIMAX_no_lead, SARIMA, Holt-Winters multiplicative, MLR_no_lead. Bates-Granger was chosen as the final method before looking at the test results.
- Survival: a theater is inactive if it has no booking in the final 28 days. Cox models are fitted with statsmodels PHReg, because lifelines' CoxPHFitter fails to converge on this data.

## Headline results
- Weekly seasonality dominates: Saturday ≈ 2.1× and Tuesday ≈ 0.6× the weekly mean. Model log demand, with D = 1.
- Weekday seasonal index: Sat 171%, Sun 154%, Tue 58%; weekday ANOVA F = 14.7 (p < 0.001), 78.3 once the monthly level is removed.
- Trend + weekday-dummy regression (reduced: trend, Tue, Wed, Sat, Sun) has the best validation MAPE (23.1%) of the 13 models.
- Best single model on test: SARIMAX_no_lead (RMSE 4,584). The simple average (4,601, best MAPE 13.8%) and Bates-Granger (4,800) are statistically tied with it (Diebold-Mariano p > 0.7). Regression-based (Granger-Ramanathan) weights overfit.
- 95.3% of theaters are still active one year after their first booking. booknow theaters have ~9× the hazard (concentrated early, so proportional hazards is violated); theaters without metadata have 2.9×.
