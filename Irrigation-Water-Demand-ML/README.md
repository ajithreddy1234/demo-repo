# Machine-Learning-Assisted Irrigation Demand Prediction and Reservoir Allocation Optimization

## Singur Reservoir Case Study — CE4400 Water Resources Engineering

This project combines **crop-water physics, machine learning and reservoir-allocation optimization** to estimate daily maize irrigation requirement and translate that demand into an operational water-allocation problem.

The study is anchored to **Singur Reservoir, Sangareddy district, Telangana** as a real water-resources case-study context. Official reservoir characteristics are used for scale and plausibility checks. Where historical operational time series are unavailable, the project uses clearly labelled stress-test scenarios rather than fabricated reservoir data.

---

## 1. Core research question

**How much irrigation water does maize require under changing weather and crop conditions, and how should a limited irrigation-water allocation be distributed through the crop season?**

The project therefore has two linked layers:

1. **Prediction** — estimate daily net irrigation water requirement, IWR (mm/day).
2. **Decision support** — convert predicted demand into reservoir-release demand and optimize releases under water-budget and delivery-capacity constraints.

---

## 2. Why Singur Reservoir?

The weather/crop study point is in Sangareddy district, and Singur is a real multipurpose project on the Manjira River in the same district. This gives the ML study a defensible Water Resources Engineering context instead of using an abstract reservoir.

Official project characteristics used in the case-study scale check:

- Gross capacity at FRL: **29.91 TMC**
- FRL: **523.60 m**
- New irrigation ayacut: **40,000 acres ≈ 16,187 ha**
- Irrigation allocation for Singur canals in the official project note: **2.00 TMC**
- Canal-head discharges:
  - Left Main Canal: **13.804 cumecs**
  - Main Branch Canal: **5.479 cumecs**
  - Right Main Canal: **1.614 cumecs**
  - Combined listed head discharge: **20.897 cumecs**
- Designed live capacity: **822.460 Mm³**
- Remote-sensing assessed live capacity (2018): **812.037 Mm³**

See `docs/SOURCES.md` and `reports/SINGUR_CASE_STUDY.md`.

---

## 3. Study design

- Crop: **Grain maize**
- Assumed sowing date: **15 June**
- Crop duration: **125 days**
- Period: **2021–2025**
- Seasons: **5**
- Daily crop-season observations: **625**
- Missing cells after processing: **0**
- Weather source: **NASA POWER Daily Point API**

A single crop and location were deliberately used so the physical and ML relationships remain consistent and interpretable.

---

## 4. Input variables

Base ML inputs:

`temp_mean_c, temp_max_c, temp_min_c, relative_humidity_pct, wind_speed_2m_m_s, solar_radiation_mj_m2_day, rainfall_mm, surface_pressure_kpa, kc, days_after_sowing, crop_stage`

`eto_mm_day` is tested separately as a physics-informed engineered feature.

Variables **not** used as ML inputs:

- `etc_mm_day`
- `effective_rain_mm`
- `net_allocation_m3_per_ha_day`

These are directly downstream of target construction and would create leakage.

---

## 5. Crop-water physics

Reference evapotranspiration is calculated using the FAO-56 Penman–Monteith framework.

Crop evapotranspiration:

`ETc = Kc × ET0`

Net irrigation water requirement:

`IWR = max(0, ETc − Peff)`

where `Peff` is effective rainfall.

Rainfall is therefore an **input that reduces reservoir irrigation demand**, not the prediction target.

Dataset summary:

- Mean ET0: **3.75 mm/day**
- Mean ETc: **2.94 mm/day**
- Mean IWR: **1.51 mm/day**
- Maximum IWR: **5.65 mm/day**
- Zero-IWR days: **248 / 625**

---

## 6. Machine-learning protocol

Chronological split:

- Training: **2021–2023**
- Validation/model selection: **2024**
- Final untouched test: **2025**

Models compared:

- Linear Regression
- Decision Tree
- Random Forest
- Random Forest + ET0
- XGBoost
- XGBoost + ET0
- MLP Neural Network + ET0

Random Forest and XGBoost hyperparameters are selected using **2024 validation RMSE only**.

The final model is locked before consulting the 2025 test season.

### Validation-selected final model

**XGBoost (raw feature set)**

2024 validation:

- RMSE: **0.2695 mm/day**
- R²: **0.9723**

After selection, XGBoost is refit on 2021–2024 and evaluated once on 2025.

### Final 2025 test

- MAE: **0.1349 mm/day**
- RMSE: **0.2201 mm/day**
- R²: **0.9804**
- Allocation-equivalent MAE: **1.349 m³/ha/day**

XGBoost + ET0 happens to obtain a slightly lower 2025 RMSE of 0.2121 mm/day, but it is not retroactively declared the final model because test data are not used for model selection.

---

## 7. 2025 irrigation-allocation result

For the validation-selected XGBoost model:

- Reference seasonal net irrigation requirement: **156.42 mm**
- Predicted seasonal net irrigation requirement: **158.07 mm**
- Reference net allocation: **1564.17 m³/ha**
- Predicted net allocation: **1580.74 m³/ha**
- Seasonal allocation difference: **16.57 m³/ha = 1.06%**
- Reference peak daily allocation: **50.92 m³/ha/day**
- Predicted peak daily allocation: **50.74 m³/ha/day**

Conversion:

`Net command-area demand (m³/day) = 10 × IWR (mm/day) × crop area (ha)`

Gross release demand additionally requires project delivery efficiency.

---

## 8. Singur case-study scale check

The official 40,000-acre ayacut is **not assumed to be entirely maize**.

The project therefore keeps a controlled **1000-ha maize block** for the operational demonstration.

- 1000 ha = **6.18%** of the official new ayacut.
- Scenario overall delivery efficiency: **70%**.
- Predicted 2025 gross seasonal release requirement: **2,258,198 m³**.
- This is only **3.99%** of the official 2.00-TMC Singur-canal irrigation allocation.
- Predicted peak gross release requirement: **72,483 m³/day = 0.839 cumecs**.
- This is **4.01%** of the combined listed Singur canal-head discharge.

Therefore the smaller 25,292 m³/day capacity used in one optimizer stress test is **not** presented as actual Singur canal-head capacity. It represents a hypothetical local/sub-command bottleneck.

---

## 9. CWC-style 10-day irrigation working table

The final 2025 daily XGBoost demand predictions are aggregated into **13 operating periods**.

The working table contains:

- period number
- start and end date
- number of days
- mean predicted IWR
- total net crop demand
- total gross release requirement
- peak gross daily release requirement

File:

`results/final/singur_10day_irrigation_working_table_2025.csv`

This follows the 10-daily/monthly planning logic used in CWC irrigation-planning guidance.

A complete historical Singur reservoir working table would also require measured:

- beginning storage
- period inflow
- drinking/industrial releases
- evaporation loss
- spill
- ending storage

Those time series are **not fabricated** in this project.

---

## 10. Reservoir-constrained release optimization

Decision variable:

`x_t = reservoir irrigation release on day t`

Constraints:

- `0 ≤ x_t ≤ daily gross irrigation demand`
- `x_t ≤ daily delivery/release capacity C`
- `Σx_t ≤ seasonal irrigation-water budget W`

Primary objective:

**Minimize the maximum relative daily irrigation deficit.**

A second optimization stage maximizes total water delivered while preserving the best achievable fairness level.

This is intentionally different from applying an arbitrary percentage reduction to every day.

---

## 11. Reservoir stress-test scenarios

All values below are **scenario assumptions**, not measured operating records.

Common demonstration assumptions:

- Maize area: **1000 ha**
- Overall delivery efficiency: **0.70**

| Scenario | Water-budget use | Minimum daily demand met | Maximum relative deficit |
|---|---:|---:|---:|
| Full supply, unconstrained release | 100.00% | 100.00% | 0.00% |
| 15% seasonal shortage only | 100.00% | 85.00% | 15.00% |
| 15% shortage + hypothetical local delivery bottleneck | 76.54% | 34.89% | 65.11% |

In the combined shortage-and-local-capacity case, approximately **450,317 m³** of the allocated seasonal water remains unused because the hypothetical sub-command delivery limit prevents water from reaching the crop at the required peak rate.

### Main engineering insight

**Seasonal water availability alone does not guarantee irrigation reliability. Water must also be deliverable at the required time and rate.**

---

## 12. Explainability

Top permutation-importance variables for the locked final model:

1. rainfall
2. Kc
3. solar radiation
4. relative humidity
5. maximum temperature

This is physically consistent with the target-generation framework:

- rainfall offsets external irrigation need,
- Kc represents crop-stage demand,
- radiation, humidity and temperature govern atmospheric evaporative demand.

---

## 13. End-to-end workflow

1. Download historical NASA POWER weather.
2. Verify units and clean data.
3. Generate maize crop calendar and Kc curve.
4. Calculate FAO-56 ET0.
5. Calculate ETc and effective rainfall.
6. Generate reference daily IWR.
7. Build the 625-row ML dataset.
8. Train and validate regression models chronologically.
9. Lock XGBoost using 2024 validation.
10. Evaluate once on untouched 2025 data.
11. Convert predicted IWR to m³/ha and command-area demand.
12. Check the scenario scale against official Singur project characteristics.
13. Aggregate demand into 10-day operating periods.
14. Optimize release schedules under seasonal-water and local-delivery constraints.
15. Generate figures, reports, model files and working tables automatically.

---

## 14. Repository structure

- `config/project_config.json` — study configuration
- `data/raw/` — NASA POWER raw data and metadata
- `data/processed/` — final crop-season ML dataset
- `docs/DATA_DICTIONARY.md` — variables, units and roles
- `docs/METHODOLOGY.md` — methodology
- `docs/SOURCES.md` — authoritative public sources
- `src/build_week2.py` — weather + hydrology pipeline
- `src/build_final.py` — ML tuning and final test
- `src/allocation_from_iwr.py` — irrigation-depth to volume conversion
- `src/reservoir_optimizer.py` — constrained release optimizer
- `src/singur_case_study.py` — official Singur scale check + 10-day table
- `results/final/` — final metrics and operating tables
- `figures/final/` — final figures
- `models/best_model.joblib` — saved validation-selected model
- `reports/FINAL_PROJECT_REPORT.md` — final technical report
- `reports/SINGUR_CASE_STUDY.md` — Singur case-study note
- `reports/RESERVOIR_OPTIMIZATION.md` — optimization note
- `tests/` — allocation and optimization sanity tests

The complete final GitHub Actions workflow rebuilds the dataset, runs tests, evaluates the ML models, creates the Singur case-study outputs, runs the optimizer and commits the generated results.

---

## 15. Limitations

- IWR is physics-derived rather than directly measured farmer/canal irrigation.
- Historical observed weather is used; operational forecasting would require weather forecasts.
- Effective rainfall is an empirical approximation.
- Soil-water storage is not dynamically observed.
- The 1000-ha maize block and 70% efficiency are scenario assumptions.
- The local bottleneck stress test is hypothetical and is not Singur's actual canal-head capacity.
- Historical Singur inflow/storage/drinking-release/evaporation/spill time series were not available in the project and were not fabricated.
- A real operating-policy model would require those observed reservoir records and operating rules.

---

## Final contribution

This project is a **reproducible Water Resources Engineering decision-support framework** that links:

**FAO-56 crop-water physics → machine-learning irrigation-demand prediction → real-reservoir scale checking → 10-day irrigation planning → constrained reservoir-allocation optimization.**

Its central engineering conclusion is simple but important:

> Accurate demand prediction is useful only when reservoir water is both available **and deliverable at the time the crop requires it**.
