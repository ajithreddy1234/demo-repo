# Final Technical Report
## Machine-Learning-Assisted Irrigation Demand Prediction and Reservoir Allocation Optimization: Singur Reservoir Case Study

## 1. Objective
Develop a reproducible Water Resources Engineering decision-support framework that:

1. estimates daily grain-maize irrigation water requirement from meteorological and crop-state variables,
2. predicts that requirement using machine learning,
3. converts the predicted demand into command-area/reservoir-release units,
4. checks the scale of the result against official Singur Project characteristics, and
5. optimizes daily releases under seasonal-water and local-delivery constraints.

The project deliberately distinguishes **official reservoir data**, **physics-derived reference targets**, **machine-learning outputs**, and **scenario assumptions**.

---

## 2. Singur Reservoir case-study context

The Singur Project is a multipurpose reservoir on the Manjira River in Sangareddy district and provides a geographically consistent reservoir context for the study.

Official characteristics used:

- Gross capacity at FRL: **29.91 TMC**
- FRL: **523.60 m**
- New irrigation ayacut: **40,000 acres ≈ 16,187 ha**
- Irrigation allocation for Singur canals: **2.00 TMC**
- Canal-head discharges:
  - Left Main Canal: **13.804 cumecs**
  - Main Branch Canal: **5.479 cumecs**
  - Right Main Canal: **1.614 cumecs**
  - Combined listed head discharge: **20.897 cumecs**
- Designed live capacity: **822.460 Mm³**
- Remote-sensing assessed live capacity (2018): **812.037 Mm³**

Authoritative sources are listed in `docs/SOURCES.md`.

---

## 3. Study configuration

- Study point: Sangareddy district, Telangana
- Crop: Grain maize
- Seasons: 2021–2025
- Crop duration: 125 days per season
- Assumed sowing date: 15 June
- Daily crop-season records: 625
- Missing cells after processing: 0
- Historical weather source: NASA POWER
- Crop-water framework: FAO-56 Penman–Monteith + crop coefficient approach

A single crop/location design was chosen to preserve physical consistency and interpretability.

---

## 4. Weather and crop inputs

Base ML features:

- mean, maximum and minimum air temperature
- relative humidity
- wind speed at 2 m
- solar radiation
- rainfall
- surface pressure
- crop coefficient Kc
- days after sowing
- crop stage

ET0 was also tested as an engineered physics-informed feature.

ETc, effective rainfall and allocation volume were excluded from the ML feature set because they are directly downstream of target construction and would create target leakage.

---

## 5. Hydrologic target generation

Reference evapotranspiration ET0 is calculated from the FAO-56 Penman–Monteith framework.

Crop evapotranspiration:

`ETc = Kc × ET0`

Daily net irrigation water requirement:

`IWR = max(0, ETc − Peff)`

where `Peff` is effective rainfall.

The target is therefore a **physics-derived reference irrigation requirement**, not measured farmer irrigation or measured dam release.

Dataset summary:

- Mean ET0: **3.75 mm/day**
- Mean ETc: **2.94 mm/day**
- Mean IWR: **1.51 mm/day**
- Maximum IWR: **5.65 mm/day**
- Zero-IWR days: **248**

---

## 6. Machine-learning protocol

Chronological evaluation:

- Training: 2021–2023
- Validation/model selection: 2024
- Final untouched test: 2025

Models compared:

- Linear Regression
- Decision Tree
- Random Forest
- Random Forest + ET0
- XGBoost
- XGBoost + ET0
- MLP Neural Network + ET0

Random Forest and XGBoost hyperparameters were selected using **2024 validation RMSE only**.

### Validation-selected final model

**XGBoost (raw features)**

2024 validation:

- RMSE: **0.2695 mm/day**
- R²: **0.9723**

The selected model was then refit on 2021–2024 and evaluated once on 2025.

### Final 2025 test

- MAE: **0.1349 mm/day**
- RMSE: **0.2201 mm/day**
- R²: **0.9804**
- Allocation-equivalent MAE: **1.349 m³/ha/day**

XGBoost + ET0 achieved a slightly lower 2025 RMSE of 0.2121 mm/day, but it was not retroactively selected because the test season was not used for model selection.

---

## 7. Final 2025 crop-demand interpretation

For the locked XGBoost model:

- Reference seasonal net requirement: **1564.17 m³/ha**
- Predicted seasonal net requirement: **1580.74 m³/ha**
- Seasonal allocation difference: **16.57 m³/ha = 1.06%**
- Reference peak daily requirement: **50.92 m³/ha/day**
- Predicted peak daily requirement: **50.74 m³/ha/day**

The conversion is:

`Net command-area demand = 10 × IWR × crop area`

Gross release demand is obtained after accounting for overall project delivery efficiency.

---

## 8. Explainability

Top permutation-importance features for the final locked model:

1. rainfall
2. Kc
3. solar radiation
4. relative humidity
5. maximum temperature

This is physically consistent: rainfall reduces external supply need, Kc represents crop-stage water demand, and radiation/humidity/temperature influence atmospheric evaporative demand.

---

## 9. Singur scale check

The official 40,000-acre ayacut is not assumed to be entirely maize.

A **1000-ha maize block** is therefore retained as a controlled operational demonstration.

- 1000 ha = **6.18%** of the official new ayacut.
- Scenario overall delivery efficiency = **70%**.
- Predicted gross 2025 seasonal release requirement = **2,258,198 m³**.
- This is **3.99%** of the official 2.00-TMC Singur canal irrigation allocation.
- Predicted peak gross release requirement = **72,483 m³/day = 0.839 cumecs**.
- This is **4.01%** of the combined listed Singur canal-head discharge.

Therefore the smaller daily capacity used in the local bottleneck stress test is explicitly **not** Singur's actual canal-head capacity.

---

## 10. 10-day irrigation working table

The 2025 daily XGBoost predictions are aggregated into **13 operating periods**, consistent with 10-daily irrigation-planning practice.

Each period contains:

- start/end date
- number of days
- mean IWR
- total net crop demand
- total gross release requirement
- peak gross release requirement

Output:

`results/final/singur_10day_irrigation_working_table_2025.csv`

This is an **irrigation-demand working table** rather than a complete historical reservoir working table.

A full historical Singur mass-balance would additionally require measured period-wise:

- beginning storage
- inflow
- drinking/industrial release
- evaporation loss
- spill
- ending storage

Those data were not fabricated.

---

## 11. Reservoir allocation optimization

Decision variable:

`x_t = irrigation release on day t`

Constraints:

- `0 ≤ x_t ≤ demand_t`
- `x_t ≤ C`, where C is a daily local delivery/release limit
- `Σx_t ≤ W`, where W is the available seasonal irrigation-water budget

Primary objective:

**Minimize the maximum relative daily irrigation deficit.**

A second stage maximizes total delivered water while preserving the best achievable fairness level.

This converts the project from demand prediction alone into a reservoir-allocation decision-support problem.

---

## 12. Stress-test scenarios

The following values are **scenario assumptions**, not measured Singur operating records:

- maize block = 1000 ha
- overall efficiency = 0.70
- seasonal water availability = 100% or 85% of predicted full gross demand
- local delivery capacity = effectively unconstrained or a deliberately restrictive hypothetical sub-command bottleneck

| Scenario | Water-budget utilization | Minimum daily demand met | Maximum relative deficit |
|---|---:|---:|---:|
| Full supply, unconstrained | 100.00% | 100.00% | 0.00% |
| 15% seasonal shortage only | 100.00% | 85.00% | 15.00% |
| 15% shortage + local bottleneck | 76.54% | 34.89% | 65.11% |

In the combined shortage-and-bottleneck case, approximately **450,317 m³** of the allocated seasonal water remains unused because the local delivery limit prevents timely release at peak-demand periods.

### Main engineering result

**Adequate seasonal water volume does not guarantee reliable irrigation. The system must also be capable of delivering water at the required time and rate.**

---

## 13. Reproducibility

The final workflow automatically:

1. installs dependencies,
2. runs allocation/optimizer tests,
3. rebuilds the NASA weather + crop-water dataset,
4. trains and evaluates the final ML models,
5. creates the Singur scale check,
6. generates the 10-day working table,
7. runs reservoir-allocation optimization,
8. writes reports/figures/results,
9. saves the final model,
10. commits generated outputs.

---

## 14. Limitations

1. The IWR target is physics-derived rather than directly measured canal/farmer irrigation.
2. Operational future use would require forecast meteorology rather than historical observed weather.
3. Effective rainfall is an empirical approximation.
4. Soil-water storage is not dynamically observed.
5. The 1000-ha crop block and 70% efficiency are scenario assumptions.
6. The local delivery bottleneck stress test is hypothetical and not the actual Singur canal-head capacity.
7. A complete historical Singur operation model requires real inflow, beginning storage, competing releases, evaporation, spill and operating-rule data.
8. Those missing operational time series were not fabricated.

---

## 15. Final conclusion

The project now forms a coherent Water Resources Engineering + AI decision-support chain:

**FAO-56 crop-water physics → XGBoost irrigation-demand prediction → Singur scale check → 10-day irrigation planning → constrained reservoir-release optimization.**

The validation-selected XGBoost model achieved **R² = 0.9804** and **RMSE = 0.2201 mm/day** on the untouched 2025 season, while the reservoir-allocation analysis demonstrates that irrigation reliability depends on both **water availability** and **timely delivery capacity**.

This is the final project scope; further development should focus on measured Singur operating data and forecast-weather integration rather than adding additional ML algorithms for their own sake.
