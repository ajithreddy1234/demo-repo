# Machine-Learning-Based Irrigation Demand Prediction for Reservoir Allocation

This CE4400 Water Resources Engineering project builds a reproducible pipeline that estimates daily net irrigation requirement for grain maize from weather and crop-state variables, then converts the prediction into allocation-ready water volume per hectare.

## Final project scope

Study area: Sangareddy, Telangana, India  
Crop: Grain maize  
Period: 2021-2025  
Resolution: Daily crop-season records  
Dataset size: 625 rows  
Target: irrigation_requirement_mm

The target is a physics-derived reference irrigation requirement. It is not measured farmer irrigation or measured dam release.

## End-to-end workflow

1. Extract daily historical weather from NASA POWER.
2. Verify units and clean the meteorological data.
3. Build a 125-day maize crop calendar and stage-dependent Kc curve.
4. Calculate FAO-56 reference evapotranspiration ET0.
5. Calculate ETc = Kc x ET0.
6. Estimate effective rainfall.
7. Generate daily IWR = max(0, ETc - Peff).
8. Convert IWR to net allocation using 1 mm over 1 ha = 10 m3.
9. Train and compare regression models.
10. Select the final model using 2024 validation only.
11. Refit on 2021-2024 and evaluate once on the untouched 2025 season.
12. Generate final model, plots, feature importance, stage-wise analysis and allocation summaries.

## Inputs used for ML

temp_mean_c, temp_max_c, temp_min_c, relative_humidity_pct, wind_speed_2m_m_s, solar_radiation_mj_m2_day, rainfall_mm, surface_pressure_kpa, kc, days_after_sowing, crop_stage

ET0 was also tested as an engineered physics-informed feature in separate model variants.

ETc, effective rainfall and allocation volume are not used as ML inputs because they are directly downstream of target construction.

## Models compared

- Linear Regression
- Decision Tree
- Random Forest
- Random Forest + ET0
- XGBoost
- XGBoost + ET0
- MLP Neural Network + ET0

Random Forest and XGBoost hyperparameters are selected using 2024 validation RMSE.

## Final model selection

The model was locked before consulting the 2025 final test.

Selected model: XGBoost  
Selection basis: lowest 2024 validation RMSE

2024 validation:
- RMSE: 0.2695 mm/day
- R2: 0.9723

After selection, XGBoost was refit on 2021-2024 and tested on 2025.

## Final 2025 test result for the locked model

- MAE: 0.1349 mm/day
- RMSE: 0.2201 mm/day
- R2: 0.9804
- Allocation-equivalent MAE: 1.349 m3/ha/day

XGBoost + ET0 obtained a slightly lower 2025 RMSE of 0.2121 mm/day, but it is not retroactively declared the final model because the final model was already selected from 2024 validation.

## Reservoir-allocation result

For the 2025 maize season:

- Reference seasonal net irrigation requirement: 156.42 mm
- Predicted seasonal net irrigation requirement: 158.07 mm
- Reference seasonal net allocation: 1564.17 m3/ha
- Predicted seasonal net allocation: 1580.74 m3/ha
- Seasonal allocation difference: 16.57 m3/ha
- Seasonal allocation difference: 1.06%
- Reference peak daily allocation: 50.92 m3/ha/day
- Predicted peak daily allocation: 50.74 m3/ha/day

For a known cultivated area A in hectares:

Net command-area demand = 10 x predicted IWR x A

Gross reservoir release additionally requires real conveyance and field-application efficiencies.

## Most influential inputs

Permutation importance on the locked final model identified:

1. rainfall_mm
2. kc
3. solar_radiation_mj_m2_day
4. relative_humidity_pct
5. temp_max_c

This is physically sensible: rainfall directly reduces external irrigation need, while Kc represents crop-stage demand and radiation/humidity/temperature control atmospheric water demand.

## Repository structure

- config/project_config.json: study area and crop configuration
- data/raw/: downloaded NASA POWER data and metadata
- data/processed/: final 625-row ML dataset
- docs/DATA_DICTIONARY.md: variables, units and roles
- docs/METHODOLOGY.md: technical methodology
- src/build_week2.py: data and hydrology pipeline
- src/build_final.py: tuning, final testing and plots
- src/allocation_from_iwr.py: converts IWR to command-area/reservoir allocation
- results/week2/: preliminary results
- results/final/: final metrics, predictions, feature importance and allocation summaries
- figures/week2/: data-development figures
- figures/final/: final model and allocation figures
- models/best_model.joblib: serialized validation-selected model
- models/best_model_metadata.json: model and feature metadata
- reports/FINAL_PROJECT_REPORT.md: complete project summary
- tests/test_allocation.py: allocation sanity tests

## Reproducibility

The final GitHub Actions workflow:

1. installs dependencies,
2. runs sanity tests,
3. rebuilds the weather/crop-water dataset,
4. trains and evaluates the final models,
5. commits generated outputs back to the repository.

The final workflow currently completes successfully.

## Important limitations

- The target is derived from FAO-based physics rather than observed farmer/canal irrigation.
- Historical observed weather is used; operational forecasting would require forecast weather.
- Effective rainfall is an empirical approximation.
- Soil moisture, deep percolation, runoff and actual canal losses are not measured.
- The current output is net crop demand. Actual dam release needs real command-area acreage and delivery efficiencies.
- A genuine PINN is not claimed because the project does not have measured soil-water states and defensible boundary/initial conditions for a governing differential equation.

## Final interpretation

This is a compact decision-support prototype for demand-based reservoir irrigation planning. It predicts crop irrigation requirement, translates that prediction into allocation units, and preserves a clear boundary between modeled crop demand and actual operational dam release.
