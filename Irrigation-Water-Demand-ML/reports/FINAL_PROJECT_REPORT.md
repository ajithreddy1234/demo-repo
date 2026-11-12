# Final Project Report - Irrigation Water Demand ML

## Project objective
Develop a reproducible machine-learning framework that estimates daily net irrigation water requirement for grain maize from meteorological and crop-state variables, then converts the predicted depth into allocation-ready water volume per hectare for reservoir/dam irrigation planning.

## Study configuration
- Study point: Sangareddy, Telangana, India
- Crop: Grain maize
- Seasons: 2021-2025
- Daily crop-season records: 625
- Crop duration: 125 days per season
- Historical weather source: NASA POWER
- Crop-water framework: FAO-56 Penman-Monteith plus crop coefficient approach

## Hydrologic target
ET0 is calculated from meteorological variables. Crop evapotranspiration is ETc = Kc x ET0. Effective rainfall is subtracted to obtain IWR = max(0, ETc - Peff).
The target is therefore a physics-derived reference irrigation requirement, not directly measured farmer irrigation or dam release.

## Reservoir allocation conversion
Net allocation (m3/ha/day) = 10 x IWR (mm/day).
For crop area A in hectares: net command-area demand = 10 x IWR x A.
Gross release additionally requires actual conveyance and application efficiencies. No undocumented area or efficiency has been invented.

## ML protocol
- Training: 2021-2023
- Validation/model selection: 2024
- Final untouched test: 2025
- Models: Linear Regression, Decision Tree, Random Forest, XGBoost, Neural Network
- Random Forest and XGBoost hyperparameters selected using 2024 validation RMSE only
- Final models refit on 2021-2024 and tested once on 2025

## Best validation candidate
- Model: XGBoost
- Validation RMSE: 0.2695 mm/day
- Validation R2: 0.9723

## Final 2025 test results
               model features  train_rows  test_rows  MAE_mm_day  RMSE_mm_day       R2  MAE_m3_per_ha_day  RMSE_m3_per_ha_day
       XGBoost + ET0  raw+ET0         500        125    0.125429     0.212114 0.981809           1.254285            2.121140
             XGBoost      raw         500        125    0.134860     0.220123 0.980410           1.348598            2.201226
 Random Forest + ET0  raw+ET0         500        125    0.156771     0.271150 0.970274           1.567715            2.711501
       Random Forest      raw         500        125    0.162584     0.287631 0.966551           1.625837            2.876308
Neural Network + ET0  raw+ET0         500        125    0.208770     0.317750 0.959179           2.087700            3.177499
       Decision Tree      raw         500        125    0.274342     0.477218 0.907923           2.743417            4.772180
   Linear Regression      raw         500        125    0.741827     0.973691 0.616683           7.418271            9.736907

## Selected final model
- Model: XGBoost + ET0
- Feature set: raw+ET0
- MAE: 0.1254 mm/day
- RMSE: 0.2121 mm/day
- R2: 0.9818
- Allocation-equivalent MAE: 1.254 m3/ha/day

## 2025 allocation interpretation
- Reference seasonal net requirement: 1564.17 m3/ha
- Predicted seasonal net requirement: 1593.13 m3/ha
- Seasonal allocation difference: 28.96 m3/ha (1.85%)
- Reference peak daily requirement: 50.92 m3/ha/day
- Predicted peak daily requirement: 50.46 m3/ha/day

## Most influential final-model inputs
Top five permutation-importance features: rainfall_mm, kc, eto_mm_day, solar_radiation_mj_m2_day, relative_humidity_pct

## Growth-stage performance
 crop_stage  days  mean_reference_iwr_mm  mean_predicted_iwr_mm  MAE_mm_day
development    35               1.486874               1.430889    0.166628
    initial    20               0.978666               0.993448    0.072655
late_season    30               0.897406               0.997684    0.167892
 mid_season    40               1.447020               1.485798    0.083918

## Limitations
1. The target is physics-derived rather than measured farmer/canal irrigation.
2. Operational future use requires forecast weather rather than historical observed weather.
3. Effective rainfall is empirical, not directly measured field infiltration.
4. Soil moisture, runoff, deep percolation and canal losses are not directly observed.
5. A genuine PINN is not claimed because measured soil-water states and defensible boundary/initial conditions are unavailable.
6. Current output is net crop demand. Gross dam release requires real command-area acreage and documented delivery efficiencies.

## Conclusion
The completed framework provides a compact, reproducible decision-support prototype linking weather and crop state to irrigation demand and reservoir-allocation units. The most important next step beyond this term project is validation against measured irrigation/canal data and coupling with forecast weather and real command-area information.