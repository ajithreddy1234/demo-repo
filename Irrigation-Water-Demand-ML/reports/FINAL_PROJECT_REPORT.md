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
All candidate models are shown for comparison, but the deployed/final model was locked using 2024 validation before consulting these test results.
               model features  train_rows  test_rows  MAE_mm_day  RMSE_mm_day       R2  MAE_m3_per_ha_day  RMSE_m3_per_ha_day
       XGBoost + ET0  raw+ET0         500        125    0.125429     0.212114 0.981809           1.254285            2.121140
             XGBoost      raw         500        125    0.134860     0.220123 0.980410           1.348598            2.201226
 Random Forest + ET0  raw+ET0         500        125    0.156771     0.271150 0.970274           1.567715            2.711501
       Random Forest      raw         500        125    0.162584     0.287631 0.966551           1.625837            2.876308
Neural Network + ET0  raw+ET0         500        125    0.208770     0.317750 0.959179           2.087700            3.177499
       Decision Tree      raw         500        125    0.274342     0.477218 0.907923           2.743417            4.772180
   Linear Regression      raw         500        125    0.741827     0.973691 0.616683           7.418271            9.736907

## Selected final model
- Model: XGBoost
- Feature set: raw
- MAE: 0.1349 mm/day
- RMSE: 0.2201 mm/day
- R2: 0.9804
- Allocation-equivalent MAE: 1.349 m3/ha/day

## 2025 allocation interpretation
- Reference seasonal net requirement: 1564.17 m3/ha
- Predicted seasonal net requirement: 1580.74 m3/ha
- Seasonal allocation difference: 16.57 m3/ha (1.06%)
- Reference peak daily requirement: 50.92 m3/ha/day
- Predicted peak daily requirement: 50.74 m3/ha/day


## Reservoir-constrained allocation optimization

To move beyond demand prediction, the final project adds a linear-programming reservoir release optimizer.

The optimizer uses the validation-selected XGBoost daily IWR predictions and converts them to gross release demand using crop area and overall delivery efficiency. Daily release is constrained by:

1. the predicted gross irrigation demand for that day,
2. a daily release/canal-capacity limit,
3. the total seasonal irrigation-water budget.

The primary optimization minimizes the maximum relative daily deficit over the season. A second stage maximizes total delivered water while preserving the best achievable fairness level.

### Stress-test scenario definition
The optimization demonstration uses:
- maize area = 1000 ha,
- overall delivery efficiency = 0.70,
- either 100% or 85% of full seasonal gross requirement,
- either an effectively unconstrained daily capacity or a cap of 1.4 times mean daily gross demand.

These are explicit stress-test assumptions, not measured operating-reservoir inputs.

### Full predicted gross seasonal requirement
For the 2025 predicted maize demand:

**2,258,198.07 m3**

### Scenario comparison

| Scenario | Available seasonal water | Optimized release | Water-budget utilization | Minimum daily demand met | Maximum relative deficit |
|---|---:|---:|---:|---:|---:|
| Full supply, unconstrained release | 2,258,198 m3 | 2,258,198 m3 | 100.00% | 100.00% | 0.00% |
| 15% seasonal shortage only | 1,919,468 m3 | 1,919,468 m3 | 100.00% | 85.00% | 15.00% |
| 15% shortage + daily release cap | 1,919,468 m3 | 1,469,152 m3 | 76.54% | 34.89% | 65.11% |

In the combined shortage-and-capacity case, **450,316.67 m3** of the allocated seasonal water remains unused. The reason is not lack of seasonal storage allocation; it is the inability to deliver enough water on peak-demand days because the daily release-capacity constraint becomes binding.

This is an important water-resources result: reservoir adequacy depends not only on total seasonal water availability, but also on whether the canal/release system can deliver water at the rate demanded by the crop.

### Why this upgrade matters
The project now answers two different questions:

1. **Prediction:** How much irrigation water will the crop require?
2. **Decision support:** Given limited seasonal water and delivery capacity, how should reservoir releases be scheduled?

This turns the work from a regression-only study into a combined ML and optimization framework for irrigation allocation.

## Most influential final-model inputs
Top five permutation-importance features: rainfall_mm, kc, solar_radiation_mj_m2_day, relative_humidity_pct, temp_max_c

## Growth-stage performance
 crop_stage  days  mean_reference_iwr_mm  mean_predicted_iwr_mm  MAE_mm_day
development    35               1.486874               1.393640    0.187536
    initial    20               0.978666               0.992459    0.095111
late_season    30               0.897406               1.020227    0.161456
 mid_season    40               1.447020               1.471012    0.088695

## Limitations
1. The target is physics-derived rather than measured farmer/canal irrigation.
2. Operational future use requires forecast weather rather than historical observed weather.
3. Effective rainfall is empirical, not directly measured field infiltration.
4. Soil moisture, runoff, deep percolation and canal losses are not directly observed.
5. A genuine PINN is not claimed because measured soil-water states and defensible boundary/initial conditions are unavailable.
6. The optimization demonstration uses scenario inputs for area, efficiency, seasonal water budget and release capacity. Real deployment requires measured reservoir storage/allocation, inflows, command-area acreage, canal capacity, delivery efficiencies and operating rules.

## Conclusion
The completed framework now links weather and crop state to irrigation-demand prediction and then to constrained reservoir-release optimization. It demonstrates both accurate 2025 demand estimation and the operational effect of seasonal shortage and daily conveyance limits. The most important next step beyond this term project is validation against measured irrigation/canal releases and replacement of stress-test inputs with real reservoir storage, inflow, command-area and operating-rule data.