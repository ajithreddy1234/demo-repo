# Week 2 Progress Report

## Study setup
- Location: Sangareddy, Telangana, India
- Coordinates: 17.624775, 78.086686
- Crop: Grain maize
- Seasons: 2021-2025
- Sowing date assumption: 15 June
- Crop duration: 125 days
- Weather source: NASA POWER Daily Point API

## Data pipeline completed
1. Downloaded daily meteorological data for 2021-2025.
2. Checked NASA missing-value sentinel and chronological completeness.
3. Verified POWER shortwave radiation units from API metadata (MJ/m²/day).
4. Calculated FAO-56 reference evapotranspiration (ET0).
5. Generated a 125-day maize Kc curve for each season.
6. Calculated ETc = Kc × ET0.
7. Estimated monthly effective rainfall using FAO empirical equations and distributed it to rainy days proportional to observed daily rainfall.
8. Generated daily reference irrigation requirement: IWR = max(0, ETc - Peff).
9. Built the processed ML dataset.
10. Ran EDA and preliminary regressions.

## Dataset results
- Processed rows: 625
- Crop seasons: 5
- Missing cells in processed dataset: 0
- Mean crop-season temperature: 25.22 °C
- Total crop-season rainfall across five seasons: 4134.12 mm
- Mean ET0: 3.75 mm/day
- Mean ETc: 2.94 mm/day
- Mean derived IWR: 1.51 mm/day
- Median derived IWR: 0.85 mm/day
- Maximum derived IWR: 5.65 mm/day
- Zero-IWR days: 248

## Preliminary ML protocol
- Training: 2021-2023 (375 rows)
- Validation: 2024 (125 rows)
- Final test: 2025, intentionally untouched in Week 2
- ML target: irrigation_requirement_mm
- Allocation output: net_allocation_m3_per_ha_day = 10 × irrigation_requirement_mm

## Best Week 2 validation model
- Model: Random Forest + ET0
- MAE: 0.178 mm/day
- RMSE: 0.298 mm/day
- R²: 0.966

## Important interpretation
The target is a physics-derived crop irrigation requirement produced from FAO crop-water equations and historical meteorology. Its intended use is reservoir/dam water-allocation support: predicted depth is converted to net volume per hectare. It is not measured farmer irrigation, and gross reservoir release would additionally require commanded crop area plus conveyance/application-efficiency information. Week 3 will tune/finalize models and evaluate exactly once on the untouched 2025 season.
