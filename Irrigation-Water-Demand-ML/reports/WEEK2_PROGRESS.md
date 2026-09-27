# Week 2 Progress Report

## Study setup
- Location: Sangareddy, Telangana, India
- Coordinates: 17.624775, 78.086686
- Crop: Grain maize
- Seasons: 2021–2025
- Sowing date assumption: 15 June
- Crop duration: 125 days
- Weather source: NASA POWER Daily Point API

## Data pipeline completed
1. Downloaded daily meteorological data for 2021–2025.
2. Checked NASA missing-value sentinel and chronological completeness.
3. Converted POWER shortwave radiation to MJ/m²/day.
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
- Mean ET0: 11.08 mm/day
- Mean ETc: 8.95 mm/day
- Mean derived IWR: 6.36 mm/day
- Median derived IWR: 5.08 mm/day
- Maximum derived IWR: 20.84 mm/day
- Zero-IWR days: 138

## Preliminary ML protocol
- Training: 2021–2023 (375 rows)
- Validation: 2024 (125 rows)
- Final test: 2025, intentionally untouched in Week 2
- Target: irrigation_requirement_mm

## Best Week 2 validation model
- Model: Random Forest + ET0
- MAE: 0.877 mm/day
- RMSE: 1.330 mm/day
- R²: 0.936

## Important interpretation
The target is a physics-derived reference irrigation requirement produced from FAO crop-water equations and historical meteorology. It is not measured farmer irrigation. Week 3 will tune/finalize models and evaluate exactly once on the untouched 2025 season.
