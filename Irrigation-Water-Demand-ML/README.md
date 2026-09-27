# Irrigation Water Demand ML — CE4400

Machine-learning-based estimation of daily irrigation water requirement for grain maize using historical meteorological data and FAO crop-water methodology.

## Study configuration
- Study point: Sangareddy, Telangana, India
- Coordinates: 17.624775 N, 78.086686 E
- Crop: Grain maize
- Seasons: 2021–2025
- Assumed sowing date: 15 June each year
- Crop duration: 125 days (20 initial + 35 development + 40 mid-season + 30 late-season)
- Weather source: NASA POWER daily point API
- Crop-water method: FAO-56 Penman-Monteith + single crop coefficient method
- Effective rainfall: FAO monthly empirical method, allocated to rainy days in proportion to daily rainfall

## Week 2 scope
1. Download daily weather data.
2. Clean and validate meteorological variables.
3. Build the maize crop calendar and Kc curve.
4. Calculate daily ET0, ETc, effective rainfall and irrigation water requirement.
5. Create a reproducible processed ML dataset.
6. Run EDA.
7. Train preliminary regression baselines on 2021–2023 and evaluate on 2024.
8. Keep 2025 untouched for the final-week test.

## Target
The regression target is `irrigation_requirement_mm` (mm/day). It is a physics-derived reference irrigation requirement, not measured farmer irrigation.

## Reproducibility
Pushing changes to the project source/config files triggers the GitHub Actions Week 2 pipeline. Generated CSVs, figures and the Week 2 report are committed back under this project folder.
