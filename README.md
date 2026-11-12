# CE4400 Smart Irrigation Demand and Reservoir Allocation Project

The completed project is located in:

`Irrigation-Water-Demand-ML/`

It combines:

- FAO-based crop-water requirement generation,
- machine-learning prediction of daily maize irrigation requirement,
- chronological model validation and untouched 2025 testing,
- conversion of predicted demand into reservoir-release units,
- reservoir-constrained release optimization under seasonal-water and daily-capacity limits.

Validation-selected final model: XGBoost

Final 2025 test:
- MAE: 0.1349 mm/day
- RMSE: 0.2201 mm/day
- R2: 0.9804

Prominent final upgrade:
- reservoir allocation optimizer using linear programming,
- full-supply, seasonal-shortage and release-capacity stress tests,
- daily optimized release schedule and reservoir-allocation diagnostics.

See `Irrigation-Water-Demand-ML/README.md` for the full methodology, verified results, figures, limitations and repository structure.
