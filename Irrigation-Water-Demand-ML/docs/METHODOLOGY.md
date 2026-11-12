# Methodology

## 1. Study definition
- Location: Sangareddy, Telangana, India
- Crop: grain maize
- Daily crop seasons: 2021-2025
- Assumed sowing date: 15 June
- Crop duration: 125 days

## 2. Weather acquisition
Daily meteorological data are requested from the NASA POWER point API using the study coordinates. API metadata are stored alongside the raw data so units can be verified rather than assumed.

## 3. Crop-water calculations
Reference evapotranspiration is calculated with the FAO-56 Penman-Monteith formulation. A stage-dependent maize crop coefficient Kc is applied:

ETc = Kc x ET0

Effective rainfall is estimated using the documented empirical monthly method and allocated across rainy days. Net daily irrigation requirement is then:

IWR = max(0, ETc - Peff)

The resulting IWR is a physics-derived reference requirement, not measured farmer irrigation.

## 4. Machine-learning design
Base inputs are meteorological variables plus crop state. A second feature set adds ET0 as a physics-informed engineered feature.

Chronological evaluation:
- Train: 2021-2023
- Validation/model selection: 2024
- Final untouched test: 2025

Models:
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- MLP neural-network baseline

Random Forest and XGBoost are tuned only against 2024 validation RMSE. The final model is locked from validation before 2025 test evaluation.

## 5. Reservoir-allocation interpretation
Predicted irrigation depth is converted as:

1 mm over 1 ha = 10 m3

Net allocation = 10 x IWR x crop area

Gross reservoir release requires actual conveyance and field-application efficiencies and is therefore computed only when those real values are supplied.

## 6. Scope boundary
This is a demand-estimation and decision-support prototype. It does not claim measured dam-release prediction, real-time reservoir optimization, or a genuine PINN because field soil-water states and defensible differential-equation boundary conditions are not available.
