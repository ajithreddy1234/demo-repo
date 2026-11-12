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

## 6. Reservoir-constrained optimization

Predicted IWR is converted into net crop demand:

Net demand (m3/day) = 10 x IWR (mm/day) x crop area (ha)

Gross release demand is then:

Gross demand = Net demand / overall delivery efficiency

For a seasonal available-water budget W and daily release-capacity C, daily releases x_t are optimized subject to:

- 0 <= x_t <= daily gross demand_t
- x_t <= C
- sum(x_t) <= W

The primary objective minimizes the maximum relative daily deficit. A second stage maximizes total delivered water while retaining the optimal fairness level.

Three scenarios are evaluated on the 2025 predicted demand:
- full supply with no effective daily-cap constraint,
- 15% seasonal shortage with no effective daily-cap constraint,
- 15% seasonal shortage together with a finite daily release cap.

This separates shortage caused by insufficient seasonal water from shortage caused by delivery/conveyance capacity.

## 7. Scope boundary
This is a demand-prediction and reservoir-allocation optimization prototype. The optimization is scenario-driven and does not claim to reproduce an actual reservoir operating policy because measured storage, inflows, command-area acreage, canal capacity and operating-rule data have not been supplied. It also does not claim a genuine PINN because field soil-water states and defensible differential-equation boundary conditions are unavailable.
