# Data Dictionary

| Variable | Unit | Source / derivation | Role |
|---|---|---|---|
| date | YYYY-MM-DD | Calendar | Index |
| season_year | year | Crop season | Split / grouping |
| days_after_sowing | day | Derived from assumed sowing date | ML input |
| crop_stage | categorical | Initial / development / mid-season / late-season | ML input |
| kc | dimensionless | FAO reference crop-coefficient curve for grain maize | ML input |
| temp_mean_c | deg C | NASA POWER T2M | ML input, ET0 |
| temp_max_c | deg C | NASA POWER T2M_MAX | ML input, ET0 |
| temp_min_c | deg C | NASA POWER T2M_MIN | ML input, ET0 |
| relative_humidity_pct | % | NASA POWER RH2M | ML input, ET0 |
| wind_speed_2m_m_s | m/s | NASA POWER WS2M | ML input, ET0 |
| solar_radiation_mj_m2_day | MJ/m2/day | NASA POWER ALLSKY_SFC_SW_DWN | ML input, ET0 |
| rainfall_mm | mm/day | NASA POWER PRECTOTCORR | ML input, effective rainfall |
| surface_pressure_kpa | kPa | NASA POWER PS | ML input, ET0 |
| eto_mm_day | mm/day | FAO-56 Penman-Monteith calculation | Engineered ML input |
| etc_mm_day | mm/day | Kc x ET0 | Target construction only |
| effective_rain_mm | mm/day | Empirical effective-rainfall method | Target construction only |
| irrigation_requirement_mm | mm/day | max(0, ETc - Peff) | Regression target |
| net_allocation_m3_per_ha_day | m3/ha/day | 10 x IWR | Reservoir-allocation output |

## Leakage control
ETc, effective rainfall and net allocation are not used as model inputs because they are directly downstream of target construction. ET0 is tested separately as a physically meaningful engineered feature.
