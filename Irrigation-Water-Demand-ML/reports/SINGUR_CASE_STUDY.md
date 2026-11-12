# Singur Reservoir Case-Study Context

## Why Singur
The selected weather/crop study point is in Sangareddy district. The Singur Project is a real multipurpose project on the Manjira River in Sangareddy and therefore provides a defensible reservoir context for the irrigation-demand model.

## Official project characteristics
- Gross capacity at FRL: 29.91 TMC
- FRL: 523.60 m
- New irrigation ayacut: 40,000 acres (16,187 ha)
- Irrigation allocation for Singur canals in the official project note: 2.00 TMC
- Canal-head discharges: Left Main 13.804 cumecs, Main Branch 5.479 cumecs, Right Main 1.614 cumecs
- Combined listed head discharge: 20.897 cumecs
- Designed live capacity: 822.460 Mm3
- Remote-sensing assessed live capacity (2018): 812.037 Mm3

## Why the model still uses a 1000-ha maize block
The official 40,000-acre ayacut is not assumed to be entirely maize. The project therefore keeps a controlled 1000-ha maize demonstration block instead of inventing a crop-distribution map.

This block is 6.18% of the official new ayacut.

## Scale check
Using the final XGBoost 2025 prediction and 70% scenario delivery efficiency:
- Predicted gross seasonal release requirement for the 1000-ha maize block: 2,258,198 m3
- This is 3.99% of the official 2.00-TMC Singur-canal irrigation allocation.
- Peak gross release requirement: 72,483 m3/day (0.839 cumecs)
- This is 4.01% of the combined listed canal-head discharge.

Therefore the previously used 25,292 m3/day optimization cap is NOT presented as Singur's actual canal-head capacity. It is retained only as a hypothetical local/sub-command delivery bottleneck stress test.

## 10-day irrigation working table
Following CWC irrigation-planning practice, the 2025 daily demand predictions are also aggregated into 10-day operating periods. The table is stored at:
results/final/singur_10day_irrigation_working_table_2025.csv

A complete historical Singur reservoir mass-balance simulation would additionally require period-wise inflow, beginning storage, drinking/industrial releases, evaporation and spill data. Those time series have not been invented in this project.
