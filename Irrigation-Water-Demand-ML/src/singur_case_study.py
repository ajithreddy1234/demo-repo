from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "final"
REPORT_DIR = ROOT / "reports"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Official Singur project characteristics from Telangana I&CAD project note.
GROSS_CAPACITY_TMC = 29.91
FRL_M = 523.60
AYACUT_ACRES = 40000.0
SINGUR_CANAL_ALLOCATION_TMC = 2.00
LEFT_MAIN_CUMECS = 13.804
MAIN_BRANCH_CUMECS = 5.479
RIGHT_MAIN_CUMECS = 1.614

# CWC remote-sensing capacity assessment (designed/live capacity context).
DESIGNED_GROSS_MCM = 847.155
DESIGNED_LIVE_MCM = 822.460
SURVEYED_LIVE_MCM_2018 = 812.037

# Demonstration block retained from the ML/optimization experiment.
SCENARIO_MAIZE_AREA_HA = 1000.0
SCENARIO_EFFICIENCY = 0.70

ACRE_TO_HA = 0.40468564224
TMC_TO_M3 = 28_316_846.592

ayacut_ha = AYACUT_ACRES * ACRE_TO_HA
singur_canal_allocation_m3 = SINGUR_CANAL_ALLOCATION_TMC * TMC_TO_M3
combined_canal_capacity_cumecs = LEFT_MAIN_CUMECS + MAIN_BRANCH_CUMECS + RIGHT_MAIN_CUMECS

alloc = pd.read_csv(RESULT_DIR / "2025_season_allocation_summary.csv").iloc[0]
pred_net_m3_per_ha = float(alloc["predicted_net_allocation_m3_per_ha"])
pred_peak_net_m3_per_ha_day = float(alloc["predicted_peak_m3_per_ha_day"])

scenario_net_season_m3 = pred_net_m3_per_ha * SCENARIO_MAIZE_AREA_HA
scenario_gross_season_m3 = scenario_net_season_m3 / SCENARIO_EFFICIENCY
scenario_peak_gross_m3_day = pred_peak_net_m3_per_ha_day * SCENARIO_MAIZE_AREA_HA / SCENARIO_EFFICIENCY
scenario_peak_gross_cumecs = scenario_peak_gross_m3_day / 86400.0

summary = pd.DataFrame([{
    "reservoir": "Mogaligundla Baga Reddy Singur Project",
    "river": "Manjira",
    "district": "Sangareddy",
    "gross_capacity_tmc": GROSS_CAPACITY_TMC,
    "frl_m": FRL_M,
    "official_new_ayacut_acres": AYACUT_ACRES,
    "official_new_ayacut_ha": ayacut_ha,
    "singur_canal_irrigation_allocation_tmc": SINGUR_CANAL_ALLOCATION_TMC,
    "singur_canal_irrigation_allocation_m3": singur_canal_allocation_m3,
    "combined_head_discharge_cumecs": combined_canal_capacity_cumecs,
    "designed_gross_capacity_mcm": DESIGNED_GROSS_MCM,
    "designed_live_capacity_mcm": DESIGNED_LIVE_MCM,
    "surveyed_live_capacity_2018_mcm": SURVEYED_LIVE_MCM_2018,
    "scenario_maize_area_ha": SCENARIO_MAIZE_AREA_HA,
    "scenario_area_as_pct_of_official_ayacut": 100.0 * SCENARIO_MAIZE_AREA_HA / ayacut_ha,
    "scenario_predicted_gross_seasonal_demand_m3": scenario_gross_season_m3,
    "scenario_demand_as_pct_of_2tmc_canal_allocation": 100.0 * scenario_gross_season_m3 / singur_canal_allocation_m3,
    "scenario_peak_gross_release_m3_day": scenario_peak_gross_m3_day,
    "scenario_peak_gross_release_cumecs": scenario_peak_gross_cumecs,
    "scenario_peak_as_pct_of_combined_head_discharge": 100.0 * scenario_peak_gross_cumecs / combined_canal_capacity_cumecs,
}])
summary.to_csv(RESULT_DIR / "singur_case_study_scale_check.csv", index=False)

# 10-day irrigation working table using the final XGBoost predictions.
pred = pd.read_csv(RESULT_DIR / "test_predictions_2025.csv", parse_dates=["date"]).sort_values("date")
iwr = pred["XGBoost"].clip(lower=0)
net = 10.0 * iwr * SCENARIO_MAIZE_AREA_HA
gross = net / SCENARIO_EFFICIENCY

daily = pd.DataFrame({
    "date": pred["date"],
    "predicted_iwr_mm_day": iwr,
    "net_demand_m3_day": net,
    "gross_release_requirement_m3_day": gross,
})
daily["period_no"] = (daily.index // 10) + 1

table = daily.groupby("period_no", as_index=False).agg(
    start_date=("date", "min"),
    end_date=("date", "max"),
    days=("date", "size"),
    mean_iwr_mm_day=("predicted_iwr_mm_day", "mean"),
    total_net_demand_m3=("net_demand_m3_day", "sum"),
    total_gross_release_requirement_m3=("gross_release_requirement_m3_day", "sum"),
    peak_gross_release_requirement_m3_day=("gross_release_requirement_m3_day", "max"),
)
table.to_csv(RESULT_DIR / "singur_10day_irrigation_working_table_2025.csv", index=False)

report = f"""# Singur Reservoir Case-Study Context

## Why Singur
The selected weather/crop study point is in Sangareddy district. The Singur Project is a real multipurpose project on the Manjira River in Sangareddy and therefore provides a defensible reservoir context for the irrigation-demand model.

## Official project characteristics
- Gross capacity at FRL: {GROSS_CAPACITY_TMC:.2f} TMC
- FRL: {FRL_M:.2f} m
- New irrigation ayacut: {AYACUT_ACRES:,.0f} acres ({ayacut_ha:,.0f} ha)
- Irrigation allocation for Singur canals in the official project note: {SINGUR_CANAL_ALLOCATION_TMC:.2f} TMC
- Canal-head discharges: Left Main {LEFT_MAIN_CUMECS:.3f} cumecs, Main Branch {MAIN_BRANCH_CUMECS:.3f} cumecs, Right Main {RIGHT_MAIN_CUMECS:.3f} cumecs
- Combined listed head discharge: {combined_canal_capacity_cumecs:.3f} cumecs
- Designed live capacity: {DESIGNED_LIVE_MCM:.3f} Mm3
- Remote-sensing assessed live capacity (2018): {SURVEYED_LIVE_MCM_2018:.3f} Mm3

## Why the model still uses a 1000-ha maize block
The official 40,000-acre ayacut is not assumed to be entirely maize. The project therefore keeps a controlled 1000-ha maize demonstration block instead of inventing a crop-distribution map.

This block is {100.0 * SCENARIO_MAIZE_AREA_HA / ayacut_ha:.2f}% of the official new ayacut.

## Scale check
Using the final XGBoost 2025 prediction and 70% scenario delivery efficiency:
- Predicted gross seasonal release requirement for the 1000-ha maize block: {scenario_gross_season_m3:,.0f} m3
- This is {100.0 * scenario_gross_season_m3 / singur_canal_allocation_m3:.2f}% of the official 2.00-TMC Singur-canal irrigation allocation.
- Peak gross release requirement: {scenario_peak_gross_m3_day:,.0f} m3/day ({scenario_peak_gross_cumecs:.3f} cumecs)
- This is {100.0 * scenario_peak_gross_cumecs / combined_canal_capacity_cumecs:.2f}% of the combined listed canal-head discharge.

Therefore the previously used 25,292 m3/day optimization cap is NOT presented as Singur's actual canal-head capacity. It is retained only as a hypothetical local/sub-command delivery bottleneck stress test.

## 10-day irrigation working table
Following CWC irrigation-planning practice, the 2025 daily demand predictions are also aggregated into 10-day operating periods. The table is stored at:
results/final/singur_10day_irrigation_working_table_2025.csv

A complete historical Singur reservoir mass-balance simulation would additionally require period-wise inflow, beginning storage, drinking/industrial releases, evaporation and spill data. Those time series have not been invented in this project.
"""
(REPORT_DIR / "SINGUR_CASE_STUDY.md").write_text(report)
print(summary.to_string(index=False))
print(table.to_string(index=False))
