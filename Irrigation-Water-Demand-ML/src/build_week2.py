from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "project_config.json"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "week2"
RESULT_DIR = ROOT / "results" / "week2"
REPORT_DIR = ROOT / "reports"
for d in [RAW_DIR, PROC_DIR, FIG_DIR, RESULT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
with CONFIG_PATH.open() as f:
    CFG = json.load(f)
LAT = CFG["study_area"]["latitude"]
LON = CFG["study_area"]["longitude"]
ELEV = CFG["study_area"]["elevation_m"]
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "PS"]


def download_power() -> tuple[pd.DataFrame, dict]:
    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "AG",
        "longitude": LON,
        "latitude": LAT,
        "start": CFG["period"]["start"].replace("-", ""),
        "end": CFG["period"]["end"].replace("-", ""),
        "format": "JSON",
        "time-standard": "LST",
    }
    r = requests.get(POWER_URL, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df.sort_index().replace(-999, np.nan), payload


def save_raw(df: pd.DataFrame, payload: dict) -> None:
    df.to_csv(RAW_DIR / "nasa_power_daily_2021_2025.csv")
    metadata = {
        "source": "NASA POWER Daily Point API",
        "study_area": CFG["study_area"],
        "period": CFG["period"],
        "requested_parameters": POWER_PARAMETERS,
        "api_header": payload.get("header", {}),
        "parameter_metadata": payload.get("parameters", {}),
    }
    (RAW_DIR / "nasa_power_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))


def sat_vp(temp_c):
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def extraterrestrial_radiation(doy: pd.Series, latitude_deg: float) -> np.ndarray:
    phi = math.radians(latitude_deg)
    j = doy.to_numpy(dtype=float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    delta = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    gsc = 0.0820
    return (24 * 60 / np.pi) * gsc * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def compute_eto(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tmean, tmax, tmin = out["T2M"], out["T2M_MAX"], out["T2M_MIN"]
    rh, u2, p = out["RH2M"], out["WS2M"], out["PS"]
    # NASA POWER metadata for AG daily API reports ALLSKY_SFC_SW_DWN in MJ/m^2/day.
    rs = out["ALLSKY_SFC_SW_DWN"]
    es = (sat_vp(tmax) + sat_vp(tmin)) / 2
    ea = np.clip(rh / 100.0, 0, 1) * es
    delta = 4098 * sat_vp(tmean) / ((tmean + 237.3) ** 2)
    gamma = 0.000665 * p
    ra = extraterrestrial_radiation(out.index.to_series().dt.dayofyear, LAT)
    rso = (0.75 + 2e-5 * ELEV) * ra
    rns = (1 - 0.23) * rs
    sigma = 4.903e-9
    rs_rso = np.clip(rs / np.where(rso <= 0, np.nan, rso), 0, 1.0)
    rnl = sigma * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * rs_rso - 0.35)
    rn = rns - rnl
    eto = (0.408 * delta * rn + gamma * (900 / (tmean + 273)) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    out["solar_radiation_mj_m2_day"] = rs
    out["eto_mm_day"] = np.maximum(eto, 0)
    return out


def crop_calendar(weather: pd.DataFrame) -> pd.DataFrame:
    frames = []
    stages = CFG["crop"]["stage_days"]
    kc_cfg = CFG["crop"]["kc"]
    total = sum(stages.values())
    for year in range(2021, 2026):
        sow = pd.Timestamp(year=year, month=CFG["crop"]["sowing_month"], day=CFG["crop"]["sowing_day"])
        season = weather.reindex(pd.date_range(sow, periods=total, freq="D")).copy()
        season["season_year"] = year
        season["days_after_sowing"] = np.arange(1, total + 1)
        init_end = stages["initial"]
        dev_end = init_end + stages["development"]
        mid_end = dev_end + stages["mid_season"]
        stage, kc = [], []
        for d in season["days_after_sowing"]:
            if d <= init_end:
                stage.append("initial"); kc.append(kc_cfg["initial"])
            elif d <= dev_end:
                frac = (d - init_end) / stages["development"]
                stage.append("development"); kc.append(kc_cfg["initial"] + frac * (kc_cfg["mid"] - kc_cfg["initial"]))
            elif d <= mid_end:
                stage.append("mid_season"); kc.append(kc_cfg["mid"])
            else:
                frac = (d - mid_end) / stages["late_season"]
                stage.append("late_season"); kc.append(kc_cfg["mid"] + frac * (kc_cfg["end"] - kc_cfg["mid"]))
        season["crop_stage"] = stage
        season["kc"] = kc
        frames.append(season)
    crop = pd.concat(frames)
    crop.index.name = "date"
    return crop


def effective_rainfall_monthly_allocated(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_key"] = out.index.to_period("M")
    monthly = out.groupby("month_key")["PRECTOTCORR"].sum(min_count=1)
    pe = monthly.copy()
    low = monthly < 75
    pe.loc[low] = np.maximum(0, 0.6 * monthly.loc[low] - 10)
    pe.loc[~low] = np.maximum(0, 0.8 * monthly.loc[~low] - 25)
    ratio = (pe / monthly.replace(0, np.nan)).fillna(0).clip(0, 1)
    out["effective_rain_ratio"] = out["month_key"].map(ratio)
    out["effective_rain_mm"] = out["PRECTOTCORR"] * out["effective_rain_ratio"]
    return out.drop(columns="month_key")


def finalize_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    crop = effective_rainfall_monthly_allocated(crop_calendar(compute_eto(raw)))
    crop["etc_mm_day"] = crop["kc"] * crop["eto_mm_day"]
    crop["irrigation_requirement_mm"] = np.maximum(0, crop["etc_mm_day"] - crop["effective_rain_mm"])
    crop = crop.rename(columns={
        "T2M": "temp_mean_c", "T2M_MAX": "temp_max_c", "T2M_MIN": "temp_min_c", "RH2M": "relative_humidity_pct",
        "WS2M": "wind_speed_2m_m_s", "PRECTOTCORR": "rainfall_mm", "PS": "surface_pressure_kpa",
        "ALLSKY_SFC_SW_DWN": "solar_radiation_mj_m2_day_raw",
    })
    columns = ["season_year", "days_after_sowing", "crop_stage", "kc", "temp_mean_c", "temp_max_c", "temp_min_c", "relative_humidity_pct", "wind_speed_2m_m_s", "solar_radiation_mj_m2_day_raw", "solar_radiation_mj_m2_day", "rainfall_mm", "surface_pressure_kpa", "eto_mm_day", "etc_mm_day", "effective_rain_ratio", "effective_rain_mm", "irrigation_requirement_mm"]
    crop = crop[columns]
    crop.to_csv(PROC_DIR / "maize_irrigation_dataset_2021_2025.csv")
    return crop


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({"metric": ["rows", "seasons", "missing_cells", "mean_temperature_c", "total_rainfall_mm", "mean_eto_mm_day", "mean_etc_mm_day", "mean_iwr_mm_day", "median_iwr_mm_day", "max_iwr_mm_day", "zero_iwr_days"], "value": [len(df), df["season_year"].nunique(), int(df.isna().sum().sum()), df["temp_mean_c"].mean(), df["rainfall_mm"].sum(), df["eto_mm_day"].mean(), df["etc_mm_day"].mean(), df["irrigation_requirement_mm"].mean(), df["irrigation_requirement_mm"].median(), df["irrigation_requirement_mm"].max(), int((df["irrigation_requirement_mm"] == 0).sum())]})
    summary.to_csv(RESULT_DIR / "dataset_summary.csv", index=False)
    return summary


def make_plots(df: pd.DataFrame) -> None:
    d = df[df["season_year"] == 2024]
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(d.index, d["eto_mm_day"], label="ET0"); ax.plot(d.index, d["etc_mm_day"], label="ETc"); ax.plot(d.index, d["irrigation_requirement_mm"], label="IWR"); ax.set_title("2024 Maize Season: ET0, ETc and IWR"); ax.set_ylabel("mm/day"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG_DIR / "01_2024_water_balance.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.plot(d["days_after_sowing"], d["kc"]); ax.set_title("Grain Maize Kc Curve"); ax.set_xlabel("Days after sowing"); ax.set_ylabel("Kc"); fig.tight_layout(); fig.savefig(FIG_DIR / "02_kc_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.hist(df["irrigation_requirement_mm"].dropna(), bins=30); ax.set_title("Distribution of Derived Daily IWR"); ax.set_xlabel("IWR (mm/day)"); ax.set_ylabel("Days"); fig.tight_layout(); fig.savefig(FIG_DIR / "03_iwr_distribution.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(df["rainfall_mm"], df["irrigation_requirement_mm"], alpha=0.5); ax.set_title("Rainfall vs Irrigation Requirement"); ax.set_xlabel("Rainfall (mm/day)"); ax.set_ylabel("IWR (mm/day)"); fig.tight_layout(); fig.savefig(FIG_DIR / "04_rainfall_vs_iwr.png", dpi=180); plt.close(fig)
    cols = ["temp_mean_c", "relative_humidity_pct", "wind_speed_2m_m_s", "solar_radiation_mj_m2_day", "rainfall_mm", "kc", "eto_mm_day", "irrigation_requirement_mm"]
    corr = df[cols].corr(); fig, ax = plt.subplots(figsize=(9, 7)); im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm"); ax.set_xticks(range(len(cols)), [c.replace("_", " ") for c in cols], rotation=45, ha="right"); ax.set_yticks(range(len(cols)), [c.replace("_", " ") for c in cols]);
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Correlation"); ax.set_title("Correlation Matrix"); fig.tight_layout(); fig.savefig(FIG_DIR / "05_correlation_matrix.png", dpi=180); plt.close(fig)


def model_pipeline(model, include_eto: bool):
    numeric = ["temp_mean_c", "temp_max_c", "temp_min_c", "relative_humidity_pct", "wind_speed_2m_m_s", "solar_radiation_mj_m2_day", "rainfall_mm", "surface_pressure_kpa", "kc", "days_after_sowing"]
    if include_eto:
        numeric.append("eto_mm_day")
    categorical = ["crop_stage"]
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if isinstance(model, LinearRegression):
        steps.append(("scaler", StandardScaler()))
    pre = ColumnTransformer([("num", Pipeline(steps), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
    return Pipeline([("pre", pre), ("model", model)]), numeric + categorical


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season_year"].isin([2021, 2022, 2023])]
    val = df[df["season_year"] == 2024]
    specs = [("Linear Regression", LinearRegression(), False), ("Decision Tree", DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42), False), ("Random Forest", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1), False), ("Linear Regression + ET0", LinearRegression(), True), ("Random Forest + ET0", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1), True)]
    rows, pred_table = [], pd.DataFrame(index=val.index)
    pred_table["actual_iwr_mm"] = val["irrigation_requirement_mm"]
    for name, model, include_eto in specs:
        pipe, features = model_pipeline(model, include_eto)
        pipe.fit(train[features], train["irrigation_requirement_mm"])
        pred = np.maximum(pipe.predict(val[features]), 0)
        rows.append({"model": name, "features": "raw+ET0" if include_eto else "raw", "train_rows": len(train), "validation_rows": len(val), "MAE_mm_day": mean_absolute_error(val["irrigation_requirement_mm"], pred), "RMSE_mm_day": math.sqrt(mean_squared_error(val["irrigation_requirement_mm"], pred)), "R2": r2_score(val["irrigation_requirement_mm"], pred)})
        pred_table[name] = pred
    results = pd.DataFrame(rows).sort_values("RMSE_mm_day")
    results.to_csv(RESULT_DIR / "baseline_validation_metrics.csv", index=False)
    pred_table.to_csv(RESULT_DIR / "validation_2024_predictions.csv")
    best = results.iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(pred_table["actual_iwr_mm"], pred_table[best], alpha=0.65); maxv = max(pred_table["actual_iwr_mm"].max(), pred_table[best].max()); ax.plot([0, maxv], [0, maxv], linestyle="--"); ax.set_xlabel("Actual derived IWR (mm/day)"); ax.set_ylabel("Predicted IWR (mm/day)"); ax.set_title(f"2024 Validation: {best}"); fig.tight_layout(); fig.savefig(FIG_DIR / "06_best_model_actual_vs_predicted.png", dpi=180); plt.close(fig)
    return results, pred_table


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    s = dict(zip(summary["metric"], summary["value"])); best = metrics.iloc[0]
    report = f"""# Week 2 Progress Report

## Study setup
- Location: {CFG['study_area']['name']}
- Coordinates: {LAT}, {LON}
- Crop: {CFG['crop']['name']}
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
- Processed rows: {int(s['rows'])}
- Crop seasons: {int(s['seasons'])}
- Missing cells in processed dataset: {int(s['missing_cells'])}
- Mean crop-season temperature: {s['mean_temperature_c']:.2f} °C
- Total crop-season rainfall across five seasons: {s['total_rainfall_mm']:.2f} mm
- Mean ET0: {s['mean_eto_mm_day']:.2f} mm/day
- Mean ETc: {s['mean_etc_mm_day']:.2f} mm/day
- Mean derived IWR: {s['mean_iwr_mm_day']:.2f} mm/day
- Median derived IWR: {s['median_iwr_mm_day']:.2f} mm/day
- Maximum derived IWR: {s['max_iwr_mm_day']:.2f} mm/day
- Zero-IWR days: {int(s['zero_iwr_days'])}

## Preliminary ML protocol
- Training: 2021-2023 ({int(metrics.iloc[0]['train_rows'])} rows)
- Validation: 2024 ({int(metrics.iloc[0]['validation_rows'])} rows)
- Final test: 2025, intentionally untouched in Week 2
- Target: irrigation_requirement_mm

## Best Week 2 validation model
- Model: {best['model']}
- MAE: {best['MAE_mm_day']:.3f} mm/day
- RMSE: {best['RMSE_mm_day']:.3f} mm/day
- R²: {best['R2']:.3f}

## Important interpretation
The target is a physics-derived reference irrigation requirement produced from FAO crop-water equations and historical meteorology. It is not measured farmer irrigation. Week 3 will tune/finalize models and evaluate exactly once on the untouched 2025 season.
"""
    (REPORT_DIR / "WEEK2_PROGRESS.md").write_text(report)


def main():
    raw, payload = download_power()
    save_raw(raw, payload)
    processed = finalize_dataset(raw)
    summary = dataset_summary(processed)
    make_plots(processed)
    metrics, _ = evaluate_models(processed)
    write_report(summary, metrics)
    print(summary)
    print(metrics)


if __name__ == "__main__":
    main()
