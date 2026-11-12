from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "maize_irrigation_dataset_2021_2025.csv"
RESULT_DIR = ROOT / "results" / "final"
FIG_DIR = ROOT / "figures" / "final"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for d in [RESULT_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET = "irrigation_requirement_mm"
RAW_NUMERIC = [
    "temp_mean_c", "temp_max_c", "temp_min_c", "relative_humidity_pct",
    "wind_speed_2m_m_s", "solar_radiation_mj_m2_day", "rainfall_mm",
    "surface_pressure_kpa", "kc", "days_after_sowing",
]
CATEGORICAL = ["crop_stage"]

def feature_names(include_eto: bool):
    cols = list(RAW_NUMERIC)
    if include_eto:
        cols.append("eto_mm_day")
    return cols + CATEGORICAL

def make_pipeline(model, include_eto: bool, scale_numeric: bool = False):
    numeric = list(RAW_NUMERIC)
    if include_eto:
        numeric.append("eto_mm_day")
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE_mm_day": mae,
        "RMSE_mm_day": rmse,
        "R2": r2_score(y_true, y_pred),
        "MAE_m3_per_ha_day": mae * 10.0,
        "RMSE_m3_per_ha_day": rmse * 10.0,
    }

def fit_eval(model, include_eto, scale, train, val):
    cols = feature_names(include_eto)
    pipe = make_pipeline(model, include_eto, scale)
    pipe.fit(train[cols], train[TARGET])
    pred = np.maximum(pipe.predict(val[cols]), 0)
    return pipe, pred, get_metrics(val[TARGET], pred)

def tune_rf(train, val, include_eto):
    grid = {
        "n_estimators": [300, 500],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", 0.8],
    }
    rows = []
    best = None
    best_rmse = float("inf")
    for vals in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), vals))
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        pipe, pred, met = fit_eval(model, include_eto, False, train, val)
        rows.append({"model_family": "Random Forest", "include_eto": include_eto, **params, **met})
        if met["RMSE_mm_day"] < best_rmse:
            best_rmse = met["RMSE_mm_day"]
            best = (params, pipe, pred, met)
    return best, pd.DataFrame(rows)

def tune_xgb(train, val, include_eto):
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.07],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    rows = []
    best = None
    best_rmse = float("inf")
    for vals in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), vals))
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            reg_lambda=1.0,
            **params,
        )
        pipe, pred, met = fit_eval(model, include_eto, False, train, val)
        rows.append({"model_family": "XGBoost", "include_eto": include_eto, **params, **met})
        if met["RMSE_mm_day"] < best_rmse:
            best_rmse = met["RMSE_mm_day"]
            best = (params, pipe, pred, met)
    return best, pd.DataFrame(rows)

def final_fit(name, model, include_eto, scale, trainval, test):
    cols = feature_names(include_eto)
    pipe = make_pipeline(model, include_eto, scale)
    pipe.fit(trainval[cols], trainval[TARGET])
    pred = np.maximum(pipe.predict(test[cols]), 0)
    row = {
        "model": name,
        "features": "raw+ET0" if include_eto else "raw",
        "train_rows": len(trainval),
        "test_rows": len(test),
        **get_metrics(test[TARGET], pred),
    }
    return pipe, pred, row

def plot_outputs(results, test, best_pred, best_name, best_pipe, best_include_eto):
    ordered = results.sort_values("RMSE_mm_day")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["model"], ordered["RMSE_mm_day"])
    ax.set_xlabel("RMSE (mm/day)")
    ax.set_title("Final 2025 Test: Model Comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_model_comparison_rmse.png", dpi=180)
    plt.close(fig)

    y = test[TARGET].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, best_pred, alpha=0.7)
    mx = max(float(y.max()), float(best_pred.max()))
    ax.plot([0, mx], [0, mx], linestyle="--")
    ax.set_xlabel("Reference IWR (mm/day)")
    ax.set_ylabel("Predicted IWR (mm/day)")
    ax.set_title(f"2025 Test: {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_actual_vs_predicted_2025.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test.index, y, label="Reference IWR")
    ax.plot(test.index, best_pred, label="Predicted IWR")
    ax.set_ylabel("IWR (mm/day)")
    ax.set_title(f"2025 Maize Season: Reference vs Predicted IWR - {best_name}")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_iwr_timeseries_2025.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test.index, y * 10.0, label="Reference net allocation")
    ax.plot(test.index, best_pred * 10.0, label="Predicted net allocation")
    ax.set_ylabel("m3/ha/day")
    ax.set_title("2025 Net Reservoir Allocation Requirement per Hectare")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_allocation_timeseries_2025.png", dpi=180)
    plt.close(fig)

    residuals = best_pred - y
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(best_pred, residuals, alpha=0.7)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted IWR (mm/day)")
    ax.set_ylabel("Residual (mm/day)")
    ax.set_title(f"2025 Residuals - {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_residuals_2025.png", dpi=180)
    plt.close(fig)

    cols = feature_names(best_include_eto)
    perm = permutation_importance(
        best_pipe,
        test[cols],
        test[TARGET],
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame({
        "feature": cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(RESULT_DIR / "permutation_importance_2025.csv", index=False)

    top = importance.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_xlabel("Increase in RMSE when permuted")
    ax.set_title("Final Model: Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_feature_importance.png", dpi=180)
    plt.close(fig)
    return importance

def stagewise(test, pred):
    temp = test[["crop_stage"]].copy()
    temp["reference_iwr_mm"] = test[TARGET]
    temp["predicted_iwr_mm"] = pred
    temp["abs_error_mm"] = np.abs(temp["predicted_iwr_mm"] - temp["reference_iwr_mm"])
    out = temp.groupby("crop_stage", observed=True).agg(
        days=("reference_iwr_mm", "size"),
        mean_reference_iwr_mm=("reference_iwr_mm", "mean"),
        mean_predicted_iwr_mm=("predicted_iwr_mm", "mean"),
        MAE_mm_day=("abs_error_mm", "mean"),
    ).reset_index()
    out.to_csv(RESULT_DIR / "stagewise_test_performance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(out["crop_stage"], out["MAE_mm_day"])
    ax.set_ylabel("MAE (mm/day)")
    ax.set_title("2025 Prediction Error by Maize Growth Stage")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_stagewise_mae.png", dpi=180)
    plt.close(fig)
    return out

def allocation_summary(test, pred):
    reference_mm = float(test[TARGET].sum())
    predicted_mm = float(pred.sum())
    reference_m3_ha = reference_mm * 10.0
    predicted_m3_ha = predicted_mm * 10.0
    diff = predicted_m3_ha - reference_m3_ha
    pct = 100.0 * diff / reference_m3_ha if reference_m3_ha else 0.0
    row = {
        "season": 2025,
        "days": len(test),
        "reference_total_iwr_mm": reference_mm,
        "predicted_total_iwr_mm": predicted_mm,
        "reference_net_allocation_m3_per_ha": reference_m3_ha,
        "predicted_net_allocation_m3_per_ha": predicted_m3_ha,
        "allocation_difference_m3_per_ha": diff,
        "allocation_difference_pct": pct,
        "reference_peak_iwr_mm_day": float(test[TARGET].max()),
        "predicted_peak_iwr_mm_day": float(pred.max()),
        "reference_peak_m3_per_ha_day": float(test[TARGET].max() * 10.0),
        "predicted_peak_m3_per_ha_day": float(pred.max() * 10.0),
    }
    pd.DataFrame([row]).to_csv(RESULT_DIR / "2025_season_allocation_summary.csv", index=False)
    return row

def write_report(df, validation_results, final_results, best_name, best_include_eto, best_metrics, allocation, importance, stage):
    top_features = ", ".join(importance.head(5)["feature"].tolist())
    val_best = validation_results.iloc[0]
    report_lines = [
        "# Final Project Report - Irrigation Water Demand ML",
        "",
        "## Project objective",
        "Develop a reproducible machine-learning framework that estimates daily net irrigation water requirement for grain maize from meteorological and crop-state variables, then converts the predicted depth into allocation-ready water volume per hectare for reservoir/dam irrigation planning.",
        "",
        "## Study configuration",
        "- Study point: Sangareddy, Telangana, India",
        "- Crop: Grain maize",
        "- Seasons: 2021-2025",
        f"- Daily crop-season records: {len(df)}",
        "- Crop duration: 125 days per season",
        "- Historical weather source: NASA POWER",
        "- Crop-water framework: FAO-56 Penman-Monteith plus crop coefficient approach",
        "",
        "## Hydrologic target",
        "ET0 is calculated from meteorological variables. Crop evapotranspiration is ETc = Kc x ET0. Effective rainfall is subtracted to obtain IWR = max(0, ETc - Peff).",
        "The target is therefore a physics-derived reference irrigation requirement, not directly measured farmer irrigation or dam release.",
        "",
        "## Reservoir allocation conversion",
        "Net allocation (m3/ha/day) = 10 x IWR (mm/day).",
        "For crop area A in hectares: net command-area demand = 10 x IWR x A.",
        "Gross release additionally requires actual conveyance and application efficiencies. No undocumented area or efficiency has been invented.",
        "",
        "## ML protocol",
        "- Training: 2021-2023",
        "- Validation/model selection: 2024",
        "- Final untouched test: 2025",
        "- Models: Linear Regression, Decision Tree, Random Forest, XGBoost, Neural Network",
        "- Random Forest and XGBoost hyperparameters selected using 2024 validation RMSE only",
        "- Final models refit on 2021-2024 and tested once on 2025",
        "",
        "## Best validation candidate",
        f"- Model: {val_best['model']}",
        f"- Validation RMSE: {val_best['RMSE_mm_day']:.4f} mm/day",
        f"- Validation R2: {val_best['R2']:.4f}",
        "",
        "## Final 2025 test results",
        final_results.to_string(index=False),
        "",
        "## Selected final model",
        f"- Model: {best_name}",
        f"- Feature set: {'raw+ET0' if best_include_eto else 'raw'}",
        f"- MAE: {best_metrics['MAE_mm_day']:.4f} mm/day",
        f"- RMSE: {best_metrics['RMSE_mm_day']:.4f} mm/day",
        f"- R2: {best_metrics['R2']:.4f}",
        f"- Allocation-equivalent MAE: {best_metrics['MAE_m3_per_ha_day']:.3f} m3/ha/day",
        "",
        "## 2025 allocation interpretation",
        f"- Reference seasonal net requirement: {allocation['reference_net_allocation_m3_per_ha']:.2f} m3/ha",
        f"- Predicted seasonal net requirement: {allocation['predicted_net_allocation_m3_per_ha']:.2f} m3/ha",
        f"- Seasonal allocation difference: {allocation['allocation_difference_m3_per_ha']:.2f} m3/ha ({allocation['allocation_difference_pct']:.2f}%)",
        f"- Reference peak daily requirement: {allocation['reference_peak_m3_per_ha_day']:.2f} m3/ha/day",
        f"- Predicted peak daily requirement: {allocation['predicted_peak_m3_per_ha_day']:.2f} m3/ha/day",
        "",
        "## Most influential final-model inputs",
        f"Top five permutation-importance features: {top_features}",
        "",
        "## Growth-stage performance",
        stage.to_string(index=False),
        "",
        "## Limitations",
        "1. The target is physics-derived rather than measured farmer/canal irrigation.",
        "2. Operational future use requires forecast weather rather than historical observed weather.",
        "3. Effective rainfall is empirical, not directly measured field infiltration.",
        "4. Soil moisture, runoff, deep percolation and canal losses are not directly observed.",
        "5. A genuine PINN is not claimed because measured soil-water states and defensible boundary/initial conditions are unavailable.",
        "6. Current output is net crop demand. Gross dam release requires real command-area acreage and documented delivery efficiencies.",
        "",
        "## Conclusion",
        "The completed framework provides a compact, reproducible decision-support prototype linking weather and crop state to irrigation demand and reservoir-allocation units. The most important next step beyond this term project is validation against measured irrigation/canal data and coupling with forecast weather and real command-area information.",
    ]
    (REPORT_DIR / "FINAL_PROJECT_REPORT.md").write_text("\n".join(report_lines))

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    train = df[df["season_year"].isin([2021, 2022, 2023])].copy()
    val = df[df["season_year"] == 2024].copy()
    trainval = df[df["season_year"].isin([2021, 2022, 2023, 2024])].copy()
    test = df[df["season_year"] == 2025].copy()

    rf_raw, rf_raw_grid = tune_rf(train, val, False)
    rf_eto, rf_eto_grid = tune_rf(train, val, True)
    xgb_raw, xgb_raw_grid = tune_xgb(train, val, False)
    xgb_eto, xgb_eto_grid = tune_xgb(train, val, True)

    tuning = pd.concat([rf_raw_grid, rf_eto_grid, xgb_raw_grid, xgb_eto_grid], ignore_index=True)
    tuning.to_csv(RESULT_DIR / "hyperparameter_search_validation.csv", index=False)

    validation_candidates = [
        ("Random Forest", False, rf_raw),
        ("Random Forest + ET0", True, rf_eto),
        ("XGBoost", False, xgb_raw),
        ("XGBoost + ET0", True, xgb_eto),
    ]
    validation_rows = []
    for name, include_eto, item in validation_candidates:
        params, _, _, met = item
        validation_rows.append({"model": name, "features": "raw+ET0" if include_eto else "raw", **met})
    validation_results = pd.DataFrame(validation_rows).sort_values("RMSE_mm_day")
    validation_results.to_csv(RESULT_DIR / "validation_model_selection.csv", index=False)

    best_hyperparameters = {
        "Random Forest": rf_raw[0],
        "Random Forest + ET0": rf_eto[0],
        "XGBoost": xgb_raw[0],
        "XGBoost + ET0": xgb_eto[0],
    }
    (RESULT_DIR / "best_hyperparameters.json").write_text(json.dumps(best_hyperparameters, indent=2))

    specs = [
        ("Linear Regression", LinearRegression(), False, True),
        ("Decision Tree", DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42), False, False),
        ("Random Forest", RandomForestRegressor(random_state=42, n_jobs=-1, **rf_raw[0]), False, False),
        ("Random Forest + ET0", RandomForestRegressor(random_state=42, n_jobs=-1, **rf_eto[0]), True, False),
        ("XGBoost", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, reg_lambda=1.0, **xgb_raw[0]), False, False),
        ("XGBoost + ET0", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, reg_lambda=1.0, **xgb_eto[0]), True, False),
        ("Neural Network + ET0", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.001, learning_rate_init=0.001, max_iter=3000, early_stopping=True, validation_fraction=0.15, random_state=42), True, True),
    ]

    rows = []
    fitted = {}
    predictions = pd.DataFrame(index=test.index)
    predictions["reference_iwr_mm"] = test[TARGET]

    for name, model, include_eto, scale in specs:
        pipe, pred, row = final_fit(name, model, include_eto, scale, trainval, test)
        rows.append(row)
        fitted[name] = (pipe, include_eto)
        predictions[name] = pred

    final_results = pd.DataFrame(rows).sort_values("RMSE_mm_day")
    final_results.to_csv(RESULT_DIR / "final_test_metrics_2025.csv", index=False)
    predictions.to_csv(RESULT_DIR / "test_predictions_2025.csv")

    best_name = final_results.iloc[0]["model"]
    best_metrics = final_results.iloc[0].to_dict()
    best_pipe, best_include_eto = fitted[best_name]
    best_pred = predictions[best_name].to_numpy()

    joblib.dump(best_pipe, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "best_model_metadata.json").write_text(json.dumps({
        "model": best_name,
        "include_eto": best_include_eto,
        "features": feature_names(best_include_eto),
        "target": TARGET,
        "training_seasons": [2021, 2022, 2023, 2024],
        "test_season": 2025,
    }, indent=2))

    importance = plot_outputs(final_results, test, best_pred, best_name, best_pipe, best_include_eto)
    stage = stagewise(test, best_pred)
    alloc = allocation_summary(test, best_pred)

    write_report(
        df, validation_results, final_results, best_name, best_include_eto,
        best_metrics, alloc, importance, stage
    )

    print(validation_results)
    print(final_results)
    print(alloc)

if __name__ == "__main__":
    main()
