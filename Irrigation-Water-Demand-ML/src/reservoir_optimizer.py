from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "results" / "final" / "test_predictions_2025.csv"
RESULT_DIR = ROOT / "results" / "final"
FIG_DIR = ROOT / "figures" / "final"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def optimize_releases(
    gross_demand_m3_day: np.ndarray,
    available_water_m3: float,
    daily_release_cap_m3_day: float,
):
    demand = np.asarray(gross_demand_m3_day, dtype=float)
    n = len(demand)

    if np.any(demand < 0):
        raise ValueError("Demand cannot be negative.")
    if available_water_m3 < 0:
        raise ValueError("Available water cannot be negative.")
    if daily_release_cap_m3_day <= 0:
        raise ValueError("Daily release cap must be positive.")

    # Variables: release_1 ... release_n, z
    # z = maximum relative deficit across all non-zero-demand days.
    c = np.zeros(n + 1)
    c[-1] = 1.0

    A = []
    b = []

    # release_t <= demand_t
    for i, d in enumerate(demand):
        row = np.zeros(n + 1)
        row[i] = 1.0
        A.append(row)
        b.append(d)

    # release_t <= daily cap
    for i in range(n):
        row = np.zeros(n + 1)
        row[i] = 1.0
        A.append(row)
        b.append(daily_release_cap_m3_day)

    # sum(release_t) <= available water
    row = np.zeros(n + 1)
    row[:n] = 1.0
    A.append(row)
    b.append(available_water_m3)

    # demand_t - release_t <= z * demand_t
    # -release_t - demand_t*z <= -demand_t
    for i, d in enumerate(demand):
        if d <= 1e-12:
            continue
        row = np.zeros(n + 1)
        row[i] = -1.0
        row[-1] = -d
        A.append(row)
        b.append(-d)

    bounds = [(0, None)] * n + [(0, 1)]

    first = linprog(
        c,
        A_ub=np.asarray(A),
        b_ub=np.asarray(b),
        bounds=bounds,
        method="highs",
    )
    if not first.success:
        raise RuntimeError(f"Release optimization failed: {first.message}")

    z_star = float(first.x[-1])

    # Second stage: among schedules with the optimal fairness level,
    # maximize total delivered water so stored water is not left unused.
    c2 = np.zeros(n + 1)
    c2[:n] = -1.0

    A2 = list(A)
    b2 = list(b)
    row = np.zeros(n + 1)
    row[-1] = 1.0
    A2.append(row)
    b2.append(z_star + 1e-9)

    second = linprog(
        c2,
        A_ub=np.asarray(A2),
        b_ub=np.asarray(b2),
        bounds=bounds,
        method="highs",
    )
    if not second.success:
        raise RuntimeError(f"Second-stage optimization failed: {second.message}")

    releases = second.x[:n]
    return releases, z_star


def run_scenario(
    area_ha: float,
    overall_efficiency: float,
    available_fraction: float,
    daily_cap_factor: float,
):
    if area_ha <= 0:
        raise ValueError("Area must be positive.")
    if not (0 < overall_efficiency <= 1):
        raise ValueError("Efficiency must be in (0,1].")
    if not (0 < available_fraction <= 1):
        raise ValueError("Available fraction must be in (0,1].")
    if daily_cap_factor <= 0:
        raise ValueError("Daily cap factor must be positive.")

    pred = pd.read_csv(PRED_PATH, parse_dates=["date"]).set_index("date")
    iwr = pred["XGBoost"].to_numpy(dtype=float)

    net_demand = 10.0 * iwr * area_ha
    gross_demand = net_demand / overall_efficiency

    full_gross_requirement = float(gross_demand.sum())
    available_water = available_fraction * full_gross_requirement
    daily_cap = daily_cap_factor * float(gross_demand.mean())

    release, max_relative_deficit = optimize_releases(
        gross_demand,
        available_water,
        daily_cap,
    )

    delivered_net = release * overall_efficiency
    demand_met_fraction = np.divide(
        release,
        gross_demand,
        out=np.ones_like(release),
        where=gross_demand > 1e-12,
    )
    deficit = np.maximum(gross_demand - release, 0.0)
    storage_remaining = available_water - np.cumsum(release)

    out = pd.DataFrame(
        {
            "predicted_iwr_mm_day": iwr,
            "net_crop_demand_m3_day": net_demand,
            "gross_release_demand_m3_day": gross_demand,
            "optimized_release_m3_day": release,
            "delivered_net_water_m3_day": delivered_net,
            "gross_deficit_m3_day": deficit,
            "demand_met_fraction": demand_met_fraction,
            "allocated_storage_remaining_m3": storage_remaining,
        },
        index=pred.index,
    )
    out.index.name = "date"
    out.to_csv(RESULT_DIR / "reservoir_stress_test_2025.csv")

    summary = {
        "scenario_area_ha": area_ha,
        "scenario_overall_efficiency": overall_efficiency,
        "scenario_available_fraction_of_full_requirement": available_fraction,
        "daily_release_cap_factor_vs_mean_gross_demand": daily_cap_factor,
        "full_season_gross_requirement_m3": full_gross_requirement,
        "available_seasonal_water_m3": available_water,
        "daily_release_cap_m3_day": daily_cap,
        "optimized_total_release_m3": float(release.sum()),
        "total_gross_deficit_m3": float(deficit.sum()),
        "water_budget_utilization_pct": 100.0 * float(release.sum()) / available_water if available_water else 0.0,
        "minimum_daily_demand_met_pct": 100.0 * float(demand_met_fraction.min()),
        "mean_daily_demand_met_pct": 100.0 * float(demand_met_fraction.mean()),
        "maximum_relative_deficit_pct": 100.0 * max_relative_deficit,
        "final_allocated_storage_remaining_m3": float(storage_remaining[-1]),
    }
    pd.DataFrame([summary]).to_csv(
        RESULT_DIR / "reservoir_stress_summary_2025.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(out.index, out["gross_release_demand_m3_day"], label="Gross release required")
    ax.plot(out.index, out["optimized_release_m3_day"], label="Optimized release")
    ax.set_ylabel("m3/day")
    ax.set_title("2025 Reservoir Stress Test: Demand vs Optimized Release")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_reservoir_demand_vs_optimized_release.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(out.index, out["allocated_storage_remaining_m3"])
    ax.set_ylabel("Allocated seasonal water remaining (m3)")
    ax.set_title("Reservoir Allocation Budget Trajectory")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_reservoir_storage_budget_trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(out.index, 100.0 * out["demand_met_fraction"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Demand met (%)")
    ax.set_title("Daily Irrigation Demand Satisfaction Under Water Constraint")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_daily_demand_satisfaction.png", dpi=180)
    plt.close(fig)

    return out, summary


def main():
    parser = argparse.ArgumentParser(
        description="Optimize reservoir release for predicted crop irrigation demand."
    )
    parser.add_argument("--area-ha", type=float, default=1000.0)
    parser.add_argument("--overall-efficiency", type=float, default=0.70)
    parser.add_argument(
        "--available-fraction",
        type=float,
        default=0.85,
        help="Stress-test seasonal water budget as a fraction of full gross demand.",
    )
    parser.add_argument(
        "--daily-cap-factor",
        type=float,
        default=1.40,
        help="Daily release cap divided by mean gross daily demand.",
    )
    args = parser.parse_args()

    _, summary = run_scenario(
        area_ha=args.area_ha,
        overall_efficiency=args.overall_efficiency,
        available_fraction=args.available_fraction,
        daily_cap_factor=args.daily_cap_factor,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
