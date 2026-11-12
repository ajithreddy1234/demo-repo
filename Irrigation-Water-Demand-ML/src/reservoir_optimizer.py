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

    report = [
        "# Reservoir-Constrained Allocation Upgrade",
        "",
        "## Why this was added",
        "The ML model estimates crop irrigation demand. This optimization layer converts that demand into an operational release schedule when the seasonal irrigation water budget is limited.",
        "",
        "## Optimization objective",
        "Minimize the maximum relative daily irrigation deficit across the season, subject to:",
        "- release on a day cannot exceed that day's gross irrigation demand,",
        "- release cannot exceed the configured daily canal/release capacity,",
        "- cumulative release cannot exceed the available seasonal irrigation-water budget.",
        "",
        "A second optimization stage maximizes total delivered water while preserving the best achievable fairness level.",
        "",
        "## Demonstration scenario",
        f"- Irrigated maize area: {area_ha:.0f} ha",
        f"- Overall delivery efficiency: {overall_efficiency:.2f}",
        f"- Available seasonal water: {100.0 * available_fraction:.1f}% of full gross requirement",
        f"- Daily release-cap factor: {daily_cap_factor:.2f} x mean gross daily demand",
        "",
        "These values are an explicit stress-test scenario, not claimed field measurements.",
        "",
        "## Results",
        f"- Full gross seasonal requirement: {full_gross_requirement:.2f} m3",
        f"- Available seasonal water: {available_water:.2f} m3",
        f"- Optimized total release: {float(release.sum()):.2f} m3",
        f"- Total gross deficit: {float(deficit.sum()):.2f} m3",
        f"- Minimum daily demand met: {100.0 * float(demand_met_fraction.min()):.2f}%",
        f"- Mean daily demand met: {100.0 * float(demand_met_fraction.mean()):.2f}%",
        f"- Maximum relative deficit: {100.0 * max_relative_deficit:.2f}%",
        f"- Final unused allocation budget: {float(storage_remaining[-1]):.2f} m3",
        "",
        "## Interpretation",
        "This layer changes the project from demand prediction alone into reservoir decision support. When water is sufficient, the optimizer can meet predicted demand. Under shortage, it distributes the constrained water budget while minimizing the worst relative daily deficit rather than applying an arbitrary release rule.",
        "",
        "## Operational use",
        "For a real reservoir deployment, the same optimizer accepts actual irrigated area, delivery efficiency, available irrigation allocation and daily release/canal capacity. Those inputs should come from the operating authority rather than being invented by the model.",
    ]
    (ROOT / "reports" / "RESERVOIR_OPTIMIZATION.md").write_text("\n".join(report))

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



def scenario_metrics(
    area_ha: float,
    overall_efficiency: float,
    available_fraction: float,
    daily_cap_factor: float,
    scenario_name: str,
):
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

    demand_met_fraction = np.divide(
        release,
        gross_demand,
        out=np.ones_like(release),
        where=gross_demand > 1e-12,
    )
    deficit = np.maximum(gross_demand - release, 0.0)

    return {
        "scenario": scenario_name,
        "available_fraction": available_fraction,
        "daily_cap_factor": daily_cap_factor,
        "full_gross_requirement_m3": full_gross_requirement,
        "available_water_m3": available_water,
        "optimized_release_m3": float(release.sum()),
        "gross_deficit_m3": float(deficit.sum()),
        "water_budget_utilization_pct": 100.0 * float(release.sum()) / available_water if available_water else 0.0,
        "minimum_daily_demand_met_pct": 100.0 * float(demand_met_fraction.min()),
        "mean_daily_demand_met_pct": 100.0 * float(demand_met_fraction.mean()),
        "maximum_relative_deficit_pct": 100.0 * max_relative_deficit,
    }

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
    comparison = pd.DataFrame([
        scenario_metrics(
            area_ha=args.area_ha,
            overall_efficiency=args.overall_efficiency,
            available_fraction=1.0,
            daily_cap_factor=10.0,
            scenario_name="Full supply, unconstrained release",
        ),
        scenario_metrics(
            area_ha=args.area_ha,
            overall_efficiency=args.overall_efficiency,
            available_fraction=0.85,
            daily_cap_factor=10.0,
            scenario_name="15% seasonal shortage only",
        ),
        scenario_metrics(
            area_ha=args.area_ha,
            overall_efficiency=args.overall_efficiency,
            available_fraction=args.available_fraction,
            daily_cap_factor=args.daily_cap_factor,
            scenario_name="15% shortage + daily release cap",
        ),
    ])
    comparison.to_csv(RESULT_DIR / "reservoir_scenario_comparison_2025.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(comparison["scenario"], comparison["minimum_daily_demand_met_pct"])
    ax.set_ylabel("Minimum daily demand met (%)")
    ax.set_title("Reservoir Constraint Scenarios")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_reservoir_scenario_comparison.png", dpi=180)
    plt.close(fig)

    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nScenario comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
