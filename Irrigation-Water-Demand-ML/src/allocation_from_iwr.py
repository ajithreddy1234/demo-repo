from __future__ import annotations

import argparse

def allocation_from_iwr(
    iwr_mm_day: float,
    area_ha: float,
    conveyance_efficiency: float | None = None,
    application_efficiency: float | None = None,
):
    if iwr_mm_day < 0:
        raise ValueError("IWR cannot be negative.")
    if area_ha <= 0:
        raise ValueError("Area must be positive.")

    net_m3_per_ha_day = 10.0 * iwr_mm_day
    net_command_area_m3_day = net_m3_per_ha_day * area_ha

    result = {
        "iwr_mm_day": iwr_mm_day,
        "area_ha": area_ha,
        "net_m3_per_ha_day": net_m3_per_ha_day,
        "net_command_area_m3_day": net_command_area_m3_day,
    }

    if conveyance_efficiency is not None or application_efficiency is not None:
        if conveyance_efficiency is None or application_efficiency is None:
            raise ValueError("Provide both efficiencies or neither.")
        if not (0 < conveyance_efficiency <= 1):
            raise ValueError("Conveyance efficiency must be in (0,1].")
        if not (0 < application_efficiency <= 1):
            raise ValueError("Application efficiency must be in (0,1].")

        overall_efficiency = conveyance_efficiency * application_efficiency
        result["overall_efficiency"] = overall_efficiency
        result["gross_reservoir_release_m3_day"] = net_command_area_m3_day / overall_efficiency

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iwr-mm", type=float, required=True)
    parser.add_argument("--area-ha", type=float, required=True)
    parser.add_argument("--conveyance-efficiency", type=float)
    parser.add_argument("--application-efficiency", type=float)
    args = parser.parse_args()

    result = allocation_from_iwr(
        args.iwr_mm,
        args.area_ha,
        args.conveyance_efficiency,
        args.application_efficiency,
    )
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
