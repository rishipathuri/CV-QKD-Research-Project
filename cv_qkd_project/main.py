from __future__ import annotations

from cv_qkd_project import config


def _print_param(name: str, value) -> None:
    print(f"{name} = {value}")


def main() -> None:
    _print_param("T_MIN", config.T_MIN)
    _print_param("T_MAX", config.T_MAX)
    _print_param("XI_MIN", config.XI_MIN)
    _print_param("XI_MAX", config.XI_MAX)
    _print_param("ETA_MIN", config.ETA_MIN)
    _print_param("ETA_MAX", config.ETA_MAX)
    _print_param("V_EL", config.V_EL)
    _print_param("BETA", config.BETA)
    _print_param("V_A_GRID_SIZE", config.V_A_GRID_SIZE)
    _print_param("V_A_MIN", config.V_A_MIN)
    _print_param("V_A_MAX", config.V_A_MAX)
    _print_param("V_A_GRID (first 5)", config.V_A_GRID[:5])
    _print_param("V_A_GRID (last 5)", config.V_A_GRID[-5:])
    _print_param("DATASET_SIZE_N", config.DATASET_SIZE_N)


if __name__ == "__main__":
    main()

