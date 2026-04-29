from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_three_experiments(
    exp1_csv: str | os.PathLike = "outputs/results/experiment1.csv",
    exp1b_csv: str | os.PathLike = "outputs/results/experiment1b_mismatch_opt.csv",
    exp2_csv: str | os.PathLike = "outputs/results/experiment2.csv",
    exp3_csv: str | os.PathLike = "outputs/results/experiment3.csv",
    out_png: str | os.PathLike = "outputs/figures/main_result.png",
) -> None:
    """
    Load results from experiments 1–3 and overlay key-rate curves vs transmittance.

    Saves an image suitable for a main result figure.
    """
    df1 = pd.read_csv(exp1_csv)
    df1b = pd.read_csv(exp1b_csv)
    df2 = pd.read_csv(exp2_csv)
    df3 = pd.read_csv(exp3_csv)

    # Consistent palette
    colors = {
        "ideal": "#1f77b4",  # blue
        "mismatch_opt": "#9467bd",  # purple
        "naive": "#d62728",  # red
        "adaptive": "#2ca02c",  # green
    }

    plt.figure(figsize=(8, 5))
    plt.plot(df1["T"], df1["K_max"], color=colors["ideal"], lw=2.0, label="Ideal (brute-force, no mismatch)")
    plt.plot(
        df1b["T"],
        df1b["K_max"],
        color=colors["mismatch_opt"],
        lw=2.0,
        label="Mismatch-optimal (brute-force, hardware-aware)",
    )
    plt.plot(df2["T"], df2["K_actual"], color=colors["naive"], lw=2.0, label="Naive (unaware, mismatch present)")
    plt.plot(df3["T"], df3["K_actual"], color=colors["adaptive"], lw=2.0, label="Adaptive (NN, mismatch aware)")

    plt.xlabel("Channel transmittance T")
    plt.ylabel("Secure key rate K (bits / channel use)")
    plt.title("CV-QKD key rate under detector mismatch: ideal vs naive vs adaptive")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()

    out_png = Path(out_png)
    os.makedirs(out_png.parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    plot_three_experiments()

