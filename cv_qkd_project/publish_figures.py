from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PublishSpec:
    """
    Copy a selected set of stable, paper-ready figures from outputs/figures/ to figures/.

    We keep `figures/` tracked and curated, and keep `outputs/figures/` as a scratch
    directory that may be overwritten by reruns.
    """

    outputs_dir: Path = Path("outputs/figures")
    paper_dir: Path = Path("figures")
    manifest_path: Path = Path("figures/manifest.json")

    # Map: output filename -> final paper filename
    mapping: dict[str, str] = None  # set in __post_init__ style below

    def __post_init__(self):
        if self.mapping is None:
            object.__setattr__(
                self,
                "mapping",
                {
                    # Main narrative
                    "main_result.png": "main_result.png",
                    # Robustness (often supplementary, but paper-worthy)
                    "robustness_ood_generalization.png": "robustness_ood_generalization.png",
                    "robustness_mismatch_asymmetry.png": "robustness_mismatch_asymmetry.png",
                    "robustness_architecture_ablation.png": "robustness_architecture_ablation.png",
                    "robustness_speed_benchmark.png": "robustness_speed_benchmark.png",
                    # Model quality / training
                    "parity_plot.png": "parity_plot.png",
                    "training_loss.png": "training_loss.png",
                    # Core validations (keep if you want them in appendix)
                    "key_rate_validation.png": "key_rate_validation.png",
                    "key_rate_vs_delta_eta.png": "key_rate_vs_delta_eta.png",
                    "key_rate_vs_VA_mismatch_levels.png": "key_rate_vs_VA_mismatch_levels.png",
                    "optimizer_validation_unimodal.png": "optimizer_validation_unimodal.png",
                    "optimizer_validation_optimal_VA_vs_T.png": "optimizer_validation_optimal_VA_vs_T.png",
                    "optimizer_validation_key_rate_vs_distance.png": "optimizer_validation_key_rate_vs_distance.png",
                },
            )


def publish(spec: PublishSpec = PublishSpec()) -> None:
    os.makedirs(spec.paper_dir, exist_ok=True)

    copied = []
    missing = []
    for src_name, dst_name in spec.mapping.items():
        src = spec.outputs_dir / src_name
        dst = spec.paper_dir / dst_name
        if not src.exists():
            missing.append(src_name)
            continue
        shutil.copy2(src, dst)
        copied.append({"from": str(src), "to": str(dst)})

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_dir": str(spec.outputs_dir),
        "paper_dir": str(spec.paper_dir),
        "copied": copied,
        "missing": missing,
        "note": "These are copied snapshots of the current outputs/figures/*.png at publish time.",
    }
    spec.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[publish_figures] copied {len(copied)} figure(s) into {spec.paper_dir}/")
    if missing:
        print(f"[publish_figures] missing {len(missing)} figure(s): {missing}")
    print(f"[publish_figures] wrote manifest: {spec.manifest_path}")


if __name__ == "__main__":
    publish()

