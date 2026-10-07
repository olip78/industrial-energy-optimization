"""Run the nested cross-fitted PV Oracle experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from energy.modeling import CrossFittedPVExperiment, load_pv_model_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()

    frame = load_pv_model_frame(arguments.project_root, year=arguments.year)
    result = CrossFittedPVExperiment(
        outer_splits=arguments.outer_splits,
        inner_splits=arguments.inner_splits,
        random_state=arguments.random_state,
    ).run(frame)
    output_dir = arguments.output_dir or (
        arguments.project_root / "artifacts" / "pv_crossfit" / str(arguments.year)
    )
    result.write(output_dir)
    print(result.average_fold_metrics)
    print(result.pooled_oof_metrics)
    print(f"Wrote artefacts to {output_dir}")


if __name__ == "__main__":
    main()
