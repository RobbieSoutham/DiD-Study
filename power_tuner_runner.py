"""CLI helper to run the CCUS power tuner on a CSV dataset.

This script expects a CSV file containing the columns used by `StudyConfig`
(e.g., Country, CCUS_sector, Year, Emissions, eor_capacity, etc.). It builds a
base `StudyConfig`, invokes the power tuner search, runs the drivers analysis,
and writes the markdown report to the chosen artifact directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from did_study.helpers.config import StudyConfig
from download_utils import download_from_mega
from power_tuner import ORIGINAL_COVARS, analyze_results, run_search, write_report


def build_base_config(df: pd.DataFrame, artifact_dir: str | None) -> StudyConfig:
    """Construct a baseline StudyConfig for the tuner search.

    The defaults lean on the StudyConfig dataclass and include the full
    ORIGINAL_COVARS list. Dose quantiles are left for the tuner to override.
    """

    cfg = StudyConfig(df=df, covariates=list(ORIGINAL_COVARS), artifact_dir=artifact_dir)
    # Keep defaults for other knobs; tuner will override fields during the grid search.
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CCUS power tuner")
    parser.add_argument(
        "data",
        type=str,
        help="Path to the input CSV (e.g., _no_prep.csv) containing panel rows",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=None,
        help=(
            "Optional remote URL to fetch the dataset if the local path is missing. "
            "Handy for hosted shares (e.g., Mega/transfer.it) when the CSV is not tracked."
        ),
    )
    parser.add_argument(
        "--target-mde",
        type=float,
        default=0.015,
        help="Target minimum detectable effect threshold",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided significance level for MDE and p-value thresholds",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Power target (1-beta) used in analytic MDE computation",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=160,
        help="Maximum number of grid candidates to evaluate",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory for tuner artifacts (defaults to StudyConfig.artifact_dir)",
    )
    return parser.parse_args()


def _download_dataset(url: str, dest: Path) -> Path:
    """Best-effort helper to retrieve a dataset, including Mega links if provided."""

    from subprocess import CalledProcessError, run
    from urllib import request

    dest.parent.mkdir(parents=True, exist_ok=True)

    if "mega.nz" in url:
        megadl = shutil.which("megadl")
        if megadl:
            cmd = [megadl, "--path", str(dest), url]
            try:
                run(cmd, check=True)
                return dest
            except CalledProcessError as exc:  # pragma: no cover - network/remote failures
                print(f"megadl failed for {url}: {exc}; trying Mega.py fallback", flush=True)

        # Fallback to Mega.py, which we patch for Python 3.12's removed coroutine
        mega_dest = download_from_mega(url, str(dest))
        if mega_dest:
            return Path(mega_dest)

        raise FileNotFoundError(
            "Mega download failed via megatools and Mega.py. Please update the pinned TLS key or download manually."
        )

    with request.urlopen(url) as resp, dest.open("wb") as f:  # nosec B310
        f.write(resp.read())
    return dest


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        if args.download_url:
            print(
                f"Dataset missing locally at {data_path}; attempting download from {args.download_url}...",
                flush=True,
            )
            try:
                _download_dataset(args.download_url, data_path)
            except Exception as exc:  # pragma: no cover - network/remote errors are environment-specific
                raise FileNotFoundError(
                    f"Download failed: {exc}. If the link requires an interactive session (e.g., Mega/transfer.it), "
                    "fetch the CSV manually and re-run."
                )
        else:
            raise FileNotFoundError(
                f"Input CSV not found: {data_path}. Provide a dataset such as '_no_prep.csv' in the repo root "
                "or pass --download-url to retrieve it automatically."
            )

    df = pd.read_csv(data_path)
    cfg_base = build_base_config(df, args.artifact_dir)

    results_df, summary = run_search(
        cfg_base,
        target_mde=args.target_mde,
        alpha=args.alpha,
        power_target=args.power,
        max_candidates=args.max_candidates,
    )
    analyze_results(results_df, summary["out_dir"])
    write_report(results_df, summary, target_mde=args.target_mde, out_dir=summary["out_dir"])
    print(f"Report written to {Path(summary['out_dir']) / 'power_tuner_report.md'}")


if __name__ == "__main__":
    main()
