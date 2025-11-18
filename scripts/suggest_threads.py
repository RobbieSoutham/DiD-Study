from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # optional


def suggest_threads(use_wcb: bool, honestdid: bool) -> int:
    cpus = os.cpu_count() or 2
    # memory in GB if psutil is available
    mem_gb: Optional[float] = None
    if psutil is not None:
        try:
            mem_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            mem_gb = None

    # base: leave one core for OS
    base = max(1, cpus - 1)

    if use_wcb or honestdid:
        # R-backed bootstraps are CPU and memory intensive.
        # Be conservative: cap at 4 on <=16GB boxes, 6 on <=32GB, else min(base, 8).
        if mem_gb is None:
            return min(base, 4)
        if mem_gb <= 16:
            return min(base, 4)
        if mem_gb <= 32:
            return min(base, 6)
        return min(base, 8)
    else:
        # Analytic-only is lighter; allow more parallelism but keep headroom.
        if mem_gb is None:
            return min(base, 8)
        if mem_gb <= 8:
            return min(base, 4)
        if mem_gb <= 16:
            return min(base, 8)
        return min(base, 12)


def main() -> None:
    ap = argparse.ArgumentParser(description="Suggest a number of parallel threads for the power tuner")
    ap.add_argument("--use-wcb", action="store_true", help="Assume R WCB will be used")
    ap.add_argument("--honestdid", action="store_true", help="Assume HonestDiD (R) will be used")
    args = ap.parse_args()

    rec = suggest_threads(args.use_wcb, args.honestdid)
    print(f"Recommended threads: {rec}")


if __name__ == "__main__":
    main()

