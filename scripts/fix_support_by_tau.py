"""
Regenerate Figure 3 (support_by_tau) with full event-time window τ ∈ [-10, +10]
using the existing results/data/support_by_tau.csv. The figure-styling agent's
earlier rebuild cropped the figure to τ ≤ 0; this script fixes it to show the
full estimation window referenced by the §4.5 prose.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
try:
    from did_study.reporting.style import apply_paper_theme, dose_colour, ref_line  # type: ignore
    apply_paper_theme()
    bar_colour = dose_colour("Large")
    _have_helpers = True
except Exception:
    sns.set_theme(style="whitegrid", context="paper")
    bar_colour = "#1f4e79"
    _have_helpers = False

CSV = HERE / "results" / "data" / "support_by_tau.csv"
OUT_LOCAL = HERE / "results" / "figures" / "support_by_tau.png"
OUT_PAPER = Path("/home/vboxuser/.openclaw/workspace/workflows/phd/papers/b_ccus_did/figures/support_by_tau.png")

df = pd.read_csv(CSV)
df = df.dropna(subset=["event_time"]).copy()
df["event_time"] = df["event_time"].astype(int)
df = df[df["event_time"].between(-10, 10)].sort_values("event_time")

fig, ax = plt.subplots(figsize=(8.0, 4.4))
ax.bar(df["event_time"], df["units"], color=bar_colour, edgecolor="white", width=0.85)
ax.set_xlabel(r"Event time ($\tau$)")
ax.set_ylabel("Number of treated units")
ax.set_xticks(np.arange(-10, 11, 1))
ax.set_yticks(np.arange(0, 8, 1))
ax.set_xlim(-10.6, 10.6)
ax.set_ylim(0, 7.6)
if _have_helpers:
    ref_line(ax, tau=-1)
else:
    ax.axvline(-1, color="#000000", linewidth=1.0, linestyle="-", alpha=0.85)
for tick_label in ax.get_xticklabels():
    tick_label.set_fontsize(9)

fig.tight_layout()
fig.savefig(OUT_LOCAL, dpi=220, bbox_inches="tight")
fig.savefig(OUT_PAPER, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT_LOCAL}")
print(f"Wrote {OUT_PAPER}")
print(f"Plotted window: tau in [{df['event_time'].min()}, {df['event_time'].max()}]")
print(df.to_string(index=False))
