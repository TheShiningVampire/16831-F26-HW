"""Plot Question 1 Part 4: BC return vs amount of expert data (Ant-v2).

Reads the sweep logs written by RUN.md and writes q1_part4.pdf next to the report.

Usage (from the hw1/ folder):
    python submit/plot_part4.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HW1 = os.path.dirname(HERE)
LOG_DIR = os.path.join(HERE, "run_logs")
OUT = os.path.join(HW1, "q1_part4.pdf")

INK = "#1f2328"
MUTED = "#6b7280"
SERIES = "#2563eb"


def scalar(path, key):
    """Return the value logged as `key : value`, or None if absent."""
    with open(path) as f:
        match = re.search(rf"^{key} : (-?[\d.]+)$", f.read(), re.MULTILINE)
    return float(match.group(1)) if match else None


def main():
    points = []
    for path in glob.glob(os.path.join(LOG_DIR, "q1_p4_bc_ant_data*.log")):
        n = int(re.search(r"data(\d+)\.log$", path).group(1))
        mean, std = scalar(path, "Eval_AverageReturn"), scalar(path, "Eval_StdReturn")
        if mean is None:
            print(f"skipping {os.path.basename(path)}: no Eval_AverageReturn")
            continue
        points.append((n, mean, std or 0.0))
    points.sort()

    if not points:
        raise SystemExit(f"No sweep logs found in {LOG_DIR} (expected q1_p4_bc_ant_data*.log)")

    ns = [p[0] for p in points]
    means = [p[1] for p in points]
    stds = [p[2] for p in points]

    expert_log = os.path.join(LOG_DIR, "q1_expert_ant.log")
    expert = scalar(expert_log, "Train_AverageReturn") if os.path.exists(expert_log) else None

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    if expert is not None:
        ax.axhline(expert, color=MUTED, linestyle="--", linewidth=1.0)
        ax.text(ns[0], expert, "  expert", color=MUTED, fontsize=8,
                va="bottom", ha="left")
        ax.axhline(0.3 * expert, color=MUTED, linestyle=":", linewidth=1.0)
        ax.text(ns[0], 0.3 * expert, "  30% of expert", color=MUTED, fontsize=8,
                va="bottom", ha="left")

    ax.errorbar(ns, means, yerr=stds, color=SERIES, linewidth=2.0,
                marker="o", markersize=6, capsize=3, elinewidth=1.2, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()

    ax.set_xlabel("Expert transitions used for training", color=INK)
    ax.set_ylabel("Evaluation return", color=INK)
    ax.set_ylim(bottom=0)

    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    for n, mean, std in points:
        pct = f"{100 * mean / expert:5.1f}%" if expert else "  n/a"
        print(f"  {n:5d} transitions: {mean:8.2f} +/- {std:7.2f}  ({pct} of expert)")


if __name__ == "__main__":
    main()
