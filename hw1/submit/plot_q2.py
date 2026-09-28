"""Plot Question 2: DAgger learning curves (Ant-v2 left, Hopper-v2 right).

Reads the DAgger logs written by RUN.md, plus the Part 3 BC logs for the BC reference
line, and writes q2_dagger.pdf next to the report.

Usage (from the hw1/ folder):
    python submit/plot_q2.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
HW1 = os.path.dirname(HERE)
LOG_DIR = os.path.join(HERE, "run_logs")
OUT = os.path.join(HW1, "q2_dagger.pdf")

INK = "#1f2328"
MUTED = "#6b7280"
SERIES = "#2563eb"

PANELS = [("ant", "Ant-v2"), ("hopper", "Hopper-v2")]


def values(path, key):
    """Every value logged as `key : value`, in the order they appear."""
    with open(path) as f:
        return [float(v) for v in re.findall(rf"^{key} : (-?[\d.]+)$", f.read(), re.MULTILINE)]


def first(path, key):
    found = values(path, key) if os.path.exists(path) else []
    return found[0] if found else None


def draw(ax, short, env):
    log = os.path.join(LOG_DIR, f"q2_dagger_{short}.log")
    if not os.path.exists(log):
        raise SystemExit(f"Missing {log}. Run the Question 2 commands in RUN.md first.")

    means = values(log, "Eval_AverageReturn")
    stds = values(log, "Eval_StdReturn")
    iters = list(range(len(means)))

    expert = first(log, "Initial_DataCollection_AverageReturn")
    bc = first(os.path.join(LOG_DIR, f"q1_bc_{short}.log"), "Eval_AverageReturn")

    if expert is not None:
        ax.axhline(expert, color=MUTED, linestyle="--", linewidth=1.0)
    if bc is not None:
        ax.axhline(bc, color=MUTED, linestyle=":", linewidth=1.2)
    ax.errorbar(iters, means, yerr=stds, color=SERIES, linewidth=2.0,
                marker="o", markersize=5, capsize=3, elinewidth=1.2, zorder=3)

    ax.set_title(env, color=INK, fontsize=10)
    ax.set_xlabel("DAgger iteration", color=INK)
    ax.set_ylabel("Evaluation return", color=INK)
    ax.set_xticks(iters)

    # Always keep a zero baseline so the curve's approach to the expert line reads as
    # saturation rather than being exaggerated by a zoomed axis.
    span = [m + s for m, s in zip(means, stds)]
    span += [v for v in (expert, bc) if v is not None]
    ax.set_ylim(bottom=0, top=max(span) * 1.08)

    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=9)

    print(f"{env}: expert {expert}, BC {bc}")
    for i, (m, s) in enumerate(zip(means, stds)):
        pct = f"{100 * m / expert:5.1f}%" if expert else "  n/a"
        print(f"  iter {i}: {m:9.2f} +/- {s:8.2f}  ({pct} of expert)")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, (short, env) in zip(axes, PANELS):
        draw(ax, short, env)

    handles = [
        Line2D([], [], color=SERIES, linewidth=2.0, marker="o", markersize=5,
               label="DAgger (mean $\\pm$ std)"),
        Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0, label="Expert"),
        Line2D([], [], color=MUTED, linestyle=":", linewidth=1.2, label="BC"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
