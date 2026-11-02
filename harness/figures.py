"""
Figures for the falsification harness.

Two panels, because the result is two claims that have to be read together.

(a) THE SCOREBOARD. Every claim the harness tests, with the fraction of draws
    that falsified it. Fourteen sit at zero. This panel exists so that the one
    that does not can be believed: a harness that tested only the claim it
    breaks would be evidence of nothing.

(b) FALSIFICATION AGAINST EDGE DENSITY. The structural reading of the
    disputed theorem, swept across the one parameter the framework leaves
    free. The curve reaches zero exactly at a complete graph, which is the
    published proposition about complete DAGs recovered rather than assumed,
    and it tracks the fraction of sampled models with two or more roots,
    which is the mechanism.

The annotation at the right of (b) is the point of the whole exercise: the
project's first test suite drew models at that single density, and so could
never have produced any other column of this chart.

Run:  python -m harness.figures
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from harness import generators as gen
from harness.properties import ALL_PROPERTIES, Config
from harness.runner import density_sweep, run_property

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")

# validated with the dataviz palette checker: all six checks pass on a light
# surface, worst adjacent CVD separation well above the floor
FINDING = "#B2182B"     # the claim that falls
CONTEXT = "#2166AC"     # the structural quantity that explains it
INK = "#14213D"
MUTED = "#6A7179"
GRID = "#DDE3EA"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
})


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def gather(trials: int, seed: int, densities: List[float]) -> Dict:
    """Run what the figure needs. Kept separate from plotting so the numbers
    can be inspected and cached rather than hidden inside a draw call."""
    cfg = Config(scm_space=gen.SCMSpace(edge_prob=0.5))

    board = []
    for i, prop in enumerate(ALL_PROPERTIES):
        rep = run_property(prop, trials, seed + i, cfg)
        evaluated = rep.passed + rep.falsified
        board.append({
            "name": rep.name,
            "source": rep.source,
            "rate": rep.falsified / evaluated if evaluated else 0.0,
            "falsified": rep.falsified,
            "evaluated": evaluated,
        })

    target = next(p for p in ALL_PROPERTIES
                  if p.__name__ == "prop_closure_unrestricted_mechanisms")
    sweep = density_sweep(target, densities, trials, seed, cfg)
    return {"scoreboard": board, "sweep": sweep,
            "config": {"trials": trials, "seed": seed}}


def draw(data: Dict, out_stem: str) -> None:
    board = sorted(data["scoreboard"], key=lambda r: (r["rate"], r["name"]))
    sweep = data["sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.3),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- panel (a): the scoreboard ---------------------------------------
    y = np.arange(len(board))
    rates = np.array([r["rate"] for r in board])
    colours = [FINDING if r > 0 else CONTEXT for r in rates]

    ax1.hlines(y, 0, rates, color=colours, linewidth=2.0, alpha=0.85)
    ax1.scatter(rates, y, s=26, color=colours, zorder=3,
                edgecolor="white", linewidth=0.8)
    # zero-rate claims get a visible marker on the axis, not an invisible one
    ax1.scatter(np.zeros(int((rates == 0).sum())), y[rates == 0], s=26,
                facecolor="white", edgecolor=CONTEXT, linewidth=1.2, zorder=4)

    ax1.set_yticks(y)
    ax1.set_yticklabels([_pretty(r["name"]) for r in board], fontsize=6.4)
    ax1.set_xlim(-0.03, 1.08)
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_xlabel("draws falsified")
    ax1.set_title("(a)  every claim the harness tests", fontsize=8.5,
                  loc="left", color=INK, pad=8)
    ax1.grid(axis="x", color=GRID, linewidth=0.5)
    ax1.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax1.spines[side].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    for yi, r in zip(y, board):
        if r["rate"] > 0:
            ax1.annotate(f"{r['rate'] * 100:.0f}%", (r["rate"], yi),
                         xytext=(6, 0), textcoords="offset points",
                         va="center", fontsize=6.8, color=FINDING, weight="bold")

    n_survive = sum(1 for r in board if r["rate"] == 0)
    ax1.annotate(f"{n_survive} of {len(board)} claims survive\n"
                 f"{data['config']['trials']} draws each",
                 xy=(0.52, 0.17), xycoords="axes fraction",
                 fontsize=6.8, color=MUTED, ha="left")

    # ---- panel (b): falsification against edge density --------------------
    x = np.array([r["edge_prob"] for r in sweep])
    rate = np.array([r["falsification_rate"] for r in sweep])
    roots = np.array([r["frac_two_or_more_roots"] for r in sweep])
    order = np.argsort(x)
    x, rate, roots = x[order], rate[order], roots[order]

    ax2.plot(x, roots, color=CONTEXT, linewidth=1.6, linestyle=(0, (4, 2)),
             marker="s", markersize=3.6, markeredgecolor="white",
             markeredgewidth=0.6, label="models with 2 or more roots")
    ax2.plot(x, rate, color=FINDING, linewidth=2.0, marker="o", markersize=4.6,
             markeredgecolor="white", markeredgewidth=0.8,
             label="draws falsifying closure")

    ax2.set_xlabel("edge probability of the sampled DAG")
    ax2.set_ylabel("fraction of draws")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_xlim(0.14, 1.06)
    ax2.set_title("(b)  the failure is structural", fontsize=8.5,
                  loc="left", color=INK, pad=8)
    ax2.grid(color=GRID, linewidth=0.5)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.legend(frameon=False, fontsize=6.8, loc="upper right",
               handlelength=2.4, labelcolor=INK)

    # the annotation that carries the argument
    ax2.scatter([1.0], [0.0], s=58, facecolor="white", edgecolor=FINDING,
                linewidth=1.6, zorder=5)
    # The white backing matters: without it the label crosses the falsification
    # curve in the low-rate stretch and both become hard to read.
    ax2.annotate("the project's first test suite\ndrew every model here,\n"
                 "where the theorem holds",
                 xy=(0.99, 0.015), xytext=(0.40, 0.17),
                 fontsize=6.6, color=INK, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                           edgecolor="none", alpha=0.92),
                 arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8,
                                 shrinkA=4, shrinkB=7,
                                 connectionstyle="arc3,rad=0.22"))

    fig.tight_layout(pad=0.9)
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{out_stem}.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_macros(data: Dict, path: str) -> None:
    """The three sweep numbers the paper quotes, so none is typed by hand."""
    sweep = sorted(data["sweep"], key=lambda r: r["edge_prob"])
    lo, hi = sweep[0], sweep[-1]
    lines = [
        "% Generated by harness/figures.py. Do not edit by hand.",
        f"\\newcommand{{\\SweepLoDensity}}{{{lo['edge_prob']:.1f}}}",
        f"\\newcommand{{\\SweepLoRate}}{{{lo['falsification_rate'] * 100:.1f}}}",
        f"\\newcommand{{\\SweepHiDensity}}{{{hi['edge_prob']:.1f}}}",
        f"\\newcommand{{\\SweepHiRate}}{{{hi['falsification_rate'] * 100:.1f}}}",
        f"\\newcommand{{\\SweepHiRoots}}"
        f"{{{hi['frac_two_or_more_roots'] * 100:.1f}}}",
        f"\\newcommand{{\\SweepPoints}}{{{len(sweep)}}}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    densities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    # trials and seed match experiments/paper_tables.py, so panel (a) and the
    # paper's Table 1 report the same run rather than two similar ones
    data = gather(trials=2000, seed=20261018, densities=densities)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "figure_data.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    draw(data, "harness_scoreboard")
    write_macros(data, os.path.join(ROOT, "paper", "sweep_numbers.tex"))
    print(f"wrote {FIGS}/harness_scoreboard.pdf and .png")
    print(f"wrote {ROOT}/paper/sweep_numbers.tex")
    print(f"wrote {RESULTS}/figure_data.json")
    for r in data["sweep"]:
        print(f"  edge_prob {r['edge_prob']:.2f}  "
              f"falsified {r['falsification_rate'] * 100:5.1f}%  "
              f"2+ roots {r['frac_two_or_more_roots'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
