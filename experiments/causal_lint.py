"""
causal-lint: is this composite variable safe to act on?

A composite variable is a weighted sum of parts, used in place of them. Total
exposure, a pooled risk score, total hours, an index. Fitting a model on top
of one always reproduces the data; acting on one is a different matter, and
nothing in the usual diagnostics tells them apart.

This reads a model and a proposed composite and answers in one screen. It
needs no interventional data and no experiment, only the model's own
coefficients, so it can be run before anything is collected.

    python -m experiments.causal_lint --demo
    python -m experiments.causal_lint --model mymodel.json

The model file is JSON:

    {
      "composite": "total_hours",
      "parts":    ["hours_maths", "hours_physics"],
      "variances": [1.0, 1.0],
      "weights":  [1.0, 1.0],
      "outcomes": {
          "final_mark": {"effects": [1.0, 0.1], "noise": 1.0}
      }
    }

Verdicts, which call for different actions:

    SAFE         act on it.
    REPAIRABLE   the weights are wrong; the fix is printed.
    IRREDUCIBLE  the outcomes disagree about what the composite should mean.
                 No weighting fixes that. Split the composite.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Sequence

import numpy as np

from core.misalignment import AuditReport, audit

BAR = "=" * 68
RULE = "-" * 68


# ---------------------------------------------------------------------------

def _bar(fraction: float, width: int = 24) -> str:
    """A text meter for the fraction of causal effect captured."""
    filled = int(round(np.clip(fraction, 0.0, 1.0) * width))
    return "#" * filled + "." * (width - filled)


def render(report: AuditReport) -> str:
    out: List[str] = [BAR, f"causal-lint: composite '{report.composite}'", BAR, ""]

    weights = "  ".join(f"{n}={w:g}" for n, w in
                        zip(report.part_names, report.weights))
    out += [f"  built from   {weights}", ""]

    badge = {"safe": "SAFE", "repairable": "UNSAFE, repairable",
             "irreducible": "UNSAFE, irreducible"}[report.verdict]
    out += [f"  VERDICT      {badge}", ""]

    out += [f"  {'outcome':<16}{'captured':>10}{'angle':>8}"
            f"{'floor':>9}{'worst':>9}", f"  {RULE[:52]}"]
    for o in report.outcomes:
        out.append(f"  {o.name:<16}{o.captured * 100:>9.1f}%{o.angle:>7.1f}d"
                   f"{o.floor:>9.4f}{o.worst_case:>9.4f}")
    out.append("")

    for o in report.outcomes:
        if o.captured > 1.0 - 1e-9:
            continue
        out.append(f"  {o.name}: captured {_bar(o.captured)} "
                   f"{o.captured * 100:.0f}%")
        detail = "  ".join(f"{n}={r:+.3f}" for n, r in
                           zip(report.part_names, o.residual))
        out.append(f"    missed effect   {detail}")
        out.append(f"    largest on      {_attribute(report.part_names, o.residual)}")
        out.append("")

    if report.recommended is not None and report.verdict != "safe":
        rec = "  ".join(f"{n}={w:+.3f}" for n, w in
                        zip(report.part_names, report.recommended))
        out += ["  SUGGESTED WEIGHTS", f"    {rec}"]
        if report.verdict == "repairable":
            out.append("    these make the composite exactly consistent")
        else:
            out.append(f"    best possible; residual misalignment "
                       f"{report.recommended_total:.4f} remains")
        out.append("")

    if report.notes:
        out.append("  NOTES")
        for n in report.notes:
            lines = _wrap(n, 60)
            out.append(f"    - {lines[0]}")
            out += [f"      {line}" for line in lines[1:]]
        out.append("")

    out.append(BAR)
    return "\n".join(out)


def _attribute(names: Sequence[str], residual: np.ndarray,
               tie: float = 0.05) -> str:
    """Name the parts carrying the missed effect.

    Naming a single variable is only honest when one clearly dominates. With
    symmetric weights the residual is often split evenly between two parts,
    and picking whichever one floating point happened to make larger would be
    an arbitrary accusation. Parts within `tie` of the largest are reported
    together.
    """
    mag = np.abs(np.asarray(residual, dtype=float))
    top = float(mag.max())
    if top <= 0:
        return "nothing"
    hits = [n for n, m in zip(names, mag) if m >= top * (1.0 - tie)]
    return hits[0] if len(hits) == 1 else " and ".join(hits) + " (evenly)"


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------

def from_spec(spec: Dict) -> AuditReport:
    names = list(spec["outcomes"].keys())
    return audit(
        B=[spec["outcomes"][k]["effects"] for k in names],
        w=np.asarray(spec["weights"], dtype=float),
        v=np.asarray(spec["variances"], dtype=float),
        v_out=[spec["outcomes"][k]["noise"] for k in names],
        part_names=spec["parts"], outcome_names=names,
        composite=spec.get("composite", "composite"),
    )


DEMOS = [
    ("A composite whose weights ignore the effects", {
        "composite": "total_hours",
        "parts": ["hours_maths", "hours_physics"],
        "variances": [1.0, 1.0], "weights": [1.0, 1.0],
        "outcomes": {"final_mark": {"effects": [1.0, 0.1], "noise": 1.0}},
    }),
    ("The same composite, reweighted by the effects", {
        "composite": "weighted_hours",
        "parts": ["hours_maths", "hours_physics"],
        "variances": [1.0, 1.0], "weights": [1.0, 0.1],
        "outcomes": {"final_mark": {"effects": [1.0, 0.1], "noise": 1.0}},
    }),
    ("One composite asked to serve two outcomes that disagree", {
        "composite": "total_hours",
        "parts": ["hours_maths", "hours_physics"],
        "variances": [1.0, 1.0], "weights": [1.0, 1.0],
        "outcomes": {"final_mark": {"effects": [1.0, 0.1], "noise": 1.0},
                     "stress_level": {"effects": [0.2, 1.0], "noise": 1.0}},
    }),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="causal-lint",
        description="Check whether a composite variable is safe to act on.")
    ap.add_argument("--model", help="JSON model specification")
    ap.add_argument("--demo", action="store_true", help="run worked examples")
    args = ap.parse_args()

    if args.demo or not args.model:
        for title, spec in DEMOS:
            print(f"\n{title}\n")
            print(render(from_spec(spec)))
        return

    with open(args.model) as fh:
        spec = json.load(fh)
    report = from_spec(spec)
    print(render(report))
    sys.exit(0 if report.verdict == "safe" else 1)


if __name__ == "__main__":
    main()
