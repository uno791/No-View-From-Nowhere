"""
The paper's two generated tables, and the macros that go with them.

The paper states sixteen claims and a shrink trace. Neither is typed by hand.
This script runs the harness at the paper's trial count, shrinks each
counterexample, and writes three files the document inputs directly:

    paper/claims_table.tex    one row per claim: name, source, verdict
    paper/shrink_trace.tex    the reduction, step by step
    paper/harness_numbers.tex the macros the prose quotes

Keeping the tables generated rather than transcribed means a reader who
re-runs the code gets a document whose tables match it, and it removes the
commonest way for a paper's numbers to drift from its code: a table edited
once and never regenerated.

Run:  python -m experiments.paper_tables
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict

import numpy as np

from core.scm import LinearGaussianSCM
from harness import generators as gen
from harness import shrink as sh
from harness.properties import ALL_PROPERTIES, Config
from harness.runner import run_property, seed_sweep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
RESULTS = os.path.join(ROOT, "results")

TRIALS = 2000
SEED = 20261018
SEED_SWEEP_SEEDS = 200
SEED_SWEEP_DRAWS = 400

# which exclusion tests apply to which reading of closure
WHICH = {
    "closure_linear_gaussian": ("gaussian_marginals",),
    "closure_unrestricted_mechanisms": ("root_independence", "ancestral_independence"),
}

# A short human reading of each claim, for the table's first column. The
# machine-readable statement lives in the property's own docstring; this is the
# phrasing a reader skimming the table needs, and it is the one thing in this
# file written rather than computed.
LABEL = {
    "intervention_is_natural":
        "an intervention commutes with the model's own structure",
    "transport_gives_interventional_law":
        "pushing the observational law along the transport gives the interventional law",
    "convex_space_axioms":
        "measures on a fixed space form a convex space under blending",
    "closure_linear_gaussian":
        "a blend of two reachable measures is reachable (linear Gaussian mechanisms)",
    "closure_unrestricted_mechanisms":
        "a blend of two reachable measures is reachable (any measurable mechanism)",
    "pushforward_is_affine":
        "an abstraction's endogenous component commutes with blending",
    "encoding_functor_laws":
        "the encoding preserves identities and composition",
    "mixing_map_round_trip":
        "the mixing map is invertible and recovers the noise exactly",
    "endogenous_component_unique":
        "the intervention transport is the only map with its defining property",
    "ic_criterion":
        "an aggregation is interventionally consistent exactly when the criterion holds",
    "extension_is_right_inverse":
        "both extension maps are right inverses of restriction",
    "kernel_extension_preserves_kl":
        "the kernel extension preserves divergence exactly",
    "restriction_does_not_increase_kl":
        "restriction to a backbone cannot increase divergence",
    "kernel_extension_fixes_own_law":
        "a subject fed its own projection recovers its own law",
    "energy_zero_iff_section":
        "disagreement energy vanishes exactly on global sections",
    "global_sections_closed_under_blending":
        "a blend of two global sections is a global section",
}


def _tex(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))


# ---------------------------------------------------------------------------

def run() -> Dict:
    cfg = Config(scm_space=gen.SCMSpace(edge_prob=0.5))
    reports, rows = [], []

    for i, prop in enumerate(ALL_PROPERTIES):
        rep = run_property(prop, TRIALS, SEED + i, cfg)
        reports.append(rep)
        evaluated = rep.passed + rep.falsified
        rows.append({
            "name": rep.name,
            "source": rep.source,
            "label": LABEL.get(rep.name, rep.statement),
            "verdict": rep.verdict,
            "evaluated": evaluated,
            "falsified": rep.falsified,
            "rate": rep.falsified / evaluated if evaluated else 0.0,
            "first_failure_at": rep.first_failure_at,
        })

    shrinks, sweeps = {}, {}
    for rep in reports:
        if rep.name not in WHICH or not rep.falsified or not rep.example:
            continue
        ex = rep.example
        scm = LinearGaussianSCM(np.array(ex["C"], dtype=float),
                                np.array(ex["noise_var"], dtype=float),
                                [f"X{i}" for i in range(len(ex["noise_var"]))])
        out = sh.shrink(sh.case_from_draw(scm, ex["intervention"], ex["lambda"]),
                        WHICH[rep.name])
        m = out.minimal
        shrinks[rep.name] = {
            "start": out.start.describe(), "minimal": m.describe(),
            "steps": out.steps, "candidates": out.candidates_tried,
            "trace": out.trace,
            "start_size": list(out.start.size()), "minimal_size": list(m.size()),
            "minimal_case": {"C": [list(r) for r in m.C],
                             "noise_var": list(m.noise_var), "kind": m.kind,
                             "targets": list(m.targets), "lam": m.lam,
                             "a": [list(r) for r in m.a], "b": [list(r) for r in m.b]},
        }
        prop = next(p for p in ALL_PROPERTIES if p.__name__.endswith(rep.name))
        sweeps[rep.name] = seed_sweep(prop, SEED_SWEEP_SEEDS, SEED_SWEEP_DRAWS, cfg)

    return {"trials": TRIALS, "seed": SEED, "rows": rows,
            "shrink": shrinks, "seed_sweep": sweeps}


# ---------------------------------------------------------------------------
# the three generated files
# ---------------------------------------------------------------------------

# A generated table carries its own \begin{tabular}, column spec and rules.
# Emitting rows alone and \input-ing them inside a tabular does not work:
# LaTeX's \input opens the first cell before TeX sees what the file starts
# with, so \multicolumn, \midrule and the paper's closing \bottomrule all end
# up misplaced. Writing the whole environment keeps the \input in ordinary
# mode, where none of that arises.

def write_claims_table(res: Dict, path: str) -> None:
    src_order = {"source": 0, "ours": 1}
    rows = sorted(res["rows"],
                  key=lambda r: (src_order.get(r["source"].split()[0], 2),
                                 r["verdict"] != "FALSIFIED", r["name"]))
    lines = ["% Generated by experiments/paper_tables.py. Do not edit by hand.",
             "\\setlength{\\tabcolsep}{5pt}",
             "\\renewcommand{\\arraystretch}{1.25}",
             "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.42\\textwidth}"
             ">{\\raggedright\\arraybackslash}p{0.15\\textwidth}lr@{}}",
             "\\toprule",
             "claim, in words & origin & verdict & first fail\\\\",
             "\\midrule",
             "\\multicolumn{4}{l}{\\textit{Claims of the published framework"
             " \\cite{dacunto2025}\\ldots}}\\\\",
             "\\addlinespace[2pt]"]
    seen_ours = False
    stripe = False
    for r in rows:
        if r["source"].startswith("ours") and not seen_ours:
            lines += ["\\addlinespace[2pt]", "\\midrule",
                      "\\multicolumn{4}{l}{\\textit{\\ldots and claims established"
                      " in our earlier work, restated as hypotheses}}\\\\",
                      "\\addlinespace[2pt]"]
            seen_ours = True
            stripe = False
        # "source Theorem 1" -> "Theorem 1";  "ours, extension maps" -> "extension maps"
        origin = r["source"].split(",", 1)[-1].strip()
        origin = origin[len("source"):].strip() if origin.startswith("source") else origin
        verdict = (f"\\falsified{{}} \\textbf{{falsified}} ({r['rate'] * 100:.0f}\\%)"
                   if r["verdict"] == "FALSIFIED" else "\\survives{} survives")
        first = f"draw {r['first_failure_at']}" if r["first_failure_at"] else "---"
        tint = "\\rowcolor{rowtint}" if stripe else ""
        stripe = not stripe
        lines.append(f"{tint}{_tex(r['label'])} & {_tex(origin)} & {verdict} & {first}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


_SIG = re.compile(
    r"n=(?P<n>\d+), edges=(?P<edges>\d+), noise=\[(?P<noise>[^\]]*)\], "
    r"(?P<kind>\w+) on \[(?P<targets>[^\]]*)\], (?P<a>\S+) vs (?P<b>\S+), "
    r"lambda=(?P<lam>\S+)")


def _fields(signature: str) -> Dict[str, str]:
    m = _SIG.match(signature)
    if not m:
        raise ValueError(f"unparsable case signature: {signature}")
    d = m.groupdict()
    d["targets"] = d["targets"].replace(" ", "")
    return d


def _short(text: str) -> str:
    """Round every number in a comma- or slash-separated list for display.
    The full-precision values stay in results/paper_tables.json."""
    def one(tok: str) -> str:
        try:
            return f"{float(tok):g}" if abs(float(tok)) >= 1e-12 else "0"
        except ValueError:
            return tok
    for sep in (",", "/"):
        if sep in text:
            return sep.join(f"{float(t):.2f}".rstrip("0").rstrip(".")
                            if _isnum(t) else one(t) for t in text.split(sep))
    return f"{float(text):.2f}".rstrip("0").rstrip(".") if _isnum(text) else text


def _isnum(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _row(label: str, move: str, f: Dict[str, str]) -> str:
    return (f"{label} & {move} & {f['n']} & {f['edges']} & "
            f"$\\{{{f['targets']}\\}}$ & {_short(f['noise'])} & "
            f"{_short(f['a'])} \\textit{{vs}} {_short(f['b'])} & "
            f"{_short(f['lam'])}\\\\")


def write_shrink_trace(res: Dict, path: str, which: str) -> None:
    """One row per accepted simplification, read off the shrinker's own trace.

    The columns are the fields of the case, so a reader can see that each row
    differs from the one above it in exactly one place, and that the moves run
    from structural (drop a variable) to cosmetic (round a number)."""
    rec = res["shrink"][which]
    lines = ["% Generated by experiments/paper_tables.py. Do not edit by hand.",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\renewcommand{\\arraystretch}{1.2}",
             "\\begin{tabular}{llcccllc}",
             "\\toprule",
             "step & move & $n$ & edges & targets & noise variances"
             " & intervention values & $\\lambda$\\\\",
             "\\midrule"]

    first = _fields(rec["trace"][0].split(":", 1)[1].split(" -> ")[0].strip())
    lines += [_row("as found", "---", first), "\\midrule"]

    for k, step in enumerate(rec["trace"], start=1):
        move, rest = step.split(":", 1)
        after = _fields(rest.split(" -> ")[1].strip())
        pretty = _tex(move.strip().lstrip("_").replace("_", " "))
        tint = "\\rowcolor{rowtint}" if k % 2 else ""
        lines.append(tint + _row(str(k), f"\\textsf{{{pretty}}}", after))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_macros(res: Dict, path: str) -> None:
    lg = next(r for r in res["rows"] if r["name"] == "closure_linear_gaussian")
    un = next(r for r in res["rows"] if r["name"] == "closure_unrestricted_mechanisms")
    survive = sum(1 for r in res["rows"] if r["verdict"] == "survives")

    def num(x: int) -> str:
        return f"{x:,}".replace(",", "{,}")

    lines = [
        "% Generated by experiments/paper_tables.py. Do not edit by hand.",
        f"\\newcommand{{\\HnClaims}}{{{len(res['rows'])}}}",
        f"\\newcommand{{\\HnSurvive}}{{{survive}}}",
        f"\\newcommand{{\\HnFalsified}}{{{len(res['rows']) - survive}}}",
        f"\\newcommand{{\\HnTrials}}{{{num(res['trials'])}}}",
        f"\\newcommand{{\\HnRateLG}}{{{lg['rate'] * 100:.0f}}}",
        f"\\newcommand{{\\HnRateUn}}{{{un['rate'] * 100:.0f}}}",
        f"\\newcommand{{\\HnFirstLG}}{{{lg['first_failure_at']}}}",
        f"\\newcommand{{\\HnFirstUn}}{{{un['first_failure_at']}}}",
    ]
    for key, tag in (("closure_linear_gaussian", "LG"),
                     ("closure_unrestricted_mechanisms", "Un")):
        s = res["shrink"][key]
        n0, e0, t0, _ = s["start_size"]
        n1, e1, t1, _ = s["minimal_size"]
        lines += [
            f"\\newcommand{{\\HnShrinkSteps{tag}}}{{{s['steps']}}}",
            f"\\newcommand{{\\HnShrinkTried{tag}}}{{{s['candidates']}}}",
            f"\\newcommand{{\\HnStartVars{tag}}}{{{n0}}}",
            f"\\newcommand{{\\HnStartEdges{tag}}}{{{e0}}}",
            f"\\newcommand{{\\HnStartTargets{tag}}}{{{t0}}}",
            f"\\newcommand{{\\HnMinVars{tag}}}{{{n1}}}",
            f"\\newcommand{{\\HnMinEdges{tag}}}{{{e1}}}",
            f"\\newcommand{{\\HnMinTargets{tag}}}{{{t1}}}",
        ]
        sw = res["seed_sweep"][key]
        lines += [
            f"\\newcommand{{\\HnSeeds{tag}}}{{{sw['seeds']}}}",
            f"\\newcommand{{\\HnSeedsFell{tag}}}{{{sw['seeds_that_falsified']}}}",
            f"\\newcommand{{\\HnMedianDraws{tag}}}{{{sw['draws_to_first_median']:g}}}",
            f"\\newcommand{{\\HnMaxDraws{tag}}}{{{sw['draws_to_first_max']}}}",
        ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    res = run()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PAPER, exist_ok=True)
    with open(os.path.join(RESULTS, "paper_tables.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    write_claims_table(res, os.path.join(PAPER, "claims_table.tex"))
    write_shrink_trace(res, os.path.join(PAPER, "shrink_trace.tex"),
                       "closure_unrestricted_mechanisms")
    write_macros(res, os.path.join(PAPER, "harness_numbers.tex"))

    survive = sum(1 for r in res["rows"] if r["verdict"] == "survives")
    print(f"{survive} of {len(res['rows'])} claims survived {res['trials']} draws each")
    for name, rec in res["shrink"].items():
        print(f"\n{name}")
        print(f"  found    {rec['start']}")
        print(f"  minimal  {rec['minimal']}")
        print(f"  {rec['steps']} steps, {rec['candidates']} candidates")
        for step in rec["trace"]:
            print(f"    {step}")
    for name, sw in res["seed_sweep"].items():
        print(f"\n{name} seed sweep: {sw['seeds_that_falsified']}/{sw['seeds']} seeds, "
              f"median {sw['draws_to_first_median']:g} draws, "
              f"max {sw['draws_to_first_max']}")
    print("\nwritten: paper/claims_table.tex, paper/shrink_trace.tex, "
          "paper/harness_numbers.tex")


if __name__ == "__main__":
    main()
