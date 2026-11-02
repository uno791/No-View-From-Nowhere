"""
Where closure is safe, and where it is not.

The scoreboard reports that closure under convex combination falsifies on
roughly a third of random draws. That number alone is not actionable. This
experiment asks the sharper question: which models and which interventions,
exactly, produce a failure, and which can never produce one?

The answer turns out to be a clean characterisation rather than a trend.
Closure fails on a draw if and only if

    (1) the model has two or more root variables, equivalently some pair of
        variables shares no ancestor, AND
    (2) the two interventions assign different means to BOTH members of such
        a pair.

Condition (1) is structural and can be read off the coefficient matrix before
anything is run. Condition (2) is about the intervention, not the model.
Neither alone is enough, and together they are exactly enough.

WHY (1) HAS THE FORM IT DOES. If a model has exactly one root variable then
every other variable has a parent, and following parents upward must
terminate at a root, which can only be that one. So the single root is an
ancestor of everything, every pair of variables shares it, and no pair is
ancestrally disjoint. There is nothing for the exclusion tests to find. A
complete DAG is the special case of this with the maximum number of edges,
so "one root" is the weaker and more useful condition.

THE ASYMMETRY. This safe region exists only under the unrestricted
mechanisms reading. Within the linear Gaussian family a mixture of two
different Gaussians is essentially never Gaussian, so closure falsifies at
every density including a complete graph. There the only safe parameters are
the mixing weights 0 and 1, where no blending occurs.

Run:  python -m experiments.safe_region
"""

from __future__ import annotations

import collections
import json
import os
from typing import Dict, List

import numpy as np

from harness import exclusions as exc
from harness import generators as gen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

SEED = 20261018
DRAWS = 20_000
DENSITIES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
STRUCTURAL = ("root_independence", "ancestral_independence")


def run(draws: int = DRAWS, seed: int = SEED) -> Dict:
    rng = np.random.default_rng(seed)

    by_roots = collections.defaultdict(lambda: [0, 0])      # n_roots -> [failed, total]
    by_pairs = collections.defaultdict(lambda: [0, 0])      # disjoint pairs -> [failed, total]
    by_moved = collections.defaultdict(lambda: [0, 0])      # both means moved -> [failed, total]
    lg_by_roots = collections.defaultdict(lambda: [0, 0])   # the linear Gaussian reading

    for _ in range(draws):
        p = float(rng.choice(DENSITIES))
        scm = gen.sample_scm(rng, gen.SCMSpace(edge_prob=p))
        a, b, _ = gen.sample_intervention_pair(rng, scm)
        lam = gen.sample_weight(rng)
        struct = gen.describe(scm)

        verdicts = exc.screen_mixture(scm, lam, a.law(), b.law())
        failed = any(v.excluded for v in verdicts if v.test in STRUCTURAL)
        failed_lg = any(v.excluded for v in verdicts if v.test == "gaussian_marginals")

        by_roots[struct.n_roots][1] += 1
        by_roots[struct.n_roots][0] += failed
        lg_by_roots[struct.n_roots][1] += 1
        lg_by_roots[struct.n_roots][0] += failed_lg

        pairs = exc.ancestrally_disjoint_pairs(scm)
        bucket = 0 if not pairs else (1 if len(pairs) == 1 else 2)
        by_pairs[bucket][1] += 1
        by_pairs[bucket][0] += failed

        if pairs:
            ma, mb = a.law()[0], b.law()[0]
            moved = any(abs(ma[i] - mb[i]) > 1e-12 and abs(ma[j] - mb[j]) > 1e-12
                        for i, j in pairs)
            by_moved[bool(moved)][1] += 1
            by_moved[bool(moved)][0] += failed

    def rows(d) -> List[Dict]:
        return [{"key": k, "total": v[1], "failed": v[0],
                 "rate": v[0] / v[1] if v[1] else 0.0}
                for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))]

    return {"draws": draws, "seed": seed,
            "by_roots": rows(by_roots), "by_disjoint_pairs": rows(by_pairs),
            "by_both_means_moved": rows(by_moved),
            "linear_gaussian_by_roots": rows(lg_by_roots)}


def report(res: Dict) -> None:
    print(f"{res['draws']} draws, seed {res['seed']}\n")

    print("UNRESTRICTED MECHANISMS, by number of root variables")
    print(f"{'roots':>7}{'models':>10}{'falsified':>12}{'rate':>9}")
    for r in res["by_roots"]:
        print(f"{r['key']:>7}{r['total']:>10}{r['failed']:>12}{r['rate'] * 100:>8.1f}%")

    print("\nBY WHETHER AN ANCESTRALLY DISJOINT PAIR EXISTS")
    label = {0: "none exist", 1: "exactly one", 2: "two or more"}
    for r in res["by_disjoint_pairs"]:
        print(f"  {label[int(r['key'])]:<14}{r['total']:>8} models"
              f"{r['failed']:>9} falsified{r['rate'] * 100:>8.1f}%")

    print("\nAMONG MODELS THAT COULD FAIL, by whether the intervention moved both means")
    for r in res["by_both_means_moved"]:
        print(f"  both moved = {str(r['key']):<6}{r['total']:>8} draws"
              f"{r['failed']:>9} falsified{r['rate'] * 100:>8.1f}%")

    print("\nLINEAR GAUSSIAN READING, by number of root variables (no safe region)")
    for r in res["linear_gaussian_by_roots"]:
        print(f"  {r['key']} root(s): {r['rate'] * 100:.1f}% of {r['total']} models")


def write_macros(res: Dict, path: str) -> None:
    """Emit the numbers the paper quotes, so none is typed by hand."""
    one_root = next(r for r in res["by_roots"] if int(r["key"]) == 1)
    none_pairs = next(r for r in res["by_disjoint_pairs"] if int(r["key"]) == 0)
    not_moved = next(r for r in res["by_both_means_moved"] if r["key"] in (False, "False"))
    moved = next(r for r in res["by_both_means_moved"] if r["key"] in (True, "True"))
    lg_rates = [r["rate"] for r in res["linear_gaussian_by_roots"] if r["total"] >= 50]

    lines = [
        "% Generated by experiments/safe_region.py. Do not edit by hand.",
        f"\\newcommand{{\\SafeDraws}}{{{res['draws']:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\SafeOneRootModels}}{{{one_root['total']:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\SafeOneRootFailed}}{{{one_root['failed']}}}",
        f"\\newcommand{{\\SafeNoPairModels}}{{{none_pairs['total']:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\SafeNoPairFailed}}{{{none_pairs['failed']}}}",
        f"\\newcommand{{\\SafeNotMovedDraws}}{{{not_moved['total']:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\SafeNotMovedRate}}{{{not_moved['rate'] * 100:.1f}}}",
        f"\\newcommand{{\\SafeMovedDraws}}{{{moved['total']:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\SafeMovedRate}}{{{moved['rate'] * 100:.1f}}}",
        f"\\newcommand{{\\SafeLGMinRate}}{{{min(lg_rates) * 100:.1f}}}",
    ]
    # one triple per root count, so the paper's table has no hand-typed cell
    for r in res["by_roots"]:
        w = _word(int(r["key"]))
        lines.append(f"\\newcommand{{\\SafeRateRoots{w}}}{{{r['rate'] * 100:.1f}}}")
        lines.append(f"\\newcommand{{\\SafeModelsRoots{w}}}"
                     f"{{{r['total']:,}}}".replace(",", "{,}"))
        lines.append(f"\\newcommand{{\\SafeFailedRoots{w}}}"
                     f"{{{r['failed']:,}}}".replace(",", "{,}"))
    lines.append(f"\\newcommand{{\\SafeMaxRoots}}"
                 f"{{{max(int(r['key']) for r in res['by_roots'])}}}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _word(k: int) -> str:
    return {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(k, str(k))


def main() -> None:
    res = run()
    report(res)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "safe_region.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    write_macros(res, os.path.join(ROOT, "paper", "safe_numbers.tex"))
    print(f"\nwritten to {RESULTS}/safe_region.json and paper/safe_numbers.tex")


if __name__ == "__main__":
    main()
