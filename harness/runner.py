"""
The scoreboard: run every property against the model space and report.

WHAT THIS PRODUCES
------------------
Three things, in this order, because each one guards the next.

1. EXCLUSION TEST CONTROL. Before any claim is tested, the exclusion tests are run
   against measures that are in CK(M) by construction. An exclusion test that fires
   there would manufacture counterexamples, so if any does the run aborts.
   A scoreboard printed without this control is not evidence of anything.

2. THE SCOREBOARD. Every property, run many times with independent draws.
   Claims that survive are reported alongside the one that does not, because
   a harness that only reports failures looks aimed at them.

3. SWEEPS. Two of them. The seed sweep reports how often, and how quickly, an
   independent run finds a counterexample, so that a single lucky draw cannot
   be mistaken for a robust result. The density sweep varies the one
   structural parameter the framework leaves free and reports the
   falsification rate against it, which is the central figure of the paper.

Usage:

    python -m harness.runner                     default run
    python -m harness.runner --trials 5000       more draws per property
    python -m harness.runner --sweeps            include both sweeps
    python -m harness.runner --quick             fast smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from harness import generators as gen
from harness import exclusions as exc
from core.scm import LinearGaussianSCM
from harness.properties import ALL_PROPERTIES, Config, Outcome

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------
# one property, many draws
# ---------------------------------------------------------------------------

@dataclass
class PropertyReport:
    name: str
    source: str
    statement: str
    trials: int
    passed: int
    falsified: int
    skipped: int
    errors: int
    first_failure_at: Optional[int] = None
    worst_residual: float = 0.0
    example: Optional[Dict] = None
    detail: str = ""

    @property
    def verdict(self) -> str:
        if self.errors:
            return "ERROR"
        if self.falsified:
            return "FALSIFIED"
        if self.passed == 0:
            return "no data"
        return "survives"


def run_property(prop, trials: int, seed: int, cfg: Config) -> PropertyReport:
    rng = np.random.default_rng(seed)
    passed = falsified = skipped = errors = 0
    first_at: Optional[int] = None
    worst = 0.0
    example: Optional[Dict] = None
    name = source = statement = detail = ""

    for i in range(trials):
        try:
            out: Outcome = prop(rng, cfg)
        except Exception as exc:                      # a crash is a finding, not a pass
            errors += 1
            if example is None:
                example = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        name, source, statement = out.name, out.source, out.statement
        if out.inapplicable:
            skipped += 1
            continue
        if out.falsified:
            falsified += 1
            if first_at is None:
                first_at, example, detail = i + 1, out.witness, out.detail
            worst = max(worst, out.residual)
        else:
            passed += 1
            worst = max(worst, out.residual) if not falsified else worst
            detail = detail or out.detail

    return PropertyReport(
        name=name or prop.__name__, source=source, statement=statement,
        trials=trials, passed=passed, falsified=falsified, skipped=skipped,
        errors=errors, first_failure_at=first_at, worst_residual=worst,
        example=example, detail=detail,
    )


# ---------------------------------------------------------------------------
# step 1: the exclusion tests' own control
# ---------------------------------------------------------------------------

def exclusion_control(trials: int, seed: int, cfg: Config) -> Dict:
    """Assert that no exclusion test excludes a measure that is in CK(M) by construction."""
    rng = np.random.default_rng(seed)
    wrong: List[str] = []
    for _ in range(trials):
        scm = gen.sample_scm(rng, cfg.scm_space)
        for v in exc.validate(scm, rng):
            wrong.append(str(v))
    return {"models_checked": trials, "false_exclusions": len(wrong), "examples": wrong[:5]}


# ---------------------------------------------------------------------------
# step 3a: seed sweep
# ---------------------------------------------------------------------------

def seed_sweep(prop, seeds: int, max_draws: int, cfg: Config, base_seed: int = 10_000) -> Dict:
    """How often, and after how many draws, does an independent run falsify?"""
    found: List[int] = []
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        hit = None
        for i in range(max_draws):
            out = prop(rng, cfg)
            if out.falsified:
                hit = i + 1
                break
        if hit is not None:
            found.append(hit)
    return {
        "seeds": seeds,
        "max_draws_per_seed": max_draws,
        "seeds_that_falsified": len(found),
        "rate": len(found) / seeds if seeds else 0.0,
        "draws_to_first_min": int(min(found)) if found else None,
        "draws_to_first_median": float(np.median(found)) if found else None,
        "draws_to_first_max": int(max(found)) if found else None,
    }


# ---------------------------------------------------------------------------
# step 3b: density sweep
# ---------------------------------------------------------------------------

def density_sweep(prop, densities, trials: int, seed: int, cfg: Config) -> List[Dict]:
    """Falsification rate against edge density, the one structural parameter
    the framework leaves free and the v1 generator silently fixed at 1.0."""
    rows = []
    for p in densities:
        space = gen.SCMSpace(
            n_vars=cfg.scm_space.n_vars, edge_prob=p,
            coeff_scale=cfg.scm_space.coeff_scale, coeff_min=cfg.scm_space.coeff_min,
            noise_var=cfg.scm_space.noise_var,
        )
        local = Config(scm_space=space, net_space=cfg.net_space,
                       intervention_kind=cfg.intervention_kind)
        rep = run_property(prop, trials, seed, local)
        struct = gen.structural_report(np.random.default_rng(seed), space, draws=600)
        rows.append({
            "edge_prob": p,
            "falsification_rate": rep.falsified / max(rep.passed + rep.falsified, 1),
            "falsified": rep.falsified,
            "evaluated": rep.passed + rep.falsified,
            "frac_two_or_more_roots": struct["frac_two_or_more_roots"],
            "frac_complete": struct["frac_complete"],
        })
    return rows


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_scoreboard(reports: List[PropertyReport]) -> None:
    width = max(len(r.name) for r in reports) + 2
    print(f"\n{'claim':<{width}}{'source':<26}{'verdict':>11}{'pass':>8}{'fail':>7}{'skip':>7}   first fail")
    print("-" * (width + 72))
    for r in sorted(reports, key=lambda x: (x.verdict != "FALSIFIED", x.name)):
        first = f"draw {r.first_failure_at}" if r.first_failure_at else ""
        print(f"{r.name:<{width}}{r.source:<26}{r.verdict:>11}"
              f"{r.passed:>8}{r.falsified:>7}{r.skipped:>7}   {first}")


def print_failure_detail(reports: List[PropertyReport]) -> None:
    for r in reports:
        if not r.falsified:
            continue
        print(f"\n--- {r.name} ---")
        print(f"  claim      {r.statement}")
        print(f"  source     {r.source}")
        print(f"  falsified  {r.falsified} of {r.passed + r.falsified} draws "
              f"({r.falsified / max(r.passed + r.falsified, 1) * 100:.1f}%), "
              f"first at draw {r.first_failure_at}")
        print(f"  detail     {r.detail}")
        if r.example:
            struct = r.example.get("structure")
            if struct:
                print(f"  structure  {struct}")
            if "intervention" in r.example:
                print(f"  witness    {r.example['intervention']}, lambda = "
                      f"{r.example.get('lambda', float('nan')):.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Falsification harness scoreboard.")
    ap.add_argument("--trials", type=int, default=1000, help="draws per property")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--control-models", type=int, default=200)
    ap.add_argument("--edge-prob", type=float, default=0.5)
    ap.add_argument("--sweeps", action="store_true", help="run the seed and density sweeps")
    ap.add_argument("--shrink", action="store_true", help="reduce each counterexample to a minimal one")
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    args = ap.parse_args()

    if args.quick:
        args.trials, args.control_models = 100, 30

    cfg = Config(scm_space=gen.SCMSpace(edge_prob=args.edge_prob))
    started = time.time()

    print("=" * 86)
    print("FALSIFICATION HARNESS")
    print(f"  seed {args.seed} | {args.trials} draws per claim | edge_prob {args.edge_prob}")
    print("=" * 86)

    # ---- step 1: are the exclusion tests sound?
    print("\n[1/3] exclusion test control: do the exclusion tests ever exclude a genuine member of CK(M)?")
    control = exclusion_control(args.control_models, args.seed, cfg)
    print(f"      {control['models_checked']} models, "
          f"{control['false_exclusions']} false exclusions")
    if control["false_exclusions"]:
        for e in control["examples"]:
            print(f"      ! {e}")
        raise SystemExit("ABORTED: an exclusion test excluded a measure that is in CK(M) by "
                         "construction. The panel is unsound; no scoreboard was produced.")
    print("      exclusion tests are sound on this sample, proceeding")

    # ---- step 2: the scoreboard
    print(f"\n[2/3] scoreboard: {len(ALL_PROPERTIES)} claims")
    reports = [run_property(p, args.trials, args.seed + i, cfg)
               for i, p in enumerate(ALL_PROPERTIES)]
    print_scoreboard(reports)
    print_failure_detail(reports)

    payload: Dict = {
        "config": {"seed": args.seed, "trials": args.trials, "edge_prob": args.edge_prob},
        "exclusion_control": control,
        "scoreboard": [asdict(r) for r in reports],
    }

    # ---- step 3: sweeps
    if args.sweeps:
        falsified = [p for p, r in zip(ALL_PROPERTIES, reports) if r.falsified]
        print(f"\n[3/3] sweeps over {len(falsified)} falsified claim(s)")
        payload["sweeps"] = {}
        for prop in falsified:
            rep = next(r for r in reports if r.name == prop(np.random.default_rng(0), cfg).name)
            print(f"\n  seed sweep: {rep.name}")
            ss = seed_sweep(prop, seeds=200, max_draws=200, cfg=cfg)
            print(f"    {ss['seeds_that_falsified']}/{ss['seeds']} independent seeds falsified "
                  f"({ss['rate'] * 100:.1f}%)")
            if ss["draws_to_first_median"]:
                print(f"    draws to first counterexample: min {ss['draws_to_first_min']}, "
                      f"median {ss['draws_to_first_median']:.0f}, max {ss['draws_to_first_max']}")

            print(f"\n  density sweep: {rep.name}")
            densities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            rows = density_sweep(prop, densities, max(args.trials // 2, 200), args.seed, cfg)
            print(f"    {'edge_prob':>10}{'falsified':>11}{'rate':>9}{'2+ roots':>11}{'complete':>10}")
            for row in rows:
                print(f"    {row['edge_prob']:>10.2f}{row['falsified']:>11}"
                      f"{row['falsification_rate'] * 100:>8.1f}%"
                      f"{row['frac_two_or_more_roots'] * 100:>10.1f}%"
                      f"{row['frac_complete'] * 100:>9.1f}%")
            payload["sweeps"][rep.name] = {"seed_sweep": ss, "density_sweep": rows}
    else:
        print("\n[3/3] sweeps skipped (pass --sweeps to run them)")

    # ---- shrink every falsifying witness to a minimal one
    if args.shrink:
        from harness import shrink as sh
        print("\n[shrink] reducing each counterexample to a minimal one")
        payload["shrink"] = {}
        which_for = {
            "closure_linear_gaussian": ("gaussian_marginals",),
            "closure_unrestricted_mechanisms": ("root_independence", "ancestral_independence"),
        }
        for rep in reports:
            if not rep.falsified or rep.name not in which_for or not rep.example:
                continue
            ex = rep.example
            if "C" not in ex:
                continue
            scm = LinearGaussianSCM(np.array(ex["C"], dtype=float),
                                    np.array(ex["noise_var"], dtype=float),
                                    [f"X{i}" for i in range(len(ex["noise_var"]))])
            case = sh.case_from_draw(scm, ex["intervention"], ex["lambda"])
            out = sh.shrink(case, which_for[rep.name])
            print(f"\n  {rep.name}")
            print(f"    found    {out.start.describe()}")
            print(f"    minimal  {out.minimal.describe()}")
            print(f"    {out.steps} steps, {out.candidates_tried} candidates tried")
            m = out.minimal
            print(f"    C = {np.array(m.C).tolist()}, noise_var = {list(m.noise_var)}")
            print(f"    {m.kind} on {list(m.targets)}: "
                  f"{[list(r) for r in m.a]} against {[list(r) for r in m.b]}, lambda = {m.lam:g}")
            payload["shrink"][rep.name] = {
                "found": out.start.describe(), "minimal": out.minimal.describe(),
                "steps": out.steps, "candidates_tried": out.candidates_tried,
                "minimal_case": {"C": [list(r) for r in m.C],
                                 "noise_var": list(m.noise_var), "kind": m.kind,
                                 "targets": list(m.targets),
                                 "a": [list(r) for r in m.a], "b": [list(r) for r in m.b],
                                 "lam": m.lam},
                "trace": out.trace,
            }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "scoreboard.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    n_fals = sum(1 for r in reports if r.falsified)
    print(f"\n{'=' * 86}")
    print(f"{len(reports) - n_fals} of {len(reports)} claims survived {args.trials} draws each. "
          f"{n_fals} falsified.")
    print(f"elapsed {time.time() - started:.1f}s | written to {out_path}")
    print("=" * 86)


if __name__ == "__main__":
    main()
