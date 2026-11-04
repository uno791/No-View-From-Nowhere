# Archive: the v1 test suite

`v1_suite.py` is the hand-written test suite from the first phase of this
project, frozen exactly as it stood on 17 October 2026. It is **not run as
part of the build and must not be edited.** It is kept because it is
evidence, not because it is code.

## Why it is kept

The suite passes. All 18 tests green, in 1.5 seconds. It covers the
invertibility of the mixing map, the naturality square, the pushforward
identity, interventional consistency, affineness, both extension maps,
the data processing inequality, and the disagreement energy.

It never tests closure under convex combination, which is the one claim in
the framework that turns out to be false.

Worse, it could not have found the failure even if it had tried. Its
network generator builds causal coefficient matrices with

    C = np.tril(rng.normal(size=(n, n)) * 0.6, k=-1)

which fills every edge below the diagonal. Sampled 2000 times it produced a
complete DAG 2000 times, so exactly one root variable every time. Closure
holds on complete DAGs with unrestricted mechanisms. The suite was drawing
exclusively from the sub-class where the untested theorem happens to be
true.

## What it is evidence of

Two things, and both belong in the paper.

First, that defects concentrate where coverage does not. The single claim
with no test is the single claim that fails. This is an old result in
software engineering and it turns out to apply to mathematics stated as
code.

Second, that the generator is the artifact. The assertion needed to falsify
closure is one line. What decides whether a suite can falsify anything is
whether it samples the space the claim quantifies over. v1 did not, and
nothing about its assertions would have told you so.

## Reproducing the audit

    python archive/audit_v1.py

prints the root-count distribution of the v1 generator alongside the same
statistic from `harness/generators.py` at several edge densities. The
contrast is the finding.

## Running it

`v1_suite.py` imports from a package named `rck`, which is what `core/` was
called at the time. That import is part of the record and the file has not
been edited, so running it needs a one-line alias rather than a change to the
file:

    python archive/run_v1.py

`audit_v1.py`, which reproduces the coverage gap, is current code and runs
directly with `python archive/audit_v1.py`.
