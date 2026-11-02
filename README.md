# No View From Nowhere: Verifying Relative Causal Knowledge

Code for the paper of that name. The paper and its supplementary material are
not in this repository; this is the model library, the verification harness and
the experiments, which is what is reusable.

Honours research project, School of Computer Science and Applied Mathematics,
University of the Witwatersrand. Harshil Vallabh (2656158), supervised by
Prof. Ritesh Ajoodha, Explainable AI Laboratory.

## What this is

A framework for multi-agent causal reasoning published at UAI 2025 makes a
set of universally quantified claims about structural causal models, causal
abstractions and networks of them. This project builds a harness that turns
each of those claims into an executable property, samples the model space the
claim quantifies over, and reports which claims survive and which fall.

Two fall, and they are the two readings of one theorem. The paper reports the
scoreboard, diagnoses the failure, characterises exactly when it can occur,
and repairs the affected result.

## Run everything

    pip install -r requirements.txt
    sh build.sh

Regenerates every number, figure and table the paper reports. Takes a few
minutes, most of it in the density sweep and the safe-region study.
`sh build.sh quick` runs the harness alone, in about fifteen seconds.

Running it creates two directories of build products: `results/` for
machine-readable JSON, and `paper/` for the LaTeX macro files and figures the
documents consume. Neither is tracked here; both are outputs.

## Layout

    core/         the model library: SCMs, interventions, abstractions,
                  network sheaves, KL divergence, misalignment.
    harness/      the new work. Generators, exclusion tests, properties,
                  runner, shrinker, figures.
    experiments/  the empirical studies and the paper's generated tables.

## The thesis, in one paragraph

The assertion that falsifies a theorem is one line. What decides whether a
test suite can falsify anything is whether it samples the space the theorem
quantifies over. The project's first suite passed 18 tests while drawing exclusively from the
sub-class where the untested theorem happens to hold, and the single claim it
never tested is the single claim that is false. The generator, not
the assertion, is the artifact.

## Integrity of the finding

The failure was first identified by hand. The harness was then written from
the theorem statements and the framework's own definitions, and it
rediscovers a minimal instance independently. The paper says so in that
order, because the reusable result is that a harness built from the
definitions catches this, not that a bug was found.

Four safeguards, all checkable by a reader:

1. **No special-casing in the generators.** They sample from Definition 1.
   There is no branch anywhere that constructs a known counterexample. The
   one change that exposes the failure is to edge density, which Definition 1
   leaves free and the first generator silently fixed at 1.0.
2. **Every claim is tested, not one.** A harness aimed at a single theorem is
   a bug reproduction. The 14 survivors are the control that makes the two
   failures credible.
3. **Our own claims are on the scoreboard too**, under the same generators,
   and whatever it finds is reported.
4. **Seed sweeps, not single runs.** Time-to-first-counterexample is reported
   as a distribution over independent seeds.

One honest qualification. The shrinker in `harness/shrink.py` is ours, not
third-party. Hypothesis would be the stronger claim since its reducer is
library code, but the case under test is a structured object that does not
map onto Hypothesis strategies without rewriting the generators to be
strategy-driven. The moves are therefore listed in one place, `MOVES`, so a
reader can confirm that none of them encodes the expected answer: none
references a root, an independence relation or a kurtosis. Moving the
generators onto Hypothesis is recorded as future work.

## Current scoreboard

    python -m harness.runner --trials 2000 --sweeps

14 of 16 claims survive 2000 draws each. Two fall, and they are the two
readings of the same published theorem:

| claim | verdict | rate |
|---|---|---|
| closure, linear Gaussian family | falsified | 100% of draws, first at draw 1 |
| closure, unrestricted mechanisms | falsified | 34% of draws, first at draw 4 |

Every other claim of the source paper, and every claim established in this
project, survives. 200 of 200 independent seeds falsify both, with medians
of 1 and 2 draws.

### Falsification against edge density

The structural reading, with unrestricted mechanisms, 2000 draws per point:

| edge probability | falsified | models with 2+ roots |
|---|---|---|
| 1.00 | 0.0% | 0.0% |
| 0.80 | 11.2% | 22.0% |
| 0.60 | 26.0% | 50.0% |
| 0.40 | 45.0% | 75.5% |
| 0.20 | 61.6% | 93.3% |

At full density the theorem holds, recovering a proposition proved by hand in
the supplementary and never given to the harness: on a complete DAG soft
interventions reach every measure, so closure holds there. Everywhere else it
fails, at a rate that tracks the fraction of models with two or more roots.

## Is a composite variable safe to act on

    python -m experiments.causal_lint --demo

A composite variable is a weighted sum of parts used in place of them: total
exposure, a pooled risk score, total hours, an index. Fitting a model on top
of one always reproduces the data. Acting on one is a different matter, and
E1 shows the two are uncorrelated at -0.069.

The criterion is that a composite is safe to act on exactly when its parts
are weighted by how much they actually cause the outcome. `core/misalignment.py`
computes how far a proposed composite is from that, which part carries the
loss, the floor on the interventional error over every possible action, and
the weights that would repair it. With several outcomes no weighting works in
general, and the best compromise is the leading eigenvector of the effect
scatter matrix in the variance-weighted metric.

    python -m experiments.lint_validation

Four correctness checks and a benefit test over 400 random models. The
closed form matches the measured error to 2e-15, the repair drives the
measured error to 5e-30, the floor is never violated, and the residual Gram
identity holds to 4e-15. On measured interventional error the designed
weights are lower than principal components on 88.0% of models, equal
weights on 86.5%, and a single-outcome fit on 81.2%, with median error ratios
of 1.38, 1.33 and 1.22.

It is not PCA. PCA was the worst of the four methods tested.

## Where closure is safe

    python -m experiments.safe_region

Over 20,000 draws the failure is characterised exactly rather than
statistically. Closure fails on a draw if and only if

1. the model has two or more root variables, equivalently some pair of
   variables shares no ancestor, **and**
2. the two interventions assign different means to both members of such a
   pair.

Models with a single root never fail: 0 of 10,325. Splitting on condition 2
gives a total separation, 0.0% against 100.0%. Both conditions are cheap, and
the first can be read off the coefficient matrix before anything is run.

This safe region exists only under the unrestricted reading. Within the
linear Gaussian family closure falsifies at every root count and every
density, because a mixture of two different Gaussians is essentially never
Gaussian.

## Shrinking: the minimal counterexamples

    python -m harness.runner --trials 2000 --shrink

The shrinker reduces each random counterexample to the simplest case that
still fails. It sees one bit per case, pass or fail, and its moves are
generic structural simplifications listed in `MOVES` in `harness/shrink.py`.

It recovers a different minimal case for each reading, and both match the
counterexamples derived by hand in the written work. That the two reductions
land on different shapes is itself evidence the reducer is not steering: a
shrinker tuned to an expected answer would return one answer twice.

**Linear Gaussian reading.** Found on a 5-variable model with 3 edges and a
2-element target set; reduced in 8 steps to

    X0 = Z0, X1 = Z1, unit noise, no edges
    1/2 do(X0 = 1)  +  1/2 do(X0 = 0)

a two-atom blend of variance 1/4 where every member of CK_LG(M) has a
Gaussian or point-mass marginal.

**Unrestricted reading.** Found on a 6-variable model with 5 edges and a
3-element target set; reduced in 11 steps to

    X0 = Z0, X1 = Z1, unit noise, no edges
    1/2 do(X0 = 1, X1 = 1)  +  1/2 do(X0 = 0, X1 = 0)

Both components have covariance exactly zero. The blend has covariance 0.25,
a correlation of 1.0, and matches lambda*lambar*s^2 with s = 1 exactly. Two
variables with no shared ancestor, made perfectly dependent by an operation
that was claimed to stay inside the set.

The full reduction, step by step, is Table 3 of the paper, emitted by
`experiments/paper_tables.py` from the shrinker's own trace.

## Status

| component | state |
|---|---|
| `core/` | model library. SCMs, interventions, abstractions, sheaves, KL, misalignment |
| `harness/` | generators, exclusion tests, 16 properties, runner, shrinker, figures |
| `experiments/` | four studies, the safe region, the KL validation, the symbolic checks, the generated tables |

The paper is 15 pages in Applied Soft Computing format and the supplementary
is 19, self-contained, holding every proof. Neither is in this repository.

## Reproducing a figure in the paper without running everything

Every number printed in the paper and the supplementary is written by the code
into a LaTeX macro file that the documents `\input`, so no value is transcribed
by hand. To regenerate one study rather than all of them:

    python -m experiments.empirical_study      # E1 to E4, Figs 4 to 7
    python -m experiments.lint_validation      # the composite criterion
    python -m experiments.kl_validation        # the KL implementation
    python -m experiments.safe_region          # the safe-region characterisation
    python -m experiments.symbolic_checks      # 16 identities, via SymPy
    python -m harness.runner --trials 2000     # the scoreboard

Seeds are constants in each file and are reported with every run, so a rerun
reproduces the published values exactly rather than approximately.

## Requirements

Python 3.10 or later with `numpy`, `scipy`, `matplotlib` and `sympy`:

    pip install -r requirements.txt

Building the documents additionally needs a LaTeX distribution with
`tikz-cd`, `booktabs` and `dblfloatfix`.
