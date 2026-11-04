#!/bin/sh
# Regenerate every number, figure and table. Builds the documents too if
# they are present beside this repository.
#
# Nothing in paper/ that this script writes should ever be edited by hand:
# the macro files (*_numbers.tex), the two generated tables (claims_table.tex,
# shrink_trace.tex) and everything in paper/figs/ are outputs. Editing one of
# them is how a paper's numbers drift away from the code that produced them.
#
# Order matters only in that the experiments must run before LaTeX does.
# The two documents are independent of each other.
#
# Usage:  sh build.sh          everything
#         sh build.sh quick    the harness only, no sweeps, no documents

set -e
cd "$(dirname "$0")"

if [ "$1" = "quick" ]; then
    python -m harness.runner --quick --shrink
    exit 0
fi

echo "== harness =========================================================="
python -m harness.runner --trials 2000 --sweeps --shrink

echo "== generated tables and figures ====================================="
python -m experiments.paper_tables
python -m harness.figures

echo "== experiments ======================================================"
python -m experiments.rq1_experiment  > /dev/null
python -m experiments.rq2_experiment  > /dev/null
python -m experiments.rq3_experiment  > /dev/null
python -m experiments.empirical_study > /dev/null
python -m experiments.safe_region
python -m experiments.lint_validation
python -m experiments.symbolic_checks
python -m experiments.kl_validation
python archive/audit_v1.py > /dev/null

# The documents are not part of this repository. If you have them beside it,
# this stage builds them from the numbers the run above just produced.
if [ -f paper/paper.tex ]; then
    echo "== documents ========================================================"
    cd paper
    latexmk -pdf -interaction=nonstopmode paper.tex
    latexmk -pdf -interaction=nonstopmode supplementary.tex
    latexmk -c > /dev/null 2>&1 || true
    echo
    echo "built paper/paper.pdf and paper/supplementary.pdf"
else
    echo
    echo "done. Macro files and figures are in paper/; no paper.tex here, so"
    echo "nothing was typeset."
fi
