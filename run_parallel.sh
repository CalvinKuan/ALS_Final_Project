#!/usr/bin/env bash
# Run all 100 cases in parallel using xargs -P.
# Usage: bash run_parallel.sh [parallel_jobs] [workers_per_job]
#
# Example (8-core machine):
#   bash run_parallel.sh 4 2   -> 4 cases at once, each using 2 CPU cores
#   bash run_parallel.sh 2 4   -> 2 cases at once, each using 4 CPU cores

JOBS=${1:-4}
WORKERS=${2:-2}

SCRIPT="$(dirname "$0")/student/optimizer.py"
EVAL="$(dirname "$0")/evaluate.py"

echo "Running $JOBS cases in parallel, $WORKERS workers each"
echo ""

seq 200 299 | xargs -P "$JOBS" -I {} \
    python3 "$SCRIPT" --case ex{} --max-workers "$WORKERS"

echo ""
echo "=== All cases done. Running evaluate.py ==="
python3 "$EVAL"
