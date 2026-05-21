#!/usr/bin/env bash
# Re-run optimizer on the worst-ADP cases with aggressive settings.
# Run from the project root: bash run_worst.sh

set -e
SCRIPT="$(dirname "$0")/student/optimizer.py"
WORST=(ex299 ex227 ex207 ex274 ex226 ex206 ex297 ex298 ex259 ex264 ex284 ex225 ex220 ex222 ex223 ex273 ex294 ex283)

for case in "${WORST[@]}"; do
    echo "=== Optimizing $case ==="
    python3 "$SCRIPT" \
        --case "$case" \
        --timeout 120 \
        --abc-aig-top-k 10 \
        --abc-aig-rounds 4 \
        --mockturtle-top-k 8 \
        --mockturtle-rounds 3 \
        --effort high
    echo ""
done

echo "=== Done with worst cases. Running evaluate.py ==="
python3 "$(dirname "$0")/evaluate.py"
