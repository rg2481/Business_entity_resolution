#!/usr/bin/env bash
# C04 runner (Claude): V2 features + 8x training sample on Codex B01 candidates. Run in WSL (Ubuntu), ONE stage at a time.
#   bash matcher/run_c04.sh prepare --country India      (then --country US)
#   bash matcher/run_c04.sh build   --country India      (then --country US)
#   bash matcher/run_c04.sh train
#   bash matcher/run_c04.sh evaluate  [--trees N]
#   bash matcher/run_c04.sh test-prepare --country France|India|US
#   bash matcher/run_c04.sh test-score   --country France|India|US  [--workers N]
#   bash matcher/run_c04.sh test-export  [--variant c04|france-b01|us-offset-reject]
#   bash matcher/run_c04.sh test-candidates [--variant us-offset-reject]   (candidate_pairs.tsv for the final package)
#   bash matcher/run_c04.sh next | summary
# Add --smoke to any stage for a tiny end-to-end check (first 3 parts, 30 trees) in matcher/runs/c04_smoke/.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source matcher/env.sh
mkdir -p matcher/runs/logs
stage="${1:-}"
if [[ -z "$stage" ]]; then sed -n '2,12p' "$0"; exit 2; fi
shift
case "$stage" in
    next|summary) python3 -B -u matcher/scripts/c04_next.py "$stage" "$@"; exit 0 ;;
    prepare) script=matcher/scripts/c04_prepare.py ;;
    build) script=matcher/scripts/c04_build.py ;;
    train) script=matcher/scripts/c04_train.py ;;
    evaluate) script=matcher/scripts/c04_evaluate.py ;;
    test-prepare) script=matcher/scripts/c04t_prepare.py ;;
    test-score) script=matcher/scripts/c04t_score.py ;;
    test-export) script=matcher/scripts/c04t_export.py ;;
    test-candidates) script=matcher/scripts/c04t_candidates.py ;;
    *) echo "Unknown stage: $stage"; sed -n '2,12p' "$0"; exit 2 ;;
esac
tag="$stage"; for a in "$@"; do case "$a" in France|India|US|c04|france-b01|us-offset-reject) tag="${tag}_$a" ;; --smoke) tag="${tag}_smoke" ;; esac; done
log="matcher/runs/logs/c04_${tag}.log"
echo "=== $(date '+%F %T') bash matcher/run_c04.sh $stage $* ===" >> "$log"
if python3 -B -u "$script" "$@" 2>&1 | tee -a "$log"; then
    python3 -B -u matcher/scripts/c04_next.py "$stage" "$@" | tee -a "$log"
else
    status=$?
    python3 -B -u matcher/scripts/c04_next.py --failed "$stage" "$@" | tee -a "$log"
    exit "$status"
fi
