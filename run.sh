#!/usr/bin/env bash
# Business entity resolution: raw TSVs -> matching_results.tsv + candidate_pairs.tsv.
# Run ONE stage at a time (each shows tqdm progress, is resumable, logs to logs/, and prints the next command):
#   bash run.sh plan      # ordered list of stages
#   bash run.sh <stage>
# Stages: retrieval/ (normalisation, Indic dictionary, GPU TF-IDF retrieval, validation split)
#         matcher/   (C04 LightGBM pair model; C28 multilingual cross-encoder re-scoring; export)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$root"
steps=(preflight bootstrap prepare-train prepare-test candidates-train-india candidates-train-us features-india features-us
       b01-train b01-evaluate candidates-test-france candidates-test-india candidates-test-us
       c04-prepare-india c04-prepare-us c04-build-india c04-build-us c04-train c04-evaluate
       c04-test-prepare-france c04-test-prepare-india c04-test-prepare-us
       c04-test-score-france c04-test-score-india c04-test-score-us c04-export c04-candidates
       c28-data c28-train c28-data-2 c28-train-2 c28-score-val c28-eval c28-score-test c28-export finalize)
stage="${1:-plan}"
if [[ "$stage" == plan ]]; then
    echo 'Run these commands in order, one at a time:'
    for s in "${steps[@]}"; do echo "  bash run.sh $s"; done
    exit 0
fi
shift || true
codex=(); c04=(); py=()
case "$stage" in
    preflight) codex=(retrieval/scripts/b01_preflight.py) ;;
    bootstrap) codex=(retrieval/scripts/b01_bootstrap_raw.py) ;;
    prepare-train) codex=(retrieval/scripts/b01_prepare.py --split train) ;;
    prepare-test) codex=(retrieval/scripts/b01_prepare.py --split test) ;;
    candidates-train-india) codex=(retrieval/scripts/b01_candidates.py --split train --country India) ;;
    candidates-train-us) codex=(retrieval/scripts/b01_candidates.py --split train --country US) ;;
    features-india) codex=(retrieval/scripts/b01_match_data.py --country India) ;;
    features-us) codex=(retrieval/scripts/b01_match_data.py --country US) ;;
    b01-train) codex=(retrieval/scripts/b01_train_stable.py) ;;
    b01-evaluate) codex=(retrieval/scripts/b01_evaluate_stable.py) ;;
    candidates-test-france) codex=(retrieval/scripts/b01_test_candidates.py --country France) ;;
    candidates-test-india) codex=(retrieval/scripts/b01_test_candidates.py --country India) ;;
    candidates-test-us) codex=(retrieval/scripts/b01_test_candidates.py --country US) ;;
    c04-prepare-india) c04=(prepare --country India) ;;
    c04-prepare-us) c04=(prepare --country US) ;;
    c04-build-india) c04=(build --country India) ;;
    c04-build-us) c04=(build --country US) ;;
    c04-train) c04=(train) ;;
    c04-evaluate) c04=(evaluate) ;;
    c04-test-prepare-france) c04=(test-prepare --country France) ;;
    c04-test-prepare-india) c04=(test-prepare --country India) ;;
    c04-test-prepare-us) c04=(test-prepare --country US) ;;
    c04-test-score-france) c04=(test-score --country France) ;;
    c04-test-score-india) c04=(test-score --country India) ;;
    c04-test-score-us) c04=(test-score --country US) ;;
    c04-export) c04=(test-export --variant us-offset-reject) ;;
    c04-candidates) c04=(test-candidates --variant us-offset-reject) ;;
    c28-data) py=(env C28_BUCKET=0 python3 -B -u matcher/scripts/c28_data.py) ;;
    c28-train) py=(env python3 -B -u matcher/scripts/c28_train.py train) ;;
    c28-data-2) py=(env C28_BUCKET=1 python3 -B -u matcher/scripts/c28_data.py) ;;
    c28-train-2) py=(env C28_TRAIN=train_b1.parquet C28_INIT=matcher/runs/c28/model C28_OUT=model2 C28_LR=2e-5 python3 -B -u matcher/scripts/c28_train.py train) ;;
    c28-score-val) py=(env C28_OUT=model2 python3 -B -u matcher/scripts/c28_train.py score val) ;;
    c28-eval) py=(env C28_SUF=_model2 python3 -B -u matcher/scripts/c28_eval.py) ;;
    c28-score-test) py=(env C28_OUT=model2 python3 -B -u matcher/scripts/c28_train.py score test) ;;
    c28-export) py=(env C28_SUF=_model2 python3 -B -u matcher/scripts/c28_export.py) ;;
    finalize) py=(bash -c 'mkdir -p output && cp matcher/runs/c28/export_pure_model2/matching_results.tsv output/ && cp matcher/runs/c04_v1/test/output_us_offset_reject/candidate_pairs.tsv output/ && ls -la output') ;;
    *) echo "Unknown stage: $stage. Run: bash run.sh plan"; exit 2 ;;
esac
next() {
    for i in "${!steps[@]}"; do
        if [[ "${steps[i]}" == "$stage" && $((i + 1)) -lt ${#steps[@]} ]]; then
            echo; echo "NEXT: bash run.sh ${steps[i + 1]}"
        fi
    done
    [[ "$stage" == finalize ]] && echo "DONE: output/matching_results.tsv and output/candidate_pairs.tsv"
    return 0
}
mkdir -p logs
if (( ${#codex[@]} )); then
    ( source retrieval/env.sh && python3 -B -u "${codex[@]}" "$@" ) 2>&1 | tee -a "logs/$stage.log"
elif (( ${#c04[@]} )); then
    bash matcher/run_c04.sh "${c04[@]}" "$@"
else
    ( source matcher/env.sh && "${py[@]}" "$@" ) 2>&1 | tee -a "logs/$stage.log"
fi
next
