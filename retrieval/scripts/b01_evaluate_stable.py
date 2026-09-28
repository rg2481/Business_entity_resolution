"""User-run stable model evaluation using the frozen B01 scoring implementation."""
import argparse
import json
import sys
from b01_common import atomic_json, code_hash, environment
environment()
from b01_match_common import settings, read_json, append_progress, stage_lock
from b01_train_stable import DEFAULT_CONFIG, leaf_audit
import b01_evaluate_matcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--base-config')
    args = parser.parse_args()
    cfg, _, _, out = settings(args.config, args.base_config)
    done = read_json(out / 'model' / 'complete.json')
    if not done.get('stability_audit', {}).get('passed'):
        raise RuntimeError('The stable model has not passed its serialized-tree bound check.')
    leaf_audit(out / 'model' / 'model.txt', cfg, done['stability_audit']['initial_log_odds'])
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], '--config', args.config]
        if args.base_config:
            sys.argv += ['--base-config', args.base_config]
        b01_evaluate_matcher.main()
    finally:
        sys.argv = old_argv
    _lock = stage_lock(out / 'evaluation')
    path = out / 'evaluation' / 'report.json'
    report = read_json(path)
    report.update(experiment='B01 stable_v2', stability_audit=done['stability_audit'],
                  comparison_protocol='Same pairs/features and held-out entities as B01. Its check result has already been reviewed; this is a diagnostic comparison, not a fresh confirmatory holdout.',
                  adapter_code=code_hash('b01_evaluate_stable.py'))
    atomic_json(path, report)
    summary = {'experiment': report['experiment'], 'threshold': report['threshold'],
               'combined': report['combined'], 'comparison_protocol': report['comparison_protocol'],
               'stability_audit': report['stability_audit'], 'france': report['france'], 'leaderboard': report['leaderboard']}
    append_progress(done['event'] + ':comparison', json.dumps(summary, indent=2))
    print('\nStable-model comparison protocol and final summary:\n' + json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
