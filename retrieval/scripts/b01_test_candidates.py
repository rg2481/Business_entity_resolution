"""User-run test-only adapter around the frozen GPU candidate generator."""
import argparse
import json
import sys
from b01_common import identity
from b01_match_common import append_progress, digest, read_json
from b01_test_common import DEFAULT_CONFIG, load_test_settings, check_country
import b01_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--country', required=True)
    args = parser.parse_args()
    cfg, _, _, out, _, model_sha, threshold = load_test_settings(args.config)
    check_country(cfg, args.country)
    event = 'test-candidates:' + digest({'plan': read_json(out / 'manifest.json'), 'country': args.country})[:16]
    append_progress(event + ':start', f'Test candidate generation started: {args.country}; optimized batch {cfg["query_batch"]}.\n'
                    f'Shared stable model is fixed at {model_sha}, threshold {threshold}; this stage performs retrieval only.\n'
                    'Retain the deduplicated union of name/address/combo top-k candidates; final scoring follows in a separate user-run stage.')
    original = sys.argv
    print('Reuse frozen preparation/retrieval. This test-run wrapper records stage results in progress.txt.', flush=True)
    try:
        sys.argv = [original[0], '--config', args.config, '--split', 'test', '--country', args.country]
        b01_candidates.main()
    finally:
        sys.argv = original
    report = read_json(out / f'candidates/test_{args.country}/report.json')
    append_progress(event + ':complete', json.dumps(report, indent=2))
    print('Test-run candidate stage recorded in progress.txt. Run the separately printed scoring command next.', flush=True)


if __name__ == '__main__':
    main()
