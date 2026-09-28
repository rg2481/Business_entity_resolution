"""Print C04 status and the exact next command (Codex style). Never starts a job."""
import json
import sys
from c04_common import settings, read_json


def status(cfg, out):
    steps = []
    for c in cfg['countries']:
        steps.append((f'prepare --country {c}', (out / 'prepare' / c / 'complete.json').is_file()))
    for c in cfg['countries']:
        steps.append((f'build --country {c}', (out / 'features' / c / 'complete.json').is_file()))
    steps.append(('train', (out / 'model' / 'complete.json').is_file()))
    steps.append(('evaluate', (out / 'evaluation' / 'report.json').is_file()))
    for c in cfg['test']['countries']:
        steps.append((f'test-prepare --country {c}', (out / 'test' / 'prepare' / c / 'complete.json').is_file()))
    for c in cfg['test']['countries']:
        steps.append((f'test-score --country {c}', (out / 'test' / 'scores' / c / 'complete.json').is_file()))
    steps.append(('test-export', (out / 'test' / 'output' / 'report.json').is_file()))
    return steps


def main(argv):
    smoke = '--smoke' in argv
    failed = '--failed' in argv
    words = [a for a in argv if not a.startswith('--') and a not in ('France', 'India', 'US', 'c04', 'france-b01', 'us-offset-reject')]
    stage = words[0] if words else 'next'
    cfg, out = settings(smoke)
    s = ' --smoke' if smoke else ''
    if failed:
        print(f'\nFAILED stage: {stage}. Read the error above, fix it, and rerun the SAME command; completed parts are reused.')
        return
    steps = status(cfg, out)
    print('\nC04 status' + (' (SMOKE run: first parts only, tiny model; not a real result)' if smoke else '') + ':')
    for name, ok in steps:
        print(f'  [{"x" if ok else " "}] {name}')
    todo = [name for name, ok in steps if not ok]
    if (out / 'evaluation' / 'report.json').is_file():
        r = read_json(out / 'evaluation' / 'report.json')
        print(f"\nC04 check macro-F0.5: " + ', '.join(f"{c} {r['countries'][c]['check']['macro_f05']:.6f} "
              f"({r['countries'][c]['check_minus_b01']:+.6f} vs B01)" for c in cfg['countries'])
              + f"; mean {r['check_mean']:.6f} vs B01 {r['b01_check_mean']:.6f}; threshold {r['threshold']:.3f}")
    if todo:
        print(f'\nNext command:\n  bash matcher/run_c04.sh {todo[0]}{s}')
    else:
        r = read_json(out / 'test' / 'output' / 'report.json')
        print(f"\nTest file: {r['file']} | official validator pass: {r['official_validator_pass']} | "
              f"S1 rows changed vs B01: {r['s1_rows_changed_vs_b01']:,}")
        print('All C04 stages are complete. Share the export report with Claude before uploading.')


if __name__ == '__main__':
    main(sys.argv[1:])
