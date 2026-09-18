"""List and run reward-coefficient sweeps with the common experiment settings."""
import argparse
import json
from pathlib import Path

from run import main as run_main


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['list', 'plan', 'run', 'smoke'])
    p.add_argument('--stage', choices=['C1', 'C2', 'joint'], required=True)
    p.add_argument('--goal', type=int, choices=[5, 6, 7])
    p.add_argument('--seed', type=int, choices=range(5))
    p.add_argument('--C1', type=float)
    p.add_argument('--C2', type=float)
    p.add_argument('--device', default='auto')
    p.add_argument('--adam-backend', default='auto', choices=['auto', 'reference', 'foreach', 'fused'])
    p.add_argument('--with-dependencies', action='store_true')
    args = p.parse_args()
    path = Path(__file__).resolve().parent/'spec/coefficient_sweeps.json'
    rows = json.loads(path.read_text(encoding='utf-8'))[args.stage]
    rows = [r for r in rows if all(getattr(args, k) is None or r[k] == getattr(args, k)
                                  for k in ['goal', 'seed', 'C1', 'C2'])]
    if args.command == 'list':
        for r in rows:
            print(r['case_id'])
        return
    if len(rows) != 1:
        p.error('select exactly one coefficient pair, goal and seed; inspect matching cases with list')
    argv = [args.command, '--case', rows[0]['case_id']]
    if args.command != 'plan':
        argv += ['--device', args.device, '--adam-backend', args.adam_backend]
        if args.with_dependencies:
            argv.append('--with-dependencies')
    run_main(argv)


if __name__ == '__main__':
    main()
