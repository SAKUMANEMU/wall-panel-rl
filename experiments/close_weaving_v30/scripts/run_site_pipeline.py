"""Generate fresh constrained scenes, then run the BC/PPO adaptation pipeline.

All child output and phase status remain on disk after the launching terminal
closes. A failed prerequisite prevents training from starting.
"""
import _bootstrap
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import traceback

from wall_panel_rl.submission import load_runtime
from wall_panel_rl.utils.config import ROOT
from run_adaptation import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--existing-scenes',type=Path,
                        help='Use an already generated frozen bank instead of regenerating it')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'pid': os.getpid(),
             'started_at': datetime.now().astimezone().isoformat(),
             'output': str(output), 'config': str(args.config.resolve())}

    def update(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        save(output/'pipeline_status.json', state)
        print(state, flush=True)

    source = ROOT/'models/current_ppo.zip'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    save(output/'runtime.json', load_runtime())
    try:
        scenes=args.existing_scenes.resolve() if args.existing_scenes else output/'scenes'
        if args.existing_scenes:
            for split in ('train','validation','test'):
                if not (scenes/f'{split}_cases.json').is_file():
                    raise FileNotFoundError(f'Missing existing {split} scene bank')
        stages = ([] if args.existing_scenes else [
            ('generate_scenes','generate_variable_cases.py',
             ['--config',str(args.config.resolve()),'--output',str(scenes)])]) + [
            ('train_and_evaluate', 'run_variable_training.py',
             ['--config', str(args.config.resolve()), '--scenes', str(scenes),
              '--output', str(output/'training')]),
        ]
        for phase, script, options in stages:
            command = [sys.executable, '-u', str(ROOT/'scripts'/script), *options]
            with (output/f'{phase}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                update(phase, child_pid=child.pid, command=command)
                code = child.wait()
            if code:
                raise RuntimeError(f'{phase} exited with status {code}; see {phase}.log')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        update('complete', status='completed', source_model_unchanged=True)
    except BaseException as exc:
        update('failed', status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
