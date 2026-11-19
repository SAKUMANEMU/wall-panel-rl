"""Recompute a v23 episode with its selected runtime, then replay recorded states."""
import _bootstrap
import argparse
from pathlib import Path
from wall_panel_rl.submission import load_agent,load_json,run_episode,save_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--seed',type=int);p.add_argument('--headless',action='store_true');a=p.parse_args()
    cases=load_json(a.run/'expanded_test_cases.json')
    case=next((c for c in cases if a.seed is None or c['scene']['seed']==a.seed),None)
    if case is None:p.error('Seed is not in the expanded test bank')
    result,recording=run_episode(load_agent(a.run/'models/selected_model.zip'),case,'enhanced',load_json(a.run/'selected_runtime.json'))
    output=a.run/'demo'/f'episode_{case["scene"]["seed"]}.json';save_json(output,recording)
    print(result,flush=True);print('Recording:',output,flush=True)
    if not a.headless:
        # Use the public replay CLI so its rendering options remain consistent.
        import subprocess,sys
        subprocess.run([sys.executable,str(Path(__file__).with_name('replay_episode.py')),str(output),'--fps','60','--playback-speed','0.6'],check=True)
if __name__=='__main__':main()
