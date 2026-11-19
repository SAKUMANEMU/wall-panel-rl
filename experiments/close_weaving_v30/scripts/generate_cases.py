"""Generate extra training scenes with the current layout rule, not a success filter."""
import _bootstrap
import argparse
from pathlib import Path
import torch
import yaml
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import load_agent, load_json, load_runtime, save_json
from wall_panel_rl.rl.vertical_corridor import fresh_vertical_corridor_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--start-seed', type=int, default=1209000)
    parser.add_argument('--max-attempts', type=int, default=300)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/generated_train_cases.json')
    args=parser.parse_args()
    if args.output.exists() or args.count<1 or args.max_attempts<args.count:
        parser.error('Use a new output file and a valid positive generation budget')
    torch.set_num_threads(1)
    runtime=load_runtime();model=load_agent()
    profile=yaml.safe_load((ROOT/'configs/scene.yaml').read_text())
    split=load_json(ROOT/'data/split.json')
    excluded=set(split['train']+split['validation']+split['demo'])
    cases=[];attempts=[]
    for seed in range(args.start_seed,args.start_seed+args.max_attempts):
        if seed in excluded:
            attempts.append({'seed':seed,'excluded_existing_split':True})
            continue
        try:
            case=fresh_vertical_corridor_case(runtime['configuration'],runtime['settings'],profile,
                seed,model,runtime['controller'])
            cases.append(case)
            attempts.append({'seed':seed,'geometry_valid':True})
        except (RuntimeError,ValueError) as exc:
            attempts.append({'seed':seed,'geometry_valid':False,'reason':str(exc)})
        print(f'{len(cases)}/{args.count} valid scenes after seed {seed}',flush=True)
        if len(cases)==args.count:
            break
    save_json(args.output,cases)
    save_json(args.output.with_suffix('.attempts.json'),attempts)
    if len(cases)!=args.count:
        raise RuntimeError('Generation budget exhausted; partial cases and all attempts are saved')


if __name__=='__main__':
    main()
