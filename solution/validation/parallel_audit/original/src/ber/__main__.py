import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Business entity resolution: train → calibrate → predict → package')
    p.add_argument('--data',type=Path,required=True,help='Dataset directory containing train/ and test/.')
    p.add_argument('--work',type=Path,default=Path('artifacts'))
    p.add_argument('--threads',type=int,default=min(8,os.cpu_count() or 2))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sample-anchors',type=int,default=20000)
    p.add_argument('--random-targets',type=int,default=100000)
    p.add_argument('--iterations',type=int,default=800)
    p.add_argument('--candidates',type=int,default=3)
    p.add_argument('--retrieval-limit',type=int,default=12)
    p.add_argument('--max-postings',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--max-targets',type=int,default=0,help='Benchmark/checkpoint only; 0 scans the full catalog.')
    p.add_argument('--team',default='YOUR_TEAM')
    p.add_argument('command',choices=['train','calibrate','predict','package','all'])
    args = p.parse_args()
    args.data,args.work = args.data.resolve(),args.work.resolve()
    if not (args.data/'train'/'train_source1.tsv').exists() or not (args.data/'test'/'test_source1.tsv').exists():
        p.error('--data must contain the supplied train/ and test/ directories.')
    if min(args.threads,args.sample_anchors,args.iterations,args.candidates,args.retrieval_limit,args.max_postings,args.batch_size)<=0 or args.random_targets<0 or args.max_targets<0:
        p.error('Counts must be positive; random-targets and max-targets may be zero.')
    args.work.mkdir(parents=True,exist_ok=True)
    from .train import train_model
    from .inference import calibrate,predict
    from .package import package
    actions = dict(train=train_model,calibrate=calibrate,predict=predict,package=package)
    if args.command=='all':
        if args.max_targets:
            p.error('Run benchmark commands individually; all requires complete data.')
        for action in actions.values():
            action(args)
    else:
        actions[args.command](args)


if __name__=='__main__':
    main()
