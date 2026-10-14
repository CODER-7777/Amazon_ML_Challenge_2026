"""Build a reduced-catalog validation fixture from supplied records only.

Training-data businesses are divided into disjoint train and evaluation sets.
French records have no labels and are used only for coverage/format checks.
Metrics on this small catalog are NOT competition-scale performance estimates.
"""
import argparse
import json
import random
from pathlib import Path
from fixture_data import FIELDS, write_tsv
from ber.core import rows


def reservoir_add(pool, row, seen, size, rng):
    if len(pool)<size:
        pool.append(row)
    else:
        j=rng.randrange(seen)
        if j<size:
            pool[j]=row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    dest=args.output
    if (dest/'scope.json').exists():
        print('Reusing completed real-data validation fixture:',dest,flush=True)
        return
    rng=random.Random(2026)
    pools,seen={},{}
    for i,r in enumerate(rows(args.data/'train/train_source1.tsv'),1):
        country=r['country']
        pool=pools.setdefault(country,[])
        seen[country]=seen.get(country,0)+1
        reservoir_add(pool,r,seen[country],500,rng)
        if i%1000000==0:
            print('Sampled from',i,'training anchors',flush=True)
    selected={}
    source_rows={split:{1:[],2:[],3:[]} for split in ('train','test')}
    for country,pool in pools.items():
        rng.shuffle(pool)
        for i,r in enumerate(pool):
            split='train' if i<400 else 'test'
            selected[r['entity_id']]=split
            source_rows[split][1].append([r[k] for k in FIELDS])
    truth={split:{} for split in ('train','test')}
    owner={}
    for r in rows(args.data/'train/train_ground_truth.tsv'):
        aid=r['source1_entity_id']
        if aid in selected:
            ids=list(filter(None,r['matched_entity_ids'].split(',')))
            truth[selected[aid]][aid]=ids
            for tid in ids:
                if tid in owner and owner[tid]!=aid:
                    raise ValueError('Gold target has multiple reference owners.')
                owner[tid]=aid
    found=set()
    for source in (2,3):
        negatives=[]
        n=0
        for i,r in enumerate(rows(args.data/'train'/f'train_source{source}.tsv'),1):
            tid=r['entity_id']
            record=[r[k] for k in FIELDS]
            if tid in owner:
                found.add(tid)
                source_rows[selected[owner[tid]]][source].append(record)
            else:
                n+=1
                reservoir_add(negatives,record,n,1250,rng)
            if i%2000000==0:
                print(f'Train S{source}: {i:,} records read',flush=True)
        rng.shuffle(negatives)
        source_rows['train'][source].extend(negatives[:1000])
        source_rows['test'][source].extend(negatives[1000:])
    if found!=set(owner):
        raise ValueError('Some sampled gold targets were not found in supplied files.')
    for r in rows(args.data/'test/test_source1.tsv'):
        if r['country']=='France':
            source_rows['test'][1].append([r[k] for k in FIELDS])
            if sum(row[3]=='France' for row in source_rows['test'][1])>=30:
                break
    for source in (2,3):
        added=0
        for r in rows(args.data/'test'/f'test_source{source}.tsv'):
            if r['country']=='France':
                source_rows['test'][source].append([r[k] for k in FIELDS])
                added+=1
                if added>=500:
                    break
    for split,sources in source_rows.items():
        for source,records in sources.items():
            write_tsv(dest/split/f'{split}_source{source}.tsv',FIELDS,records)
    write_tsv(dest/'train/train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],[[aid,','.join(ids)] for aid,ids in truth['train'].items()])
    (dest/'known_test_truth.json').write_text(json.dumps(truth['test']))
    scope=dict(not_competition_submission=True,seed=2026,
               description='Reduced catalog: disjoint businesses sampled from supplied training data; 30 unlabeled French reference rows and 1000 French targets from supplied test data. Scores are diagnostic only.',
               counts={split:{str(k):len(v) for k,v in sources.items()} for split,sources in source_rows.items()},
               labeled_test_anchors=len(truth['test']))
    (dest/'scope.json').write_text(json.dumps(scope,indent=2))
    print(json.dumps(scope,indent=2),flush=True)


if __name__=='__main__':
    main()
