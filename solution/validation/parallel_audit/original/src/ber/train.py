"""Entity-disjoint sampled supervised training against the full anchor index."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from .core import FEATURE_NAMES, Retriever, build_indexes, features, fingerprint, fold, prepare, rows, stable_hash


def sample_records(data, work, sample_anchors, random_targets, seed):
    path = work / 'sample.tsv'
    owners_path, meta_path = work / 'sample_owners.json', work / 'sample_complete.json'
    files = [data/'train'/f'train_{s}.tsv' for s in ('source1', 'source2', 'source3', 'ground_truth')]
    signature = dict(files=[fingerprint(p) for p in files], sample_anchors=sample_anchors, random_targets=random_targets, seed=seed)
    if meta_path.exists() and json.loads(meta_path.read_text())['signature'] == signature:
        return path, json.loads(owners_path.read_text())
    count = sum(json.loads((work/'index_train'/'complete.json').read_text())['countries'].values())
    proportion = min(1., sample_anchors / count)
    selected_owners = {}
    selected_anchors = 0
    for r in rows(files[3]):
        if stable_hash(f"sample:{seed}:{r['source1_entity_id']}") / 2**64 < proportion:
            selected_anchors += 1
            for target in filter(None, r['matched_entity_ids'].split(',')):
                if target in selected_owners and selected_owners[target] != r['source1_entity_id']:
                    raise ValueError('A target belongs to multiple S1 entities; reverse-assignment assumption fails.')
                selected_owners[target] = r['source1_entity_id']
    extra_ids = set()
    owners = {}
    random_rate = min(1., random_targets / max(1, count * 5))
    fields = ['entity_id', 'business_name', 'business_address', 'country']
    with open(path.with_suffix('.building'), 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for source in files[1:3]:
            for i, r in enumerate(rows(source), 1):
                target = r['entity_id']
                if target in selected_owners or stable_hash(f'extra:{seed}:{target}') / 2**64 < random_rate:
                    writer.writerow({k: r[k] for k in fields})
                    if target in selected_owners:
                        owners[target] = selected_owners[target]
                    else:
                        owners[target] = None
                        extra_ids.add(target)
                if i % 1000000 == 0:
                    print(f'Sampling {source.name}: {i:,} read, {len(owners):,} retained', flush=True)
    # Resolve randomly sampled records using gold labels, never assume they are negative.
    for r in rows(files[3]):
        for target in filter(None, r['matched_entity_ids'].split(',')):
            if target in extra_ids:
                if owners[target] is not None and owners[target] != r['source1_entity_id']:
                    raise ValueError('Multiple gold owners for a sampled target.')
                owners[target] = r['source1_entity_id']
    path.with_suffix('.building').replace(path)
    owners_path.write_text(json.dumps(owners))
    meta_path.write_text(json.dumps(dict(signature=signature, selected_anchors=selected_anchors, targets=len(owners)), indent=2))
    print(f'Sampled {selected_anchors:,} complete entity groups and {len(extra_ids):,} additional targets.', flush=True)
    return path, owners


def train_model(args):
    data, work = args.data, args.work
    work.mkdir(parents=True, exist_ok=True)
    build_indexes(data/'train'/'train_source1.tsv', work/'index_train')
    sample, owners = sample_records(data, work, args.sample_anchors, args.random_targets, args.seed)
    retriever = Retriever(work/'index_train', args.candidates, args.retrieval_limit, args.max_postings)
    x, y, partitions = [], [], []
    covered, eligible, injected = 0, 0, 0
    start = time.monotonic()
    for i, raw in enumerate(rows(sample), 1):
        target = prepare(raw)
        owner = owners[target['id']]
        partition = fold(owner or target['id'], args.seed)
        candidates = retriever.retrieve(target)
        found = any(a['id'] == owner for a in candidates)
        if owner:
            eligible += 1
            covered += found
        # Extra gold positives teach the classifier; this never changes validation
        # or inference candidates, and is excluded from the blocking-recall count.
        if partition == 'train' and owner and not found:
            anchor = retriever.get(target['country'], owner)
            if anchor:
                candidates.append(anchor)
                injected += 1
        for anchor in candidates:
            # Both ends' gold business groups are held out together.
            if fold(anchor['id'], args.seed) != partition:
                continue
            x.append(features(anchor, target))
            y.append(int(anchor['id'] == owner))
            partitions.append(partition == 'valid')
        if i % 5000 == 0:
            print(f'Training features: {i:,} targets, {len(x):,} pairs, {i/(time.monotonic()-start):.1f} targets/s', flush=True)
    retriever.close()
    x = np.asarray(x, dtype=np.float32)
    y, valid = np.asarray(y, dtype=np.int8), np.asarray(partitions, dtype=bool)
    if not len(x) or len(set(y[~valid])) < 2 or len(set(y[valid])) < 2:
        raise ValueError('Need both labels in both partitions. Increase --sample-anchors / --random-targets.')
    model = CatBoostClassifier(iterations=args.iterations, depth=7, learning_rate=.06,
                               loss_function='Logloss', eval_metric='AUC', random_seed=args.seed,
                               thread_count=args.threads, l2_leaf_reg=5, verbose=100,
                               allow_writing_files=False)
    model.fit(x[~valid], y[~valid], eval_set=(x[valid], y[valid]), early_stopping_rounds=80)
    model.save_model(str(work/'model.cbm'))
    config = dict(seed=args.seed, candidates=args.candidates, retrieval_limit=args.retrieval_limit,
                  max_postings=args.max_postings, features=FEATURE_NAMES, threads=args.threads)
    (work/'model_config.json').write_text(json.dumps(config, indent=2))
    report = dict(train_pairs=int((~valid).sum()), validation_pairs=int(valid.sum()),
                  train_positives=int(y[~valid].sum()), validation_positives=int(y[valid].sum()),
                  sampled_blocking_recall=covered/max(1, eligible), injected_training_positives=injected,
                  trees=int(model.tree_count_), elapsed_seconds=time.monotonic()-start,
                  settings=dict(sample_anchors=args.sample_anchors,random_targets=args.random_targets,iterations=args.iterations,seed=args.seed,candidates=args.candidates,retrieval_limit=args.retrieval_limit,max_postings=args.max_postings,threads=args.threads),
                  note='Pair validation is entity-disjoint. Run calibrate on the FULL training catalog for macro F0.5 and the final threshold.')
    (work/'training_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
