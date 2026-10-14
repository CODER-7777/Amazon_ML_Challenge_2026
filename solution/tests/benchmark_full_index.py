"""Measure sampled retrieval against the actual full training anchor catalog."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from ber.core import Retriever, build_indexes, prepare, rows

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--sample',type=Path,required=True)
p.add_argument('--work',type=Path,required=True)
a=p.parse_args()
a.work.mkdir(parents=True,exist_ok=True)
started=time.monotonic()
build_indexes(a.data/'train/train_source1.tsv',a.work/'index_train_full')
build_seconds=time.monotonic()-started
truth={r['source1_entity_id']:list(filter(None,r['matched_entity_ids'].split(','))) for r in rows(a.sample/'train/train_ground_truth.tsv')}
truth.update(json.loads((a.sample/'known_test_truth.json').read_text()))
owners={target:anchor for anchor,ids in truth.items() for target in ids}
retriever=Retriever(a.work/'index_train_full')
counts=Counter()
by_country={}
started=time.monotonic()
for split in ('train','test'):
    for source in (2,3):
        for raw in rows(a.sample/split/f'{split}_source{source}.tsv'):
            if raw['country']=='France':
                continue
            target=prepare(raw)
            candidates=retriever.retrieve(target)
            counts['targets']+=1
            counts['candidates']+=len(candidates)
            if target['id'] in owners:
                country=by_country.setdefault(raw['country'],Counter())
                found=any(record['id']==owners[target['id']] for record in candidates)
                counts['gold_links']+=1
                counts['retrieved_gold_links']+=found
                country['gold_links']+=1
                country['retrieved_gold_links']+=found
            if counts['targets']%1000==0:
                print(f"Full-index retrieval: {counts['targets']:,} targets; {counts['targets']/(time.monotonic()-started):.1f}/s",flush=True)
elapsed=time.monotonic()-started
retriever.close()
report=dict(scope='Sampled targets against all 2,206,821 supplied training anchors. Blocking-only, not final-model or leaderboard performance.',
            counts=dict(counts),country_counts={c:dict(n) for c,n in by_country.items()},
            blocking_recall=counts['retrieved_gold_links']/counts['gold_links'],
            candidates_per_target=counts['candidates']/counts['targets'],index_build_seconds=build_seconds,
            retrieval_seconds=elapsed,targets_per_second=counts['targets']/elapsed,
            estimated_10m_target_retrieval_hours=10000000/(counts['targets']/elapsed)/3600)
(a.work/'full_index_benchmark.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
