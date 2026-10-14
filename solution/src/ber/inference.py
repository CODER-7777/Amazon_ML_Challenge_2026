"""Resumable inference, full-catalog calibration and streaming export."""
import ast
from collections import deque
import csv
import hashlib
import itertools
import json
import os
import signal
import sqlite3
import time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from . import core
from .core import Retriever, build_indexes, features, file_sha256, fingerprint, fold, prepare, rows

# These two historical files were checked against the original scorer. Only
# scheduling differed. All other signature fields must still match on resume.
LEGACY_V1_CODE_HASHES = {
    '184688236461863eee9c28d5f661e6f20c4105041ba15ed7d3da55b924155a9b',
    'dd320ea9f12ffb5827a53f59633bf36fb10c3a065620fadcf2746fe835cfad1f',
}
COMPATIBLE_V1_SCORING_HASH = '45014a8f256ed301966fe1d52fa631b350c82a923d3d9f3e5b888d144bc2ee88'
_worker_retriever = None


def _init_worker(index_path, candidates, limit, max_postings):
    global _worker_retriever
    # The parent handles Ctrl+C and shuts down the worker pool.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_retriever = Retriever(index_path, candidates, limit, max_postings)


def _prepare_with_retriever(raw, retriever):
    target = prepare(raw)
    candidates = retriever.retrieve(target)
    matrix = [features(anchor, target) for anchor in candidates]
    return target, candidates, matrix


def _process_chunk(chunk):
    return [_prepare_with_retriever(raw, _worker_retriever) for raw in chunk]


def _create_pool(workers, index, config):
    import multiprocessing
    # Spawn avoids inheriting the parent's SQLite handles or numerical-library
    # threads. Each worker opens its own read-only anchor index.
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = '1'
        return multiprocessing.get_context('spawn').Pool(
            workers, initializer=_init_worker,
            initargs=(index, config['candidates'], config['retrieval_limit'], config['max_postings']))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _prepared_rows(source, skip, retriever, pool, workers, limit=0):
    raw_rows = itertools.islice(rows(source), skip, None)
    if limit:
        raw_rows = itertools.islice(raw_rows, limit)
    if pool is None:
        for raw in raw_rows:
            yield _prepare_with_retriever(raw, retriever)
        return
    pending = deque()
    def submit():
        chunk = list(itertools.islice(raw_rows, 128))
        if chunk:
            pending.append(pool.apply_async(_process_chunk, (chunk,)))
        return bool(chunk)
    for _ in range(workers * 2):
        if not submit():
            break
    # Consume in submission order so committed row numbers describe a contiguous
    # prefix of each source. Bound queued work to keep memory and prefetch small.
    while pending:
        result = pending.popleft()
        while True:
            try:
                prepared = result.get(timeout=1)
                break
            except __import__('multiprocessing').TimeoutError:
                pass
        for item in prepared:
            yield item
        submit()


def load_truth(data, seed):
    return {r['source1_entity_id']: set(filter(None, r['matched_entity_ids'].split(',')))
            for r in rows(data/'train'/'train_ground_truth.tsv')
            if fold(r['source1_entity_id'], seed) == 'valid'}


def load_model(work):
    model = CatBoostClassifier()
    model.load_model(str(work/'model.cbm'))
    return model, json.loads((work/'model_config.json').read_text())


def scoring_code_hash():
    # Check scoring semantics, not worker scheduling/count. Source segments are
    # stable across Python versions, unlike the fields in ast.dump().
    source = Path(__file__).read_text()
    nodes = {node.name: node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.FunctionDef)}
    digest = hashlib.sha256(Path(core.__file__).read_bytes())
    for name in ('_prepare_with_retriever', 'flush', 'load_model', 'load_truth'):
        digest.update(name.encode() + b'\0')
        digest.update(ast.get_source_segment(source, nodes[name]).encode() + b'\0')
    return digest.hexdigest()


def compatible_legacy_signature(previous, current):
    old, new = dict(previous), dict(current)
    old_code, new_code = old.pop('code', None), new.pop('code', None)
    return (old_code in LEGACY_V1_CODE_HASHES
            and new_code == COMPATIBLE_V1_SCORING_HASH and old == new)


def open_scores(path, signature):
    c = sqlite3.connect(path)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=NORMAL')
    c.execute('PRAGMA cache_size=-32768')
    c.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)')
    previous = c.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
    encoded = json.dumps(signature, sort_keys=True)
    if previous and previous[0] != encoded:
        if not compatible_legacy_signature(json.loads(previous[0]), signature):
            c.close()
            raise ValueError(f'{path} belongs to different data/model/config. Use a new work directory or rename this score database.')
        with c:
            c.execute('INSERT INTO meta VALUES (?,?)',
                      (f'signature_before_parallel:{time.time_ns()}', previous[0]))
            c.execute("UPDATE meta SET value=? WHERE key='signature'", (encoded,))
        print('Migrated verified v1 checkpoint signature; committed scores preserved.', flush=True)
    c.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('signature', encoded))
    c.execute('CREATE TABLE IF NOT EXISTS pairs(anchor TEXT,target TEXT,probability REAL,selected INTEGER,margin REAL,correct INTEGER,PRIMARY KEY(anchor,target)) WITHOUT ROWID')
    c.commit()
    return c


def scan(args, split, truth=None):
    work, data = args.work, args.data
    build_indexes(data/split/f'{split}_source1.tsv', work/f'index_{split}')
    model, config = load_model(work)
    sources = [data/split/f'{split}_source{k}.tsv' for k in (2,3)]
    signature = dict(model=hashlib.sha256((work/'model.cbm').read_bytes()).hexdigest(),
                     code=scoring_code_hash(), config=config,
                     data=[fingerprint(data/split/f'{split}_source{k}.tsv') for k in (1,2,3)], split=split)
    if truth is not None:
        signature['ground_truth'] = fingerprint(data/'train'/'train_ground_truth.tsv')
    db = open_scores(work/f'scores_{split}.sqlite', signature)
    start, new_count = time.monotonic(), 0
    unfinished = False
    workers = getattr(args, 'workers', 1)
    pool = retriever = None
    try:
        if workers > 1:
            pool = _create_pool(workers, work/f'index_{split}', config)
        else:
            retriever = Retriever(work/f'index_{split}', config['candidates'], config['retrieval_limit'], config['max_postings'])
        print(f'Retrieval workers: {workers}; model scoring threads: {args.threads}', flush=True)
        for source in sources:
            if db.execute('SELECT 1 FROM meta WHERE key=?',(source.name+':done',)).fetchone():
                print(f'Reusing scores for {source.name}',flush=True)
                continue
            last = db.execute('SELECT value FROM meta WHERE key=?',(source.name+':rows',)).fetchone()
            skip = int(last[0]) if last else 0
            print(f'Resuming {source.name} after {skip:,} committed rows.', flush=True)
            batch, matrix, last_row = [], [], skip
            def flush(row_number):
                nonlocal batch,matrix
                probabilities = model.predict_proba(np.asarray(matrix,dtype=np.float32),thread_count=args.threads)[:,1] if matrix else np.array([])
                output,offset = [],0
                for target,candidates in batch:
                    scores = probabilities[offset:offset+len(candidates)]
                    offset += len(candidates)
                    if not candidates:
                        continue
                    order = sorted(range(len(candidates)),key=lambda j:(-scores[j],candidates[j]['id']))
                    best = order[0]
                    gap = float(scores[best]-scores[order[1]]) if len(order)>1 else float(scores[best])
                    for j,anchor in enumerate(candidates):
                        aid = anchor['id']
                        if truth is None or aid in truth:
                            correct = int(target['id'] in truth[aid]) if truth is not None else 0
                            output.append((aid,target['id'],float(scores[j]),int(j==best),gap if j==best else 0.,correct))
                with db:
                    db.executemany('INSERT OR REPLACE INTO pairs VALUES (?,?,?,?,?,?)',output)
                    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(source.name+':rows',str(row_number)))
                batch,matrix = [],[]
            remaining = max(0, args.max_targets-new_count) if args.max_targets else 0
            prepared = _prepared_rows(source, skip, retriever, pool, workers, remaining)
            for i, (target, candidates, mat) in enumerate(prepared, skip + 1):
                batch.append((target,candidates))
                matrix.extend(mat)
                new_count += 1
                last_row = i
                if len(batch)>=args.batch_size:
                    flush(i)
                if new_count%5000==0:
                    print(f'{split}: {source.name} row {i:,}; {new_count/(time.monotonic()-start):.1f} targets/s',flush=True)
                if args.max_targets and new_count>=args.max_targets:
                    flush(i)
                    unfinished = True
                    break
            if batch:
                flush(last_row)
            if unfinished:
                break
            with db:
                db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(source.name+':done','1'))
        complete = all(db.execute('SELECT 1 FROM meta WHERE key=?',(p.name+':done',)).fetchone() for p in sources)
        print(f'Processed {new_count:,} new targets in {time.monotonic()-start:.1f}s. Complete={complete}',flush=True)
        return db,complete
    except BaseException:
        db.close()
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        if retriever is not None:
            retriever.close()


def calibrate(args):
    _,config = load_model(args.work)
    truth = load_truth(args.data,config['seed'])
    if not truth:
        raise ValueError('Validation partition is empty.')
    db,complete = scan(args,'train',truth)
    if not complete:
        db.close()
        print('Checkpoint saved. Resume calibrate without --max-targets before submitting.',flush=True)
        return
    keys = list(truth)
    positions = {key:i for i,key in enumerate(keys)}
    counts = np.asarray([len(truth[k]) for k in keys],dtype=np.int32)
    positives = int(counts.sum())
    coverage,candidate_count = db.execute('SELECT COALESCE(SUM(correct),0),COUNT(*) FROM pairs').fetchone()
    grid = []
    for margin in (0.,.025,.05):
        for threshold in (.40,.50,.60,.65,.70,.75,.80,.825,.85,.875,.90,.925,.95,.965,.975,.985,.995):
            tp = np.zeros(len(keys),dtype=np.int32)
            predicted = np.zeros(len(keys),dtype=np.int32)
            query = 'SELECT anchor,SUM(correct),COUNT(*) FROM pairs WHERE selected=1 AND probability>=? AND margin>=? GROUP BY anchor'
            for anchor,correct,total in db.execute(query,(threshold,margin)):
                j = positions[anchor]
                tp[j],predicted[j] = correct,total
            denominator = 1.25*tp+(predicted-tp)+.25*(counts-tp)
            scores = np.divide(1.25*tp,denominator,out=np.zeros(len(keys)),where=denominator>0)
            scores[(counts==0)&(predicted==0)] = 1.
            grid.append(dict(threshold=threshold,margin=margin,macro_f05=float(scores.mean()),
                             precision=float(tp.sum()/max(1,predicted.sum())),recall=float(tp.sum()/max(1,positives))))
    best = max(grid,key=lambda x:(x['macro_f05'],x['threshold'],x['margin']))
    report = dict(**best,validation_anchors=len(keys),validation_singletons=int((counts==0).sum()),
                  blocking_recall=coverage/max(1,positives),candidate_pairs=candidate_count,
                  candidates_per_anchor=candidate_count/len(keys),full_catalog=True,grid=grid,
                  model_sha256=hashlib.sha256((args.work/'model.cbm').read_bytes()).hexdigest(),
                  note='Threshold-tuned validation, not a test/leaderboard estimate. Entity-disjoint training; France has no labeled validation data.')
    (args.work/'calibration.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='grid'},indent=2),flush=True)
    db.close()


def validate_outputs(output,db):
    csv.field_size_limit(2147483647)
    expected = db.execute('SELECT entity_id FROM anchors ORDER BY entity_id')
    count = 0
    for anchor,match,candidate in itertools.zip_longest(expected,rows(output/'matching_results.tsv'),rows(output/'candidate_pairs.tsv')):
        if anchor is None or match is None or candidate is None:
            raise ValueError('Output row count differs from Source 1.')
        if list(match)!=['source1_entity_id','matched_entity_ids'] or list(candidate)!=['source1_entity_id','candidate_entity_ids']:
            raise ValueError('Incorrect output headers.')
        if match['source1_entity_id']!=anchor[0] or candidate['source1_entity_id']!=anchor[0]:
            raise ValueError('Missing or duplicated anchor.')
        m = list(filter(None,match['matched_entity_ids'].split(',')))
        c = list(filter(None,candidate['candidate_entity_ids'].split(',')))
        if len(m)!=len(set(m)) or len(c)!=len(set(c)) or not set(m)<=set(c):
            raise ValueError('Duplicate IDs or a match outside the candidate set.')
        if any(not t.startswith(('S2-','S3-')) for t in c):
            raise ValueError('Invalid target prefix.')
        count += 1
    return count


def predict(args):
    calibration_path = args.work/'calibration.json'
    if not calibration_path.exists():
        raise ValueError('Run full calibrate first; no unvalidated threshold is substituted.')
    calibration = json.loads(calibration_path.read_text())
    if calibration['model_sha256']!=hashlib.sha256((args.work/'model.cbm').read_bytes()).hexdigest():
        raise ValueError('Model changed after calibration; calibrate the new model.')
    db,complete = scan(args,'test')
    if not complete:
        db.close()
        print('Partial checkpoint saved. No submission exported. Resume predict without --max-targets.',flush=True)
        return
    db.execute('CREATE TABLE IF NOT EXISTS anchors(entity_id TEXT PRIMARY KEY) WITHOUT ROWID')
    with db:
        db.executemany('INSERT OR IGNORE INTO anchors VALUES (?)',((r['entity_id'],) for r in rows(args.data/'test'/'test_source1.tsv')))
    output = args.work/'output'
    output.mkdir(exist_ok=True)
    threshold,margin = calibration['threshold'],calibration['margin']
    matches,candidates,previous = [],[],None
    pairs_count,matches_count,singleton_count = 0,0,0
    with open(output/'matching_results.tsv','w',encoding='utf-8',newline='') as mf,open(output/'candidate_pairs.tsv','w',encoding='utf-8',newline='') as cf:
        mw,cw = csv.writer(mf,delimiter='\t'),csv.writer(cf,delimiter='\t')
        mw.writerow(['source1_entity_id','matched_entity_ids'])
        cw.writerow(['source1_entity_id','candidate_entity_ids'])
        def emit():
            nonlocal singleton_count
            mw.writerow([previous,','.join(matches)])
            cw.writerow([previous,','.join(candidates)])
            singleton_count += int(not matches)
        query = 'SELECT a.entity_id,p.target,p.probability,p.selected,p.margin FROM anchors a LEFT JOIN pairs p ON p.anchor=a.entity_id ORDER BY a.entity_id,p.target'
        for anchor,target,probability,selected,gap in db.execute(query):
            if anchor!=previous:
                if previous is not None:
                    emit()
                previous,matches,candidates = anchor,[],[]
            if target is not None:
                candidates.append(target)
                pairs_count += 1
                if selected and probability>=threshold and gap>=margin:
                    matches.append(target)
                    matches_count += 1
        if previous is not None:
            emit()
    anchors = validate_outputs(output,db)
    report = dict(anchors=anchors,candidate_pairs=pairs_count,matches=matches_count,
                  predicted_singletons=singleton_count,candidates_per_anchor=pairs_count/max(1,anchors),
                  threshold=threshold,margin=margin,full_test=True,model_sha256=calibration['model_sha256'],
                  calibration_sha256=file_sha256(calibration_path),
                  output_sha256={name:file_sha256(output/name) for name in ('matching_results.tsv','candidate_pairs.tsv')},
                  validation='PASS: all anchors exactly once; actual scored candidates; matches subset; valid prefixes. Target existence follows from scoring supplied target rows only.')
    (args.work/'prediction_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    db.close()
