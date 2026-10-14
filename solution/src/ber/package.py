"""Create the required final ZIP only after training, calibration and inference."""
import json
import hashlib
import re
import zipfile
from pathlib import Path
from .core import file_sha256


def package(args):
    work = args.work
    needed = ['model.cbm','model_config.json','training_report.json','calibration.json','prediction_report.json','output/matching_results.tsv','output/candidate_pairs.tsv']
    for name in needed:
        if not (work/name).is_file():
            raise ValueError(f'Missing {name}; finish the pipeline before packaging.')
    calibration = json.loads((work/'calibration.json').read_text())
    prediction = json.loads((work/'prediction_report.json').read_text())
    training = json.loads((work/'training_report.json').read_text())
    if not calibration.get('full_catalog') or not prediction.get('full_test'):
        raise ValueError('Only complete calibration and inference can be packaged.')
    current_hash = hashlib.sha256((work/'model.cbm').read_bytes()).hexdigest()
    if calibration['model_sha256'] != current_hash or prediction['model_sha256'] != current_hash:
        raise ValueError('Model, calibration and predictions must belong to the same run.')
    if prediction.get('calibration_sha256') != file_sha256(work/'calibration.json'):
        raise ValueError('Calibration changed after prediction. Run predict again before packaging.')
    for filename in ('matching_results.tsv','candidate_pairs.tsv'):
        if prediction.get('output_sha256', {}).get(filename) != file_sha256(work/'output'/filename):
            raise ValueError(f'{filename} changed after validation. Run predict again before packaging.')
    team = re.sub(r'[^A-Za-z0-9_-]+','_',args.team).strip('_')
    if not team or team.upper() in {'YOUR_TEAM','TEAM_NAME','YOUR_TEAM_NAME'}:
        raise ValueError('Set --team to your registered team name.')
    root = Path(__file__).resolve().parents[2]
    name = work/f'{team}_submission.zip'
    method = f'''# Business Entity Resolution — {args.team}

## Method
We use only the supplied business name, address and country fields. Unicode
normalization, punctuation handling, legal-suffix variants and address abbreviation
normalization create comparable views. Country labels are discovered dynamically;
France is included even though it has no labeled training examples.

## Candidate generation
Disk-backed SQLite FTS5 indexes contain Source 1 anchors. Each Source 2/3 record
queries separate name-token, name-character-trigram and address-token routes.
Frequent terms are excluded from retrieval. A cheap lexical filter limits the
candidate set before classifier inference. Reverse retrieval allows arbitrarily
many records to match an anchor. Every pair scored at final inference is recorded
in candidate_pairs.tsv; it is not reconstructed from accepted matches.

## Matching and validation
A CatBoost classifier learns from {training['train_pairs']:,} training pairs using
name/address similarities, token overlap, numeric agreement/conflict and missingness.
Hard negatives come from retrieval. Gold positives missed by blocking may be added
only to classifier training, never to validation or test candidates. A deterministic
entity-level split reserves 20% of reference businesses. Labeled target records
belonging to held-out businesses are excluded from classifier training. The full
training target catalog is scanned for threshold calibration, including singleton
anchors and records outside the sampled classifier-training pool.

At inference a target is assigned to its highest-scoring S1 candidate only when
probability and optional competitor-margin thresholds pass. This reflects the
provided deduplicated reference source and permits zero or many targets per anchor.

## Measured results
- Threshold-tuned validation macro F0.5: {calibration['macro_f05']:.6f}
- Validation anchors: {calibration['validation_anchors']:,}
- Blocking recall on validation: {calibration['blocking_recall']:.6f}
- Acceptance probability threshold: {calibration['threshold']}
- Competitor margin: {calibration['margin']}
- Test anchors: {prediction['anchors']:,}
- Actual test candidate pairs: {prediction['candidate_pairs']:,}
- Candidates per test anchor: {prediction['candidates_per_anchor']:.3f}
- Accepted test matches: {prediction['matches']:,}

These are local validation results, not leaderboard scores. The threshold was chosen
on validation, so the reported maximum is not an unbiased held-out test estimate.
No labeled French evaluation is available. Heavy name corruption, aliases combined
with absent addresses, and very common names remain potential failure cases. This
submission contains one measured baseline, not an untested claim of optimality.

## Reproduction
See code/business_entity_resolution/README.md and the saved JSON reports. The model,
configuration, code and pinned dependencies are included. The original supplied
dataset is the only separate input. No external entity lookup or augmentation is used.
'''
    readme = '''# Reproduce the submitted predictions
Use Python 3.10–3.12. Extract the original provided dataset so DATASET contains
train/train_source1.tsv and test/test_source1.tsv (and the remaining supplied TSVs).
Run from this code/business_entity_resolution directory:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m ber --data /path/to/DATASET --work artifacts predict
```
This uses the included trained model and calibrated thresholds, builds a local
Source 1 search index, scans all test targets, and recreates both TSV files under
artifacts/output/. Re-running resumes committed inference progress. The index and
score database need local disk space; retain artifacts between runs. No GPU is required.

To retrain and recalibrate, use a fresh artifacts directory:
```bash
PYTHONPATH=src python -m ber --data /path/to/DATASET --work retrained train
PYTHONPATH=src python -m ber --data /path/to/DATASET --work retrained calibrate
PYTHONPATH=src python -m ber --data /path/to/DATASET --work retrained predict
```
The saved training report records actual training settings. Use those settings to
reproduce a non-default training run. A full catalog scan is substantial: benchmark
with --max-targets 1000 and then resume without that flag. Partial scans cannot be
exported or packaged as submissions. SQLite FTS5 support is required.
'''
    with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for filename in needed:
            destination = filename if filename.startswith('output/') else 'code/business_entity_resolution/artifacts/'+filename
            archive.write(work/filename,destination)
        for path in (root/'src').rglob('*.py'):
            archive.write(path,'code/business_entity_resolution/'+str(path.relative_to(root)))
        archive.write(root/'requirements.txt','code/business_entity_resolution/requirements.txt')
        archive.writestr('code/business_entity_resolution/README.md',readme)
        archive.writestr('Documentation_template.md',method)
    print(f'LEADERBOARD: {work / "output/matching_results.tsv"}',flush=True)
    print(f'FINAL PACKAGE: {name}',flush=True)
