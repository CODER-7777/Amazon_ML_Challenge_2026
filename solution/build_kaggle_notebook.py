"""Build the Kaggle training notebook; no training is performed here."""
import json
from pathlib import Path

cells = []


def md(source):
    cells.append(dict(cell_type='markdown', metadata={}, source=source))


def code(source):
    cells.append(dict(cell_type='code', metadata={}, source=source,
                      execution_count=None, outputs=[]))


md('''# Amazon ML Challenge — Kaggle training
Attach your PRIVATE dataset containing solution_code.zip and provided_dataset.zip.
Kaggle may extract the ZIP files automatically; this notebook supports both layouts.
Select Accelerator: None and enable Internet to install the pinned dependencies.
Use Save Version → Save & Run All for a saved background run.

This notebook trains the existing CPU CatBoost model, saves its artifacts, and
benchmarks candidate search. It does not launch the full calibration/test scans:
the existing single-process search may exceed Kaggle's 12-hour session limit.
Training artifacts are not leaderboard submissions.
''')

code('''from pathlib import Path
import os, sys, json, shutil, subprocess, zipfile

INPUT = Path('/kaggle/input')
BASE = Path('/kaggle/working/amazon_ml')
CODE = BASE / 'solution'
WORK = BASE / 'artifacts'
RESULTS = Path('/kaggle/working/training_results')
THREADS = min(4, os.cpu_count() or 2)
SAMPLE_ANCHORS = 20000
RANDOM_TARGETS = 100000
ITERATIONS = 800
for directory in (CODE, WORK, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)
assert (3, 10) <= sys.version_info[:2] <= (3, 12), (
    'The pinned dependencies require Python 3.10–3.12. '
    'Select a compatible notebook environment before proceeding.')
assert INPUT.exists(), 'Run this notebook on Kaggle with your private dataset attached.'
print('CPU threads:', THREADS)
print('Python:', sys.version.split()[0])
print('Free working disk GiB:', round(shutil.disk_usage(BASE).free / 2**30, 1))

def safe_extract(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError('Unsafe path in ZIP')
        bundle.extractall(destination)

bundles = list(INPUT.rglob('solution_code.zip'))
if bundles:
    assert len(bundles) == 1, 'Attach only one solution_code.zip.'
    safe_extract(bundles[0], CODE)
else:
    entries = list(INPUT.glob('**/src/ber/__main__.py'))
    assert len(entries) == 1, 'Add the private dataset containing solution_code.zip.'
    source = entries[0].parents[2]
    for folder in ('src', 'tests', 'utils'):
        shutil.copytree(source / folder, CODE / folder, dirs_exist_ok=True)
    shutil.copy2(source / 'requirements.txt', CODE / 'requirements.txt')

anchors = [p for p in INPUT.rglob('train_source1.tsv') if p.parent.name == 'train']
if not anchors:
    archives = list(INPUT.rglob('provided_dataset.zip'))
    assert len(archives) == 1, 'Add the private dataset containing provided_dataset.zip.'
    # Extract outside /kaggle/working so source data is not saved as notebook output.
    extracted = Path('/tmp/amazon_ml_dataset')
    safe_extract(archives[0], extracted)
    anchors = [p for p in extracted.rglob('train_source1.tsv') if p.parent.name == 'train']
assert len(anchors) == 1, 'Expected exactly one supplied dataset.'
DATA = anchors[0].parent.parent
required = [DATA / 'train' / f'train_{name}.tsv'
            for name in ('source1', 'source2', 'source3', 'ground_truth')]
required += [DATA / 'test' / f'test_source{i}.tsv' for i in (1, 2, 3)]
assert all(p.is_file() for p in required), 'The supplied dataset is incomplete.'
print('Dataset:', DATA)
print('Dataset GiB:', round(sum(p.stat().st_size for p in required) / 2**30, 2))
''')

md('Install the pinned packages in a private package directory. No business lookup API is used.')
code('''DEPS = Path('/tmp/amazon_ml_deps')
PYTHON = Path(sys.executable)
DEPS.mkdir(parents=True, exist_ok=True)
subprocess.run([str(PYTHON), '-m', 'pip', '--version'], check=True)
subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
                '--target', str(DEPS), '-r', str(CODE / 'requirements.txt')], check=True)
ENVIRONMENT = dict(os.environ, PYTHONPATH=os.pathsep.join((str(DEPS), str(CODE / 'src'))))
subprocess.run([str(PYTHON), '-c',
                'import numpy, catboost, rapidfuzz; '
                'print("numpy:", numpy.__version__, "catboost:", catboost.__version__)'],
               env=ENVIRONMENT, check=True)

def run_process(command):
    collected = []
    with (WORK / 'run.log').open('a') as log:
        log.write('\\n' + ' '.join(command) + '\\n')
        process = subprocess.Popen(command, env=ENVIRONMENT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                collected.append(line)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError('Command failed; see amazon_ml/artifacts/run.log.')
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return ''.join(collected)

def run(stage, *extra):
    return run_process([str(PYTHON), '-u', '-m', 'ber', '--data', str(DATA),
                        '--work', str(WORK), '--threads', str(THREADS),
                        *map(str, extra), stage])

run_process([str(PYTHON), '-m', 'unittest', 'discover', '-s',
             str(CODE / 'tests'), '-p', 'test_core.py', '-v'])
''')

md('Train. Initial indexing and feature generation use the CPU and can take longer than fitting the classifier. Progress appears in the log. Running this cell again retrains the classifier.')
code('''run('train', '--sample-anchors', SAMPLE_ANCHORS,
    '--random-targets', RANDOM_TARGETS, '--iterations', ITERATIONS)

def save_training_results():
    for name in ('model.cbm', 'model_config.json', 'training_report.json', 'run.log'):
        shutil.copy2(WORK / name, RESULTS / name)
    shutil.copy2(CODE / 'requirements.txt', RESULTS / 'requirements.txt')
    for name in ('benchmark_log.txt', 'benchmark_summary.json'):
        if (RESULTS / name).exists():
            print('Including:', name)
    archive = shutil.make_archive('/kaggle/working/training_results', 'zip', RESULTS)
    print('Saved:', archive)

save_training_results()
print('Training complete. The next cell measures search speed.')
''')

md('Benchmark only 2,000 validation targets. This saves a partial checkpoint and does not select the final threshold or create submission files. The runtime estimate is approximate: country mix, I/O, and later stages can change it.')
code('''import re
benchmark_log = run('calibrate', '--max-targets', 2000)
(RESULTS / 'benchmark_log.txt').write_text(benchmark_log)
match = re.search(r'Processed ([\\d,]+) new targets in ([\\d.]+)s', benchmark_log)
if match:
    processed = int(match.group(1).replace(',', ''))
    seconds = float(match.group(2))
    rate = processed / max(seconds, 0.001)
    # Counts of targets in the supplied complete training and test catalogs.
    remaining = 20_289_808 - processed
    estimate = remaining / max(rate, 0.001) / 3600
    summary = dict(benchmark_targets=processed, scan_seconds=seconds,
                   targets_per_second=rate, estimated_remaining_scan_hours=estimate,
                   note='Estimate excludes index building, setup, threshold tuning, '
                        'export and packaging. Not an accuracy evaluation.')
    (RESULTS / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('STOP HERE: assess this speed before scheduling full calibration and inference.')
else:
    print('Read benchmark_log.txt for measured progress.')
save_training_results()
''')

md('''After the saved version succeeds, open its Output tab and download training_results.zip.
It contains the trained model, configuration, training report and benchmark logs.
Keep the notebook and attached dataset private.

This ZIP is NOT the competition submission. The leaderboard file is
matching_results.tsv, which requires completed calibration and test inference.
Save & Run All runs a new session from the beginning; do not also run a duplicate
interactive training job. Closing the browser after starting the saved run does
not remove its 12-hour execution limit.
''')

for i, cell in enumerate(cells):
    cell['id'] = f'kaggle-{i:02d}'
    if cell['cell_type'] == 'code':
        compile(cell['source'], cell['id'], 'exec')
notebook = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1] / 'cloud_upload/Amazon_ML_Kaggle.ipynb'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(notebook, indent=1))
print('Generated and syntax-checked:', path)
