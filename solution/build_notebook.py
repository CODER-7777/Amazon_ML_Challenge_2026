"""Create cloud execution artifacts from the validated source tree."""
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parent
out=root.parent/'cloud_upload'
out.mkdir(exist_ok=True)
validator=root.parent/'6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py'
with zipfile.ZipFile(out/'solution_code.zip','w',zipfile.ZIP_DEFLATED) as z:
    for folder in ('src','tests'):
        for path in sorted((root/folder).rglob('*.py')):
            z.write(path,str(path.relative_to(root)))
    z.write(root/'requirements.txt','requirements.txt')
    z.write(validator,'utils/validate_submission.py')
cells=[]
def markdown(text):
    cells.append(dict(cell_type='markdown',metadata={},source=text))
def code(text):
    cells.append(dict(cell_type='code',metadata={},source=text,execution_count=None,outputs=[]))
markdown('''# Amazon ML Challenge — SageMaker execution
Upload this notebook and `solution_code.zip` into the same JupyterLab folder.
Use Python 3.10–3.12, a CPU space with roughly 32 GB RAM, and 50 GB storage as a
starting configuration. The full-catalog benchmark below determines runtime.
Set your dataset location and team name, then run cells in order. Existing tests
verify correctness; they do not establish full-competition model accuracy.
''')
code('''from pathlib import Path
TEAM = "YOUR_TEAM"  # your registered team name
DATA_DIRECTORY = ""  # existing directory containing train/ and test/, if available
LOCAL_DATA_ZIP = ""  # alternatively an uploaded dataset zip's path
S3_DATA_URI = ""  # alternatively s3://your-bucket/your-dataset.zip
BASE = Path.cwd() / "amazon_ml_run"
THREADS = 8
SAMPLE_ANCHORS = 20000
RANDOM_TARGETS = 100000
ITERATIONS = 800
assert TEAM.upper() not in {"YOUR_TEAM", "YOUR_TEAM_NAME", "TEAM_NAME"}, "Enter your registered team name."
BASE.mkdir(exist_ok=True)
''')
code('''import os, sys, subprocess, zipfile, shutil
assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Select a Python 3.10–3.12 image."
CODE = BASE / "solution"
CODE.mkdir(exist_ok=True)
def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError("Unsafe ZIP path")
        bundle.extractall(destination)
safe_extract(Path.cwd() / "solution_code.zip", CODE)
ENV = BASE / "env"
PYTHON = ENV / "bin/python"
if not PYTHON.exists():
    subprocess.run([sys.executable, "-m", "venv", str(ENV)], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(CODE / "requirements.txt")], check=True)
print("Free disk GB:", round(shutil.disk_usage(BASE).free / 2**30, 1))
''')
code('''from urllib.parse import urlparse
if DATA_DIRECTORY:
    DATA = Path(DATA_DIRECTORY).expanduser().resolve()
else:
    if LOCAL_DATA_ZIP:
        archive = Path(LOCAL_DATA_ZIP).expanduser().resolve()
    else:
        assert S3_DATA_URI.startswith("s3://"), "Set DATA_DIRECTORY, LOCAL_DATA_ZIP, or S3_DATA_URI."
        import boto3  # uses your SageMaker execution role
        uri = urlparse(S3_DATA_URI)
        archive = BASE / "provided_dataset.zip"
        client = boto3.client("s3")
        expected = client.head_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))["ContentLength"]
        if not archive.exists() or archive.stat().st_size != expected:
            client.download_file(uri.netloc, uri.path.lstrip("/"), str(archive))
    destination = BASE / "data"
    destination.mkdir(exist_ok=True)
    archive_stamp = str(archive.resolve()) + ":" + str(archive.stat().st_size) + ":" + str(archive.stat().st_mtime_ns)
    marker = destination / ".archive_stamp"
    if not marker.exists():
        safe_extract(archive, destination)
        marker.write_text(archive_stamp)
    else:
        assert marker.read_text() == archive_stamp, "Archive changed. Choose a fresh BASE directory."
    matches = [p for p in destination.rglob("train_source1.tsv") if p.parent.name == "train"]
    assert len(matches) == 1, "Expected one supplied dataset."
    DATA = matches[0].parent.parent
required = [DATA / "train" / f"train_{x}.tsv" for x in ("source1", "source2", "source3", "ground_truth")]
required += [DATA / "test" / f"test_source{x}.tsv" for x in (1, 2, 3)]
assert all(p.is_file() for p in required), "Missing dataset files."
WORK = BASE / "artifacts"
WORK.mkdir(exist_ok=True)
print("Dataset:", DATA)
print("Checkpoints:", WORK)
''')
code('''ENVIRONMENT = dict(os.environ, PYTHONPATH=str(CODE / "src"))
def run_process(command):
    with (WORK / "run.log").open("a") as log:
        log.write("\\n" + " ".join(command) + "\\n")
        process = subprocess.Popen(command, env=ENVIRONMENT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError("Command failed; inspect run.log before continuing.")
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

def run(stage, *extra):
    run_process([str(PYTHON), "-u", "-m", "ber", "--data", str(DATA), "--work", str(WORK), "--threads", str(THREADS), *map(str, extra), stage])

run_process([str(PYTHON), "-m", "unittest", "discover", "-s", str(CODE / "tests"), "-p", "test_core.py", "-v"])
''')
markdown('Train the classifier. This is the first full-data setup step; the search index is reused later.')
code('run("train", "--sample-anchors", SAMPLE_ANCHORS, "--random-targets", RANDOM_TARGETS, "--iterations", ITERATIONS)')
markdown('Benchmark validation speed on 2,000 targets. A checkpoint is saved; no partial calibration is published. Full training and test catalogs contain about 10.32 million and 9.97 million target records respectively. Inspect throughput before the full scans. Retrieval is single-process; classifier threads do not parallelize retrieval.')
code('run("calibrate", "--max-targets", 2000)')
markdown('Finish full validation and threshold selection. Re-running this cell resumes committed progress.')
code('run("calibrate")')
markdown('Run full test inference. It includes every Source 1 row, including France and singletons.')
code('run("predict")')
markdown('Run the supplied official validator, including target-ID existence checks. This needs more RAM than the streaming checks built into the pipeline.')
code('''run_process([str(PYTHON), str(CODE / "utils/validate_submission.py"),
             "--matching", str(WORK / "output/matching_results.tsv"),
             "--candidate", str(WORK / "output/candidate_pairs.tsv"),
             "--test-dir", str(DATA / "test"), "--check-ids"])
''')
markdown('Package measured results, code, model, and both TSVs. The methodology is generated now from actual run metrics.')
code('''run("package", "--team", TEAM)
from IPython.display import FileLink, display
print("Leaderboard upload:")
display(FileLink(str((WORK / "output/matching_results.tsv").relative_to(Path.cwd()))))
print("Final submission ZIP:")
for archive in WORK.glob("*_submission.zip"):
    display(FileLink(str(archive.relative_to(Path.cwd()))))
''')
markdown('Download the outputs, then stop the SageMaker application when compute is no longer needed. Retain the space/checkpoints until backups are safe; deleting the space removes its storage. Closing the browser does not stop compute billing.')
for i,cell in enumerate(cells):
    cell['id']=f'cell-{i:02d}'
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
path=out/'Amazon_ML_SageMaker.ipynb'
path.write_text(json.dumps(notebook,indent=1))
for cell in cells:
    if cell['cell_type']=='code':
        compile(cell['source'],cell['id'],'exec')
print('Generated and syntax-checked:',path)
print('Source bundle:',out/'solution_code.zip')
