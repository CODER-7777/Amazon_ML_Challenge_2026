📢 Update: candidate_pairs.tsv is part of your final submission
Blocking has to scale. Amazon resolves business entities across billions of records, so comparing every record with every other one is not an option. Your blocking / candidate-generation step must cut the search space to a small candidate set per Source 1 entity.
Candidate generation counts toward the final ranking. We will review your candidate_pairs.tsv and the code that produces it when deciding final rankings, alongside your matching_results.tsv score. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard.
Business Entity Resolution Challenge
In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers, and the challenge of determining which records refer to the same business is known as Entity Resolution (ER). Your challenge is to build an ML solution that, given business records from 3 independent data sources with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

Source 1 is the deduplicated reference source. Your task is to find all matching records from Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero, one, or many records from Source 2 and Source 3.

Download Data Set: Click here to Download
File Format
All files in this challenge are tab-separated (.tsv), and your submissions must be tab-separated too. Tabs are used because business addresses and the ID list columns both contain commas. Read them with an explicit tab separator, for example:

import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
Reading a .tsv without sep="\t" will silently produce a single column containing the whole line.

Data Description:
Each source file (*_source1.tsv, *_source2.tsv, *_source3.tsv) has the following columns:

entity_id: Unique identifier for the record. The prefix indicates the source — S1-, S2-, or S3-.
business_name: Name of the business entity (may contain abbreviations, legal suffixes, typos, transliterations)
business_address: Address of the business (may contain partial addresses, format variations, missing components, landmark-based references)
country: Country label for the record. The training data covers US and India. The test set additionally contains a third country, France, that does not appear in the training data. Treat country as an open set of string labels: do not hard-code, filter, or one-hot your pipeline to only {US, India}, and remember that every test entity — France included — must appear in your submission.
There is no separate source column — a record's source is given by its entity_id prefix (S1-/S2-/S3-) and by which file it appears in.

The ground truth file (train_ground_truth.tsv) has two columns:

source1_entity_id: The entity_id of a Source 1 record
matched_entity_ids: Comma-separated list of matching entity_ids from Source 2 and/or Source 3 (empty when the entity has no matches)
Noise Patterns to Expect:

Name variations: Abbreviations (Corp vs. Corporation, Pvt vs. Private, Ltd vs. Limited), legal suffix inconsistencies, DBA/trade names, punctuation differences (& vs. "and"), word-order transpositions, typos
Address variations: Abbreviations (Rd vs. Road, St vs. Street), transliteration variants, missing components (no PIN code, no state), landmark-based references (Near SBI ATM), municipal numbering formats, component reordering
Dataset Details:
Training Dataset: Business records across 3 sources with ground truth matching labels
Test Set: Business records across 3 sources without matching labels
File Descriptions:
Training files

dataset/train/train_source1.tsv: Source 1 training records (the deduplicated reference source)
dataset/train/train_source2.tsv: Source 2 training records
dataset/train/train_source3.tsv: Source 3 training records
dataset/train/train_ground_truth.tsv: Ground truth matching labels for the training set
Test files

dataset/test/test_source1.tsv: Source 1 test records. Generate matches for every entity in this file.
dataset/test/test_source2.tsv: Source 2 test records
dataset/test/test_source3.tsv: Source 3 test records
No ground truth is provided for the test set. To measure your own performance, hold out a validation split from the training data and score it yourself using the F_0.5 formula given below.

Output Format:
Your solution produces two tab-separated files, both placed in the output/ folder of your final submission package (see Final Submission Package below):

matching_results.tsv — your final entity matches. This is the only file scored on the leaderboard — it is what you upload to the Portal during the challenge.
candidate_pairs.tsv — the candidate set your blocking / candidate-generation stage produced, before your final matching model narrowed it down.
matching_results.tsv
Your final entity matches:

Column	Description
source1_entity_id	The entity_id of a Source 1 record
matched_entity_ids	Comma-separated list of matching entity_ids from Source 2 and/or Source 3
Example (columns separated by a single tab, ID lists separated by commas with no quoting):

source1_entity_id	matched_entity_ids
S1-00001	        S2-00047,S2-00193,S3-00812
S1-00002	        S3-00004
S1-00003	
Important:

Every Source 1 entity in the test set must have exactly one row
Leave matched_entity_ids empty for entities with no matches (singletons)
No duplicate entity IDs within a single ID list
ID lists must only contain Source 2 or Source 3 IDs that exist in the test set
candidate_pairs.tsv
The candidate set from your blocking stage — every Source 2 / Source 3 record you considered a plausible match for each Source 1 entity, before your final matching model narrowed it down. This is the exact set of records you feed into your matching model for inference — the final candidate list just before the ML model scores them, not the raw output of an early blocking pass you later filter further. If your pipeline has several blocking/filtering stages, candidate_pairs.tsv is the last one: whatever your model actually runs inference over. Every ID in matching_results.tsv should therefore appear here.

It is not scored on the leaderboard; we use it to analyse blocking quality (recall ceiling, reduction ratio) and to verify your pipeline.

Column	Description
source1_entity_id	The entity_id of a Source 1 record
candidate_entity_ids	Comma-separated list of candidate entity_ids from Source 2 and/or Source 3
Example:

source1_entity_id	candidate_entity_ids
S1-00001	        S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	        S3-00004
S1-00003	
Same rules as matching_results.tsv: one row per Source 1 entity, candidate_entity_ids empty when blocking found no candidates, S2-/S3- IDs only, no duplicates within a list. Your final matches should be a subset of your candidates (a matched ID that never appeared as a candidate signals a pipeline bug — the validator warns about it).

Validate before submitting: a helper script utils/validate_submission.py (stdlib only, no dependencies) checks both files against every rule above so you can catch a rejection locally instead of spending a submission on it. Run it from this student_resource/ directory:

python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
It prints PASS (exit 0) when the files are safe to submit, or a numbered list of issues to fix (exit 1). It only reads your output files and the test source files; it does not compute your score.

Final Submission Package:
In addition to your live leaderboard uploads, every team submits a single zip archive with your code and outputs. We use it to reproduce your results, audit your blocking, and check the fair-play and model-license rules — the top teams' packages are reviewed in detail before the final rankings are confirmed.

Structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file you upload to the leaderboard)
│   └── candidate_pairs.tsv         # your blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all your source code
│       ├── README.md               # how to reproduce end-to-end (data → blocking → matching → output)
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # your methodology write-up (this filled-in template)
output/ — the two TSV files described above: matching_results.tsv and candidate_pairs.tsv.
code/business_entity_resolution/ — a self-contained, runnable copy of your pipeline. Put all source under src/, and include a README.md with exact run instructions plus a requirements.txt (or equivalent environment file) pinning versions. Anyone should be able to regenerate both output files from the training/test data using only what is in this folder.
Methodology document — fill in the provided Documentation_template.md and drop it straight into the zip (the filled-in .md is fine; a .pdf export works too). No need to rename it.
Constraints:
Format your output exactly as described above. Submissions that fail validation will not be evaluated. You should see a SCORED status with your F_0.5 score if the output is correctly formatted.
matched_entity_ids must only reference entities from Source 2 or Source 3. Self-matches to Source 1, and IDs that do not exist in the test set, will be rejected.
Every Source 1 entity must appear in your submission. Missing entities will cause rejection.
Duplicate entity IDs in any ID list will cause rejection, as will duplicate source1_entity_id rows.
Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.
Evaluation Criteria:
Submissions are evaluated using F_β Score (β = 0.5) — a precision-heavy metric that penalizes false merges (matching two different businesses) more than missed matches.

Formula:

F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
Computed as a macro-average: F_0.5 is calculated per Source 1 entity, then averaged across all Source 1 entities in the evaluation set.

Singletons are included in that average. A Source 1 entity with no true matches scores 1.0 when you correctly predict an empty list, and 0.0 when you predict any match for it. Correctly identifying singletons therefore earns credit, and false merges on them are penalised.

Why precision-heavy? In real-world entity resolution, merging two distinct businesses (false positive) is more damaging than missing a link (false negative). F_0.5 weights precision 2× over recall.

Example:

Your model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
Ground truth says S1-00001 matches [S2-00047, S3-00812]
Precision = 2/3, Recall = 2/2 = 1.0
F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714
Leaderboard Information:
Public Leaderboard: During the challenge, rankings will be based on a subset of the test set to provide real-time feedback on your model's performance.
Private Leaderboard: After the challenge ends, the private leaderboard will be revealed, which uses the remaining portion of the test set for evaluation.
Final Rankings: The final decision will be based on the private leaderboard.
You submit predictions for the full test set in both cases; the split is applied during scoring.

Submission Requirements:
Leaderboard (during the challenge): upload matching_results.tsv in the Portal — tab-separated, with the exact column names described above. This is what drives the public and private leaderboards.
Final submission package: submit the single zip described in Final Submission Package above — output/ with both matching_results.tsv (final matches) and candidate_pairs.tsv (your candidate-generation / blocking set fed to the model), code/business_entity_resolution/ (runnable pipeline), and your methodology document. All teams must submit it; the top teams' packages are reviewed before the final rankings are confirmed.
Your methodology document must describe:
Methodology used
Candidate generation/blocking strategy
Model architecture and feature engineering
Any other relevant information about the approach
A template for this documentation is provided in Documentation_template.md. There is no page limit — prioritise clarity and technical depth over brevity.

"cat > /home/vivekananda/Amazon_ML_Challenge_2026/solution/build_notebook.py <<'PY'
\"\"\"Bundle reviewable source into a self-contained cloud notebook.\"\"\"
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/'cloud_upload'
DEST.mkdir(exist_ok=True)
archive=io.BytesIO()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ('src','tests'):
        for p in sorted((ROOT/folder).rglob('*.py')):
            z.write(p,str(p.relative_to(ROOT)))
    z.write(ROOT/'requirements.txt','requirements.txt')
payload=base64.b64encode(archive.getvalue()).decode()
(DEST/'solution_code.zip').write_bytes(archive.getvalue())
cells=[]
def md(text):
    cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(text):
    cells.append(dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=None,outputs=[]))
md('"'''"'# Amazon ML Challenge 2026 — run on SageMaker AI
Run cells in order. This notebook contains the complete source code. Set your team
name and dataset location below. Use a CPU JupyterLab space with about 32 GB RAM
and 50 GB storage as a starting configuration. Measure throughput in the benchmark
cell before committing to the full scans. No leaderboard score has been established.

Training, full validation, full test inference and packaging are separate stages.
The final methodology is generated from measured results only after all stages finish.
'"'''"')
code('"'''"'from pathlib import Path
BASE = Path.cwd() / \"amazon_ml_run\"
TEAM = \"YOUR_TEAM\"  # replace with your registered team name
S3_DATA_URI = \"s3://YOUR_BUCKET/amazon_ml_dataset.zip\"
LOCAL_DATA_ZIP = \"\"  # alternatively, an uploaded zip's local path; takes precedence
DATA_DIRECTORY = \"\"  # alternatively, existing dataset/ containing train/ and test/
THREADS = 8
SAMPLE_ANCHORS = 20000
RANDOM_TARGETS = 100000
ITERATIONS = 800
BASE.mkdir(parents=True, exist_ok=True)
'"'''"')
code(f'"'''"'import base64, io, zipfile, sys, subprocess, os, shutil
SOLUTION = BASE / \"solution\"
SOLUTION.mkdir(exist_ok=True)
SOURCE_ZIP = base64.b64decode({payload"'!r})
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as bundle:
    bundle.extractall(SOLUTION)  # this fixed bundle contains only the supplied solution source
assert (3,10) <= sys.version_info[:2] <= (3,12), "Choose a Python 3.10–3.12 kernel for these pinned dependencies."
ENV = BASE / "env"
PYTHON = ENV / "bin" / "python"
if not PYTHON.exists():
    subprocess.run([sys.executable,"-m","venv",str(ENV)],check=True)
subprocess.run([str(PYTHON),"-m","pip","install","--disable-pip-version-check","-r",str(SOLUTION/"requirements.txt")],check=True)
print("Code:", SOLUTION)
print("Free disk GB:",round(shutil.disk_usage(BASE).free/2**30,1))
'"'"'''"')
code('"'''"'import csv
from urllib.parse import urlparse
if DATA_DIRECTORY:
    DATA = Path(DATA_DIRECTORY).expanduser().resolve()
else:
    if LOCAL_DATA_ZIP:
        dataset_zip = Path(LOCAL_DATA_ZIP).expanduser().resolve()
    else:
        assert \"YOUR_BUCKET\" not in S3_DATA_URI, \"Set S3_DATA_URI in the first cell.\"
        uri = urlparse(S3_DATA_URI)
        assert uri.scheme == \"s3\" and uri.netloc and uri.path.strip(\"/\")
        import boto3  # supplied by the SageMaker image; uses its execution role
        s3 = boto3.client(\"s3\")
        dataset_zip = BASE / \"amazon_ml_dataset.zip\"
        expected_size = s3.head_object(Bucket=uri.netloc,Key=uri.path.lstrip(\"/\"))[\"ContentLength\"]
        if not dataset_zip.exists() or dataset_zip.stat().st_size "'!= expected_size:
            print("Downloading supplied data from S3...")
            s3.download_file(uri.netloc,uri.path.lstrip("/"),str(dataset_zip))
    extracted = BASE / "data"
    extracted.mkdir(exist_ok=True)
    marker = extracted / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(dataset_zip) as z:
            for member in z.infolist():
                destination = (extracted/member.filename).resolve()
                if not destination.is_relative_to(extracted.resolve()):
                    raise ValueError("Unsafe archive path")
            z.extractall(extracted)
        marker.touch()
    sources = [p for p in extracted.rglob("train_source1.tsv") if p.parent.name=="train"]
    assert len(sources)==1, "Expected exactly one supplied dataset."
    DATA = sources[0].parent.parent
required = [DATA/"train"/f"train_{name}.tsv" for name in ("source1","source2","source3","ground_truth")]
required += [DATA/"test"/f"test_source{k}.tsv" for k in (1,2,3)]
assert all(p.is_file() for p in required), "Dataset is incomplete."
WORK = BASE / "artifacts"
WORK.mkdir(exist_ok=True)
print("Dataset:",DATA)
print("Checkpoint directory:",WORK)
'"'"'''"')
code('"'''"'ENVIRONMENT = dict(os.environ, PYTHONPATH=str(SOLUTION/\"src\"))
def run_process(command):
    process = subprocess.Popen(command,env=ENVIRONMENT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:
            print(line,end=\"\",flush=True)
        if process.wait():
            raise RuntimeError(\"Command failed. Read the output above before continuing.\")
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

def run(stage,*extra):
    run_process([str(PYTHON),\"-u\",\"-m\",\"ber\",\"--data\",str(DATA),\"--work\",str(WORK),\"--threads\",str(THREADS),*map(str,extra),stage])

run_process([str(PYTHON),\"-m\",\"unittest\",\"discover\",\"-s\",str(SOLUTION/\"tests\"),\"-v\"])
'"'''"')
md('"'''"'Train the pair classifier. This builds the full training anchor index, samples
complete labeled business groups plus additional records, and mines hard negatives.
It prints real progress. The index is reused on later runs.'"'''"')
code('"'''"'run(\"train\",\"--sample-anchors\",SAMPLE_ANCHORS,\"--random-targets\",RANDOM_TARGETS,\"--iterations\",ITERATIONS)
'"'''"')
md('"'''"'Benchmark the full-catalog validation scan. It saves a resumable checkpoint,
but deliberately does not write a final calibration or submission from partial data.
Review targets/second: the training catalog has about 10.32 million target records
and the test catalog has about 9.97 million. Retrieval is currently single-process;
more CPU threads mainly help classifier training/scoring.'"'''"')
code('"'''"'run(\"calibrate\",\"--max-targets\",2000)
'"'''"')
md('"'''"'Continue full validation. This tunes the exact macro F0.5 metric with singletons.
Re-run the same cell after a process interruption to resume. Keep the same data,
source code and model. Model/code changes require a new score database/work directory.'"'''"')
code('"'''"'run(\"calibrate\")
import json
validation = json.loads((WORK/\"calibration.json\").read_text())
print({k:v for k,v in validation.items() if k"'!="grid"})
'"'"'''"')
md('"'''"'Run test inference. Every test Source 1 entity, including France and predicted
singletons, receives an output row. Every candidate scored by the model is recorded.
No output is exported until the full test scan completes.'"'''"')
code('"'''"'run(\"predict\")
'"'''"')
md('"'''"'Create the final ZIP. The methodology is written here from the actual run's
reports; there are no invented experiment results. Set TEAM in the first cell first.'"'''"')
code('"'''"'assert TEAM and TEAM.upper() not in {\"YOUR_TEAM\",\"TEAM_NAME\"}, \"Set your registered team name.\"
run(\"package\",\"--team\",TEAM)
from IPython.display import FileLink, display
leaderboard = WORK/\"output\"/\"matching_results.tsv\"
packages = list(WORK.glob(\"*_submission.zip\"))
print(\"LEADERBOARD UPLOAD: matching_results.tsv\")
display(FileLink(str(leaderboard.relative_to(Path.cwd()))))
print(\"FINAL PACKAGE: your team submission ZIP\")
for package in packages:
    display(FileLink(str(package.relative_to(Path.cwd()))))
'"'''"')
md('"'''"'Download the two indicated artifacts. Stop the SageMaker JupyterLab space when
you finish using compute; persistent EBS storage and S3 can still incur charges.
An AWS Builder Center profile is your challenge-registration identity, not a promise
of free SageMaker compute. Confirm your credits and applicable rates in AWS Billing.
'"'''"')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
for i,c in enumerate(cells):
    c['id']=f'cell-{i:02d}'
path=DEST/'Amazon_ML_SageMaker.ipynb'
path.write_text(json.dumps(notebook,indent=1))
print(path)
print('Code bundle:',DEST/'solution_code.zip')
PY
python3 /home/vivekananda/Amazon_ML_Challenge_2026/solution/build_notebook.py"'


python3 - <<'PY'
from pathlib import Path
root=Path('/home/vivekananda/Amazon_ML_Challenge_2026/solution/src/ber')
p=root/'train.py'
s=p.read_text()
s=s.replace("note='Pair validation", "settings=dict(sample_anchors=args.sample_anchors,random_targets=args.random_targets,iterations=args.iterations,seed=args.seed,candidates=args.candidates,retrieval_limit=args.retrieval_limit,max_postings=args.max_postings,threads=args.threads),\n                  note='Pair validation")
p.write_text(s)
p=root/'inference.py'
s=p.read_text().replace("code=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()", "code=hashlib.sha256(Path(core.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()")
s=s.replace("threshold=threshold,margin=margin,full_test=True,", "threshold=threshold,margin=margin,full_test=True,model_sha256=calibration['model_sha256'],")
p.write_text(s)
p=root/'package.py'
s=p.read_text().replace('import json\n','import json\nimport hashlib\n')
s=s.replace("team = re.sub", "current_hash = hashlib.sha256((work/'model.cbm').read_bytes()).hexdigest()\n    if calibration['model_sha256'] != current_hash or prediction['model_sha256'] != current_hash:\n        raise ValueError('Model, calibration and predictions must belong to the same run.')\n    team = re.sub")
p.write_text(s)
p=root/'core.py'
s=p.read_text().replace("self.connections, self.frequencies = {}, {}", "self.connections, self.frequencies = {}, {}\n        self.countries = set(json.loads((self.directory/'complete.json').read_text())['countries'])")
s=s.replace("if country in self.connections:", "if country not in self.countries:\n            return None\n        if country in self.connections:")
p.write_text(s)
for path in root.glob('*.py'):
 compile(path.read_text(),str(path),'exec')
 print('Syntax OK:',path.name)
PY


| Finding | Evidence |
|---|---|
| Submission deadline **27 September 2026, 11:59 PM IST**; opening September 25 | `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`, p. 1; current official event page agrees |
| Maximum **5 submissions per team per day**, over three days; maintain version history | Same guidelines, p. 1 |
| Candidate-generation efficiency affects final ranking beyond leaderboards; avoid all-pairs comparison | `6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf`, p. 1; `PLAN.md:1–3` |
| `candidate_pairs.tsv` means the actual candidate set scored by the final matching model; final matches are its subset | Emails PDF, p. 4; `PLAN.md:75` |
| Both TSV outputs, complete runnable code, pinned environment and methodology belong in final ZIP | Emails PDF, p. 5; README submission-package section |
| Model must have MIT/Apache-2.0 license and at most 8B parameters | Emails PDF, p. 6; README constraints |
| No business-identity database/API lookup, geocoding API or Internet data augmentation | Emails PDF, pp. 7–8; README fair-play section |
| Simultaneous portal logins prohibited per participant | Guidelines PDF, p. 2 |

| Paper | What I inspected | Useful idea and limitation here |
|---|---|---|
| **[Deep Entity Matching with Pre-Trained Language Models — Ditto](https://arxiv.org/pdf/2004.00584)**, Li et al., 2020 | Full-text sections 2–3: record serialization, pair classification, domain features; discussion and experimental appendix | Serialize named fields and train a Transformer to classify candidate pairs. The paper specifically highlights street numbers as valuable organization-matching evidence. **Application:** rerank retrieved candidates with name/address jointly, and retain explicit numeric-agreement features. Original tested LMs are English; its reported F1 is not evidence of transfer to French or macro F0.5. |
| **[Deep Learning for Blocking in Entity Matching: A Design Space Exploration — DeepBlocker](https://nantang.github.io/research/pubs/deepblocker.pdf)**, Thirumuruganathan et al., 2021 | Full-text framework, sections 3.5–3.8 and 5.1/5.4, selected comparison tables | Learn record embeddings and retrieve nearest neighbors. The study evaluates **recall against candidate-set size**, and shows complementarity between learned and traditional blocking. **Application:** measure a union of lexical, structural and dense retrieval routes; keep only routes that improve the useful trade-off. Synthetic examples and random negatives are methodological options, not necessary prerequisites. The original architectures and findings do not establish that dense retrieval will beat lexical retrieval here. |
| **[Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation](https://arxiv.org/html/2207.04122v2)**, Wang et al., 2022 preprint | Full-text sections II–IV: contrastive representations, pseudo-labeling, cutoff augmentation, clustered negatives; selected VI experiments | Train embeddings to bring equivalent records together and separate confusable records; clustered negatives are harder than arbitrary pairs. **Application:** use supplied gold-linked positives and near-miss negatives retrieved from supplied training records. Local perturbations are optional; dropping the wrong address number can destroy identity information. Pseudo-labels can reinforce mistakes, and unlabeled duplicates create false negatives, so mask known equivalent records. Reported experiments use different label budgets, domains and metrics. |
| **[Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)**, Paulsen et al., 2023 | Full-text sections 3.2–3.4, selected recall/output-size comparisons, limitations | Top-k lexical retrieval offers controllable candidate counts; its experiments show that strong TF-IDF/BM25 blocking is a serious competitor to neural blocking. **Application:** establish word and character n-gram retrieval before expensive neural training, retaining rare business/address tokens. Important limitation explicitly discussed: fixed k can omit matches when an entity has more than k counterparts. Sparkly’s distributed software is optional; the retrieval principle is the useful starting point. |
| **[Unsupervised Cross-lingual Representation Learning at Scale — XLM-R](https://aclanthology.org/2020.acl-main.747.pdf)**, Conneau et al., 2020 | Full-text model/data section, language coverage and evaluation description | Multilingual pretraining supports cross-language transfer; French is covered. **Application:** motivates testing a multilingual encoder/backbone against English-only choices. It is **not an entity-resolution paper**: its NLI/NER/QA results do not prove French business-identity matching accuracy. Use it as architectural evidence, then validate the actual task. Root is checking specific model/checkpoint licenses separately. |
| **[Collective Entity Resolution in Relational Data](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf)**, Bhattacharya and Getoor, 2007 | Full-text relational similarity/collective clustering methods, selected experimental discussion including transitive closure | Joint reasoning can use relationships to disambiguate similar names; naive relationship similarity can mislead. **Application:** investigate consistency among high-confidence S1/S2/S3 matches only after a strong pairwise baseline. This paper uses genuine relational information such as coauthor links; shared address text is not equivalent evidence. Its experimental discussion also shows transitive closure has dataset-dependent effects. Do not use unrestricted connected components: a wrong bridge can join unrelated businesses. |


| Paper | What I inspected | Useful idea and limitation here |
|---|---|---|
| **[Deep Entity Matching with Pre-Trained Language Models — Ditto](https://arxiv.org/pdf/2004.00584)**, Li et al., 2020 | Full-text sections 2–3: record serialization, pair classification, domain features; discussion and experimental appendix | Serialize named fields and train a Transformer to classify candidate pairs. The paper specifically highlights street numbers as valuable organization-matching evidence. **Application:** rerank retrieved candidates with name/address jointly, and retain explicit numeric-agreement features. Original tested LMs are English; its reported F1 is not evidence of transfer to French or macro F0.5. |
| **[Deep Learning for Blocking in Entity Matching: A Design Space Exploration — DeepBlocker](https://nantang.github.io/research/pubs/deepblocker.pdf)**, Thirumuruganathan et al., 2021 | Full-text framework, sections 3.5–3.8 and 5.1/5.4, selected comparison tables | Learn record embeddings and retrieve nearest neighbors. The study evaluates **recall against candidate-set size**, and shows complementarity between learned and traditional blocking. **Application:** measure a union of lexical, structural and dense retrieval routes; keep only routes that improve the useful trade-off. Synthetic examples and random negatives are methodological options, not necessary prerequisites. The original architectures and findings do not establish that dense retrieval will beat lexical retrieval here. |
| **[Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation](https://arxiv.org/html/2207.04122v2)**, Wang et al., 2022 preprint | Full-text sections II–IV: contrastive representations, pseudo-labeling, cutoff augmentation, clustered negatives; selected VI experiments | Train embeddings to bring equivalent records together and separate confusable records; clustered negatives are harder than arbitrary pairs. **Application:** use supplied gold-linked positives and near-miss negatives retrieved from supplied training records. Local perturbations are optional; dropping the wrong address number can destroy identity information. Pseudo-labels can reinforce mistakes, and unlabeled duplicates create false negatives, so mask known equivalent records. Reported experiments use different label budgets, domains and metrics. |
| **[Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)**, Paulsen et al., 2023 | Full-text sections 3.2–3.4, selected recall/output-size comparisons, limitations | Top-k lexical retrieval offers controllable candidate counts; its experiments show that strong TF-IDF/BM25 blocking is a serious competitor to neural blocking. **Application:** establish word and character n-gram retrieval before expensive neural training, retaining rare business/address tokens. Important limitation explicitly discussed: fixed k can omit matches when an entity has more than k counterparts. Sparkly’s distributed software is optional; the retrieval principle is the useful starting point. |
| **[Unsupervised Cross-lingual Representation Learning at Scale — XLM-R](https://aclanthology.org/2020.acl-main.747.pdf)**, Conneau et al., 2020 | Full-text model/data section, language coverage and evaluation description | Multilingual pretraining supports cross-language transfer; French is covered. **Application:** motivates testing a multilingual encoder/backbone against English-only choices. It is **not an entity-resolution paper**: its NLI/NER/QA results do not prove French business-identity matching accuracy. Use it as architectural evidence, then validate the actual task. Root is checking specific model/checkpoint licenses separately. |
| **[Collective Entity Resolution in Relational Data](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf)**, Bhattacharya and Getoor, 2007 | Full-text relational similarity/collective clustering methods, selected experimental discussion including transitive closure | Joint reasoning can use relationships to disambiguate similar names; naive relationship similarity can mislead. **Application:** investigate consistency among high-confidence S1/S2/S3 matches only after a strong pairwise baseline. This paper uses genuine relational information such as coauthor links; shared address text is not equivalent evidence. Its experimental discussion also shows transitive closure has dataset-dependent effects. Do not use unrestricted connected components: a wrong bridge can join unrelated businesses. |