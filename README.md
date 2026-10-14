# Amazon ML Challenge 2026 — Business Entity Resolution

> **Team:** Mad SVMs &nbsp;•&nbsp; **Task:** resolve noisy business records across 3 independent sources &nbsp;•&nbsp; **Metric:** macro F<sub>0.5</sub> &nbsp;•&nbsp; **Engine:** disk-backed FTS5 blocking + CatBoost reranker

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white">
  <img alt="CatBoost" src="https://img.shields.io/badge/Model-CatBoost%201.2.8-FFCC00">
  <img alt="SQLite" src="https://img.shields.io/badge/Blocking-SQLite%20FTS5-003B57?logo=sqlite&logoColor=white">
  <img alt="RapidFuzz" src="https://img.shields.io/badge/Similarity-RapidFuzz%203.14-8A2BE2">
  <img alt="License" src="https://img.shields.io/badge/License-MIT%2FApache--2.0-brightgreen">
  <img alt="External data" src="https://img.shields.io/badge/External%20lookup-NONE-critical">
</p>

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [The problem](#2-the-problem)
3. [Results at a glance](#3-results-at-a-glance)
4. [System architecture](#4-system-architecture)
5. [Training pipeline](#5-training-pipeline)
6. [Candidate generation (blocking)](#6-candidate-generation-blocking)
7. [Feature engineering](#7-feature-engineering)
8. [The matching model](#8-the-matching-model)
9. [Calibration & thresholding](#9-calibration--thresholding)
10. [Inference & packaging](#10-inference--packaging)
11. [Correctness, safety & resume guarantees](#11-correctness-safety--resume-guarantees)
12. [Repository layout](#12-repository-layout)
13. [Reproducing everything](#13-reproducing-everything)
14. [Validation & testing](#14-validation--testing)
15. [Design decisions & hard-won lessons](#15-design-decisions--hard-won-lessons)
16. [Fair play, licensing & constraints](#16-fair-play-licensing--constraints)
17. [References](#17-references)

---

## 1. TL;DR

We built a **two-stage entity-resolution system** that finds, for every Source 1
reference business, the matching records in Source 2 and Source 3.

- **Stage 1 — Blocking.** Every Source 1 record is written into a **per-country,
  disk-backed SQLite FTS5 index**. Each target (S2/S3) record fires **three
  independent retrieval routes** (name tokens, name character-trigrams, address
  tokens), merged into a tiny candidate pool. No all-pairs comparison. Ever.
- **Stage 2 — Matching.** A **CatBoost pair classifier** (33 handcrafted
  string/numeric-agreement features) reranks the candidates. A target is assigned
  to its single best Source 1 anchor only if it clears a **probability threshold**
  *and* a **competitor-margin threshold** — a precision-first rule that directly
  mirrors F<sub>0.5</sub>.

Everything is **CPU-only**, **resumable**, **checkpointed**, **hash-guarded**, and
trained **only on the supplied data**. No external entity databases, no geocoders,
no internet augmentation.

---

## 2. The problem

Business identity data arrives from independent providers as **partial, noisy
fragments** with no shared key. The same real-world company can appear as
`ACME Corp., 12 Lake Rd.` in one source and `Acme Corporation, 12 Lake Road` in
another — or as a DBA name with a completely different address grammar.

| Source | Role |
|---|---|
| **Source 1 (S1)** | Deduplicated **reference** catalog. The anchor set. |
| **Source 2 (S2)** | Noisy records to be resolved *into* S1. |
| **Source 3 (S3)** | Second noisy source, also resolved into S1. |

For each S1 anchor we must output **zero, one, or many** matching S2/S3 records.

**Noise we had to survive**

- Name: legal suffixes (`Corp`/`Corporation`, `Pvt`/`Private`), `&` vs `and`,
  transliteration (`École`→`Ecole`), typos, word order, DBA/trade names.
- Address: abbreviations (`Rd`/`Road`, `St`/`Street`), missing components,
  landmark references (`Near SBI ATM`), municipal number formats, reordering.
- **Open-set countries.** Train = `US`, `India`. Test *also* contains `France`
  with **zero training labels**. Any pipeline that one-hot-encodes country or
  filters to known labels silently fails.
- **Singletons are scored.** Predicting an empty list for a truly-unmatched S1
  entity earns a full **1.0**; a false merge earns **0.0**.

**Metric** — precision-heavy macro **F<sub>0.5</sub>**, averaged *per S1 entity*:

```
F_0.5 = (1.25 · P · R) / (0.25 · P + R)
```

Precision is weighted **2×**. Merging two distinct businesses is far more costly
than missing a link, so the entire design leans conservative.

---

## 3. Results at a glance

Full-catalog run (model `ceb636bb…`), as recorded in the shipped JSON reports:

| Metric | Value |
|---|---|
| **Validation macro F<sub>0.5</sub>** (threshold-tuned) | **0.881305** |
| Validation precision | 0.961125 |
| Validation recall | 0.811013 |
| Blocking recall (validation) | 0.886180 |
| Validation anchors | 441,368 |
| Validation singletons | 24,742 |
| Candidates / anchor (validation) | 11.39 |
| **Test anchors** (every S1 entity emitted) | **1,732,544** |
| **Test candidate pairs** scored | **25,661,438** |
| Candidates / test anchor | 14.81 |
| **Accepted test matches** | **5,252,862** |
| Predicted singletons | 152,318 |
| Acceptance threshold / margin | 0.75 / 0.05 |

Classifier-training snapshot (from `training_report.json`):

| Metric | Value |
|---|---|
| Training pairs / validation pairs | 277,889 / 34,881 |
| Training positives / validation positives | 109,602 / 24,401 |
| Sampled blocking recall | 0.882810 |
| Gold positives *injected* into training only | 12,930 |
| Trees | 800 |
| Wall-clock training | ~1,108 s |

> **Honesty note.** The threshold was tuned on validation, so 0.8813 is a
> threshold-tuned local estimate, **not** an unbiased held-out/leaderboard score.
> France has no labeled evaluation data. These numbers describe one measured
> baseline — not a claim of optimality.

---

## 4. System architecture

```
                       ┌──────────────────────────────────────────────┐
   provided_dataset    │  Source 1  (train OR test)                    │
   train/*.tsv  ──────▶│   S1 anchors ──▶ normalize ──▶ per-country    │
   test/*.tsv          │                    FTS5 index (SQLite)       │
                       └──────────────────────────────────────────────┘
                                          │  index_train / index_test
                                          ▼
   ┌──────────────┐   candidates    ┌────────────────────┐
   │  Retriever   │◀────────────────│  3 retrieval routes │
   │ (per country)│  name·trigram·  │  bm25 fusion        │
   └──────┬───────┘   address       └────────────────────┘
          │  ≤ N candidates/target
          ▼
   ┌──────────────┐   33 features   ┌────────────────────┐
   │   features   │────────────────▶│  CatBoost reranker  │
   └──────────────┘                 └─────────┬──────────┘
                                              │ p(anchor == owner)
                                              ▼
                              ┌───────────────────────────────┐
                              │ best-candidate + margin gate  │
                              │ p ≥ threshold ∧ gap ≥ margin  │
                              └───────────────┬───────────────┘
                                              ▼
                     output/matching_results.tsv   (leaderboard file)
                     output/candidate_pairs.tsv   (blocking audit file)
                                              ▼
                                    <team>_submission.zip
```

**Pipeline stages** (`python -m ber <stage>`):

```
train ──▶ calibrate ──▶ predict ──▶ package
  │            │            │           │
  │            │            │           └─ integrity-checked ZIP (code+output+docs)
  │            │            └─ full test scan → matching_results + candidate_pairs
  │            └─ full train-catalog scan → tune macro F0.5 → threshold + margin
  └─ build index, sample gold groups, mine hard negatives, fit CatBoost
```

---

## 5. Training pipeline

Training is **sampled, entity-disjoint and leakage-audited**. `ber/train.py`:

### 5.1 Build the anchor index
`build_indexes(data/train/train_source1.tsv, work/index_train)` shards Source 1
by **normalized country** into one SQLite FTS5 file per country. A `complete.json`
marker stores a `(size, mtime, version)` fingerprint so an interrupted or repeated
run **reuses** the finished index instead of rebuilding.

### 5.2 Sample complete entity groups + hard negatives
`sample_records(...)`:
- Draws `--sample-anchors` (default 20,000) complete **labelled groups** using a
  stable hash, so every gold positive of a selected anchor is retained.
- Draws `--random-targets` (default 100,000) **random** S2/S3 records as extra
  negatives — but resolves each against the gold file so a random record that
  *is* a true match is labelled positive, never a false negative.
- Enforces **reverse-assignment**: a target belonging to two different S1 anchors
  raises an error rather than corrupting labels.
- Writes `sample.tsv`, `sample_owners.json`, and a fingerprinted
  `sample_complete.json` so sampling is reproducible and skippable.

### 5.3 Mine candidates + engineer pairs
For each sampled target, the `Retriever` returns its candidate anchors. Pairs are
turned into 33-dimensional feature vectors. **Gold positives missed by blocking**
are *injected* into the candidate list for **training rows only** — this teaches
the classifier what a missed true match looks like **without ever inflating
blocking-recall or validation**.

### 5.4 Entity-disjoint validation split
A deterministic `fold(entity_id, seed)` assigns **20% of gold business groups** to
validation. Critically, *both endpoints* of a pair must share the partition, so no
business identity leaks between train and validation. IDs are **never** features.

### 5.5 Fit the classifier
```python
CatBoostClassifier(iterations=800, depth=7, learning_rate=0.06,
                   loss_function='Logloss', eval_metric='AUC',
                   l2_leaf_reg=5, random_seed=42, thread_count=8)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=80)
```
Model, config, feature names and a `training_report.json` are persisted. Training
refuses to proceed unless **both partitions contain both labels**.

> **Why sampling?** The full training catalog holds millions of anchors. Sampling
> complete groups keeps gold structure intact while making iteration cheap, and
> the trained model is later used against the **whole** catalog during calibration
> and inference.

---

## 6. Candidate generation (blocking)

Blocking sets the **recall ceiling** — and Amazon explicitly ranks final
submissions partly on candidate-set size. Our blocker is intentionally small and
precise.

**Index.** Per country: an FTS5 table (`unicode61 remove_diacritics 2`) over three
columns — `n` (normalized *core* name, legal suffixes stripped), `g` (sorted
character trigrams of the core name), `a` (normalized address). A `fts5vocab`
table supplies document frequencies.

**Three retrieval routes per target** (`Retriever.retrieve`):

| Route | Column | Weight in `bm25` | Ranked by |
|---|---|---|---|
| Name tokens | `n` | 3.0 | frequency-rare terms first |
| Name trigrams | `g` | 1.0 | fuzzy/typo robustness |
| Address tokens | `a` | 2.0 | landmark/number signal |

- **Rare-term selection.** Only the `count` rarest tokens per route are queried —
  common tokens add noise and cost, not signal.
- **Frequency pruning.** Terms with a document frequency above `--max-postings`
  (default 3,000) are dropped from the vocabulary at load time.
- **bm25 fusion**, then a **RapidFuzz lexical rescore**
  (`0.82·name + 0.18·address`, plus the mirror `0.80·address + 0.20·name`) keeps the
  top `--candidates` (default 3) above a dynamic `--lexical-floor` (0.32) and a
  relative cutoff (`best − 0.16`).
- **Reverse retrieval.** A *target* queries *anchors*, so any number of records
  may map to one anchor — one-to-many is native, not a special case.
- **Open-set safe.** A target whose normalized country has no index yet simply
  returns no candidates; unknown/French/unseen countries never crash the pipeline.

Every pair the model actually scores is written to **`candidate_pairs.tsv`**, so
the blocking audit file is a faithful record — never reconstructed after the fact.

---

## 7. Feature engineering

33 interpretable features, computed on normalized core name + address
(`core.FEATURE_NAMES`). No embeddings, no learned tokens — fast, explainable and
dependency-light.

| Family | Features |
|---|---|
| **Name similarity** | `ratio`, `token_sort`, `token_set`, `WRatio`, `compact_ratio` |
| **Name token overlap** | `token_jaccard`, `token_overlap`, `trigram_jaccard` |
| **Name flags** | `equal`, `length_ratio` |
| **Address similarity** | `ratio`, `token_sort`, `token_set`, `WRatio` |
| **Address overlap** | `token_jaccard`, `token_overlap` |
| **Address flags** | `equal`, `length_ratio`, `missing` |
| **Numbers** | `number_jaccard`, `number_overlap`, `number_conflict` |
| **Postcodes** | `postcode_overlap`, `postcode_conflict` |
| **First house number** | `first_number_equal`, `first_number_conflict` |
| **Interaction / shape** | `both_fields_agree`, `name_address_min`, word-counts (4), `source3` |

Normalization (`core.normalize`) folds Unicode (NFKD + strip combining), strips
URLs/TLDs, maps `&`→`and`, applies address abbreviation maps (`street`→`st`, …),
and distills a **core name** by removing legal suffixes and stopwords. This is what
makes `Ltd`≈`Limited` and `École`≈`Ecole` collapse to the same tokens.

---

## 8. The matching model

| | |
|---|---|
| **Algorithm** | Gradient-boosted decision trees (CatBoost) |
| **Task** | Binary: does this (anchor, target) pair share an owner? |
| **Input** | 33 features per candidate pair |
| **Trees** | 800 (early stopping, wait=80) |
| **Depth** | 7 |
| **Learning rate** | 0.06 |
| **Loss / metric** | Logloss / AUC |
| **Regularization** | `l2_leaf_reg=5` |
| **License** | Apache-2.0 (CatBoost) — well under the 8B-parameter cap |

CatBoost was chosen for strong tabular performance, native CPU speed, tiny
artifacts (~1.7 MB), and a permissive license. The model reranks blocking output;
it never sees the whole catalog at once.

**Decision rule at inference** — a target is accepted for its best anchor only if:

```
p_best ≥ threshold   AND   (p_best − p_runner_up) ≥ margin
```

The **margin** term suppresses ambiguous anchors whose competitors score almost
as high — a cheap, effective precision guard that aligns with F<sub>0.5</sub>.

---

## 9. Calibration & thresholding

`ber/inference.py::calibrate` **scans the entire training catalog** (not just the
sampled pool) with gold labels, then grid-searches:

- **thresholds:** `.40 .50 .60 .65 .70 .75 .80 .825 .85 .875 .90 .925 .95 .965 .975 .985 .995`
- **margins:** `0 .025 .05`

For every `(threshold, margin)` it reconstructs the exact competition metric —
**macro F<sub>0.5</sub> with singletons** — from the persisted per-pair
`(correct, selected, probability, margin)` records, and selects the best. The
chosen grid point becomes `calibration.json`, which records precision, recall,
blocking recall, candidate counts, and the **model SHA-256**. `predict` refuses to
run if the model hash changed after calibration.

> Full-catalog scan is heavy. `--max-targets N` runs a **checkpoint-only benchmark**
> that deliberately refuses to emit a calibration; resume without the flag to
> finish. (Measured ~78 targets/s single-process on the cloud box → the full scan
> is multi-hour by design, which is why checkpoints and resume exist.)

---

## 10. Inference & packaging

`predict` scans **all** test targets and streams `matching_results.tsv` and
`candidate_pairs.tsv`, using a `LEFT JOIN` over the anchor table so **every S1
entity gets exactly one row** — including France and predicted singletons (empty
match list). `validate_outputs` asserts: one row per anchor, correct headers,
matches ⊆ candidates, only `S2-`/`S3-` prefixes, no duplicates.

`package` builds `<team>_submission.zip` containing the two TSVs, the full
runnable `code/business_entity_resolution/` tree, pinned `requirements.txt`, a
generated `README.md`, and the filled `Documentation_template.md` — and **refuses
to package** unless model, calibration, predictions and output hashes all agree.

---

## 11. Correctness, safety & resume guarantees

This is where most of the engineering effort lives. The pipeline is built to be
**interruptible and tamper-evident**:

- **Resumable scans.** Progress is committed to a SQLite score DB every batch
  (`batch_size`, default 1024) with per-source `:rows` / `:done` markers. Kill it
  at any time; resume continues from the committed contiguous prefix.
- **No partial submissions.** A scan that stops early saves a checkpoint but
  **cannot** export or package — `package` checks `full_catalog` / `full_test`.
- **Signature guard.** The score DB stores a JSON signature of
  `{model hash, code hash, config, data fingerprint, split, ground-truth}`. Any
  mismatch aborts unless it is a *verified legacy v1* checkpoint (an explicit,
  audited allow-list of historical code hashes) that is migrated forward.
- **Scoring-code hash.** Instead of hashing brittle `ast.dump()` output, the guard
  hashes stable **source segments** of `_prepare_with_retriever`, `flush`,
  `load_model`, `load_truth`, plus `core.py` — so worker count / scheduling changes
  don't invalidate valid checkpoints, but real scoring changes do.
- **Parallel workers.** Optional `--workers` uses a **spawn** pool (no inherited
  SQLite handles, `OMP/MKL/OPENBLAS` pinned to 1 thread each). Results are consumed
  **in submission order** so committed row counts always describe a contiguous
  prefix.
- **Signal handling.** Workers ignore SIGINT; the parent owns Ctrl-C and tears the
  pool down cleanly.
- **Tamper detection.** Tests actively prove that appended output rows, a changed
  threshold, and a mutated model are all **rejected** at packaging time.

---

## 12. Repository layout

```
Amazon_ML_Challenge_2026/
├── README.md                     ← you are here
├── PLAN.md                       ← challenge brief + research notes
├── solution/
│   ├── requirements.txt          ← numpy, catboost, rapidfuzz (pinned)
│   ├── src/ber/
│   │   ├── core.py               ← normalize, FTS5 index, Retriever, features, metric
│   │   ├── train.py              ← sampling + pair mining + CatBoost fit
│   │   ├── inference.py          ← resumable scan, calibrate, predict
│   │   ├── package.py            ← integrity-checked submission ZIP
│   │   └── __main__.py           ← CLI: train | calibrate | predict | package | all
│   ├── tests/
│   │   ├── test_core.py          ← normalization, metric, indexing, one-to-many
│   │   ├── test_pipeline.py      ← real end-to-end, resume, packaging, ZIP repro
│   │   ├── fixture_data.py       ← deterministic synthetic dataset
│   │   ├── prepare_real_subset.py← reduced real-catalog validation fixture
│   │   └── benchmark_full_index.py← full-index blocking recall/throughput
│   ├── build_notebook.py         ← generates the SageMaker notebook
│   ├── build_kaggle_notebook.py  ← generates the Kaggle notebook
│   └── validation/               ← diagnostic runs (synthetic + real subset)
├── cloud_upload/                 ← Kaggle / SageMaker notebooks + code bundle
├── artifacts/                    ← trained model, reports, outputs (git-ignored)
├── 6ab10eb3b23ba_student_resource/ ← official problem statement + validator
└── *.pdf                         ← official guidelines & comms
```

---

## 13. Reproducing everything

**Requirements:** Python 3.10–3.12, SQLite with **FTS5**, ~20 GB free disk.
CPU only. No GPU.

```bash
python -m pip install -r solution/requirements.txt
export PYTHONPATH=solution/src
DATA=/path/to/dataset      # contains train/ and test/ with the supplied TSVs
WORK=solution/artifacts

# 1) train the pair classifier
python -m ber --data "$DATA" --work "$WORK" train

# 2) full-catalog calibration → threshold + margin  (hours, resumable)
python -m ber --data "$DATA" --work "$WORK" calibrate

# 3) full test inference → the two output TSVs       (hours, resumable)
python -m ber --data "$DATA" --work "$WORK" predict

# 4) build the integrity-checked submission ZIP
python -m ber --data "$DATA" --work "$WORK" --team "Mad SVMs" package
```

Benchmark any scan first without committing to a full run:

```bash
python -m ber --data "$DATA" --work "$WORK" --max-targets 2000 calibrate
```

Run the tests:

```bash
cd solution
PYTHONPATH=src python -m unittest discover -s tests -v
```

Primary run settings: `seed=42, sample_anchors=20000, random_targets=100000,
iterations=800, candidates=3, retrieval_limit=12, max_postings=3000, threads=8`.

---

## 14. Validation & testing

The test suite is **not** a toy — it runs the real pipeline on real
CatBoost + SQLite and verifies hard guarantees:

| Test | Guarantees |
|---|---|
| `test_official_validator_and_country_coverage` | Passes the **official** validator; France anchors are present *and* evaluated; one-to-many survives export |
| `test_calibration_metric_matches_independent_set_formula` | Reproduced macro F<sub>0.5</sub> matches an **independent set-based** computation to 12 dp |
| `test_calibration_resume_does_not_publish_partial_results` | A `--max-targets` run writes **no** calibration; a resumed run matches a fresh run exactly |
| `test_prediction_resume_and_inference_audit` | Resumed predictions are **byte-identical**; `candidate_pairs.tsv` equals the actual scored pairs |
| `test_zip_reproduces_identical_outputs` | The packaged ZIP retrained/predicted from its own bundled code reproduces **byte-identical** outputs |
| `test_packaging_rejects_modified_outputs` | Appended output rows are **rejected** |
| `test_packaging_rejects_changed_threshold` | A tampered threshold is **rejected** |
| `test_stale_model_is_rejected` | A mutated model invalidates a stale calibration |

Two extra diagnostic runs ship under `solution/validation/`:

- **Synthetic fixture** — full pipeline on deterministic data (macro F<sub>0.5</sub> ≈ 0.9928).
- **Real-subset** — 500 businesses/country reservoir-sampled from supplied data
  (macro F<sub>0.5</sub> ≈ 0.9688, precision 0.996, blocking recall 0.971). Labelled
  diagnostics only; **not** competition-scale estimates.

---

## 15. Design decisions & hard-won lessons

- **Precision first, because the metric is.** The margin gate and high default
  threshold exist because F<sub>0.5</sub> punishes false merges ~2× harder.
- **Injecting gold positives, carefully.** Blocking misses shouldn't become
  training blind spots — so missed positives are injected into *training candidates
  only*, never into validation or the blocking-recall numerator.
- **Entity-disjoint or it doesn't count.** Splitting pairs at random leaks identity.
  We split **businesses**, and require both endpoints on the same side.
- **Open-set country, by construction.** Country is a normalized *key into an index*,
  not a one-hot feature. France works with zero labels.
- **IDs are never features.** `stable_hash(entity_id)` is only used for *partitioning*.
- **Checkpoints are a feature, not an afterthought.** Multi-hour scans are normal;
  every stage resumes, and the signature guard prevents mixing incompatible runs.
- **The audit file is real.** `candidate_pairs.tsv` is streamed from the exact
  scored set, so the blocking audit can't drift from inference.

---

## 16. Fair play, licensing & constraints

- **No external lookup.** No commercial ER APIs, government registries, geocoders,
  or internet augmentation. Only the supplied data + offline normalization. This
  is a hard challenge rule and a design invariant.
- **Permissive stack.** CatBoost (Apache-2.0), NumPy (BSD), RapidFuzz (MIT),
  SQLite (public domain). Well within the MIT/Apache-2.0 + ≤8B-parameter rule.
- **Pinned environment.** `numpy==1.26.4`, `catboost==1.2.8`, `rapidfuzz==3.14.1`.
- **Auditable output.** Matches ⊆ candidates, valid prefixes, one row per anchor,
  no duplicates — enforced at predict time and by the official validator.

---

## 17. References

The blocking/feature design was informed by the entity-matching literature:

- **Ditto** — Li et al., *Deep Entity Matching with Pre-Trained Language Models* (2020). Motivated joint name+address reranking and explicit street-number evidence.
- **DeepBlocker** — Thirumuruganathan et al., *Deep Learning for Blocking in Entity Matching* (2021). Recall-vs-candidate-size framing.
- **Sparkly** — Paulsen et al., *A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching* (2023). Strong lexical blocking as a serious baseline.
- **Sudowoodo** — Wang et al., *Contrastive Self-supervised Learning for Data Integration* (2022). Hard-negative mining intuition.
- **XLM-R** — Conneau et al. (2020). Motivation to keep the pipeline multilingual/open-set.
- **Collective ER** — Bhattacharya & Getoor (2007). Caution against naive transitive clustering.

---

<p align="center"><sub>Built for the Amazon ML Challenge 2026 · CPU-only · supplied data only · no external lookup</sub></p>
