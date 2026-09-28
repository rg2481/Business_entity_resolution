# Business Entity Resolution at Scale

**Amazon ML Challenge 2026.** We matched **1.73 million businesses** to **~10 million noisy records** from two other
data sources, across **3 countries and 10 writing systems** (Latin + 9 Indic scripts). The pipeline combines GPU TF-IDF
retrieval, a LightGBM pair model and a fine-tuned multilingual transformer.

| | |
|---|---|
| **Final team score** | **0.982171** public macro F0.5, up from a 0.963571 baseline |
| **This repo, own pipeline only** | 0.980736 public (model 1); local held-out F0.5 **0.9888** (model 2) |
| **Scale** | 1,732,544 reference businesses · 9,969,589 records · 242.6M candidate pairs scored |
| **Stack** | Python · PyTorch/CUDA · LightGBM · Hugging Face Transformers · Polars/PyArrow · scikit-learn |
| **Hardware** | one laptop: RTX 5080 (16 GB), 15 GB RAM (WSL2) |

---

## The problem

Three sources describe the same real businesses:
- **Source 1:** clean reference records.
- **Sources 2 and 3:** noisy copies of those records, plus decoys.

For every Source-1 business, return all Source-2/3 records that are the same business, using only **name + address**.
The noise is realistic:
- typos and letter/digit swaps (`PLUM8ERS`);
- dropped or added legal words (`Pvt Ltd`, `LLC`, `SARL`);
- web handles (`amicaledulamitie.com`);
- names transliterated into Hindi, Tamil, Bengali and other scripts;
- reordered addresses and empty fields.

About 26–44% of records match *nothing*. Many of them are deliberate **sibling decoys**: the same name and street as
a real business, but a different house number.

**Metric:** F0.5 per reference business, averaged over all 1.73M.
- Precision counts twice as much as recall, so a false merge costs about as much as missing half the true matches.
- A business with no true match scores 1 only if we predict nothing for it.

## Approach

```
7 raw TSVs
  │  normalisation: Unicode/accents (Latin only), legal forms, DBA/web handles,
  │  learned Indic→Latin dictionary (1,347 tokens), state/street aliases
  ▼
Blocking (retrieval/): GPU char-trigram TF-IDF, per country, record-centric
  │  union of top-10 by name, by address, by name+address → ~24 candidates per record
  │  recall 98.7–99.0% · 99.996% of all pairs pruned
  ▼
Pair model C04 (matcher/): LightGBM on 92 features
  │  string similarities + how the name/number was perturbed + cross-record "twin" evidence
  ▼
Cross-encoder C28 (matcher/): multilingual-e5-small fine-tuned on raw "name | address" pairs
  │  re-scores each record's best candidate; logistic stack of LightGBM + transformer
  ▼
Decision: one owner per record (argmax), threshold chosen on held-out data, US near-number rule
  ▼
output/matching_results.tsv  +  output/candidate_pairs.tsv
```

### Key ideas

1. **Record-centric retrieval.** The ground truth is a partition: each record belongs to at most one business. So
   each record searches the reference businesses of its own country; we never search the other way.
2. **Model the generator's noise, not just similarity.** EDA catalogued the noise families. The 31 features we
   added describe *how* a record differs from its reference:
   - word swaps versus typos;
   - dropped digits versus shifted house numbers;
   - acronyms;
   - rare-word priors;
   - "twin" counts across records.

   Local F0.5 rose from 0.971 to 0.984.
3. **Find the train/test shift with controlled experiments.**
   - Local gains were only half-reflected on the leaderboard.
   - Uploads that changed one country or one rule at a time located the cause: the test data holds many more
     sibling decoys.
4. **Use a model that reads text differently.**
   - A cross-encoder that reads both raw records side by side catches decoys the tree model trusted.
   - Local F0.5 went from 0.9844 to **0.9888**; leaderboard from 0.9725 to 0.9807, using this pipeline alone.
5. **Evidence before every upload.** Every idea was scored on a labeled held-out split first:
   - query isolation, so no validation record is trained on;
   - threshold chosen on one half and reported on the other.

   Raw samples were read before any rule.

The full story, including what did **not** work (XGBoost swaps, global house-number rules, extra transliteration),
is in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

## Results

| Stage | Local held-out F0.5 | Public leaderboard |
|---|---|---|
| B01: TF-IDF retrieval + LightGBM (61 features) | 0.9714 | 0.9636 |
| C04: + 31 perturbation-aware features, 8x data | 0.9844 | 0.9700 |
| C04 + US sibling-decoy rule | – | 0.9725 |
| **C28: + multilingual cross-encoder (this repo's final stage)** | **0.9888** | **0.9807** (model 1) |
| Team best: C28 + blend with a teammate's model + France self-training | – | **0.9822** |

## How to run

**Requirements:** Linux or WSL2, Python 3.10, an NVIDIA GPU with ≥ 12 GB and CUDA 12.x, about 15 GB RAM and about
60 GB free disk. The full run takes about 11 hours on one laptop GPU; every stage is resumable.

**1. Environment**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cuda.txt     # PyTorch CUDA build
pip install -r requirements.txt
```

**2. Data.** The competition data is not included. Place the official kit here:

```text
student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
student_resource/utils/validate_submission.py
```

**3. Run the stages in order.** `bash run.sh plan` lists them. Each stage shows progress, writes `logs/<stage>.log`
and prints the next command.

| Stages | What happens | Time |
|---|---|---|
| `preflight`, `bootstrap`, `prepare-train`, `prepare-test` | CUDA check; TSV → Parquet; normalisation; Indic dictionary | ~25 min |
| `candidates-train-india`, `candidates-train-us` | GPU retrieval on train | ~2.6 h |
| `features-india`, `features-us`, `b01-train`, `b01-evaluate` | held-out split, base features, baseline | ~35 min |
| `candidates-test-{france,india,us}` | GPU retrieval on test | ~1.4 h |
| `c04-prepare-*`, `c04-build-*`, `c04-train`, `c04-evaluate` | C04 features and LightGBM; threshold on tune half | ~45 min |
| `c04-test-prepare-*`, `c04-test-score-*`, `c04-export`, `c04-candidates` | score 242.6M test pairs; candidate file | ~50 min |
| `c28-data`, `c28-train`, `c28-data-2`, `c28-train-2` | build text pairs; fine-tune the cross-encoder in two rounds (25% + 25% of data) | ~1 h |
| `c28-score-val`, `c28-eval` | score validation; fit the stack; choose threshold on tune half | ~15 min |
| `c28-score-test`, `c28-export`, `finalize` | score test winners; export; official validator | ~2 h |

Final files: `output/matching_results.tsv` and `output/candidate_pairs.tsv`, one row per test business. The export
stage runs the official validator with `--check-ids`. GPU nearest-neighbour ties can change a few rows between runs.

## Repository layout

```text
run.sh                     single launcher: 36 ordered, resumable stages
retrieval/                 normalisation, Indic dictionary, GPU TF-IDF retrieval, held-out split, base features
  scripts/b01_*.py         (b01 = baseline experiment id)
  configs/, env.sh
matcher/                   pair models and decisions
  scripts/c04_*.py         LightGBM pair model: 31 perturbation-aware features, training, evaluation
  scripts/c04t_*.py        test scoring, export, US decoy rule, candidate file
  scripts/c28_*.py         multilingual cross-encoder: data, fine-tuning, stacking, export
  configs/c04.json, run_c04.sh, env.sh
docs/EXPERIMENT_LOG.md     leaderboard timeline, what worked, what did not, lessons
```

## Validation protocol

- Reference businesses are split by a hash of their ID; 10,000 held-out businesses per country are halved into
  **tune** and **check**.
- Any record whose candidates touch a held-out business is excluded from training (**query isolation**).
- Its whole candidate competition, distractors included, stays in validation.
- Thresholds are chosen on *tune* with false positives weighted 1.9x (test has about 1.9x more distractors), and
  reported on *check*.
- The evaluator reproduces the official per-business F0.5.

## Team and tools

Built as a team for the Amazon ML Challenge 2026. The work used AI coding assistants (Anthropic Claude and OpenAI
Codex), which the challenge rules permit for writing code. Every model in the pipeline is trained locally on the
provided data. The pipeline makes no external API calls and uses no external data. Pretrained weights:
`intfloat/multilingual-e5-small` (MIT licence).
