# PrivacySIM: Evaluating LLM Simulation of User Privacy Behavior

## Hardware requirements

All of our experiments involving open-weight models used 8x A100 40GB. Please modify configs/models.yaml to meet your hardware requirements.

## Quickstart (quickly get results from the full pipeline)

The fastest path that exercises every stage (sim → eval → plots) with one open-weight model and ~50 prompts:

```bash
# 1. environment (see Setup below for prerequisites)
uv venv --python 3.10 --seed
source .venv/bin/activate
uv pip install 'vllm==0.19.0' 'torch==2.10.0' --torch-backend=cu128
uv pip install -r requirements.txt

# 2. datasets
bash scripts/download_datasets.sh
python scripts/process_datasets.py

# 3. sanity-check the test suite (no GPU needed, ~1s)
pytest tests/

# 4. one model, 50 prompts, end-to-end
python src/pipeline.py run --model gemma4_32B --limit 50
```

You should now see plots at `data/runs/gemma4_32B/plots__limit_50/*.{png,pdf}` and an additional file `data/runs/gemma4_32B/run__limit_50.json`. Drop `--limit 50` to run the full 6,400 prompts.

## Repository Structure

```text
.
├── configs/
│   └── models.yaml                  # Per-model backend, sampling params, prompt template
├── data/
│   ├── <study>/                     # Per-study raw + cleaned CSVs (5 studies)
│   │   └── data_processed.csv
│   ├── full_dataset.csv             # All 2,000 users, no filter, no sample
│   ├── dataset.csv                  # 1,000 sampled users, healthcare-filtered
│   ├── dataset_with_prompts.csv     # 6,400 prompt rows (baseline)
│   ├── dataset_with_prompts_<theory>.csv  # ×3 theory variants
│   ├── persona_clustering/          # RQ2 cluster labels + per-model per-cluster metrics
│   └── runs/<model>/                # Per-model sim outputs, eval summary, plots, run.json
├── docs/
│   └── SCHEMA.md                    # Column-by-column schema for every CSV the pipeline writes
├── notebooks/
│   └── prompt_viewer.ipynb          # Inspect an individual prompt by uuid
├── scripts/
│   ├── download_datasets.sh         # Fetch the 5 raw user-study datasets
│   ├── process_datasets.py          # process / finalize / prompts (3 stages)
│   └── plot_models_summary.py       # Cross-model summary figure
├── src/
│   ├── pipeline.py                  # Orchestrator (sim → eval → plots)
│   ├── user_simulation.py           # vLLM batch backend
│   ├── openai_user_simulation.py    # OpenAI Batch API backend
│   ├── claude_user_simulation.py    # Anthropic Message Batches backend
│   ├── gemini_user_simulation.py    # Google Gemini Batch API backend
│   ├── _batch_sim_common.py         # Shared scaffolding for the 3 closed backends
│   ├── evaluation.py                # alignment / Spearman / tolerance accuracy
│   ├── plots.py                     # Per-model plots from evaluation_summary.csv
│   ├── persona_clustering.py        # RQ2 stage 1: assign cluster labels
│   └── per_cluster_eval.py          # RQ2 stage 2: per-cluster metrics
├── tests/                           # pytest suite (20 tests, ~0.3s, no GPU)
├── .github/workflows/ci.yml         # CI: py_compile + pytest on Ubuntu, no GPU
├── requirements.txt                 # Top-level deps (extends vllm)
└── requirements.lock                # Frozen transitive deps for byte-reproducible installs
```

## Setup

We use uv as our package manager. You can install uv by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To set up the environment, run the following (we used Python 3.10.12, CUDA 12.8 driver, 8× A100 40 GB):

```bash
uv venv --python 3.10 --seed
source .venv/bin/activate
# We pin vllm 0.19.0 + torch cu128 because the codebase was developed/tested
# against this combo. 
uv pip install 'vllm==0.19.0' 'torch==2.10.0' --torch-backend=cu128
uv pip install -r requirements.txt
```

Use the lockfile generated from a known-good environment instead:

```bash
uv pip install -r requirements.lock --torch-backend=cu128
```

> [!IMPORTANT]
> vLLM model downloads can be 50+ GB. If your `~/.cache/huggingface` partition is small, point `HF_HOME` at a partition with enough space *before* running the pipeline:
>
> ```bash
> export HF_HOME=/path/to/large/disk/hf_cache
> ```

### Datasets

Step 1 — download all five raw user-study datasets and place under `data/`:

```bash
bash scripts/download_datasets.sh
```

The download script skips any file already present. Source links for each dataset:

- **LLM-based chatbot privacy norms** — [WeberLab-UW/AIES_2025_ChatbotPrivacy](https://github.com/WeberLab-UW/AIES_2025_ChatbotPrivacy/blob/main/data/clean/cleaned_factorial_vignette_survey_data_2025.csv) → `data/llm_chatbot/`
- **LLM health consultation** — [Cristliu/LLMHealthPrivacy_UserStudy](https://github.com/Cristliu/LLMHealthPrivacy_UserStudy/blob/main/redacted%20datasets%20and%20statistical%20analysis%20scripts/Data/REDACTED_Data.xlsx) → `data/llm_health/`
- **LLM-based agent permissions** — [llm-platform-security/ai-agent-permissions](https://github.com/llm-platform-security/ai-agent-permissions/blob/main/data/user_study.json) → `data/agent_permissions/`
- **Chatbot group chat** — [csienslab/bot-among-us](https://github.com/csienslab/bot-among-us) (`Chatbot User Study Cleaned.csv` and `question_id_mapping.csv`) → `data/chatbot_group_chat/`
- **LLM-based conversational agent** — [OSF project (Answers.csv)](https://osf.io/2vqws/) → `data/llm_conversational_agent/`

Step 2 — run the dataset pipeline. `scripts/process_datasets.py` contains three stages by default:

1. **process** — turn each raw study into `data/<study>/data_processed.csv` (schema: `user_id`, `domain`, `demographics`, `previous_experiences`, `privacy_attitudes`, `questionnaire`, `responses`).
2. **finalize** — combine the per-study CSVs, sample 200 users per domain (1000 total) with `random_state=42`, add a deterministic `uuid5` column, and write `data/dataset.csv`.
3. **prompts** — expand each user into the powerset of available persona facets (8 prompt types) and write `data/dataset_with_prompts.csv` plus the three privacy-theory variants (`*_privacy_calculus.csv`, `*_bounded_rationality.csv`, `*_pmt.csv`).

```bash
# Run all three stages
python scripts/process_datasets.py

# Run a single stage
python scripts/process_datasets.py --stage process
python scripts/process_datasets.py --stage finalize
python scripts/process_datasets.py --stage prompts

# Limit the 'process' stage to one or more studies
python scripts/process_datasets.py --stage process --dataset llm_chatbot llm_health
```

## Running the Pipeline

Model configs live in `configs/models.yaml`. The `defaults:` block is split by backend (`defaults.vllm`, `defaults.openai`, `defaults.claude`, `defaults.gemini`); each model inherits from the block matching its `backend` field (default: `vllm`) and can override any key. `src/pipeline.py` orchestrates the three stages — user simulation, evaluation, plots — for any configured model, dispatching the sim stage to the matching backend script (`src/user_simulation.py`, `src/openai_user_simulation.py`, `src/claude_user_simulation.py`, or `src/gemini_user_simulation.py`).

Closed-source backends require credentials in the environment:

- GPT (OpenAI): `OPENAI_API_KEY`
- Claude (Anthropic): `ANTHROPIC_API_KEY`
- Gemini (Google): `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

Run one model end-to-end (sim → eval → plots, fail-fast):

```bash
python src/pipeline.py run --model qwen3.5_27B
python src/pipeline.py run --model gpt-5.4-mini
```

Run just a subset of the stages:

```bash
python src/pipeline.py run --model qwen3.5_27B --steps eval,plots
```

Sweep every configured model (sequential):

```bash
python src/pipeline.py run --all --steps eval,plots
```

Run a single domain (useful for quick iteration on one domain without clobbering the full-run artifacts):

```bash
# CGC-only end-to-end
python src/pipeline.py run --model qwen3.5_27B --domain "chatbot group chat"

# Eval + plots only on the CGC subset (after editing src/evaluation.py or src/plots.py)
python src/pipeline.py run --model qwen3.5_27B --domain "chatbot group chat" --steps eval,plots
```

When `--domain` is set, artifacts are written to suffixed paths (`outputs__<slug>.csv`, `evaluation_summary__<slug>.csv`, `plots__<slug>/`) so the unsuffixed full-dataset artifacts are preserved. Valid domain values (case-insensitive): `ai agent permissions`, `llm chatbot`, `llm conversational agents`, `chatbot group chat`, `llm healthcare consultation`. `--domain` is not supported for OpenAI models (they always run over the full dataset).

After every successful sim, the pipeline writes a `run.json` along with `outputs.csv` capturing git SHA, hostname, GPU model, vLLM/torch versions, sampling params, and wall-clock timing. 

List configured models and which outputs exist on disk (one row per model × theory, including the baseline):

```bash
python src/pipeline.py list
```

Adding a new model is a small addition to `configs/models.yaml` — no new shell script required. For OpenAI models, set `backend: openai` on the entry.

### Output layout

All per-model artifacts live under `data/runs/<model>/`:

```text
data/runs/<model>/
  outputs.csv               # baseline sim output (model responses)
  evaluation_summary.csv    # baseline eval output
  plots/
    alignment_by_domain.png
    ccc_by_domain.png
    spearman_by_domain.png
    tolerance_by_domain.png
  # When run with --theory <t>, baseline artifacts are preserved and a parallel
  # set is written with the theory suffix:
  outputs__<theory>.csv
  evaluation_summary__<theory>.csv
  plots__<theory>/
    ...
  # When run with --domain <d> (vLLM only), the same pattern applies with the
  # domain slug; combining --theory and --domain stacks as __<theory>__<domain>.
  # Closed-model batch backends (OpenAI / Claude / Gemini) additionally write
  # side artifacts that carry the same suffix as outputs.csv:
  batch_response.jsonl              # baseline
  batch_errors.jsonl
  missing_rows.json
  batch_artifacts/                  # chunked resume cache
    batch_response_part_<i>.jsonl
    ...
  batch_response__<theory>.jsonl    # per-theory equivalents
  batch_errors__<theory>.jsonl
  missing_rows__<theory>.json
  batch_artifacts__<theory>/
    ...
```

### Running evaluation / plots directly

In case you want more granular control over the evaluation or plotting stages, you can run them directly on any existing `outputs.csv`:

```bash
python src/evaluation.py \
    --input-csv data/runs/gpt-5.4-mini/outputs.csv \
    --summary-csv data/runs/gpt-5.4-mini/evaluation_summary.csv

python src/plots.py \
    --input-csv data/runs/gpt-5.4-mini/evaluation_summary.csv \
    --output-dir data/runs/gpt-5.4-mini/plots
```

`src/evaluation.py` does not filter by `prompt_type`; it evaluates all prompt types present in the input CSV.

For a column-by-column reference of every CSV the pipeline writes (raw datasets → `outputs.csv` → `evaluation_summary.csv` → per-cluster metrics) plus the `run.json`, see [docs/SCHEMA.md](docs/SCHEMA.md).

### Generating the main paper figure

`paper/figures/main_results.pdf` is the cross-model summary figure (prompt types on the x-axis, one bar per model, plus a black "Average" line). It is produced by `scripts/plot_models_summary.py`, which also generates `paper/figures/model_subplots.pdf` (a 1×4 per-model breakdown for Gemini 3.1 Pro, Gemini 3 Flash, GPT-5.4, and Sonnet 4.6):

```bash
python scripts/plot_models_summary.py
```

The script reads `data/runs/<model>/evaluation_summary.csv` for each model listed in its `MODELS` constant. To add or reorder models, edit that list.

### RQ2: per-cluster privacy persona analysis

The RQ2 analysis groups users into a 5-quadrant `(stance × exposure)` taxonomy and reports per-cluster simulation quality. It is two stages:

```bash
# Stage 1: assign each of the 1,000 sampled users to a privacy-persona cluster
# (writes data/persona_clustering/cluster_labels.csv + cluster_centroids.csv).
python src/persona_clustering.py

# Stage 2: aggregate a model's per-row evaluation by cluster × prompt_type.
# --view 5domain: prompt types {none, PE, PA, PE+PA} across all 5 domains.
# --view 3domain: AAP / CGC / LCA only, all 8 prompt types (incl. demographics).
python src/per_cluster_eval.py --model gemma4_32B --view 5domain
python src/per_cluster_eval.py --model gemma4_32B --view 3domain
```

Stage 2 reads `data/runs/<model>/outputs.csv` (so the model's pipeline run must have completed first) and writes `data/persona_clustering/<model>__per_cluster_metrics__<view>.csv`.

### RQ3: Theory-informed prompting

Run with a privacy-theory-informed prompt variant:

```bash
# PMT-framed prompts, end-to-end
python src/pipeline.py run --model qwen3.5_27B --theory pmt

# Privacy Calculus, eval + plots only
python src/pipeline.py run --model qwen3.5_27B --theory privacy_calculus --steps eval,plots
```

Three theory variants are supported: `privacy_calculus`, `bounded_rationality`, `pmt`. Each adds a theory-specific `PRIVACY DECISION LENS` block to every prompt while keeping the persona, questionnaire, and target output format byte-identical to the baseline. The variants are generated by the `prompts` stage of `scripts/process_datasets.py` alongside the baseline; they live at:

```text
data/dataset_with_prompts.csv                        # baseline (no theory)
data/dataset_with_prompts_privacy_calculus.csv
data/dataset_with_prompts_bounded_rationality.csv
data/dataset_with_prompts_pmt.csv
```

When `--theory` is set, sim/eval/plots artifacts are written to theory-suffixed paths (`outputs__<theory>.csv`, `evaluation_summary__<theory>.csv`, `plots__<theory>/`) under `data/runs/<model>/`, so the baseline (unsuffixed) artifacts are preserved. The pipeline picks the right input CSV automatically from the `--theory` flag — no manual `--input-csv` override needed. `--theory` works on all four backends (vLLM, OpenAI, Claude, Gemini) and can be combined with `--domain` on vLLM, in which case suffixes stack as `__<theory>__<domain>` (e.g. `outputs__pmt__chatbot_group_chat.csv`).

## Licenses and attribution

This repository ships two separately licensed artifacts:

- **The code** in `src/`, `scripts/`, `tests/`, `notebooks/`, `configs/`, and `docs/` is released under the [MIT License](LICENSE).
- **The combined PrivacySIM dataset** (the participant responses fetched and reshaped by the pipeline, hosted on Hugging Face) is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). Per-row content remains under each source study's original license (MIT, CC0 1.0, or CC BY 4.0). Full per-study attribution, license texts (MIT verbatim where required), copyright notices, and an indication of modifications are in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Anyone who runs `bash scripts/download_datasets.sh` and shares the resulting `data/` directory must preserve that attribution file alongside it.

The two licenses are independent — the MIT code can be used to process the CC BY 4.0 data without either license being affected by the other.
