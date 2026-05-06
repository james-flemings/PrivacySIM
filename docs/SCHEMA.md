# CSV / JSON schemas

Every artifact the pipeline writes is documented here, in the order it appears
in the data flow:

```
raw study data
    │
    ▼   scripts/process_datasets.py --stage process
data/<study>/data_processed.csv   (one per study)
    │
    ▼   scripts/process_datasets.py --stage finalize
data/full_dataset.csv             (all 2,000 users, no filter, no sample)
data/dataset.csv                  (1,000 sampled users, healthcare-filtered)
    │
    ▼   scripts/process_datasets.py --stage prompts
data/dataset_with_prompts.csv     (6,400 prompt-rows, baseline)
data/dataset_with_prompts_<theory>.csv   (×3, theory variants)
    │
    ▼   src/pipeline.py run --model <m>
data/runs/<m>/outputs.csv         (sim output)
data/runs/<m>/run.json            (provenance sidecar)
    │
    ▼   src/evaluation.py
data/runs/<m>/evaluation_summary.csv
    │
    ▼   src/plots.py
data/runs/<m>/plots/*.{png,pdf}
    │
    ▼   src/persona_clustering.py + src/per_cluster_eval.py (RQ2)
data/persona_clustering/cluster_labels.csv
data/persona_clustering/cluster_centroids.csv
data/persona_clustering/<m>__per_cluster_metrics__<view>.csv
```

---

## `data/<study>/data_processed.csv`

Per-study, after raw-input cleaning. One row per user. Five files: `llm_chatbot`,
`llm_health`, `agent_permissions`, `chatbot_group_chat`, `llm_conversational_agent`.

| Column | Type | Description |
|---|---|---|
| `user_id` | int | 1-indexed within study; not unique across studies. |
| `domain` | str | One of: `LLM chatbot`, `LLM healthcare consultation`, `AI agent permissions`, `Chatbot group chat`, `LLM conversational agents`. |
| `demographics` | str(dict) \| NA | `{label: value}` of demographic fields the source study released; NA when the study didn't collect any. Stringified Python dict (or JSON, for `chatbot_group_chat`). |
| `previous_experiences` | str(dict) | `{question_text: user_response}` of AI-tool exposure / familiarity items. |
| `privacy_attitudes` | str(dict) | `{question_text: user_response}` of stated privacy-concern items. |
| `questionnaire` | str | The full questionnaire text the user actually saw — `Q1.`, `Q2.`, …, with shared instructions kept once. |
| `responses` | str(dict) | `{Q1: response, Q2: response, …}` — Q-keyed map of the user's answers. |
| `preferences` | str(dict) | Like `responses` but keyed by **comprehensive question text** instead of Q-numbers: each key repeats whatever shared scenario context Q_i depends on, but skips pure scene-setting (e.g., the year-2049 / MIRA backstory in `agent_permissions`). |

`previous_experiences` / `privacy_attitudes` / `responses` / `preferences` /
`demographics` are stored as Python dict reprs (single-quoted) for four of the
five studies; `chatbot_group_chat` uses JSON (double-quoted). Both forms parse
with `ast.literal_eval`.

---

## `data/full_dataset.csv` and `data/dataset.csv`

Combined across all studies; same columns as `data_processed.csv` plus a
deterministic `uuid`.

| Column | Type | Description |
|---|---|---|
| `uuid` | str | `uuid5(NAMESPACE_URL, "<user_id>||<domain>")` — stable across runs, lets downstream artifacts join back to a user identity. |
| `user_id`, `domain`, … | as above | Inherited from `data_processed.csv`. |

`full_dataset.csv` has 2,000 rows: every user from every per-study CSV with no
filtering and no sampling. `dataset.csv` has 1,000 rows: `full_dataset.csv` after
(a) restricting healthcare to higher-sensitivity attributes (mental/medical)
with informative previous-experience values and (b) sampling 200 per domain
(`random_state=42`).

Per-study row counts:

| `domain` | `full_dataset.csv` | `dataset.csv` |
|---|---|---|
| `LLM healthcare consultation` | 846 | 200 |
| `Chatbot group chat` | 374 | 200 |
| `LLM chatbot` | 300 | 200 |
| `LLM conversational agents` | 277 | 200 |
| `AI agent permissions` | 203 | 200 |
| **Total** | **2,000** | **1,000** |

---

## `data/dataset_with_prompts.csv` (and `dataset_with_prompts_<theory>.csv`)

Each user from `dataset.csv` is expanded into the **powerset** of available
persona facets (8 prompt types when all three facets are present, fewer when a
study didn't release demographics). Same columns as `dataset.csv` plus:

| Column | Type | Description |
|---|---|---|
| `prompt_id` | int | 1-indexed unique identifier for this row. |
| `prompt_type` | str | `none`, `demographics`, `previous_experiences`, `privacy_attitudes`, or `+`-joined combinations (e.g., `previous_experiences+privacy_attitudes`). |
| `available_columns` | str(JSON list) | Subset of `["demographics", "previous_experiences", "privacy_attitudes"]` that this user has parseable values for. |
| `included_columns` | str(JSON list) | Subset of `available_columns` actually inserted into the prompt for this row. |
| `llm_prompt` | str | The fully assembled prompt the model receives — `ROLE` + `USER PRIVACY PERSONA` + (optional `PRIVACY DECISION LENS`) + `TASK & CONSTRAINTS` + `TARGET OUTPUT FORMAT` + `QUESTIONNAIRE`. |

The four files (`baseline` + 3 privacy-theory variants) share `prompt_id`,
`uuid`, and `prompt_type`; only `llm_prompt` differs (theories add a
`PRIVACY DECISION LENS` block; the rest is byte-identical).

---

## `data/runs/<model>/outputs.csv`

The dataset_with_prompts row plus the model's response. Same columns as the
input plus:

| Column | Type | Description |
|---|---|---|
| `model_thoughts` | str | (vLLM only) Reasoning trace extracted by `vllm.reasoning.ReasoningParser`. Empty for backends that don't surface reasoning. |
| `final_answer` | str | The model's answer text — for closed backends, parsed out of the provider-specific JSON via `extract_final_answer()` in each `*_user_simulation.py`. |
| `raw_response` | str(JSON) | (closed backends only) The full provider response payload (OpenAI Responses API body, Anthropic Message, or Gemini generateContent response). Use this if `final_answer` parsing missed something. |
| `source_row_index` | int | (closed backends only) The 0-indexed row of the input prompts CSV this response maps to. |

---

## `data/runs/<model>/run.json`

Provenance sidecar emitted by `src/pipeline.py` after every successful sim
step. JSON object with the following keys:

| Key | Description |
|---|---|
| `model_alias` | The configs/models.yaml alias (e.g., `qwen3.5_27B`). |
| `backend` | `vllm` / `openai` / `claude` / `gemini`. |
| `model_id` | The provider-side model identifier (e.g., `Qwen/Qwen3.5-27B`). |
| `wall_clock` | `{started_at_utc, finished_at_utc, elapsed_seconds}` for the sim step. |
| `host` | `{hostname, platform, python_version, gpus}` (gpus from nvidia-smi). |
| `git` | `{sha, dirty}` — the repo state when the sim was launched. |
| `package_versions` | Frozen versions of vllm / torch / pandas / etc. |
| `config` | The merged per-model config (after `defaults.<backend>` overlay). |
| `qualifiers` | `{domain, theory, limit}` — what variant produced this run. |
| `outputs.sim_csv` | Repo-relative path to the `outputs.csv` this metadata describes. |

---

## `data/runs/<model>/evaluation_summary.csv`

One row per `(prompt_type, domain)` plus pooled rows for each prompt_type. See
the docstring at the top of [src/evaluation.py](../src/evaluation.py) for the
exact metric definitions.

| Column | Type | Description |
|---|---|---|
| `prompt_type` | str | One of the 8 prompt types (or `none`). |
| `domain` | str | Either a study domain name, or `ALL_DOMAINS` (5-domain pool), or `3DOMAIN_INTERSECTION` (the 3 demographics-bearing studies). |
| `rows` | int | Users in this `(prompt_type, domain)` cell. |
| `rows_with_any_match` | int | Users with at least one Q_i where both user and model responses parsed and mapped. |
| `row_coverage` | float | `rows_with_any_match / rows`. |
| `parse_error_rows` | int | Users where either the user-response dict or the model-answer dict failed to parse. |
| `total_compared_questions` | int | Sum of comparable Q_i across users in this cell. |
| `total_unmapped` | int | Sum of Q_i where one side parsed but didn't map to a numeric label. |
| `alignment_score` | float | Per-user mean of `100 × clip(1 − mean_i(\|model_i − user_i\| / domain_range), 0, 1)`. Reported in `[0, 100]`. |
| `spearman_rank_correlation` | float | Per-user mean Spearman ρ over each user's K paired (user, model) values. NaN-on-zero-variance users are dropped from the mean and counted in `users_dropped`. |
| `tolerance_accuracy` | float | Per-user mean of `100 × |{i: \|model_i − user_i\| ≤ τ × domain_range}| / K` at `τ = 0.15`. The headline metric used in the paper figures. |
| `users_dropped` | int | Users excluded from the `spearman_rank_correlation` mean due to fully-tied responses. |

---

## `data/runs/<model>/plots/*.{png,pdf}`

Three plots, all per-`(prompt_type × domain)` line charts at `tolerance_accuracy_tau015_by_domain`,
`alignment_by_domain`, `spearman_by_domain`. Generated by `src/plots.py` from
`evaluation_summary.csv`.

---

## `data/persona_clustering/cluster_labels.csv`

Per-user cluster assignment from `src/persona_clustering.py`. Joins to other
artifacts on `uuid`.

| Column | Description |
|---|---|
| `uuid` | Joins to `dataset.csv` / `outputs.csv`. |
| `user_id`, `domain` | Inherited from `dataset.csv`. |
| `stated_concern` | 0..1 score over `privacy_attitudes` items. |
| `revealed_concern` | 0..1 score over questionnaire `responses` (NOT used as a cluster predictor; would be circular). |
| `pe_exposure` | 0..1 AI-tool exposure score over `previous_experiences` items. |
| `stated_z`, `revealed_z`, `pe_z` | Within-domain z-scores of the three above. |
| `stated_polarity_flipped` | True if the empirical polarity check flipped `stated_concern` for this domain. |
| `cluster` | One of `high / high`, `low / high`, `high / mid`, `mid / mid`, `low / low`, or `Unassigned` (no parseable `stated_z`). |

---

## `data/persona_clustering/<model>__per_cluster_metrics__<view>.csv`

Per-`(cluster × prompt_type)` aggregation produced by `src/per_cluster_eval.py`.

| Column | Description |
|---|---|
| `model`, `view` | Echoed from CLI args (`--view` ∈ {`5domain`, `3domain`}). |
| `cluster` | One of the 5 quadrants (Unassigned rows are excluded). |
| `prompt_type` | Same set as `evaluation_summary.csv`. |
| `n_rows`, `sum_compared` | Sample size in this cell. |
| `alignment_score`, `spearman`, `tolerance_accuracy` | Same definitions as in `evaluation_summary.csv`. |
| `signed_bias` | Per-user mean of `(model_i − user_i) / domain_range`, averaged over Q_i. Positive = simulator over-predicts. |
| `users_dropped` | Same definition as in `evaluation_summary.csv`. |
