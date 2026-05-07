"""End-to-end processor for the five user-study datasets.

Three pipeline stages, all chained by default:

  1. process  — read raw dataset under data/<study>/ and write the canonical
                data_processed.csv used by the rest of the pipeline.
  2. finalize — combine the per-study data_processed.csv files into
                data/full_dataset.csv (every user from every study, no filter
                and no sampling), then restrict the healthcare study to
                higher-sensitivity attributes (mental/medical) with
                informative previous experiences, sample 200 users per domain
                (1,000 total), add a deterministic uuid5 column, and write
                data/dataset.csv.
  3. prompts  — expand each user into the powerset of available persona
                facets (8 prompt types), build the LLM prompt for each, and
                write data/dataset_with_prompts.csv plus the three
                privacy-theory variants.

Usage:
    python scripts/process_datasets.py                          # all stages
    python scripts/process_datasets.py --stage process          # just stage 1
    python scripts/process_datasets.py --stage process --dataset llm_chatbot
    python scripts/process_datasets.py --stage finalize         # just stage 2
    python scripts/process_datasets.py --stage prompts          # just stage 3
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import uuid
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

OUTPUT_COLUMNS = [
    "user_id",
    "domain",
    "demographics",
    "previous_experiences",
    "privacy_attitudes",
    "questionnaire",
    "responses",
    # Per-question, self-contained {question_text: user_response} dict. Each
    # question repeats whatever shared instructions / scenario context Q_i
    # depends on, but skips pure scene-setting (e.g. the year-2049 / MIRA
    # backstory in agent_permissions). Keys carry no Q-numbering.
    "preferences",
]


# ---------------------------------------------------------------------------
# 1. LLM chatbot (Weber Lab AIES 2025)
# ---------------------------------------------------------------------------
def process_llm_chatbot() -> Path:
    src = DATA_DIR / "llm_chatbot" / "cleaned_factorial_vignette_survey_data_2025.csv"
    out = DATA_DIR / "llm_chatbot" / "data_processed.csv"
    df = pd.read_csv(src)

    def create_vignette(role, location, purpose, content, consent, anonymity, privacy):
        return (
            f"Your ChatGPT conversations were shared with a(n) {role} based in {location} "
            f"for the purpose of {purpose}. {content} were shared. You were {consent}. "
            f"Your conversations were {anonymity} before being shared. "
            f"Personally identifying information from your conversations was {privacy}."
        )

    previous_experience_mapping = {
        "Usage": "How frequently do you use ChatGPT?",
        "year_start_using": "What is the date of your first use of ChatGPT?",
        "account_type": "What is your ChatGPT subscription type?",
        "Q14": "Tasks ChatGPT is Used For",
        "Q14_13_TEXT": 'If you selected "Other" for tasks, please elaborate.',
        "Q15": "Topics Discussed with ChatGPT",
        "Q15_26_TEXT": 'If you selected "Other" for topics, please elaborate.',
    }
    privacy_attitudes_mapping = {
        "concerned_pc": "Online companies collect too much personal information.  Please rate your agreement on a scale of 0 to 100 (0 = Strongly Disagree, 50 = Neutral, 100 = Strongly Agree).",
        "trust_pc": "In general, I trust websites.  Please rate your agreement on a scale of 0 to 100 (0 = Strongly Disagree, 50 = Neutral, 100 = Strongly Agree).",
        "privacy_important_pc": "In general, I believe privacy is important.  Please rate your agreement on a scale of 0 to 100 (0 = Strongly Disagree, 50 = Neutral, 100 = Strongly Agree).",
        "concerns_gpt_pc": "I have privacy concerns about conversations with ChatGPT.  Please rate your agreement on a scale of 0 to 100 (0 = Strongly Disagree, 50 = Neutral, 100 = Strongly Agree).",
        "private_gpt_pc": "In general, I believe that my conversations with chatbots like ChatGPT will remain private.  Please rate your agreement on a scale of 0 to 100 (0 = Strongly Disagree, 50 = Neutral, 100 = Strongly Agree).",
    }

    questionnaire_intro = (
        "You will be shown a series of scenarios describing the use of your ChatGPT history. "
        "For each scenario, please indicate your agreement with the following statement: "
        "'This is an appropriate use of my ChatGPT data.' Please rate your agreement on a scale "
        "of 0 to 100 (0 = Strongly Disagree, 50 = Neutral, 100 = Strongly Agree)."
    )

    rng = random.Random(42)
    rows = []

    for user_id, (_, row) in enumerate(df.iterrows(), start=1):
        previous_experiences = {
            label: row[col] for col, label in previous_experience_mapping.items()
        }
        privacy_attitudes = {
            label: row[col] for col, label in privacy_attitudes_mapping.items()
        }

        sampled = rng.sample(range(30), 10)
        lines, responses = [], {}
        for q_num, scenario_idx in enumerate(sampled, start=1):
            qid = f"Q{q_num}"
            vignette = create_vignette(
                str(row[f"role_shown_{scenario_idx}"]),
                str(row[f"location_shown_{scenario_idx}"]),
                str(row[f"purpose_shown_{scenario_idx}"]),
                str(row[f"content_shown_{scenario_idx}"]),
                str(row[f"consent_shown_{scenario_idx}"]),
                str(row[f"anonymity_shown_{scenario_idx}"]),
                str(row[f"privacy_shown_{scenario_idx}"]),
            )
            lines.append(f"{qid}. {vignette}")
            answer = row[f"Q{scenario_idx}_P"]
            responses[qid] = answer

        # preferences carries every vignette this user answered in the
        # original study, not just the 10 we sampled into questionnaire.
        preferences = {}
        for scenario_idx in range(30):
            vignette = create_vignette(
                str(row[f"role_shown_{scenario_idx}"]),
                str(row[f"location_shown_{scenario_idx}"]),
                str(row[f"purpose_shown_{scenario_idx}"]),
                str(row[f"content_shown_{scenario_idx}"]),
                str(row[f"consent_shown_{scenario_idx}"]),
                str(row[f"anonymity_shown_{scenario_idx}"]),
                str(row[f"privacy_shown_{scenario_idx}"]),
            )
            preferences[
                f"{vignette} Please rate your agreement with the following "
                "statement about the above scenario: 'This is an appropriate "
                "use of my ChatGPT data.' Rate from 0 to 100 (0 = Strongly "
                "Disagree, 50 = Neutral, 100 = Strongly Agree)."
            ] = row[f"Q{scenario_idx}_P"]

        rows.append(
            {
                "user_id": user_id,
                "domain": "LLM chatbot",
                "demographics": pd.NA,
                "previous_experiences": previous_experiences,
                "privacy_attitudes": privacy_attitudes,
                "questionnaire": questionnaire_intro + "\n" + "\n".join(lines),
                "responses": responses,
                "preferences": preferences,
            }
        )

    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# 2. LLM health consultation (Cristliu)
# ---------------------------------------------------------------------------
def process_llm_health() -> Path:
    src = DATA_DIR / "llm_health" / "REDACTED_Data.xlsx"
    out = DATA_DIR / "llm_health" / "data_processed.csv"
    df = pd.read_excel(src)

    stripped_to_col = {str(c).strip(): c for c in df.columns}

    attribute_cols = [
        "basic information",
        "lifestyle",
        "mental health information",
        "medical Information",
    ]
    # All four attributes are processed here so full_dataset.csv contains every
    # healthcare user. The sampled dataset.csv applies a higher-sensitivity
    # filter (mental/medical only) at finalize time — basic/lifestyle cluster
    # at ceiling and leave little variance for the persona to explain.

    attribute_examples = {
        "basic information": "age, gender, place of residence, etc",
        "lifestyle": "dietary habits, exercise habits, smoking and drinking habits, etc.",
        "mental health information": "stress levels, emotional fluctuations, psychotherapy, experiences",
        "medical Information": "blood pressure metrics, disease symptoms, disease history, diagnoses",
    }

    recipient_direct_cols = [
        "Your family members",
        "Healthcare providers",
        "Government agencies",
        "LLM service providers",
    ]
    recipient_indicator_cols = [
        "Recipient-Family members",
        "Recipient-Healthcare providers",
        "Recipient-Government agencies",
        "Recipient-LLM service providers",
    ]
    recipient_from_indicator = {
        "Recipient-Family members": "Your family members",
        "Recipient-Healthcare providers": "Healthcare providers",
        "Recipient-Government agencies": "Government agencies",
        "Recipient-LLM service providers": "LLM service providers",
    }
    condition_cols = [
        "If you have given consent",
        "If you are notified before collection",
        "If information is kept confidential and not stored",
        "If information will be stored",
        "If information is used for health monitoring",
        "If information is shared with others",
    ]
    # Original survey also asked recipient-specific follow-up conditions.
    # They are not universally answered, so we keep them out of the shared
    # questionnaire text and just surface them in `preferences` for the rows
    # that did answer them.
    extra_condition_cols = [
        "If it complies with information protection regulations",
        "If information is provided for academic research",
        "If information is used in public health surveys",
        "If used for information analysis and service improvement",
    ]
    previous_experiences_cols = [
        "Which of the following best describes who you are when using the healthcare chatbot?",
        "Did you provide any personal information, such as age, in the consultation experience? Why？",
        "Do you have any healthcare consultation experience with any AI-large-scale-model except in our survey?",
        "Do you have any experience with any AI-large-scale-model except in our survey?",
        "Which AI-large-scale-model platforms have you used?",
        "Have you ever heard of AI-large-scale-model?",
        "Do you have any healthcare consultation experience online?",
        "Why do you choose to use LLM-based healthcare consultation services?",
        "Why don't you want to use LLM-based healthcare consultation services?",
        "Have you experienced a health privacy breach?",
    ]
    privacy_attitudes_cols = [
        "Is online healthcare consultation more likely to compromise privacy than asking a doctor face-to-face?",
        "You will use reliable LLM-based healthcare consultation services in the future",
        "You understand health privacy",
        "You are concerned about health privacy",
    ]
    response_score_map = {
        -2: "Completely unacceptable",
        -1: "Somewhat unacceptable",
        0: "Neutral",
        1: "Somewhat acceptable",
        2: "Completely acceptable",
    }

    def pick_marked_one(row, names):
        for name in names:
            actual = stripped_to_col.get(name)
            if actual is not None and pd.notna(row[actual]) and row[actual] == 1:
                return name
        return None

    def extract_row_dict(row, names):
        out_dict = {}
        for name in names:
            actual = stripped_to_col.get(name)
            if actual is not None:
                out_dict[name] = row[actual]
        return out_dict

    def map_response_label(value):
        if pd.isna(value):
            return value
        if value in response_score_map:
            return response_score_map[value]
        try:
            return response_score_map.get(int(value), value)
        except (TypeError, ValueError):
            return value

    def build_numbered_questions(row, names, start_idx, punct):
        lines, responses, idx = [], {}, start_idx
        for name in names:
            actual = stripped_to_col.get(name)
            if actual is None:
                continue
            qid = f"Q{idx}"
            lines.append(f"{qid}. {name}{punct}")
            value = row[actual]
            if pd.notna(value):
                responses[qid] = map_response_label(value)
            idx += 1
        return lines, responses, idx

    HC_OPTIONS = (
        "Your options are 'Completely unacceptable', 'Somewhat unacceptable', "
        "'Neutral', 'Somewhat acceptable', and 'Completely acceptable'."
    )

    rows = []
    for user_id, (_, row) in enumerate(df.iterrows(), start=1):
        previous_experiences = extract_row_dict(row, previous_experiences_cols)
        privacy_attitudes = extract_row_dict(row, privacy_attitudes_cols)

        attribute = pick_marked_one(row, attribute_cols)
        if attribute is None:
            continue

        examples = attribute_examples.get(attribute, "e.g., examples")
        selected_indicator = pick_marked_one(row, recipient_indicator_cols)
        scenario_preamble = (
            "Imagine you use the AI-large-scale-model for healthcare consultation, "
            f"and the conversation mentions your {attribute} (e.g, {examples})."
        )

        questionnaire_lines = [
            (
                f"{scenario_preamble} "
                "How acceptable is it for the following recipients to access this information?\n"
                f"{HC_OPTIONS}"
            )
        ]

        idx = 1
        recipient_questions, responses, idx = build_numbered_questions(
            row, recipient_direct_cols, idx, "."
        )
        questionnaire_lines.extend(recipient_questions)

        # Build per-question preferences. Q1-Q4 (recipients) all repeat the
        # scenario preamble; Q5+ (conditions) repeat the recipient-specific
        # preamble that the questionnaire only states once.
        preferences: dict[str, object] = {}
        for name, qid in zip(recipient_direct_cols, [f"Q{i}" for i in range(1, idx)]):
            if qid not in responses:
                continue
            preferences[
                f"{scenario_preamble} How acceptable is it for {name} to "
                f"access this information? {HC_OPTIONS}"
            ] = responses[qid]

        if selected_indicator is not None:
            selected_recipient = recipient_from_indicator[selected_indicator]
            questionnaire_lines.append("")
            questionnaire_lines.append(
                f"How acceptable is it for {selected_recipient} to access your {attribute} "
                "under the following conditions?\n"
                f"{HC_OPTIONS}"
            )
            cond_start_idx = idx
            condition_questions, condition_responses, idx = build_numbered_questions(
                row, condition_cols, idx, ","
            )
            questionnaire_lines.extend(condition_questions)
            responses.update(condition_responses)

            for cond_name, qid in zip(
                condition_cols, [f"Q{i}" for i in range(cond_start_idx, idx)]
            ):
                if qid not in condition_responses:
                    continue
                preferences[
                    f"{scenario_preamble} How acceptable is it for "
                    f"{selected_recipient} to access your {attribute} "
                    f"under the following condition: '{cond_name}'? {HC_OPTIONS}"
                ] = condition_responses[qid]

            # Surface the extra recipient-specific conditions the user
            # actually answered into `preferences` (they fall outside the
            # 6-condition questionnaire shown to the LLM).
            for cond_name in extra_condition_cols:
                actual = stripped_to_col.get(cond_name)
                if actual is None:
                    continue
                value = row[actual]
                if pd.isna(value):
                    continue
                preferences[
                    f"{scenario_preamble} How acceptable is it for "
                    f"{selected_recipient} to access your {attribute} "
                    f"under the following condition: '{cond_name}'? {HC_OPTIONS}"
                ] = map_response_label(value)

        rows.append(
            {
                "user_id": user_id,
                "domain": "LLM healthcare consultation",
                "demographics": pd.NA,
                "previous_experiences": previous_experiences,
                "privacy_attitudes": privacy_attitudes,
                "questionnaire": "\n".join(questionnaire_lines) + '\n"',
                "responses": responses,
                "preferences": preferences,
            }
        )

    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# 3. AI agent permissions (llm-platform-security)
# ---------------------------------------------------------------------------
def process_agent_permissions() -> Path:
    src = DATA_DIR / "agent_permissions" / "user_study.json"
    out = DATA_DIR / "agent_permissions" / "data_processed.csv"
    with src.open("r", encoding="utf-8") as f:
        user_study = json.load(f)

    QUESTIONNAIRE_PREFIX = """In year 2049, AI personal assistants are an essential part of people’s daily lives. These AI assistants handle everything - from helping with work, to managing finances, to organizing travel, and meal planning.
-> "Create a training plan for my upcoming marathon"
\t- The AI assistant might consider user’s health metrics and calendar to create a training plan
-> "Plan a stopover in Spain for a two-day vacation with my family on my way back to travel from Japan"
\t- The AI assistant might consider user’s calendar, their interests, and family info to suggest a plan
-> "Can you compose a bedtime story that my children will like?"
\t- The AI assistant might consider children’s interests to create a story
-> "Recommend a TV show or movie I’ll love"
\t- The AI assistant might consider user’s viewing history and their mood for recommendations
Now imagine that you’re living in year 2049! Your job is to train a personal assistant named "MIRA" for yourself. You will be presented several scenarios to personalize Mira.
Training MIRA to learn your data sharing preferences
Mira learns your data sharing preferences and automatically manages permissions on your behalf, so you never have to decide each one individually! Please go through the following scenarios and express your preferences! Let's begin with the queries:
"""

    PREVIOUS_EXPERIENCE_KEYS = {
        "aiFamiliarity": "On a scale from 1 (Not at all familiar) to 5 (Very familiar), how familiar are you with existing AI tools, such as ChatGPT and Gemini, in terms of your knowledge and usage? [1. Not at all familiar, 2. Slightly familiar, 3. Somewhat familiar, 4. Moderately familiar, 5. Very familiar]",
        "aiFrequency": "How often do you use these tools? [1. Never, 2. Rarely (e.g., once a month or less), 3. Occasionally (e.g., 1-2 times a week), 4. Frequently (e.g., several times a week), 5. Daily]",
        "aiPurposes": "For what purposes do you use these tools?",
    }
    PRIVACY_ATTITUDE_KEYS = {
        "aiTrust": "On a scale from 1 (Not at all) to 5 (Full trust), how much do you trust these tools to not violate your privacy?",
        "privacyImportance": "On a scale from 1 (Not important) to 5 (Extremely Important), how important is privacy to you when using digital tools or services, including AI tools?",
        "privacyDetails": "If you have any privacy concerns while using these tools, what are they?",
        "concern_domain": "Select 3 domains you care the most about your privacy when using AI tools:",
    }
    AGE_MAPPING = {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55 or older"}

    AP_OPTIONS = (
        "Options: 'Yes, always share', 'Yes, but ask me next time', "
        "'No, but ask me next time', or 'No, never share'."
    )

    def build_questionnaire(task3_items, seed):
        rng = random.Random(seed)
        remaining = list(task3_items)
        blocks, responses, q_num = [], {}, 1

        while q_num <= 10 and remaining:
            query_item = rng.choice(remaining)
            remaining.remove(query_item)
            answer_items = list(query_item.get("answer", {}).items())
            if not answer_items:
                continue

            slots_left = 10 - q_num + 1
            if len(answer_items) <= slots_left:
                selected = answer_items
            else:
                keys = set(rng.sample([k for k, _ in answer_items], slots_left))
                selected = [item for item in answer_items if item[0] in keys]

            user_query = query_item["query"].strip()
            block_lines = [
                f"USER: {user_query}",
                "MIRA: Absolutely USER! I need the data listed below to address your query.",
                'Please express your data sharing and accessing preferences. The permission preferences options are "Yes, always share", "Yes, but ask me next time", "No, but ask me next time", and "No, never share". Be careful, Mira or apps may collect incorrect or unnecessary data!',
                "",
            ]
            for data_key, permission in selected:
                app_name, data_name = data_key.split(",", 1)
                app_name, data_name = app_name.strip(), data_name.strip()
                block_lines.append(
                    f"Q{q_num}. Data: {data_name}; App/MIRA: {app_name}"
                )
                value = (
                    permission.strip() if isinstance(permission, str) else permission
                )
                responses[f"Q{q_num}"] = value
                q_num += 1
                if q_num > 10:
                    break
            blocks.append("\n".join(block_lines))

        if q_num <= 10:
            raise ValueError("Unable to sample 10 questionnaire answers for one of the records.")

        # preferences spans every (query, data, app) the user answered in the
        # original task3 — typically 60-90 items per user — not just the 10
        # we sampled into questionnaire.
        preferences = {}
        for query_item in task3_items:
            user_query = query_item["query"].strip()
            for data_key, permission in query_item.get("answer", {}).items():
                app_name, data_name = data_key.split(",", 1)
                app_name, data_name = app_name.strip(), data_name.strip()
                value = (
                    permission.strip() if isinstance(permission, str) else permission
                )
                # Skip the year-2049 / MIRA scene-setting; preserve the
                # query + data + app trio that the question actually depends on.
                preferences[
                    f"You ask your personal AI assistant: '{user_query}'. To "
                    f"address this request, your AI assistant wants to access "
                    f"'{data_name}' from '{app_name}'. How willing are you to "
                    f"share this data? {AP_OPTIONS}"
                ] = value

        return QUESTIONNAIRE_PREFIX + "\n\n" + "\n\n".join(blocks), responses, preferences

    rows = []
    for user_id, record in enumerate(user_study, start=1):
        questionnaire, responses, preferences = build_questionnaire(
            record["task3"], seed=f"task3-{record.get('uuid', user_id)}"
        )
        age_code = record.get("age")
        rows.append(
            {
                "user_id": user_id,
                "domain": "AI agent permissions",
                "demographics": {
                    "education": record.get("education"),
                    "age": AGE_MAPPING.get(age_code, "unspecified age range"),
                    "gender": record.get("gender"),
                },
                "previous_experiences": {
                    label: record.get(field)
                    for field, label in PREVIOUS_EXPERIENCE_KEYS.items()
                },
                "privacy_attitudes": {
                    label: record.get(field)
                    for field, label in PRIVACY_ATTITUDE_KEYS.items()
                },
                "questionnaire": questionnaire,
                "responses": responses,
                "preferences": preferences,
            }
        )

    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# 4. Chatbot group chat (csienslab)
# ---------------------------------------------------------------------------
def process_chatbot_group_chat() -> Path:
    src = DATA_DIR / "chatbot_group_chat" / "Chatbot%20User%20Study%20Cleaned.csv"
    qid_src = DATA_DIR / "chatbot_group_chat" / "question_id_mapping.csv"
    out = DATA_DIR / "chatbot_group_chat" / "data_processed.csv"

    df = pd.read_csv(src)
    qid_mapping_df = pd.read_csv(qid_src)
    qid_to_text = dict(zip(qid_mapping_df["QuestionID"], qid_mapping_df["QuestionText"]))

    demographics_cols = [
        "QID63", "QID64", "QID64_4_TEXT", "QID65", "QID66", "QID67", "QID67_6_TEXT",
    ]
    previous_experiences_cols = [
        "QID5", "QID6", "QID7", "QID11_TEXT", "QID12", "QID13", "QID13_5_TEXT", "QID14",
        "QID16", "QID17", "QID19_TEXT", "QID20", "QID22", "QID23", "QID25_TEXT", "QID26",
    ]
    privacy_attitudes_cols = ["QID27", "QID28", "QID29", "QID30"]

    QID5_PLACEHOLDER = "[QID5-ChoiceGroup-SelectedChoices]"
    QID_TEXT_OVERRIDES = {
        "QID19_TEXT": "Regarding private group chats - Please elaborate on your answer to the previous question (how often you check for the presence of chatbots when joining private groups).",
        "QID25_TEXT": "Regarding public group chats - Please elaborate on your answer to the previous question (how often you check for the presence of chatbots when joining public groups).",
    }

    def resolve_question_text(col, row):
        text = QID_TEXT_OVERRIDES.get(col, qid_to_text.get(col, col))
        if QID5_PLACEHOLDER in text:
            qid5_value = row.get("QID5")
            if pd.notna(qid5_value):
                text = text.replace(QID5_PLACEHOLDER, str(qid5_value))
        return text

    responses_mapping = {
        "Q1": "QID33", "Q2": "QID34", "Q3": "QID35", "Q4": "QID36",
        "Q5": "QID38", "Q6": "QID39", "Q7": "QID40",
    }
    likert_cols = ["QID33", "QID34", "QID35", "QID36", "QID38", "QID39", "QID40"]
    likert_preamble = (
        "Please rate your level of agreement with the following statements about chatbots in group chats.\n"
        "Your options are 'Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', "
        "'Somewhat agree', and 'Strongly agree'.\n"
    )
    questionnaire_text = (
        likert_preamble
        + "\n"
        + "\n".join(f"Q{i}. {qid_to_text[c]}" for i, c in enumerate(likert_cols, start=1))
    )

    rows = []
    for idx, row in df.iterrows():
        def collect(cols):
            return {
                resolve_question_text(c, row): (row[c] if pd.notna(row[c]) else None)
                for c in cols
                if c in df.columns
            }

        demographics = collect(demographics_cols)
        previous_experiences = collect(previous_experiences_cols)
        privacy_attitudes = collect(privacy_attitudes_cols)

        responses = {
            q_key: (row[col] if pd.notna(row[col]) else None)
            for q_key, col in responses_mapping.items()
            if col in df.columns
        }

        cgc_options = (
            "Options: 'Strongly disagree', 'Somewhat disagree', "
            "'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree'."
        )
        preferences = {}
        for q_key, qid_col in responses_mapping.items():
            if qid_col not in df.columns or q_key not in responses:
                continue
            statement = qid_to_text.get(qid_col, qid_col)
            preferences[
                f"Please rate your level of agreement with the following "
                f"statement about chatbots in group chats: '{statement}'. "
                f"{cgc_options}"
            ] = responses[q_key]

        rows.append(
            {
                "user_id": idx + 1,
                "domain": "Chatbot group chat",
                "demographics": json.dumps(demographics, ensure_ascii=False),
                "previous_experiences": json.dumps(previous_experiences, ensure_ascii=False),
                "privacy_attitudes": json.dumps(privacy_attitudes, ensure_ascii=False),
                "questionnaire": questionnaire_text,
                "responses": json.dumps(responses, ensure_ascii=False),
                "preferences": json.dumps(preferences, ensure_ascii=False),
            }
        )

    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# 5. LLM conversational agents (OSF Answers.csv)
# ---------------------------------------------------------------------------
def process_llm_conversational_agent() -> Path:
    src = DATA_DIR / "llm_conversational_agent" / "Answers.csv"
    out = DATA_DIR / "llm_conversational_agent" / "data_processed.csv"
    df = pd.read_csv(src)

    def resolve_unique_column(frame, fragment):
        matches = [c for c in frame.columns if fragment in c]
        if not matches:
            raise KeyError(f"No column matched fragment: {fragment}")
        if len(matches) > 1:
            raise KeyError(f"Multiple columns matched fragment '{fragment}': {matches}")
        return matches[0]

    chatbot_usage_col = resolve_unique_column(
        df, "Have you ever used such chatbot service (e.g., ChatGPT, Gemini, Copilot)?"
    )
    df = df[df[chatbot_usage_col].astype(str).str.strip().str.lower() != "no"].reset_index(drop=True)

    demographics_columns = ["age", "country/region"]
    missing = [c for c in demographics_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing demographics columns in source dataframe: {missing}")

    previous_experience_fragments = [
        "Are you using your own local chatbot instance based on only locally stored LLM model(s)",
        "Among the following, for what purpose do you use your own local chatbot instance? - Selected Choice",
        "Among the following, for what purpose do you use your own local chatbot instance? - Other (please specifiy) - Text",
        "Why do you have your own local chatbot instance and not only use chatbot services (e.g., ChatGPT, Copilot)? - Selected Choice",
        "Why do you have your own local chatbot instance and not only use chatbot services (e.g., ChatGPT, Copilot)? - Other (please specifiy) - Text",
        "Please select the chatbots you usually use. - Selected Choice",
        "Please select the chatbots you usually use. - Other (please specify, use commas if multiple answers) - Text",
        "How frequently do you use chatbots?",
        "On the days you are interacting with chatbots, how long does this interactions last on average (per day)?",
        "Among the following devices, on which one do you usually interact with chatbots? Several selections are possible. - Selected Choice",
        "Among the following devices, on which one do you usually interact with chatbots? Several selections are possible. - Other (please specify) - Text",
        "Among the following, for what purpose do you usually use chatbots? Several selections are possible. - Selected Choice",
        "Among the following, for what purpose do you usually use chatbots? Several selections are possible. - Other - Text",
    ]
    privacy_attitudes_fragments = [
        "It usually bothers me when online companies ask me for personal information.",
        "When online companies ask me for personal information, I sometimes think twice before providing it.",
        "It bothers me to give personal information to so many online companies.",
        "I am concerned that online companies are collecting too much personal information about me.",
        "Consumer online privacy is really a matter of consumers",
        "Consumer control of personal information lies at the heart of consumer privacy.",
        "Companies seeking information online should disclose the way the data are collected, processed, and used.",
        "A good consumer online privacy policy should have a clear and conspicuous disclosure.",
    ]

    previous_experience_columns = [
        resolve_unique_column(df, f) for f in previous_experience_fragments
    ]
    privacy_attitudes_columns = [
        resolve_unique_column(df, f) for f in privacy_attitudes_fragments
    ]

    privacy_attitudes_scale = (
        "[Strongly agree, Disgree, Moderately disagree, Neither agree nor disagree, "
        "Moderately agree, Agree, Strongly Agree]"
    )
    pa_prompt_prefix = "Please indicate to what extent you agree with each of the following statements"

    def add_scale_to_privacy_label(label):
        if pa_prompt_prefix in label:
            return label.replace(
                pa_prompt_prefix,
                f"{pa_prompt_prefix} {privacy_attitudes_scale}",
                1,
            )
        return label

    privacy_attitudes_rename_map = {
        col: add_scale_to_privacy_label(col) for col in privacy_attitudes_columns
    }

    questionnaire_text = (
        "In the following part of the survey, we will ask you question about how frequently you share different type of personal data with chatbots. To evaluate this frequency, please think about the number of times where it would have been relevant to share a given type of data. For example, you should select 'I usually share' if you usually share a given type of data when it is relevant for your chatbot usage at the moment. If you never shared a given type of data in any context, please just answer 'I never shared'\n"
        "For all the following types of data related to identifiers and general information, please indicate how frequently you share such information with a chatbot.\n"
        "Your options are 'I never shared', 'I rarely shared', 'I sometimes shared', 'I often shared', and 'I usually shared'.\n"
        "Information about your...\n"
        "Q1. Personal identifiers (e.g., name, ID card number, email address, phone number)\n"
        "Q2. Account credentials (i.e., usernames, passwords)\n"
        "Q3. Location and mobility (e.g., homeplace, workplace, current location)\n"
        "Q4. Ethnicity and/or citizenship\n"
        "Q5. Criminal records\n\n"
        "For all the following types of data about standard of living and opinions, please indicate how frequently you share such information with a chatbot?\n"
        "Your options are 'I never shared', 'I rarely shared', 'I sometimes shared', 'I often shared', and 'I usually shared'.\n"
        "Information about your...\n"
        "Q6. Family life and relationship (e.g., civil status, number of children)\n"
        "Q7. Wealth details (e.g., salary, savings amount, land ownership)\n"
        "Q8. Job / Education\n"
        "Q9. Religion\n"
        "Q10. Political views"
    )

    response_fragment_map = {
        "Q1": "please indicate how frequently you share such information with a chatbot.\n\nInformation about your... - Personal identifiers",
        "Q2": "please indicate how frequently you share such information with a chatbot.\n\nInformation about your... - Account credentials",
        "Q3": "please indicate how frequently you share such information with a chatbot.\n\nInformation about your... - Location and mobility",
        "Q4": "please indicate how frequently you share such information with a chatbot.\n\nInformation about your... - Ethnicity and/or citizenship",
        "Q5": "please indicate how frequently you share such information with a chatbot.\n\nInformation about your... - Criminal records",
        "Q6": "please indicate how frequently you share such information with a chatbot?\n\n\n\nInformation about your... - Family life and relationship",
        "Q7": "please indicate how frequently you share such information with a chatbot?\n\n\n\nInformation about your... - Wealth details",
        "Q8": "please indicate how frequently you share such information with a chatbot?\n\n\n\nInformation about your... - Job / Education",
        "Q9": "please indicate how frequently you share such information with a chatbot?\n\n\n\nInformation about your... - Religion",
        "Q10": "please indicate how frequently you share such information with a chatbot?\n\n\n\nInformation about your... - Political views",
    }
    response_column_map = {
        q_key: resolve_unique_column(df, frag)
        for q_key, frag in response_fragment_map.items()
    }
    response_columns = list(response_column_map.values())

    # The original survey also asked two further question groups (lifestyle &
    # health, personal characteristics & emotions) — 10 more frequency items
    # per user. They aren't part of the LLM-facing questionnaire/responses,
    # but `preferences` should surface every item the user actually answered.
    extra_response_fragments = {
        "physical_health": ("Physical health condition (e.g., diagnosis, medication, symptoms, menstrual cycle details)", "lifestyle and health"),
        "mental_health": ("Mental health", "lifestyle and health"),
        "lifestyle_habits": ("Lifestyle and habits (e.g., sport activities, hobbies, sleeping time, eating habits)", "lifestyle and health"),
        "recreational": ("Recreational consumption (i.e., alcohol, smoking habits, recreational drugs)", "lifestyle and health"),
        "sexual_dating": ("Sexual and dating activities", "lifestyle and health"),
        "sexual_orientation": ("Sexual orientation", "personal characteristics and emotions"),
        "mental_state": ("Mental state and personality (e.g., mood, emotions, mindest, personality traits)", "personal characteristics and emotions"),
        "gender": ("Gender", "personal characteristics and emotions"),
        "age": ("Age (e.g., age category, age in years, birthdate)", "personal characteristics and emotions"),
        "physical_traits": ("Physical traits (e.g., size, skin/hair/eye color, weight)", "personal characteristics and emotions"),
    }
    extra_response_column_map = {
        key: (
            resolve_unique_column(
                df,
                f"please indicate how frequently you share such information with a chatbot.\n\n\n\nInformation about your... - {label}",
            ),
            label,
            group,
        )
        for key, (label, group) in extra_response_fragments.items()
    }
    extra_response_columns = [info[0] for info in extra_response_column_map.values()]

    response_values = df[response_columns]
    nan_like_mask = response_values.isna() | response_values.astype(str).apply(
        lambda col: col.str.strip().str.lower().eq("nan")
    )
    df = df[~nan_like_mask.any(axis=1)].reset_index(drop=True)

    # Per-question text for the preferences dict. Q1-Q5 share the
    # "identifiers / general information" framing; Q6-Q10 share the "standard
    # of living and opinions" framing. We repeat the framing in every key
    # because the user said comprehensive context per question.
    LCA_OPTIONS = (
        "Options: 'I never shared', 'I rarely shared', 'I sometimes shared', "
        "'I often shared', 'I usually shared'."
    )
    LCA_QUESTION_TEXT = {
        "Q1": "Personal identifiers (e.g., name, ID card number, email address, phone number)",
        "Q2": "Account credentials (i.e., usernames, passwords)",
        "Q3": "Location and mobility (e.g., homeplace, workplace, current location)",
        "Q4": "Ethnicity and/or citizenship",
        "Q5": "Criminal records",
        "Q6": "Family life and relationship (e.g., civil status, number of children)",
        "Q7": "Wealth details (e.g., salary, savings amount, land ownership)",
        "Q8": "Job / Education",
        "Q9": "Religion",
        "Q10": "Political views",
    }

    def _build_lca_preferences(record_dict: dict, extra_record: dict) -> dict:
        out_prefs: dict = {}
        for q_key in [f"Q{i}" for i in range(1, 11)]:
            value = record_dict.get(q_key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            data_type = LCA_QUESTION_TEXT[q_key]
            group_label = (
                "identifiers and general information"
                if q_key in {"Q1", "Q2", "Q3", "Q4", "Q5"}
                else "standard of living and opinions"
            )
            out_prefs[
                f"How frequently do you share information about your "
                f"{data_type} with a chatbot? Please think about the number "
                f"of times where it would have been relevant to share this "
                f"type of data, which falls under '{group_label}'. {LCA_OPTIONS}"
            ] = value
        # Add the lifestyle/health and characteristics/emotions items the
        # user also answered in the original survey but that aren't shown
        # to the LLM through questionnaire/responses.
        for _, (_, label, group_label) in extra_response_column_map.items():
            value = extra_record.get(label)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            out_prefs[
                f"How frequently do you share information about your "
                f"{label} with a chatbot? Please think about the number "
                f"of times where it would have been relevant to share this "
                f"type of data, which falls under '{group_label}'. {LCA_OPTIONS}"
            ] = value
        return out_prefs

    response_records = (
        df[response_columns]
        .rename(columns={v: k for k, v in response_column_map.items()})
        .to_dict(orient="records")
    )
    extra_records = (
        df[extra_response_columns]
        .rename(columns={info[0]: info[1] for info in extra_response_column_map.values()})
        .to_dict(orient="records")
    )

    out_df = pd.DataFrame(
        {
            "user_id": range(1, len(df) + 1),
            "domain": ["LLM conversational agents"] * len(df),
            "demographics": df[demographics_columns].to_dict(orient="records"),
            "previous_experiences": df[previous_experience_columns].to_dict(orient="records"),
            "privacy_attitudes": df[privacy_attitudes_columns]
            .rename(columns=privacy_attitudes_rename_map)
            .to_dict(orient="records"),
            "questionnaire": [questionnaire_text] * len(df),
            "responses": response_records,
            "preferences": [
                _build_lca_preferences(rec, extra)
                for rec, extra in zip(response_records, extra_records)
            ],
        },
        columns=OUTPUT_COLUMNS,
    )
    out_df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Stage 2: combine per-study CSVs, sample, add uuid -> data/dataset.csv
# ---------------------------------------------------------------------------
NUM_SAMPLE = 200

# Stable namespace so the user uuids reproduce across runs.
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "eval_llms_priv_norms/finalize_dataset")

_BAD_TOKENS = {
    "(skipped)", "prefer not to say", "not provided", "not provided.",
    "nan", "none", "null", "n/a", "na", "",
}


def _is_informative_previous_experiences(value):
    """Healthcare's previous_experiences is a stringified dict; only keep rows
    where at least one value is something other than 'skipped'/'nan'/etc."""
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return text.lower() not in _BAD_TOKENS

    if isinstance(parsed, dict):
        return any(str(v).strip().lower() not in _BAD_TOKENS for v in parsed.values())
    if isinstance(parsed, list):
        return any(str(v).strip().lower() not in _BAD_TOKENS for v in parsed)
    return str(parsed).strip().lower() not in _BAD_TOKENS


def _row_uuid5(row):
    """Deterministic uuid5 derived from (user_id, domain). Stable across runs
    so downstream artifacts can be joined back to a user identity."""
    key = f"{row.get('user_id', '')}||{row.get('domain', '')}"
    return str(uuid.uuid5(_UUID_NAMESPACE, key))


def _add_uuid_column(df: pd.DataFrame) -> pd.DataFrame:
    if "uuid" in df.columns:
        df = df.drop(columns=["uuid"])
    df.insert(0, "uuid", df.apply(_row_uuid5, axis=1))
    return df


def _load_combined_per_study() -> pd.DataFrame:
    """Load all five per-study data_processed.csv files and concat them.

    Order matches the original notebook's load order so dataset.csv (and
    downstream prompt_id) reproduces byte-for-byte after sampling.
    """
    csv_files = [
        DATA_DIR / "llm_health" / "data_processed.csv",
        DATA_DIR / "agent_permissions" / "data_processed.csv",
        DATA_DIR / "llm_chatbot" / "data_processed.csv",
        DATA_DIR / "chatbot_group_chat" / "data_processed.csv",
        DATA_DIR / "llm_conversational_agent" / "data_processed.csv",
    ]
    missing = [f for f in csv_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing per-study CSVs (run the 'process' stage first): {missing}"
        )
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def build_full_dataset(combined: pd.DataFrame | None = None) -> Path:
    """Write data/full_dataset.csv: every user from every per-study CSV, no
    filter and no sampling. Same schema as dataset.csv (uuid first column),
    just no row dropping.
    """
    out = DATA_DIR / "full_dataset.csv"
    if combined is None:
        combined = _load_combined_per_study()
    _add_uuid_column(combined.copy()).to_csv(out, index=False)
    return out


def finalize_dataset() -> Path:
    out = DATA_DIR / "dataset.csv"
    combined = _load_combined_per_study()

    # Side-effect: also emit the unfiltered, unsampled combined dataset so
    # users have access to the full per-study pool before our 200/domain cap.
    build_full_dataset(combined)

    sampled_parts = []
    for domain_name, domain_df in combined.groupby("domain", sort=False):
        source_df = domain_df
        if domain_name == "LLM healthcare consultation":
            # Restrict the experimental sample to higher-sensitivity attributes
            # (mental/medical) and to records with informative previous
            # experiences. full_dataset.csv keeps every healthcare user.
            mask = source_df["questionnaire"].str.contains(
                "your medical Information|your mental health information",
                regex=True, na=False,
            ) & source_df["previous_experiences"].apply(_is_informative_previous_experiences)
            source_df = source_df[mask]
        sample_n = min(NUM_SAMPLE, len(source_df))
        sampled_parts.append(source_df.sample(n=sample_n, random_state=42))

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    _add_uuid_column(sampled_df).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Stage 3: expand persona-facet powerset, build LLM prompts -> dataset_with_prompts*.csv
# ---------------------------------------------------------------------------
PERSONA_FIELDS = [
    ("demographics", "Demographics"),
    ("previous_experiences", "Previous AI Experiences"),
    ("privacy_attitudes", "Privacy Attitudes"),
]

THEORY_LENSES = {
    "privacy_calculus": """PRIVACY DECISION LENS — PRIVACY CALCULUS
Evaluate each scenario as this person performing a deliberate cost–benefit calculation about disclosure. For each request:
1. Identify the perceived BENEFITS of sharing — service utility, personalization quality, convenience, time saved, social or relational value.
2. Identify the perceived RISKS or COSTS of sharing — loss of control over the data, downstream reuse or profiling, embarrassment, identifiability, future harm, regulatory or employer exposure.
3. Weigh the two through the lens of this person's stated attitudes and prior experiences; their answer should reflect whichever side they judge to dominate.
4. When benefits clearly outweigh perceived risks for them, lean toward acceptance; when risks dominate, lean toward refusal; when the two are comparable, lean toward a conditional or intermediate option.
Pick the answer this person would arrive at after this tradeoff, not the answer a privacy-maximizing or utility-maximizing agent would give.""",
    "bounded_rationality": """PRIVACY DECISION LENS — BOUNDED RATIONALITY
Evaluate each scenario as this person under realistic cognitive constraints, not as a perfectly rational analyst:
- People rarely enumerate every risk. They rely on mental shortcuts: \"does this look like the kind of thing I usually allow?\", \"is this the default?\", \"the system asked, so it probably needs it\", \"everyone seems to use this\".
- They satisfice — accept the first option that feels good enough — rather than optimize.
- Stated privacy attitudes and concrete behavior can diverge (the privacy paradox). This person's answer in a specific scenario may not perfectly match their abstract stance, particularly when the request feels routine, low-friction, or socially normal.
- If the request feels familiar given their previous AI experiences, the heuristic favors acceptance even when their attitudes are cautious. If the request feels unfamiliar, intrusive, or salient, the heuristic flips toward refusal.
Pick the option this person — as a cognitively bounded, heuristic-driven decision-maker — would most plausibly land on in the moment, not the option a fully deliberative version of them would compute.""",
    "pmt": """PRIVACY DECISION LENS — PROTECTION MOTIVATION THEORY
Evaluate each scenario as this person running two parallel appraisals:

THREAT APPRAISAL
- Perceived severity: how harmful would exposure of this specific data be for this person, given their profile (e.g., medical, financial, identity, employment-relevant)?
- Perceived vulnerability: how likely does this person believe a harmful outcome — leak, misuse, profiling, secondary use — actually is in this context?

COPING APPRAISAL
- Response efficacy: does refusing, redacting, or limiting disclosure actually reduce the threat, in this person's belief?
- Self-efficacy: does this person feel able to refuse or push back without losing the service or the relationship?
- Response cost: what do they give up — utility, convenience, social standing, continuity of care — by protecting?

Combine the appraisals. High threat × high coping efficacy → protective behavior (refusal or restriction). Low threat, or high threat but low coping efficacy / high response cost → likely acceptance, sometimes reluctantly. Intermediate cases → conditional acceptance. Pick the answer this person — given their stated attitudes, demographics, and previous experiences — would arrive at after this combined appraisal.""",
}


def _clean_text(value):
    if pd.isna(value):
        return "Not provided."
    text = str(value).strip()
    return text if text else "Not provided."


def _is_informative_text(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in _BAD_TOKENS


def _placeholderize(obj):
    if isinstance(obj, dict):
        return {k: _placeholderize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_placeholderize(v) for v in obj]
    if isinstance(obj, bool):
        return "<boolean>"
    if isinstance(obj, int):
        return "<integer>"
    if isinstance(obj, float):
        return "<number>"
    if obj is None:
        return "<null_or_value>"
    return "<answer>"


def _format_structure_example(raw_value):
    raw_text = _clean_text(raw_value)
    if raw_text == "Not provided.":
        return '{"Q1": "<answer>", "Q2": "<answer>", "...": "<answer>"}'

    parsed = None
    try:
        parsed = json.loads(raw_text)
    except Exception:
        try:
            parsed = ast.literal_eval(raw_text)
        except Exception:
            parsed = None

    if parsed is None:
        return '{"Q1": "<answer>", "Q2": "<answer>", "...": "<answer>"}'
    return json.dumps(_placeholderize(parsed), indent=2, ensure_ascii=False)


def _powerset_columns(columns):
    return [c for r in range(len(columns) + 1) for c in combinations(columns, r)]


def _available_persona_columns(row):
    return [
        column_name
        for column_name, _ in PERSONA_FIELDS
        if _is_informative_text(row.get(column_name))
    ]


def _build_response_prompt(row, included_columns, theory_lens=None):
    questionnaire = _clean_text(row.get("questionnaire"))
    response_format_example = _format_structure_example(
        row.get("responses", row.get("response"))
    )
    included_set = set(included_columns)

    persona_lines = [
        "USER PRIVACY PERSONA",
        "Internalize the user's traits and answer each scenario as this specific person would. Draw on whichever aspects of their profile are most relevant to the particular scenario.",
        f"- Domain Context: {row.get('domain', 'Unknown')}",
    ]
    for column_name, display_name in PERSONA_FIELDS:
        value = (
            _clean_text(row.get(column_name))
            if column_name in included_set
            else "Not provided."
        )
        persona_lines.append(f"- {display_name}: {value}")
    persona_section = "\n".join(persona_lines)

    if theory_lens is None:
        lens_section = ""
    else:
        if theory_lens not in THEORY_LENSES:
            raise ValueError(
                f"Unknown theory_lens '{theory_lens}'. Expected one of {list(THEORY_LENSES)}."
            )
        lens_section = "\n\n" + THEORY_LENSES[theory_lens]

    return f"""ROLE
You are an expert AI simulator acting as a specific human participant in a research study. Your task is to realistically answer a questionnaire about data-sharing scenarios exactly as this specific person would.

{persona_section}{lens_section}

TASK & CONSTRAINTS
1. Answer all data-sharing scenarios in the QUESTIONNAIRE below from the perspective of the User Persona.
2. Return ONLY valid, parseable JSON.
3. DO NOT wrap your response in markdown code blocks (e.g., do not use ```json ... ```).
4. DO NOT add any commentary, greetings, or explanations outside the JSON object.
5. Keep field names, nesting, and data types strictly aligned with the TARGET OUTPUT FORMAT.

TARGET OUTPUT FORMAT (STRUCTURE ONLY - DO NOT COPY VALUES)
{response_format_example}

QUESTIONNAIRE
{questionnaire}
"""


def _build_prompts_dataframe(df, theory_lens=None):
    records = []
    for _, row in df.iterrows():
        base_record = row.to_dict()
        available_columns = _available_persona_columns(row)
        for included_cols in _powerset_columns(available_columns):
            prompt_type = "none" if not included_cols else "+".join(included_cols)
            record = base_record.copy()
            record["prompt_type"] = prompt_type
            record["available_columns"] = json.dumps(available_columns)
            record["included_columns"] = json.dumps(list(included_cols))
            record["llm_prompt"] = _build_response_prompt(
                row, list(included_cols), theory_lens=theory_lens
            )
            records.append(record)
    out_df = pd.DataFrame(records)
    if "prompt_id" in out_df.columns:
        out_df = out_df.drop(columns=["prompt_id"])
    out_df.insert(0, "prompt_id", range(1, len(out_df) + 1))
    return out_df


def gen_response_prompts() -> list[Path]:
    src = DATA_DIR / "dataset.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run the 'finalize' stage first."
        )
    df = pd.read_csv(src)

    written = []

    baseline_path = DATA_DIR / "dataset_with_prompts.csv"
    _build_prompts_dataframe(df, theory_lens=None).to_csv(baseline_path, index=False)
    written.append(baseline_path)

    for theory in THEORY_LENSES:
        theory_path = DATA_DIR / f"dataset_with_prompts_{theory}.csv"
        _build_prompts_dataframe(df, theory_lens=theory).to_csv(theory_path, index=False)
        written.append(theory_path)

    return written


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
PROCESSORS = {
    "llm_chatbot": process_llm_chatbot,
    "llm_health": process_llm_health,
    "agent_permissions": process_agent_permissions,
    "chatbot_group_chat": process_chatbot_group_chat,
    "llm_conversational_agent": process_llm_conversational_agent,
}

STAGES = ["process", "finalize", "prompts", "all"]


def _row_count(path: Path) -> int:
    return sum(1 for _ in path.open()) - 1


def _run_process(dataset_filter):
    selected = list(PROCESSORS) if "all" in dataset_filter else dataset_filter
    for name in selected:
        out = PROCESSORS[name]()
        print(f"[process:{name}] wrote {_row_count(out)} rows -> {out.relative_to(REPO_ROOT)}")


def _run_finalize():
    out = finalize_dataset()
    full_out = DATA_DIR / "full_dataset.csv"
    if full_out.exists():
        print(f"[finalize] wrote {_row_count(full_out)} rows -> {full_out.relative_to(REPO_ROOT)}")
    print(f"[finalize] wrote {_row_count(out)} rows -> {out.relative_to(REPO_ROOT)}")


def _run_prompts():
    for out in gen_response_prompts():
        print(f"[prompts] wrote {_row_count(out)} rows -> {out.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process raw user-study datasets, finalize the sampled dataset, and build LLM prompts."
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Pipeline stage to run (default: all = process -> finalize -> prompts).",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(PROCESSORS.keys()) + ["all"],
        default=["all"],
        help="Per-study filter for the 'process' stage (ignored otherwise).",
    )
    args = parser.parse_args()

    if args.stage in ("process", "all"):
        _run_process(args.dataset)
    if args.stage in ("finalize", "all"):
        _run_finalize()
    if args.stage in ("prompts", "all"):
        _run_prompts()


if __name__ == "__main__":
    main()
