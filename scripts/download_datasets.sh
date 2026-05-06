#!/usr/bin/env bash
# Download all six user-study datasets used by the project into their
# expected locations under data/. Skips any file that already exists so
# the script is safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

fetch() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "[skip] $dest already exists"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[get ] $dest"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

# 1. LLM-based chatbot privacy norms
fetch \
  "https://raw.githubusercontent.com/WeberLab-UW/AIES_2025_ChatbotPrivacy/refs/heads/main/data/clean/cleaned_factorial_vignette_survey_data_2025.csv" \
  "$DATA_DIR/llm_chatbot/cleaned_factorial_vignette_survey_data_2025.csv"

# 2. LLM health consultation
fetch \
  "https://raw.githubusercontent.com/Cristliu/LLMHealthPrivacy_UserStudy/refs/heads/main/redacted%20datasets%20and%20statistical%20analysis%20scripts/Data/REDACTED_Data.xlsx" \
  "$DATA_DIR/llm_health/REDACTED_Data.xlsx"

# 3. LLM-based agent permissions
fetch \
  "https://raw.githubusercontent.com/llm-platform-security/ai-agent-permissions/refs/heads/main/data/user_study.json" \
  "$DATA_DIR/agent_permissions/user_study.json"

# 4. Chatbot group chat (two files)
fetch \
  "https://raw.githubusercontent.com/csienslab/bot-among-us/refs/heads/main/Chatbot%20User%20Study%20Cleaned.csv" \
  "$DATA_DIR/chatbot_group_chat/Chatbot%20User%20Study%20Cleaned.csv"
fetch \
  "https://raw.githubusercontent.com/csienslab/bot-among-us/refs/heads/main/question_id_mapping.csv" \
  "$DATA_DIR/chatbot_group_chat/question_id_mapping.csv"

# 5. LLM-based conversational agent
fetch \
  "https://osf.io/2vqws/download" \
  "$DATA_DIR/llm_conversational_agent/Answers.csv"

echo "Done."
