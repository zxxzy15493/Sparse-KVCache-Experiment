cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
LOG_TIME=$(date '+%m%d_%H%M')
LOG_FILE="./eff_log/quest_${LOG_TIME}.log"
mkdir -p ./eff_log
bash ./scripts/llama.sh >"${LOG_FILE}" 2>&1



LOG_TIME=$(date '+%m%d_%H%M')
LOG_FILE="./eff_log/quest_${LOG_TIME}.log"
mkdir -p ./eff_log
bash ./scripts/qwen.sh >"${LOG_FILE}" 2>&1



LOG_TIME=$(date '+%m%d_%H%M')
LOG_FILE="./eff_log/quest_${LOG_TIME}.log"
mkdir -p ./eff_log
bash ./scripts/budget_llama.sh >"${LOG_FILE}" 2>&1



LOG_TIME=$(date '+%m%d_%H%M')
LOG_FILE="./eff_log/quest_${LOG_TIME}.log"
mkdir -p ./eff_log
bash ./scripts/budget_qwen.sh >"${LOG_FILE}" 2>&1


