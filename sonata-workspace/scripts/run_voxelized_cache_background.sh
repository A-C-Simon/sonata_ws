#!/usr/bin/env bash
# Запуск вокселизации в фоне: можно закрыть терминал, процесс продолжит работу.
# Лог пишется в logs/voxelized_cache_<timestamp>.log
#
# Пример:
#   cd /workspace/sonata-workspace
#   ./scripts/run_voxelized_cache_background.sh --data_path /workspace/dataset/SemanticKITTI/dataset --skip_map --skip_existing
#
# Смотреть прогресс: tail -f logs/voxelized_cache_*.log

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/voxelized_cache_${STAMP}.log"

echo "Log file: $LOG_FILE"
echo "Command: python scripts/run_voxelized_cache_semantickitti.py $*"
echo "Run: tail -f $LOG_FILE"
echo ""

# nohup: игнорируем SIGHUP, вывод в лог, процесс в фоне
cd "$REPO_ROOT"
nohup python scripts/run_voxelized_cache_semantickitti.py "$@" >> "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "${LOG_DIR}/voxelized_cache_${STAMP}.pid"
echo "PID: $PID — можно закрыть терминал. Прогресс: tail -f $LOG_FILE"
