#!/bin/bash
# 1) Создать ветку с вашими изменениями и закоммитить только код репо (без dataset и т.п.)
# 2) Подтянуть обновления с origin/main и влить их в вашу ветку
# Запускать из корня репозитория: cd /workspace/sonata-workspace && ./scripts/git_save_branch_and_pull_main.sh

set -e
BRANCH="${1:-pipeline-and-baseline}"

echo "=== 1) Текущая ветка и статус ==="
git branch
git status -s | head -20

echo ""
echo "=== 1.5) Git user (если ещё не задан) ==="
if ! git config user.email >/dev/null 2>&1; then
  git config user.email "user@local"
  git config user.name "Local User"
  echo "  Задан локальный user.email/user.name (только для этого репо). При желании смените: git config user.email '...'"
fi
echo ""
echo "=== 2) Переходим на ветку $BRANCH (создаём, если нет) и коммитим только файлы репозитория ==="
git rev-parse --verify "$BRANCH" >/dev/null 2>&1 && git checkout "$BRANCH" || git checkout -b "$BRANCH"

# Добавляем только файлы внутри репо (не ../dataset и не соседние каталоги)
git add \
  .cursorignore \
  .gitignore \
  README_BASELINE.md \
  set_paths.sh \
  data/map_from_scans.py \
  data/semantickitti.py \
  docs/PIPELINE_DEPTH_PRO_TO_SONATA.md \
  scripts/run_depthpro_to_sonata.py \
  scripts/run_lidar_ground_truth_and_train.sh \
  scripts/reset_step2_and_rerun.py \
  scripts/watch_pipeline_progress.sh \
  scripts/export_pointcloud_to_ply.py \
  scripts/git_save_branch_and_pull_main.sh \
  VoxFormerDepthPro/paths_config.py \
  VoxFormerDepthPro/scripts/2_run_depth_pro.py \
  VoxFormerDepthPro/scripts/3_depth_to_pointcloud.py \
  2>/dev/null || true
# Доп. файлы, если есть
git add VoxFormerDepthPro/run_pipeline.sh 2>/dev/null || true

git status -s
if git diff --cached --quiet 2>/dev/null; then
  echo "Нет изменений для коммита (возможно уже закоммичено). Продолжаем."
else
  git commit -m "Pipeline Depth Pro, LiDAR baseline, map_only_map_world, README_BASELINE, watch script, .gitignore"
fi

echo ""
echo "=== 3) Подтягиваем обновления с origin/main ==="
git fetch origin main
git merge origin/main -m "Merge origin/main into $BRANCH" || {
  echo "Конфликты при слиянии. Разрешите их вручную:"
  echo "  git status"
  echo "  # отредактируйте файлы с конфликтами, затем:"
  echo "  git add ."
  echo "  git commit -m 'Resolve merge with main'"
  exit 1
}

echo ""
echo "Готово. Вы на ветке $BRANCH с вашими изменениями + обновлениями из main."
echo "Переключиться на main: git checkout main"
echo "Снова на вашу ветку:  git checkout $BRANCH"
