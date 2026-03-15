#!/bin/bash
# Останавливает пайплайн Depth Pro (run_depthpro_to_sonata.py и дочерний процесс),
# чтобы не начиналось обучение (шаги 7–8). Данные шагов 1–4 (и при необходимости 5–6)
# остаются на диске.
# Запуск: ./scripts/stop_pipeline_before_training.sh

echo "Останавливаем пайплайн Depth Pro..."
PARENT=$(pgrep -f "run_depthpro_to_sonata.py" | head -1)
CHILD=$(pgrep -f "4_assign_labels_from_voxels.py" | head -1)

if [ -n "$CHILD" ]; then
  echo "  Останавливаем шаг 4 (PID $CHILD)..."
  kill "$CHILD" 2>/dev/null || true
  sleep 2
fi
if [ -n "$PARENT" ]; then
  echo "  Останавливаем run_depthpro_to_sonata.py (PID $PARENT)..."
  kill "$PARENT" 2>/dev/null || true
fi
sleep 1
if pgrep -f "run_depthpro_to_sonata.py" >/dev/null || pgrep -f "4_assign_labels" >/dev/null; then
  echo "  Процессы ещё работают, отправляем SIGKILL..."
  pkill -f "4_assign_labels_from_voxels.py" 2>/dev/null || true
  pkill -f "run_depthpro_to_sonata.py" 2>/dev/null || true
fi
echo "Готово. Пайплайн остановлен."
echo "Дальше: объединить код с main (./scripts/git_save_branch_and_pull_main.sh), затем при необходимости дозапустить шаги 5–6 и 7–8 вручную."
