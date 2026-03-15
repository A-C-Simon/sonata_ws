# Sonata-LiDiff: бейзлайн (LiDAR) и пайплайн Depth Pro

Описание двух вариантов запуска: обучение на **лидарных** данных SemanticKITTI и на данных, полученных из **RGB через Depth Pro**.

---

## Общее

- **Переменные:** `dataset` — корень с SemanticKITTI; `OUT` — корень выходов (часто тот же путь).  
  Пример: `export dataset=/workspace/dataset OUT=/workspace/dataset`.
- Все команды — из **корня репозитория** `sonata-workspace`.

---

# Часть 1. Бейзлайн на LiDAR

Обучение на **оригинальном лидаре** SemanticKITTI (velodyne), без RGB и Depth Pro.

## Что делаем

1. Строим ground-truth карты из лидарных сканов (`map_from_scans`).
2. Обучаем диффузию и refinement по датасету с этими картами.

**Вход:** `$dataset/SemanticKITTI/dataset/sequences/00..10/` (velodyne, labels, poses, calib).  
**Выход:** чекпоинты `checkpoints/diffusion_lidar`, `checkpoints/refinement_lidar`.

## Запуск бейзлайна

### 1. Путь к датасету

```bash
cd /workspace/sonata-workspace
export dataset=/workspace/dataset
```

### 2. Ground-truth по лидару (map_from_scans)

**Экономный режим (рекомендуется)** — только `map_world.npz` на seq, несколько GB вместо сотен GB:

```bash
python data/map_from_scans.py \
  -p "$dataset/SemanticKITTI/dataset/sequences" \
  -o "$dataset/SemanticKITTI/dataset" \
  -v 0.1 -b torch \
  -s 00 01 02 03 04 05 06 07 08 09 10 \
  --save_only_map_world
```

Без `--save_only_map_world` создаётся по одному .npz на каждый скан (очень много места).

### 3. Обучение по LiDAR-датасету

```bash
python training/train_diffusion.py \
  --data_path "$dataset/SemanticKITTI/dataset" \
  --output_dir checkpoints/diffusion_lidar \
  --log_dir logs/diffusion_lidar

python training/train_refinement.py \
  --data_path "$dataset/SemanticKITTI/dataset" \
  --output_dir checkpoints/refinement_lidar \
  --log_dir logs/refinement_lidar
```

### Одной командой (скрипт)

```bash
export dataset=/workspace/dataset
./scripts/run_lidar_ground_truth_and_train.sh
```

При необходимости добавьте в скрипт `--save_only_map_world` в вызов `map_from_scans.py`.

---

# Часть 2. Пайплайн Depth Pro (RGB → Sonata)

Обучение на облаках, полученных из **RGB** через оценку глубины (Depth Pro) и конвертацию в точки.

## Схема пайплайна

```
SemanticKITTI (image_2, labels, poses, calib)
    → [1] preprocess/labels/*.npy
    → [2] depth/sequences/*.npy        ← Depth Pro по RGB
    → [3] lidar_pro/sequences/*.bin
    → [4] lidar_pro_labeled/labels/*.label
    → [5] sonata_depth_pro/sequences/ (velodyne + labels + poses + calib)
    → [6] sonata_depth_pro/ground_truth/ (map_from_scans)
    → [7–8] обучение Sonata-LiDiff
```

## Шаги по порядку

| # | Шаг | Вход | Выход |
|---|-----|------|--------|
| 1 | Препроцессинг меток (воксели) | KITTI labels, calib, poses | `VoxFormerDepthPro/preprocess/labels/00..10/*.npy` |
| 2 | **Depth Pro по RGB** | `image_2/*.png` | `VoxFormerDepthPro/depth/sequences/00..10/*.npy` |
| 3 | Глубины → облака | depth + calib | `VoxFormerDepthPro/lidar_pro/sequences/*.bin` |
| 4 | Метки на облаках | lidar_pro + preprocess | `VoxFormerDepthPro/lidar_pro_labeled/labels/*.label` |
| 5 | Сборка датасета Sonata | lidar_pro, lidar_pro_labeled, KITTI | `$OUT/sonata_depth_pro/sequences/00..10/` |
| 6 | Ground-truth карты | sonata_depth_pro/sequences | `sonata_depth_pro/ground_truth/` |
| 7 | Обучение диффузии | sonata_depth_pro | `checkpoints/diffusion_depthpro/` |
| 8 | Обучение refinement | sonata_depth_pro | `checkpoints/refinement_depthpro/` |

## Запуск пайплайна Depth Pro

Полный прогон (шаги 1–8):

```bash
cd /workspace/sonata-workspace
export dataset=/workspace/dataset OUT=/workspace/dataset
python scripts/run_depthpro_to_sonata.py
```

Скрипт сам пропускает уже готовые шаги (например, не перезапускает Depth Pro для seq, где уже есть полный набор .npy).

**Требования:** установленный Depth Pro (например `pip install .` в `/workspace/ml-depth-pro`), чекпоинт `depth_pro.pt` в `DEPTH_PRO_ROOT` (по умолчанию `/workspace/ml-depth-pro`). При необходимости задайте `export DEPTH_PRO_ROOT=/path/to/ml-depth-pro`.

## Перезапуск только шага 2 (Depth Pro) для части seq

Если шаг 2 оборвался или нужно пересчитать глубины для 00 и 01:

```bash
rm -rf $OUT/VoxFormerDepthPro/depth/sequences/00
rm -rf $OUT/VoxFormerDepthPro/depth/sequences/01
export dataset=/workspace/dataset OUT=/workspace/dataset
python scripts/run_depthpro_to_sonata.py
```

Или используйте скрипт:

```bash
export dataset=/workspace/dataset OUT=/workspace/dataset
python scripts/reset_step2_and_rerun.py 00 01
```

После этого пайплайн заново выполнит шаг 2 для 00 и 01, затем 3–4–5 (скрипты сами пропустят готовые seq), потом 6–7–8.

## Ground-truth для sonata_depth_pro (шаг 6)

После шагов 2–5 можно строить карты по облакам из Depth Pro. Экономный вариант — только `map_world.npz` на seq:

```bash
python data/map_from_scans.py \
  -p "$OUT/sonata_depth_pro/sequences" \
  -o "$OUT/sonata_depth_pro" \
  -v 0.1 -b torch \
  -s 00 01 02 03 04 05 06 07 08 09 10 \
  --save_only_map_world
```

Если пайплайн запущен до конца, шаг 6 вызывается из `run_depthpro_to_sonata.py` автоматически (без `--save_only_map_world`; при желании можно добавить флаг в скрипт).

## Где что лежит (Depth Pro)

| Что | Путь |
|-----|------|
| Глубины | `$OUT/VoxFormerDepthPro/depth/sequences/00..10/` |
| Облака из глубин | `$OUT/VoxFormerDepthPro/lidar_pro/sequences/00..10/` |
| Метки на облаках | `$OUT/VoxFormerDepthPro/lidar_pro_labeled/labels/00..10/` |
| Датасет для обучения | `$OUT/sonata_depth_pro/sequences/`, `$OUT/sonata_depth_pro/ground_truth/` |
| Чекпоинты (Depth Pro) | `checkpoints/diffusion_depthpro/`, `checkpoints/refinement_depthpro/` |

---

# Сводка: LiDAR vs Depth Pro

| | Бейзлайн (LiDAR) | Пайплайн (Depth Pro) |
|---|------------------|------------------------|
| Вход | velodyne (лидар) | image_2 (RGB) |
| Карты / датасет | SemanticKITTI/dataset + ground_truth | sonata_depth_pro + ground_truth |
| Запуск | map_from_scans → train_* с `--data_path .../SemanticKITTI/dataset` | `run_depthpro_to_sonata.py` или шаги 2–6 вручную, затем train_* с `--data_path .../sonata_depth_pro` |
| Чекпоинты | diffusion_lidar, refinement_lidar | diffusion_depthpro, refinement_depthpro |

Бейзлайн можно запускать сразу; пайплайн Depth Pro — после установки Depth Pro и при наличии RGB. Оба варианта используют один и тот же код обучения (train_diffusion, train_refinement), различается только `--data_path` и источник облаков (лидар или Depth Pro).
