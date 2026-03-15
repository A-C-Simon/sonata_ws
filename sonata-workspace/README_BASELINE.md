# Sonata-LiDiff: бейзлайн LiDAR и пайплайн RGB → Depth Pro

Два варианта: обучение на **лидарных** данных SemanticKITTI и на облаках из **RGB через Depth Pro**.

---

## Общее

- **Переменные:** `dataset` — корень с данными (например SemanticKITTI); при необходимости `OUT` — корень выходов.  
Пример: `export dataset=/workspace/dataset OUT=/workspace/dataset`.
- Все команды выполняются из **корня репозитория** `sonata-workspace`.

---

# Часть 1. Бейзлайн на LiDAR (полный план)

Обучение на **оригинальном лидаре** SemanticKITTI (velodyne), без RGB и Depth Pro. Используется вокселизованный кэш на основе глобальной карты `map_world` (план B) для ускорения обучения.

## Входные данные

- `$dataset/SemanticKITTI/dataset/sequences/00..10/` — velodyne (`.bin`), labels (`.label`), `poses.txt`, `calib.txt`.

## Выход

- Карты: `ground_truth/XX/map_world.npz` (одна на seq).
- Кэш: `voxelized_cache/XX/{scan_id}.npz` (один на кадр).
- Чекпоинты: `checkpoints/diffusion_lidar`, `checkpoints/refinement_lidar`.

## План реализации — вариант 1: минимум места, один проход map_from_scans


| #   | Шаг                              | Описание                                                                                                                           |
| --- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **map_from_scans**               | Один раз с `--save_only_map_world` → `ground_truth/XX/map_world.npz`.                                                              |
| 2   | **precompute_voxelized_dataset** | С `--backend torch` (и при желании `--max_map_points`): карта → СК каждого кадра, вокселизация, запись `voxelized_cache/XX/*.npz`. |
| 3   | **train_diffusion**              | С `--voxelized_cache_dir` (датасет читает готовые `.npz`).                                                                         |
| 4   | **train_refinement**             | На том же датасете.                                                                                                                |


Подробнее про шаги 1–2 и формат кэша: `docs/VOXELIZED_CACHE_PIPELINE.md` и `docs/LIDAR_BASELINE_VOXELIZATION.md`.

---

## Запуск бейзлайна LiDAR

### 1. Путь к датасету

```bash
cd /workspace/sonata-workspace
export dataset=/workspace/dataset
```

### 2. Карты по лидару (шаг 1)

Один `map_world.npz` на seq (рекомендуется):

```bash
python data/map_from_scans.py \
  -p "$dataset/SemanticKITTI/dataset/sequences" \
  -o "$dataset/SemanticKITTI/dataset" \
  --voxel_size 0.1 \
  --backend torch \
  --save_only_map_world \
  --sequences 00 01 02 03 04 05 06 07 08 09 10
```

Без `--save_only_map_world` создаётся по одному `.npz` на каждый скан (очень много места).

### 3. Вокселизованный кэш (шаг 2, план B)

После появления `ground_truth/XX/map_world.npz`:

```bash
python scripts/run_voxelized_cache_semantickitti.py \
  --data_path "$dataset/SemanticKITTI/dataset" \
  --skip_map
```

Если карт ещё нет — без `--skip_map` скрипт сначала вызовет map_from_scans, затем прекомпут. Для ускорения прекомпута на GPU:

```bash
python scripts/run_voxelized_cache_semantickitti.py \
  --data_path "$dataset/SemanticKITTI/dataset" \
  --skip_map \
  --backend torch
```

Подробный вывод по кадрам: запуск напрямую `data/precompute_voxelized_dataset.py` с `--verbose` (см. `docs/VOXELIZED_CACHE_PIPELINE.md`).

### 4. Обучение диффузии (шаг 3)

С кэшем обучение идёт быстрее (без вокселизации в датасете):

```bash
python training/train_diffusion.py \
  --data_path "$dataset/SemanticKITTI/dataset" \
  --voxelized_cache_dir "$dataset/SemanticKITTI/dataset/voxelized_cache" \
  --output_dir checkpoints/diffusion_lidar \
  --log_dir logs/diffusion_lidar
```

### 5. Обучение refinement (шаг 4)

```bash
python training/train_refinement.py \
  --data_path "$dataset/SemanticKITTI/dataset" \
  --output_dir checkpoints/refinement_lidar \
  --log_dir logs/refinement_lidar
```

### Всё одной командой (скрипт)

Полный пайплайн (карты → кэш → диффузия → refinement):

```bash
export dataset=/workspace/dataset
python scripts/run_lidar_to_sonata.py
```

В скрипте уже используются `--save_only_map_world`, прекомпут кэша и `--voxelized_cache_dir` при обучении диффузии.

---

# Часть 2. Пайплайн RGB → Depth Pro → Sonata

Обучение на облаках, полученных из **RGB** через оценку глубины (Depth Pro) и конвертацию в точки. Вход — изображения SemanticKITTI (`image_2`), не лидар.

## Схема пайплайна

```
SemanticKITTI (image_2, labels, poses, calib)
    → [1] preprocess/labels/*.npy
    → [2] depth/sequences/*.npy        ← Depth Pro по RGB
    → [3] lidar_pro/sequences/*.bin   ← глубины → облака
    → [4] lidar_pro_labeled/labels/*.label
    → [5] sonata_depth_pro/sequences/ (velodyne + labels + poses + calib)
    → [6] sonata_depth_pro/ground_truth/ (map_from_scans)
    → [7] (опционально) voxelized_cache для sonata_depth_pro
    → [8–9] обучение Sonata-LiDiff (diffusion + refinement)
```

## Шаги по порядку


| #   | Шаг                           | Вход                                | Выход                                                     |
| --- | ----------------------------- | ----------------------------------- | --------------------------------------------------------- |
| 1   | Препроцессинг меток (воксели) | KITTI labels, calib, poses          | `VoxFormerDepthPro/preprocess/labels/00..10/*.npy`        |
| 2   | **Depth Pro по RGB**          | `image_2/*.png`                     | `VoxFormerDepthPro/depth/sequences/00..10/*.npy`          |
| 3   | Глубины → облака              | depth + calib                       | `VoxFormerDepthPro/lidar_pro/sequences/*.bin`             |
| 4   | Метки на облаках              | lidar_pro + preprocess              | `VoxFormerDepthPro/lidar_pro_labeled/labels/*.label`      |
| 5   | Сборка датасета Sonata        | lidar_pro, lidar_pro_labeled, KITTI | `$OUT/sonata_depth_pro/sequences/00..10/`                 |
| 6   | Ground-truth карты            | sonata_depth_pro/sequences          | `sonata_depth_pro/ground_truth/` (map_world или per-scan) |
| 7   | (опц.) Вокселизованный кэш    | как в бейзлайне LiDAR               | `sonata_depth_pro/voxelized_cache/`                       |
| 8   | Обучение диффузии             | sonata_depth_pro                    | `checkpoints/diffusion_depthpro/`                         |
| 9   | Обучение refinement           | sonata_depth_pro                    | `checkpoints/refinement_depthpro/`                        |


## Запуск пайплайна Depth Pro

Полный прогон (шаги 1–9):

```bash
cd /workspace/sonata-workspace
export dataset=/workspace/dataset OUT=/workspace/dataset
python scripts/run_depthpro_to_sonata.py
```

Скрипт пропускает уже готовые шаги (например, не перезапускает Depth Pro для seq, где есть полный набор `.npy`).

**Требования:** установленный Depth Pro (например `pip install .` в `/workspace/ml-depth-pro`), чекпоинт `depth_pro.pt` в `DEPTH_PRO_ROOT` (по умолчанию `/workspace/ml-depth-pro`). При необходимости: `export DEPTH_PRO_ROOT=/path/to/ml-depth-pro`.

## Перезапуск только шага 2 (Depth Pro) для части seq

Если шаг 2 оборвался или нужно пересчитать глубины для 00 и 01:

```bash
rm -rf $OUT/VoxFormerDepthPro/depth/sequences/00
rm -rf $OUT/VoxFormerDepthPro/depth/sequences/01
export dataset=/workspace/dataset OUT=/workspace/dataset
python scripts/run_depthpro_to_sonata.py
```

## Ground-truth для sonata_depth_pro (шаг 6)

После шагов 2–5 можно строить карты по облакам из Depth Pro. Экономный вариант — только `map_world.npz` на seq:

```bash
python data/map_from_scans.py \
  -p "$OUT/sonata_depth_pro/sequences" \
  -o "$OUT/sonata_depth_pro" \
  --voxel_size 0.1 --backend torch \
  --sequences 00 01 02 03 04 05 06 07 08 09 10 \
  --save_only_map_world
```

При полном прогоне `run_depthpro_to_sonata.py` шаг 6 вызывается автоматически (при необходимости добавьте в скрипт `--save_only_map_world` и/или прекомпут кэша по аналогии с LiDAR).

## Где что лежит (Depth Pro)


| Что                   | Путь                                                                      |
| --------------------- | ------------------------------------------------------------------------- |
| Глубины по RGB        | `$OUT/VoxFormerDepthPro/depth/sequences/00..10/`                          |
| Облака из глубин      | `$OUT/VoxFormerDepthPro/lidar_pro/sequences/00..10/`                      |
| Метки на облаках      | `$OUT/VoxFormerDepthPro/lidar_pro_labeled/labels/00..10/`                 |
| Датасет для обучения  | `$OUT/sonata_depth_pro/sequences/`, `$OUT/sonata_depth_pro/ground_truth/` |
| Чекпоинты (Depth Pro) | `checkpoints/diffusion_depthpro/`, `checkpoints/refinement_depthpro/`     |


Подробнее про пайплайн Depth Pro: `docs/PIPELINE_DEPTH_PRO_TO_SONATA.md`.

---

# Сводка: LiDAR vs RGB (Depth Pro)


|                 | Бейзлайн (LiDAR)                                                                  | Пайплайн (RGB → Depth Pro)                                                                           |
| --------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Вход            | velodyne (лидар)                                                                  | image_2 (RGB)                                                                                        |
| Карты / датасет | SemanticKITTI/dataset + ground_truth + voxelized_cache                            | sonata_depth_pro + ground_truth (и при необходимости voxelized_cache)                                |
| Запуск          | map_from_scans → precompute_voxelized_dataset → train_* с `--voxelized_cache_dir` | `run_depthpro_to_sonata.py` или шаги 2–6 вручную, затем train_* с `--data_path .../sonata_depth_pro` |
| Чекпоинты       | diffusion_lidar, refinement_lidar                                                 | diffusion_depthpro, refinement_depthpro                                                              |


Бейзлайн LiDAR можно запускать сразу при наличии velodyne; пайплайн Depth Pro — после установки Depth Pro и при наличии RGB. Оба варианта используют один и тот же код обучения (train_diffusion, train_refinement), различается только `--data_path` и источник облаков (лидар или Depth Pro).