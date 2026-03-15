# Как появляются датасет VoxFormerDepthPro и sonata_depth_pro

Пайплайн: **SemanticKITTI (RGB)** → **Depth Pro** → **облака точек** → **разметка из вокселей** → **датасет Sonata** → **карты и обучение**.

Переменные окружения: `dataset` — корень с SemanticKITTI, `OUT` — корень выходов (часто тот же путь).  
Пример: `export dataset=/workspace/dataset OUT=/workspace/dataset`.

---

## Входные данные

- **SemanticKITTI**: `$dataset/SemanticKITTI/dataset/sequences/00..10/`
  - `image_2/` — RGB-кадры
  - `velodyne/` — лидар (для эталонов, не для нашего пайплайна)
  - `labels/` — семантические метки (нужны для вокселей)
  - `poses.txt`, `calib.txt`

---

## 1) Датасет VoxFormerDepthPro (`$OUT/VoxFormerDepthPro/`)

Всё, что лежит в `VoxFormerDepthPro`, — это промежуточные и итоговые артефакты пайплайна до обучения Sonata.

### Шаг 1 — препроцессинг меток (воксели)

- **Скрипт:** `VoxFormerDepthPro/scripts/1_prepare_labels.py`
- **Вход:** SemanticKITTI `labels/`, `calib.txt`, `poses.txt`
- **Выход:** `VoxFormerDepthPro/preprocess/labels/00..10/*.npy`  
  Разметка в формате вокселей для последующего переноса на облака.

### Шаг 2 — Depth Pro по RGB (карты глубины)

- **Скрипт:** `VoxFormerDepthPro/scripts/2_run_depth_pro.py` (вызов из репозитория ml-depth-pro)
- **Вход:** `SemanticKITTI/dataset/sequences/XX/image_2/*.png`
- **Выход:** `VoxFormerDepthPro/depth/sequences/00..10/*.npy`  
  Один файл глубины на кадр.  
  При нехватке места или падении процесса шаг можно перезапустить: пайплайн доделает только те последовательности, где число `.npy` меньше числа изображений (или можно удалить папки `depth/sequences/00` и `01` и перезапустить шаг 2 «с нуля»).

### Шаг 3 — глубины → облака точек

- **Скрипт:** `VoxFormerDepthPro/scripts/3_depth_to_pointcloud.py`
- **Вход:** `VoxFormerDepthPro/depth/sequences/XX/*.npy` и `SemanticKITTI/.../calib.txt`
- **Выход:** `VoxFormerDepthPro/lidar_pro/sequences/00..10/*.bin`  
  Облака в формате KITTI (x,y,z, intensity и т.д.).

### Шаг 4 — разметка из вокселей на облаках

- **Скрипт:** `VoxFormerDepthPro/scripts/4_assign_labels_from_voxels.py`
- **Вход:** `lidar_pro/sequences/` (облака) и **SemanticKITTI** `dataset/sequences/XX/voxels/*.label` (воксельные метки).
- **Выход:** `VoxFormerDepthPro/lidar_pro_labeled/labels/00..10/*.label`  
  Семантические метки для каждой точки (по соответствию с вокселями).

**Важно:** воксельные метки берутся из папки `voxels/` датасета SemanticKITTI. В официальном датасете SSC-разметка (воксели) есть только для **каждого 5-го кадра** (000000, 000005, 000010, …). Поэтому для каждой последовательности получается примерно 1/5 от числа кадров (например, ~909 из 4541 для seq 00). Кадры без файла в `voxels/` скрипт пропускает — для них `.label` не создаётся. Это ограничение датасета, а не ошибка пайплайна.

Итог раздела: **датасет VoxFormerDepthPro** — это каталог `$OUT/VoxFormerDepthPro/` с подкаталогами `preprocess/`, `depth/`, `lidar_pro/`, `lidar_pro_labeled/`. Они появляются последовательно на шагах 1–4.

---

## 2) Датасет sonata_depth_pro (`$OUT/sonata_depth_pro/`)

Это уже формат, в котором Sonata-LiDiff обучается: те же последовательности, но собранные в одну структуру с картами и путями, ожидаемыми скриптами обучения.

### Шаг 5 — сборка дерева датасета Sonata

- **Код:** в `run_depthpro_to_sonata.py` (шаг 5), без отдельного скрипта.
- **Вход:**
  - облака: `VoxFormerDepthPro/lidar_pro/sequences/XX/*.bin`
  - метки: `VoxFormerDepthPro/lidar_pro_labeled/labels/XX/*.label`
  - позы и калибровка: `SemanticKITTI/dataset/sequences/XX/poses.txt`, `calib.txt`
- **Выход:** `$OUT/sonata_depth_pro/sequences/00..10/`
  - `velodyne/` — симлинки или копии `.bin` из `lidar_pro`
  - `labels/` — симлинки или копии `.label` из `lidar_pro_labeled`
  - `poses.txt`, `calib.txt` — копии из KITTI

Так **появляется сам датасет sonata_depth_pro**: он просто собирается из уже готовых артефактов VoxFormerDepthPro и KITTI.

### Шаг 6 — карты для обучения (map_from_scans)

- **Скрипт:** `data/map_from_scans.py`
- **Вход:** `sonata_depth_pro/sequences/` (облака, метки, позы, калибровка)
- **Выход:** в `sonata_depth_pro/`:
  - `.pkl` и/или другие артефакты карт
  - каталоги/файлы вида `gt_*` — ground-truth карты для обучения

После шага 6 датасет sonata_depth_pro готов для шагов 7–8 (диффузия и дообучение).

---

## Краткая схема

```
SemanticKITTI (image_2, labels, poses, calib)
    → [1] preprocess/labels/*.npy
    → [2] depth/sequences/*.npy        ← Depth Pro по RGB (шаг, который падал из‑за памяти)
    → [3] lidar_pro/sequences/*.bin
    → [4] lidar_pro_labeled/labels/*.label
    → [5] sonata_depth_pro/sequences/ (velodyne + labels + poses + calib)
    → [6] sonata_depth_pro/*.pkl, gt_* (map_from_scans)
    → [7–8] обучение Sonata-LiDiff
```

---

## Два сценария: LiDAR сейчас и Depth Pro потом

Можно не ждать завершения шагов 2–3–4: сначала обучить Sonata-LiDiff на **лидарном** датасете, затем использовать результаты Depth Pro отдельно.

### A. LiDAR-датасет (SemanticKITTI velodyne) — можно запускать сразу

1. **Сгенерировать ground truth по лидару:**
   ```bash
   cd /workspace/sonata-workspace
   export dataset=/workspace/dataset
   DATA="$dataset"
   python data/map_from_scans.py \
     -p "$DATA/SemanticKITTI/dataset/sequences" \
     -o "$DATA/SemanticKITTI/dataset" \
     -v 0.1 -b torch \
     -s 00 01 02 03 04 05 06 07 08 09 10
   ```
   Выход: `$DATA/SemanticKITTI/dataset/ground_truth/` (карты по оригинальным velodyne).

2. **Обучить Sonata-LiDiff с корнем по LiDAR:**
   ```bash
   cd /workspace/sonata-workspace
   python training/train_diffusion.py \
     --data_path "$DATA/SemanticKITTI/dataset" \
     --output_dir checkpoints/diffusion_lidar \
     --log_dir logs/diffusion_lidar
   python training/train_refinement.py \
     --data_path "$DATA/SemanticKITTI/dataset" \
     --output_dir checkpoints/refinement_lidar \
     --log_dir logs/refinement_lidar
   ```

Так вы получаете модель, обученную на лидарных картах, без ожидания Depth Pro.

### B. Depth Pro — после шагов 2–3–4–5

Когда шаги 2–3–4 (и сборка 5) будут готовы:

1. **Ground truth по облакам из Depth Pro:**
   ```bash
   python data/map_from_scans.py \
     -p "$OUT/sonata_depth_pro/sequences" \
     -o "$OUT/sonata_depth_pro" -v 0.1 -b torch \
     -s 00 01 02 03 04 05 06 07 08 09 10
   ```

2. **Обучить (или дообучить) с корнем sonata_depth_pro:**
   ```bash
   python training/train_diffusion.py \
     --data_path "$OUT/sonata_depth_pro" \
     --output_dir checkpoints/diffusion_depthpro \
     --log_dir logs/diffusion_depthpro
   python training/train_refinement.py ...
   ```

Итого: сначала запускаете map_from_scans и обучение на LiDAR; параллельно или позже прогоняете 2–3–4–5 и используете sonata_depth_pro для обучения/информации с Depth Pro.

---

## Перезапуск шага 2 с нуля (например, после нехватки памяти)

Чтобы заново прогнать Depth Pro только для последовательностей 00 и 01:

```bash
export dataset=/workspace/dataset OUT=/workspace/dataset
rm -rf /workspace/dataset/VoxFormerDepthPro/depth/sequences/00
rm -rf /workspace/dataset/VoxFormerDepthPro/depth/sequences/01
cd /workspace/sonata-workspace
python scripts/run_depthpro_to_sonata.py
```

Пайплайн пропустит шаг 1 (препроцессинг уже есть), заново выполнит шаг 2 для 00 и 01, затем шаги 3–4 (при необходимости), 5 и дальше. Чтобы перезапустить только шаг 2 вручную для одной последовательности:

```bash
cd /workspace/ml-depth-pro
python /workspace/sonata-workspace/VoxFormerDepthPro/scripts/2_run_depth_pro.py --seq 00 --device cuda
```

(Аналогично для `--seq 01`.)
