# P04 — жестовый эпизод

Скрипт `Вставленный код(20260928-154548).py` запускает инференс P04 по видео с использованием MediaPipe Pose Heavy и определяет жестовые эпизоды по движению кистей и локтей.

## 1. Установить зависимости

В терминале VS Code:

```bash
pip install mediapipe opencv-python numpy Pillow
```

## 2. Подготовить файлы

Нужны:

- входное видео;
- модель `pose_landmarker_heavy.task`.

## 3. Указать пути в начале скрипта

Изменить:

```python
VIDEO_PATH = r"C:\path\to\video.mp4"

POSE_MODEL_PATH = r"C:\path\to\pose_landmarker_heavy.task"

OUTPUT_VIDEO = r"C:\path\to\output\p04_result.mp4"

OUTPUT_CSV = r"C:\path\to\output\p04_result.csv"

OUTPUT_JSON = r"C:\path\to\output\p04_result_results.json"
```

## 4. Запустить

```bash
python "Вставленный код(20260928-154548).py"
```

## Результат

После обработки будут сохранены:

- `.mp4` — видео с оверлеем;
- `.csv` — список жестовых эпизодов с временем начала, окончания, длительностью и длиной пути;
- `.json` — события P04 в формате `bs_profiling_cut_v1`.

## Важно

Основные параметры детектора находятся в начале скрипта:

```python
GESTURE_START_SPEED = 0.18
GESTURE_END_SPEED = 0.075

START_HOLD_FRAMES = 3
END_HOLD_FRAMES = 7

MIN_EPISODE_SECONDS = 0.35
MIN_EPISODE_PATH = 0.08

MIN_VISIBILITY = 0.50
```

Для обычного инференса их менять не нужно, если используется текущая откалиброванная версия.

Если нужен предпросмотр во время обработки:

```python
SHOW_PREVIEW = True
```

По умолчанию:

```python
SHOW_PREVIEW = False
```
