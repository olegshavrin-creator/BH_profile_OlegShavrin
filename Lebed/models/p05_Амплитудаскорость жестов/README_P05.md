# P05 — амплитуда / скорость жестов

Скрипт `p05_mediapipe_pose_with_json_and_frequency_GRAPH_BEAUTIFUL.py` запускает инференс P05 по видео с использованием MediaPipe Pose Heavy.

## 1. Установить зависимости

В терминале VS Code:

```bash
pip install mediapipe opencv-python numpy Pillow matplotlib
```

## 2. Подготовить файлы

Нужны:

- входное видео;
- модель `pose_landmarker_heavy.task`.

## 3. Указать пути в начале скрипта

Изменить:

```python
VIDEO_PATH = r"C:\path\to\video.mp4"

MODEL_PATH = r"C:\path\to\pose_landmarker_heavy.task"

OUTPUT_PATH = r"C:\path\to\output\p05_result.mp4"
```

JSON и PNG-график создаются автоматически рядом с `OUTPUT_PATH`.

## 4. Запустить

```bash
python "p05_mediapipe_pose_with_json_and_frequency_GRAPH_BEAUTIFUL.py"
```

Во время обработки открывается окно предпросмотра.

- `Q` — остановить обработку;
- `ESC` — остановить обработку.

## Результат

После обработки будут сохранены:

- `.mp4` — видео с оверлеем;
- `_results.json` — события P05 с временем начала/окончания, амплитудой и скоростью;
- `_gesture_frequency.png` — график частоты жестикуляции.

## Важно

Основные пороги детектора находятся в начале скрипта:

```python
GESTURE_START_SPEED = 0.18
GESTURE_STOP_SPEED = 0.08
STOP_HOLD_FRAMES = 5
MIN_VISIBILITY = 0.50
SPEED_ALPHA = 0.30
```

Для обычного инференса их менять не нужно, если используется текущая откалиброванная версия.
