# P09 — смена позы / раскачивание корпуса

Скрипт `p09_shoulders_v2_with_json_and_frequency.py` запускает инференс P09 по видео с использованием MediaPipe Pose Heavy и готового `.joblib` bundle.

## 1. Установить зависимости

В терминале VS Code:

```bash
pip install mediapipe opencv-python numpy pandas joblib tqdm matplotlib scikit-learn
```

## 2. Подготовить файлы

Нужны:

- входное видео;
- `pose_landmarker_heavy.task`;
- обученный bundle `shoulders_v3_clean.joblib`.

## 3. Указать пути в начале скрипта

Изменить:

```python
TARGET_VIDEO_PATH = r"C:\path\to\video.mp4"

POSE_LANDMARKER_PATH = r"C:\path\to\pose_landmarker_heavy.task"

V2_MODEL_PATH = r"C:\path\to\shoulders_v3_clean.joblib"

OUTPUT_VIDEO = r"C:\path\to\output\p09_result.mp4"

OUTPUT_CSV = r"C:\path\to\output\p09_result.csv"
```

JSON и PNG-график создаются автоматически рядом с `OUTPUT_CSV`.

## 4. Указать нейтральный интервал

Нужно выбрать участок видео, где человек спокойно сидит без смены позы.

Например:

```python
TARGET_NEUTRAL_INTERVALS = [
    (2.0, 4.0),
]
```

Здесь `2.0–4.0` — секунды видео.

Если отдельная нейтраль для раскачивания не нужна, оставить:

```python
TARGET_ROCKING_NEUTRAL_INTERVALS = None
```

## 5. Запустить

```bash
python "p09_shoulders_v2_with_json_and_frequency(2).py"
```

## Результат

После обработки будут сохранены:

- `.mp4` — видео с оверлеем;
- `.csv` — диагностические данные по кадрам;
- `_results.json` — события P09 с временем начала/окончания;
- `_p09_frequency.png` — график частоты смен позы.

## Важно

Для нового видео обязательно проверь `TARGET_NEUTRAL_INTERVALS`. Неправильно выбранный нейтральный участок может изменить адаптивные пороги и привести к ложным событиям P09.

Данный момент можно использовать в готовом продукте, в начале интервью обычно всегода говорят - "посмотрите в камеру, расслабьтесь" и т.д.
