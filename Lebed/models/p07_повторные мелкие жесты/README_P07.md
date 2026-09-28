# P07 — повторные мелкие движения

Скрипт `p07_v5_vscode.py` запускает **инференс P07** — поиск повторных мелких движений рук на видео.

Для обычного запуска модель уже обучена. Переобучать её не нужно.

---

## 1. Что должно лежать в папке проекта

Рекомендуемая структура:

```text
P07/
├── p07_v5_vscode.py
├── requirements.txt
│
├── models/
│   ├── pose_landmarker_heavy.task
│   ├── hand_landmarker.task
│   ├── p07_repetition_rf_v5.joblib
│   └── p07_repetition_rf_v5_calibration.json
│
├── videos/
│   └── my_video.mp4
│
└── output/
```

### Для чего нужны файлы

**`p07_v5_vscode.py`**  
Основной скрипт для запуска P07 в VS Code.

**`pose_landmarker_heavy.task`**  
Модель MediaPipe Pose. Нужна для точек плеч, локтей и кистей.

**`hand_landmarker.task`**  
Модель MediaPipe Hands. Нужна для точек кистей и пальцев.

**`p07_repetition_rf_v5.joblib`**  
Главный обученный классификатор P07.  
Внутри него находится Random Forest и сохранённые пороги START/END.  
**Для инференса этот файл обязателен.**

**`p07_repetition_rf_v5_calibration.json`**  
Отдельный файл с результатами калибровки.  
Для обычного инференса он не обязателен, потому что нужные пороги уже сохранены внутри `.joblib`. Его можно оставить в папке `models/` как дополнительный файл/резерв.

**Калибровочные видео**  
`Nikolaev...mp4`, `My_Interview.mp4` и другие обучающие видео нужны только для переобучения или повторной калибровки.  
Для обычного инференса они не нужны.

**Google Colab версия скрипта**  
Это запасной вариант для запуска в Google Colab.  
Если работаешь в VS Code на компьютере — запускай `p07_v5_vscode.py`.

---

## 2. Установить зависимости

Открой терминал VS Code в папке проекта:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Если `requirements.txt` нет:

```bash
pip install mediapipe opencv-python numpy scikit-learn joblib Pillow matplotlib
```

---

## 3. Положить видео

Видео, которое нужно обработать, положи в:

```text
videos/
```

Например:

```text
videos/my_video.mp4
```

---

## 4. Указать видео в скрипте

В начале `p07_v5_vscode.py` найди:

```python
TARGET_VIDEO = str(
    VIDEOS_DIR / "video6044313067708227319.mp4"
)
```

Замени только имя файла:

```python
TARGET_VIDEO = str(
    VIDEOS_DIR / "my_video.mp4"
)
```

---

## 5. Указать имена выходных файлов

Лучше дать выходным файлам имя твоего видео:

```python
OUTPUT_VIDEO = str(
    OUTPUT_DIR / "P07_my_video.mp4"
)

OUTPUT_CSV = str(
    OUTPUT_DIR / "P07_my_video_events.csv"
)

OUTPUT_SCORE_CSV = str(
    OUTPUT_DIR / "P07_my_video_scores.csv"
)
```

### Важно

В исходной версии `OUTPUT_CSV` и `OUTPUT_SCORE_CSV` имеют одинаковое имя.  
Из-за этого CSV с вероятностями может перезаписать CSV со списком событий.

Поэтому для запуска лучше использовать **два разных имени**, как в примере выше:

```text
P07_my_video_events.csv
P07_my_video_scores.csv
```

`OUTPUT_JSON` и `OUTPUT_GRAPH` менять не обязательно — они формируются автоматически от имени `OUTPUT_VIDEO`.

---

## 6. Проверить пути к моделям

Если папки расположены как в примере выше, эти строки менять не нужно:

```python
POSE_MODEL_PATH = str(
    MODELS_DIR / "pose_landmarker_heavy.task"
)

HAND_MODEL_PATH = str(
    MODELS_DIR / "hand_landmarker.task"
)

RF_MODEL_PATH = str(
    MODELS_DIR / "p07_repetition_rf_v5.joblib"
)
```

---

## 7. Запустить инференс

В терминале:

```bash
python p07_v5_vscode.py
```

---

## 8. Что появится после обработки

В папке `output/` будут:

```text
P07_my_video.mp4
```

Видео с оверлеем P07 и временем начала/окончания события.

```text
P07_my_video_events.csv
```

Список найденных P07-событий:
- начало;
- конец;
- длительность.

```text
P07_my_video_scores.csv
```

Вероятность P07 во времени и используемые START/END thresholds.

```text
P07_my_video_results.json
```

Итоговые события P07 в JSON.

```text
P07_my_video_p07_timeline.png
```

График вероятности P07 с отмеченными START/END.

---

## Самое короткое руководство

Для нового видео нужно сделать только это:

1. Положить видео в `videos/`.
2. Положить модели в `models/`.
3. В `TARGET_VIDEO` написать имя нового видео.
4. Задать разные имена для `OUTPUT_CSV` и `OUTPUT_SCORE_CSV`.
5. Запустить:

```bash
python p07_v5_vscode.py
```

Для обычного инференса обучающие видео и Google Colab версия скрипта не нужны.

p.s. на всякий случай приложен файл .ipynb инференса для google collab
