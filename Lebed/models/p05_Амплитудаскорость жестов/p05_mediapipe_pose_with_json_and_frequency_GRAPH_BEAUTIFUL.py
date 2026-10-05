#pip install matplotlib
import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


# ============================================================
# НАСТРОЙКИ
# ============================================================

VIDEO_PATH = r"D:\Стажировка Братские сердца\Мультимодалка\Видео от команды\video.mp4"

# MediaPipe Pose Landmarker
MODEL_PATH = r"C:\interview\openface\pose_landmarker_heavy.task"

OUTPUT_PATH = r"C:\interview\openface\video_geasture.mp4"

OUTPUT_JSON = str(
    Path(OUTPUT_PATH).with_name(
        Path(OUTPUT_PATH).stem + "_results.json"
    )
)

OUTPUT_FREQUENCY_GRAPH = str(
    Path(OUTPUT_PATH).with_name(
        Path(OUTPUT_PATH).stem + "_gesture_frequency.png"
    )
)

# Аналитика частоты жестикуляции.
# Каждую секунду смотрим, сколько новых жестов началось
# за предыдущие 20 секунд, и переводим это в жесты/мин.
# На сам P05 detector это НЕ влияет.
FREQUENCY_WINDOW_SEC = 20.0
FREQUENCY_STEP_SEC = 1.0

RUN_STARTED_PERF = time.perf_counter()



# ------------------------------------------------------------
# ПОРОГИ ЖЕСТА
#
# Скорость нормализована относительно ширины плеч.
#
# Например:
# speed = 0.20 означает примерно
# 0.20 ширины плеч в секунду.
# ------------------------------------------------------------

GESTURE_START_SPEED = 0.18
GESTURE_STOP_SPEED = 0.08

# Сколько кадров скорость должна быть ниже STOP,
# чтобы считать жест законченным
STOP_HOLD_FRAMES = 5

# Минимальная visibility точки
MIN_VISIBILITY = 0.50

# Сглаживание скорости.
# Больше значение -> быстрее реагирует.
# Меньше -> сильнее сглаживает.
SPEED_ALPHA = 0.30

# Прозрачность панели
PANEL_ALPHA = 0.50


# ============================================================
# LANDMARK ID
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def distance_2d(a, b):
    """
    Евклидово расстояние между двумя точками x/y.
    """
    return np.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def landmark_xy(landmark):
    return np.array([
        landmark.x,
        landmark.y
    ], dtype=np.float32)


def get_visibility(landmark):
    """
    Не у всех реализаций landmark visibility обязательно присутствует,
    поэтому используем безопасный вариант.
    """
    return getattr(landmark, "visibility", 1.0)


def find_windows_font():
    """
    Ищем шрифт Windows с поддержкой кириллицы.
    """

    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            return font_path

    return None


# ============================================================
# СОСТОЯНИЕ ОДНОЙ РУКИ
# ============================================================

class HandGestureTracker:

    def __init__(self):
        self.prev_wrist = None

        self.smoothed_speed = 0.0

        self.is_gesture = False
        self.stop_counter = 0

        self.gesture_count = 0

        # ----------------------------------------------------
        # Параметры текущего жеста
        # ----------------------------------------------------

        self.start_position = None

        self.min_x = None
        self.max_x = None

        self.min_y = None
        self.max_y = None

        self.current_amplitude = 0.0

        self.current_path = 0.0

        self.peak_speed = 0.0

        # Последний завершённый жест
        self.last_amplitude = 0.0
        self.last_peak_speed = 0.0
        self.last_path = 0.0


    def reset_current_gesture(self, wrist):

        self.start_position = wrist.copy()

        self.min_x = wrist[0]
        self.max_x = wrist[0]

        self.min_y = wrist[1]
        self.max_y = wrist[1]

        self.current_amplitude = 0.0
        self.current_path = 0.0
        self.peak_speed = 0.0


    def update(
        self,
        wrist,
        shoulder_width,
        fps,
        visible=True
    ):

        # ----------------------------------------------------
        # Если кисть плохо видна
        # ----------------------------------------------------

        if not visible or shoulder_width <= 0:

            self.prev_wrist = None

            return {
                "speed": 0.0,
                "amplitude": self.current_amplitude,
                "path": self.current_path,
                "peak_speed": self.peak_speed,
                "gesture": self.is_gesture,
            }


        # ----------------------------------------------------
        # Первый кадр
        # ----------------------------------------------------

        if self.prev_wrist is None:

            self.prev_wrist = wrist.copy()

            return {
                "speed": 0.0,
                "amplitude": self.current_amplitude,
                "path": self.current_path,
                "peak_speed": self.peak_speed,
                "gesture": self.is_gesture,
            }


        # ----------------------------------------------------
        # Перемещение кисти между кадрами
        # ----------------------------------------------------

        frame_distance = distance_2d(
            wrist,
            self.prev_wrist
        )


        # ----------------------------------------------------
        # НОРМАЛИЗАЦИЯ
        #
        # frame_distance / shoulder_width
        #
        # Получаем перемещение относительно ширины плеч.
        # ----------------------------------------------------

        normalized_distance = (
            frame_distance / shoulder_width
        )


        # ----------------------------------------------------
        # Скорость:
        #
        # расстояние за кадр * FPS
        #
        # Единица:
        # "ширин плеч в секунду"
        # ----------------------------------------------------

        raw_speed = normalized_distance * fps


        # ----------------------------------------------------
        # EMA сглаживание скорости
        # ----------------------------------------------------

        self.smoothed_speed = (
            SPEED_ALPHA * raw_speed
            +
            (1.0 - SPEED_ALPHA) * self.smoothed_speed
        )


        speed = self.smoothed_speed


        # ====================================================
        # НАЧАЛО ЖЕСТА
        # ====================================================

        if not self.is_gesture:

            if speed >= GESTURE_START_SPEED:

                self.is_gesture = True

                self.stop_counter = 0

                self.gesture_count += 1

                self.reset_current_gesture(wrist)


        # ====================================================
        # ЖЕСТ ИДЁТ
        # ====================================================

        if self.is_gesture:

            # Путь кисти
            self.current_path += normalized_distance

            # Минимальные / максимальные координаты
            self.min_x = min(self.min_x, wrist[0])
            self.max_x = max(self.max_x, wrist[0])

            self.min_y = min(self.min_y, wrist[1])
            self.max_y = max(self.max_y, wrist[1])


            # ------------------------------------------------
            # Амплитуда текущего жеста
            # ------------------------------------------------

            dx = (
                self.max_x -
                self.min_x
            ) / shoulder_width

            dy = (
                self.max_y -
                self.min_y
            ) / shoulder_width


            self.current_amplitude = np.sqrt(
                dx ** 2 +
                dy ** 2
            )


            # ------------------------------------------------
            # Пиковая скорость
            # ------------------------------------------------

            self.peak_speed = max(
                self.peak_speed,
                speed
            )


            # =================================================
            # ПРОВЕРКА ЗАВЕРШЕНИЯ
            # =================================================

            if speed <= GESTURE_STOP_SPEED:

                self.stop_counter += 1

            else:

                self.stop_counter = 0


            if self.stop_counter >= STOP_HOLD_FRAMES:

                self.is_gesture = False

                self.stop_counter = 0

                self.last_amplitude = self.current_amplitude
                self.last_peak_speed = self.peak_speed
                self.last_path = self.current_path


        self.prev_wrist = wrist.copy()


        return {
            "speed": speed,
            "amplitude": self.current_amplitude,
            "path": self.current_path,
            "peak_speed": self.peak_speed,
            "gesture": self.is_gesture,
        }


# ============================================================
# РИСОВАНИЕ SKELETON
# ============================================================

def draw_pose_arm(
    frame,
    shoulder,
    elbow,
    wrist,
    width,
    height
):

    points = []

    for p in [shoulder, elbow, wrist]:

        x = int(p.x * width)
        y = int(p.y * height)

        points.append((x, y))


    # Соединения
    cv2.line(
        frame,
        points[0],
        points[1],
        (255, 255, 255),
        max(1, width // 500)
    )

    cv2.line(
        frame,
        points[1],
        points[2],
        (255, 255, 255),
        max(1, width // 500)
    )


    # Landmarks
    radius = max(3, width // 250)

    for point in points:

        cv2.circle(
            frame,
            point,
            radius,
            (255, 255, 255),
            -1
        )


# ============================================================
# РУССКИЙ ДИНАМИЧЕСКИЙ OVERLAY
# ============================================================

def draw_overlay(
    frame,
    left_data,
    right_data,
    left_count,
    right_count
):

    height, width = frame.shape[:2]


    # ========================================================
    # ДИНАМИЧЕСКИЙ РАЗМЕР
    # ========================================================

    # Размер шрифта относительно высоты видео
    font_size = int(height * 0.0176)

    font_size = max(
        13,
        min(font_size, 27)
    )


    title_font_size = int(font_size * 1.15)


    font_path = find_windows_font()

    if font_path is None:
        raise RuntimeError(
            "Не найден шрифт Windows с поддержкой русского языка."
        )


    font = ImageFont.truetype(
        font_path,
        font_size
    )

    title_font = ImageFont.truetype(
        font_path,
        title_font_size
    )


    # ========================================================
    # ПАРАМЕТРЫ ПАНЕЛИ
    # ========================================================

    panel_width = int(width * 0.256)

    panel_width = max(
        248,
        min(panel_width, 480)
    )


    margin = int(
        min(width, height) * 0.020
    )

    padding = int(
        font_size * 0.8
    )

    line_height = int(
        font_size * 1.45
    )


    lines_count = 12

    panel_height = (
        padding * 2
        +
        line_height * lines_count
    )


    # --------------------------------------------------------
    # Справа сверху
    # --------------------------------------------------------

    x1 = width - panel_width - margin
    y1 = margin

    x2 = width - margin
    y2 = y1 + panel_height


    # ========================================================
    # ПРОЗРАЧНАЯ ПАНЕЛЬ
    # ========================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )


    frame = cv2.addWeighted(
        overlay,
        PANEL_ALPHA,
        frame,
        1.0 - PANEL_ALPHA,
        0
    )


    # ========================================================
    # PIL ДЛЯ КИРИЛЛИЦЫ
    # ========================================================

    image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )

    draw = ImageDraw.Draw(image)


    text_x = x1 + padding
    text_y = y1 + padding


    # ========================================================
    # ЗАГОЛОВОК
    # ========================================================

    draw.text(
        (text_x, text_y),
        "P05 — Амплитуда / скорость жестов",
        font=title_font,
        fill=(255, 255, 255)
    )

    text_y += int(
        line_height * 1.5
    )


    # ========================================================
    # ЛЕВАЯ РУКА
    # ========================================================

    left_state = (
        "ЖЕСТ"
        if left_data["gesture"]
        else "покой"
    )


    left_lines = [

        "Левая рука",

        f"Скорость: {left_data['speed']:.2f}",

        f"Амплитуда: {left_data['amplitude']:.2f}",

        f"Пик скорости: {left_data['peak_speed']:.2f}",

        f"Состояние: {left_state}",
    ]


    for i, text in enumerate(left_lines):

        current_font = (
            title_font
            if i == 0
            else font
        )

        draw.text(
            (text_x, text_y),
            text,
            font=current_font,
            fill=(255, 255, 255)
        )

        text_y += line_height


    text_y += int(
        line_height * 0.35
    )


    # ========================================================
    # ПРАВАЯ РУКА
    # ========================================================

    right_state = (
        "ЖЕСТ"
        if right_data["gesture"]
        else "покой"
    )


    right_lines = [

        "Правая рука",

        f"Скорость: {right_data['speed']:.2f}",

        f"Амплитуда: {right_data['amplitude']:.2f}",

        f"Пик скорости: {right_data['peak_speed']:.2f}",

        f"Состояние: {right_state}",
    ]


    for i, text in enumerate(right_lines):

        current_font = (
            title_font
            if i == 0
            else font
        )

        draw.text(
            (text_x, text_y),
            text,
            font=current_font,
            fill=(255, 255, 255)
        )

        text_y += line_height


    text_y += int(
        line_height * 0.25
    )


    # ========================================================
    # СЧЁТЧИК
    # ========================================================

    draw.text(
        (text_x, text_y),
        f"Жестов слева: {left_count}",
        font=font,
        fill=(255, 255, 255)
    )

    text_y += line_height

    draw.text(
        (text_x, text_y),
        f"Жестов справа: {right_count}",
        font=font,
        fill=(255, 255, 255)
    )


    # PIL -> OpenCV
    frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    return frame



# ============================================================
# P05 -> JSON / АНАЛИТИКА ЧАСТОТЫ
# ============================================================
#
# ВАЖНО:
# Ниже добавлена только регистрация результатов.
# Исходная логика P05 НЕ меняется:
# GESTURE_START_SPEED, GESTURE_STOP_SPEED, STOP_HOLD_FRAMES,
# MIN_VISIBILITY, SPEED_ALPHA и HandGestureTracker.update()
# остаются прежними.
# ============================================================

def format_human_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def human_duration(seconds):
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    if hours > 0:
        return f"{hours}ч {minutes}м {secs}с"
    if minutes > 0:
        return f"{minutes}м {secs}с"
    return f"{secs}с"


class P05JsonTracker:
    """
    Пассивно наблюдает за уже готовым tracker.is_gesture.

    START/END определяются НЕ этим классом, а существующим
    HandGestureTracker. Поэтому добавление JSON не меняет
    точность исходного P05.
    """

    def __init__(self):
        self.previous_state = {
            "left": False,
            "right": False,
        }

        self.active = {
            "left": None,
            "right": None,
        }

        self.cuts = []


    def _start(self, hand, timestamp, data):
        self.active[hand] = {
            "start_sec": float(timestamp),
            "speed_samples": [
                float(data.get("speed", 0.0))
            ],
        }


    def _append_speed(self, hand, data):
        current = self.active[hand]

        if current is None:
            return

        # Если pose/рука на этом кадре реально обновились,
        # сохраняем уже рассчитанную сглаженную скорость.
        if bool(data.get("gesture", False)):
            current["speed_samples"].append(
                float(data.get("speed", 0.0))
            )


    def _finish(self, hand, timestamp, tracker, forced=False):
        current = self.active[hand]

        if current is None:
            return

        start_sec = float(current["start_sec"])
        end_sec = max(start_sec, float(timestamp))
        duration_sec = max(0.0, end_sec - start_sec)

        speed_samples = np.asarray(
            current["speed_samples"],
            dtype=np.float64
        )

        mean_speed = (
            float(np.mean(speed_samples))
            if len(speed_samples) > 0
            else 0.0
        )

        if forced:
            amplitude = float(tracker.current_amplitude)
            peak_speed = float(tracker.peak_speed)
            path = float(tracker.current_path)
        else:
            # Эти значения исходный HandGestureTracker
            # уже сохранил при обычном END.
            amplitude = float(tracker.last_amplitude)
            peak_speed = float(tracker.last_peak_speed)
            path = float(tracker.last_path)

        hand_name = (
            "левая рука"
            if hand == "left"
            else "правая рука"
        )

        self.cuts.append(
            {
                "source": "mediapipe",
                "pattern": "P05",
                "name": "амплитуда/скорость жестов",

                "hand": hand,
                "hand_name": hand_name,

                "t_start": round(start_sec, 3),
                "t_end": round(end_sec, 3),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(duration_sec, 3),

                # Всё, что уже измеряет исходный скрипт.
                "amplitude": round(amplitude, 6),
                "mean_speed": round(mean_speed, 6),
                "peak_speed": round(peak_speed, 6),
                "path": round(path, 6),

                "speed_unit": "shoulder_widths_per_second",
                "amplitude_unit": "shoulder_widths",
                "path_unit": "shoulder_widths",

                # Поля совместимости с Alyona_interview_results.json
                "clip_pad_sec": 0.0,
                "clip_file": None,
                "clip": None,
                "scenario_label": None,
                "scenario_kind": None,
            }
        )

        self.active[hand] = None


    def update(
        self,
        timestamp,
        left_tracker,
        right_tracker,
        left_data,
        right_data
    ):
        for hand, tracker, data in [
            ("left", left_tracker, left_data),
            ("right", right_tracker, right_data),
        ]:
            current_state = bool(tracker.is_gesture)
            previous_state = bool(self.previous_state[hand])

            if current_state and not previous_state:
                self._start(
                    hand,
                    timestamp,
                    data
                )

            elif current_state and previous_state:
                self._append_speed(
                    hand,
                    data
                )

            elif (not current_state) and previous_state:
                self._finish(
                    hand,
                    timestamp,
                    tracker,
                    forced=False
                )

            self.previous_state[hand] = current_state


    def finalize(
        self,
        timestamp,
        left_tracker,
        right_tracker
    ):
        if self.active["left"] is not None:
            self._finish(
                "left",
                timestamp,
                left_tracker,
                forced=True
            )

        if self.active["right"] is not None:
            self._finish(
                "right",
                timestamp,
                right_tracker,
                forced=True
            )

        self.cuts.sort(
            key=lambda item: (
                item["start_sec"],
                item["hand"]
            )
        )


def build_gesture_frequency(
    cuts,
    duration_sec,
    window_sec=FREQUENCY_WINDOW_SEC,
    step_sec=FREQUENCY_STEP_SEC
):
    """
    Частота жестикуляции = сколько НОВЫХ жестов началось
    за предыдущее окно времени, пересчитанное в жесты/мин.

    Например, window=20 сек:
    4 старта за последние 20 сек -> 12 жестов/мин.
    """

    duration_sec = max(0.0, float(duration_sec))

    starts = {
        "left": np.asarray(
            [
                cut["start_sec"]
                for cut in cuts
                if cut["hand"] == "left"
            ],
            dtype=np.float64
        ),
        "right": np.asarray(
            [
                cut["start_sec"]
                for cut in cuts
                if cut["hand"] == "right"
            ],
            dtype=np.float64
        ),
    }

    sample_times = np.arange(
        0.0,
        duration_sec + step_sec,
        step_sec,
        dtype=np.float64
    )

    if len(sample_times) == 0:
        sample_times = np.asarray([0.0])

    points = []

    for t in sample_times:
        window_start = t - window_sec

        left_count = int(
            np.sum(
                (starts["left"] > window_start)
                &
                (starts["left"] <= t)
            )
        )

        right_count = int(
            np.sum(
                (starts["right"] > window_start)
                &
                (starts["right"] <= t)
            )
        )

        left_rate = left_count * 60.0 / window_sec
        right_rate = right_count * 60.0 / window_sec
        total_rate = left_rate + right_rate

        points.append(
            {
                "time_sec": round(float(t), 3),
                "time_human": format_human_time(t),
                "left_gestures_per_min": round(left_rate, 3),
                "right_gestures_per_min": round(right_rate, 3),
                "total_gestures_per_min": round(total_rate, 3),
            }
        )

    return {
        "method": "trailing_rolling_window",
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "unit": "gestures_per_minute",
        "meaning": (
            "Число новых жестов, начавшихся за предыдущие "
            f"{window_sec:.0f} секунд, пересчитанное в жесты/мин."
        ),
        "points": points,
    }




def format_axis_time(
    seconds
):
    """
    Формат времени для оси X строго MM:SS.
    Примеры: 00:30, 01:00, 01:30.
    """
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def save_frequency_graph(
    frequency_data,
    output_path
):
    """
    Более красивый и понятный график частоты жестикуляции.

    Что изменено:
    - по оси X подписи через 30 секунд;
    - формат времени MM:SS: 00:30, 01:00, 01:30;
    - ось Y оставлена без смысловых изменений;
    - более читаемые линии, сетка и легенда.
    """

    import matplotlib.pyplot as plt

    points = frequency_data.get("points", [])

    if not points:
        points = [
            {
                "time_sec": 0.0,
                "left_gestures_per_min": 0.0,
                "right_gestures_per_min": 0.0,
                "total_gestures_per_min": 0.0,
            }
        ]

    time_sec = np.asarray(
        [p["time_sec"] for p in points],
        dtype=np.float64
    )

    left_rate = np.asarray(
        [p["left_gestures_per_min"] for p in points],
        dtype=np.float64
    )

    right_rate = np.asarray(
        [p["right_gestures_per_min"] for p in points],
        dtype=np.float64
    )

    total_rate = np.asarray(
        [p["total_gestures_per_min"] for p in points],
        dtype=np.float64
    )

    fig, ax = plt.subplots(
        figsize=(16, 8),
        dpi=140
    )

    # Основные линии
    ax.plot(
        time_sec,
        total_rate,
        linewidth=3.2,
        label="Всего"
    )

    ax.plot(
        time_sec,
        left_rate,
        linewidth=2.2,
        label="Левая рука"
    )

    ax.plot(
        time_sec,
        right_rate,
        linewidth=2.2,
        label="Правая рука"
    )

    # Мягкая заливка под общей линией
    ax.fill_between(
        time_sec,
        total_rate,
        alpha=0.12
    )

    ax.set_title(
        "P05 — частота жестикуляции во времени",
        fontsize=16,
        pad=18
    )

    ax.set_xlabel(
        "Время видео",
        fontsize=12
    )

    ax.set_ylabel(
        "Жестов/мин",
        fontsize=12
    )

    subtitle = (
        "Скользящее окно: "
        f'{frequency_data.get("window_sec", 20):.0f} с'
        " | Шаг: "
        f'{frequency_data.get("step_sec", 1):.0f} с'
        " | Единица: жестов/мин"
    )

    fig.text(
        0.125,
        0.94,
        subtitle,
        fontsize=10
    )

    # --------------------------------------------------------
    # Ось X: шаг 30 секунд, формат MM:SS
    # --------------------------------------------------------
    max_time = float(np.max(time_sec)) if len(time_sec) > 0 else 0.0

    tick_step = 30.0
    x_ticks = np.arange(
        0.0,
        max_time + tick_step,
        tick_step,
        dtype=np.float64
    )

    if len(x_ticks) == 0:
        x_ticks = np.asarray([0.0])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [format_axis_time(v) for v in x_ticks],
        rotation=0,
        fontsize=10
    )

    # Ось Y без смысловых изменений
    y_max = float(np.max([
        np.max(total_rate) if len(total_rate) else 0.0,
        np.max(left_rate) if len(left_rate) else 0.0,
        np.max(right_rate) if len(right_rate) else 0.0,
        1.0,
    ]))

    y_upper = max(
        5.0,
        np.ceil(y_max / 5.0) * 5.0
    )

    ax.set_ylim(0, y_upper)

    # Более приятная сетка
    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=0.28,
        linestyle="--"
    )

    # Убираем верхнюю и правую рамки для чистоты
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10
    )
    legend.get_frame().set_alpha(0.9)

    note = (
        "Синяя линия показывает общую частоту жестикуляции: "
        "чем выше линия, тем чаще начинались новые жесты "
        "в предыдущие 20 секунд."
    )

    fig.text(
        0.125,
        0.02,
        note,
        fontsize=9
    )

    plt.tight_layout(
        rect=[0.03, 0.05, 0.99, 0.92]
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close(fig)


def build_results_json(

    p05_tracker,
    frequency_data,
    fps,
    frame_count,
    frames_processed,
    video_duration_sec,
    processed_duration_sec,
    run_duration_sec
):
    p05_count = len(
        p05_tracker.cuts
    )

    pattern_names = [
        ("P01", "моргание"),
        ("P02", "кивок"),
        ("P03", "голова"),
        ("P04", "жест"),
        ("P05", "ампл.жеста"),
        ("P06", "рука у лица"),
        ("P07", "мелкие движ."),
        ("P08", "положение рук"),
        ("P09", "смена позы"),
        ("P10", "уголки губ"),
        ("P11", "брови/губы"),
        ("P12", "взгляд"),
    ]

    patterns_hud = [
        {
            "code": code,
            "name": name,
            "count": (
                p05_count
                if code == "P05"
                else 0
            ),
        }
        for code, name in pattern_names
    ]

    if run_duration_sec > 0:
        speed_ratio = (
            processed_duration_sec
            /
            run_duration_sec
        )

        effective_fps = (
            frames_processed
            /
            run_duration_sec
        )
    else:
        speed_ratio = 0.0
        effective_fps = 0.0

    if processed_duration_sec > 0:
        wall_sec_per_video_min = (
            run_duration_sec
            /
            (
                processed_duration_sec
                /
                60.0
            )
        )
    else:
        wall_sec_per_video_min = 0.0

    return {
        "schema": "bs_profiling_cut_v1",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "tracker": (
            "MediaPipe Pose Landmarker — P05 "
            "amplitude/speed"
        ),

        "video": str(VIDEO_PATH),
        "video_name": Path(VIDEO_PATH).name,

        "timing": {
            "video_duration_sec": round(
                video_duration_sec,
                3
            ),
            "video_duration_human": human_duration(
                video_duration_sec
            ),
            "video_duration_min": round(
                video_duration_sec / 60.0,
                2
            ),
            "run_duration_sec": round(
                run_duration_sec,
                2
            ),
            "run_duration_human": human_duration(
                run_duration_sec
            ),
            "run_duration_min": round(
                run_duration_sec / 60.0,
                2
            ),
            "processed_duration_sec": round(
                processed_duration_sec,
                3
            ),
            "processed_duration_human": human_duration(
                processed_duration_sec
            ),
            "processed_full_video": bool(
                frame_count <= 0
                or
                frames_processed >= frame_count
            ),
            "speed_ratio": round(
                speed_ratio,
                3
            ),
            "wall_sec_per_video_min": round(
                wall_sec_per_video_min,
                1
            ),
            "frames_total": int(
                frame_count
            ),
            "frames_processed": int(
                frames_processed
            ),
            "video_fps": round(
                float(fps),
                3
            ),
            "effective_fps": round(
                effective_fps,
                3
            ),
            "run_finished_local": datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            ),
            "note": (
                "P05 detector unchanged; JSON logger observes "
                "existing HandGestureTracker states."
            ),
        },

        "duration_processed_sec": round(
            processed_duration_sec,
            3
        ),

        "video_file_duration_sec": round(
            video_duration_sec,
            3
        ),

        "timeline_file": None,
        "timeline_segments": [],

        "patterns_hud": patterns_hud,

        "cuts": p05_tracker.cuts,

        # Дополнительный числовой ряд:
        # в будущем по нему удобно автоматически искать,
        # где жестикуляция усилилась или ослабла.
        "gesture_frequency": frequency_data,

        "gesture_frequency_graph": str(
            OUTPUT_FREQUENCY_GRAPH
        ),

        "overlay_timeline_mp4": str(
            OUTPUT_PATH
        ),

        "notes": [
            (
                "P05 = амплитуда/скорость жестов. "
                "Исходная логика detector не изменена."
            ),
            (
                "Каждый cut содержит start/end/duration, руку, "
                "amplitude, mean_speed, peak_speed и path."
            ),
            (
                "gesture_frequency = скользящая частота стартов "
                "жестов; она не влияет на P05 detector."
            ),
            (
                "clip_file/clip = null, потому что этот скрипт "
                "не вырезает отдельные clips."
            ),
        ],
    }


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise RuntimeError(
        f"Не удалось открыть видео: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("=" * 60)

print("P05 — Амплитуда / скорость жестов")

print("=" * 60)

print(
    f"Видео: {VIDEO_PATH}"
)

print(
    f"Разрешение: {width}x{height}"
)

print(
    f"FPS: {fps:.2f}"
)

print(
    f"Кадров: {frame_count}"
)

print("=" * 60)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACKERS
# ============================================================

left_tracker = HandGestureTracker()
right_tracker = HandGestureTracker()

# Пассивный logger; на детекцию не влияет.
p05_json_tracker = P05JsonTracker()


frame_index = 0


# ============================================================
# PROCESSING
# ============================================================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # ----------------------------------------------------
        # Timestamp для MediaPipe VIDEO mode
        # ----------------------------------------------------

        timestamp_ms = int(
            frame_index * 1000 / fps
        )


        # ----------------------------------------------------
        # OpenCV BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # Данные по умолчанию
        left_data = {
            "speed": 0.0,
            "amplitude": 0.0,
            "path": 0.0,
            "peak_speed": 0.0,
            "gesture": False,
        }

        right_data = left_data.copy()


        # ====================================================
        # ЕСЛИ ПОЗА НАЙДЕНА
        # ====================================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]


            ls = landmarks[LEFT_SHOULDER]
            rs = landmarks[RIGHT_SHOULDER]

            le = landmarks[LEFT_ELBOW]
            re = landmarks[RIGHT_ELBOW]

            lw = landmarks[LEFT_WRIST]
            rw = landmarks[RIGHT_WRIST]


            # ------------------------------------------------
            # Ширина плеч
            # ------------------------------------------------

            left_shoulder_xy = landmark_xy(ls)
            right_shoulder_xy = landmark_xy(rs)

            shoulder_width = distance_2d(
                left_shoulder_xy,
                right_shoulder_xy
            )


            # ------------------------------------------------
            # Wrist coordinates
            # ------------------------------------------------

            left_wrist_xy = landmark_xy(lw)
            right_wrist_xy = landmark_xy(rw)


            # ------------------------------------------------
            # Visibility
            # ------------------------------------------------

            left_visible = (
                get_visibility(lw)
                >= MIN_VISIBILITY
            )

            right_visible = (
                get_visibility(rw)
                >= MIN_VISIBILITY
            )


            # =================================================
            # LEFT
            # =================================================

            left_data = left_tracker.update(
                wrist=left_wrist_xy,
                shoulder_width=shoulder_width,
                fps=fps,
                visible=left_visible
            )


            # =================================================
            # RIGHT
            # =================================================

            right_data = right_tracker.update(
                wrist=right_wrist_xy,
                shoulder_width=shoulder_width,
                fps=fps,
                visible=right_visible
            )


            # ------------------------------------------------
            # Рисуем руки
            # ------------------------------------------------

            if left_visible:

                draw_pose_arm(
                    frame,
                    ls,
                    le,
                    lw,
                    width,
                    height
                )


            if right_visible:

                draw_pose_arm(
                    frame,
                    rs,
                    re,
                    rw,
                    width,
                    height
                )


        # ====================================================
        # P05 JSON LOGGER
        # ====================================================

        timestamp_sec = (
            frame_index
            /
            fps
        )

        p05_json_tracker.update(
            timestamp=
                timestamp_sec,

            left_tracker=
                left_tracker,

            right_tracker=
                right_tracker,

            left_data=
                left_data,

            right_data=
                right_data,
        )


        # ====================================================
        # OVERLAY
        # ====================================================

        frame = draw_overlay(

            frame,

            left_data,
            right_data,

            left_tracker.gesture_count,
            right_tracker.gesture_count
        )


        # ====================================================
        # SAVE
        # ====================================================

        writer.write(frame)


        # ====================================================
        # PREVIEW
        # ====================================================

        # Для большого видео уменьшаем окно просмотра
        preview_width = min(
            width,
            1280
        )

        scale = (
            preview_width / width
        )

        preview_height = int(
            height * scale
        )


        preview = cv2.resize(
            frame,
            (
                preview_width,
                preview_height
            )
        )


        cv2.imshow(
            "P05 - Gestures",
            preview
        )


        # Q или ESC для остановки
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


        frame_index += 1


        # ----------------------------------------------------
        # Прогресс
        # ----------------------------------------------------

        if frame_index % 100 == 0:

            percent = (
                frame_index
                / frame_count
                * 100
            )

            print(
                f"Обработано: "
                f"{frame_index}/{frame_count} "
                f"({percent:.1f}%)"
            )


# ============================================================
# FINISH
# ============================================================

processed_duration_sec = (
    frame_index
    /
    fps
    if
    fps > 0
    else
    0.0
)

video_duration_sec = (
    frame_count
    /
    fps
    if
    (
        frame_count > 0
        and
        fps > 0
    )
    else
    processed_duration_sec
)


# Если видео закончилось во время активного жеста.
p05_json_tracker.finalize(
    timestamp=
        processed_duration_sec,

    left_tracker=
        left_tracker,

    right_tracker=
        right_tracker,
)


gesture_frequency = (
    build_gesture_frequency(
        cuts=
            p05_json_tracker.cuts,

        duration_sec=
            processed_duration_sec,

        window_sec=
            FREQUENCY_WINDOW_SEC,

        step_sec=
            FREQUENCY_STEP_SEC,
    )
)


graph_created = (
    save_frequency_graph(
        gesture_frequency,
        OUTPUT_FREQUENCY_GRAPH
    )
)


run_duration_sec = (
    time.perf_counter()
    -
    RUN_STARTED_PERF
)


results_json = (
    build_results_json(
        p05_tracker=
            p05_json_tracker,

        frequency_data=
            gesture_frequency,

        fps=
            fps,

        frame_count=
            frame_count,

        frames_processed=
            frame_index,

        video_duration_sec=
            video_duration_sec,

        processed_duration_sec=
            processed_duration_sec,

        run_duration_sec=
            run_duration_sec,
    )
)


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as json_file:

    json.dump(
        results_json,
        json_file,
        ensure_ascii=False,
        indent=2
    )


cap.release()
writer.release()
cv2.destroyAllWindows()


print()
print("=" * 60)

print("Обработка завершена")

print(
    f"Жестов левой рукой: "
    f"{left_tracker.gesture_count}"
)

print(
    f"Жестов правой рукой: "
    f"{right_tracker.gesture_count}"
)

print(
    f"Результат: {OUTPUT_PATH}"
)

print(
    f"JSON: {OUTPUT_JSON}"
)

if graph_created:

    print(
        f"График частоты: {OUTPUT_FREQUENCY_GRAPH}"
    )

else:

    print(
        "График частоты: не создан "
        "(установи matplotlib: pip install matplotlib)"
    )

print(
    f"P05 событий в JSON: "
    f"{len(p05_json_tracker.cuts)}"
)

print("=" * 60)