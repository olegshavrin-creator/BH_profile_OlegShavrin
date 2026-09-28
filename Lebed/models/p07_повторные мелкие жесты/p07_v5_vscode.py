# ============================================================
# P07 V5 — INFERENCE / ПРОВЕРКА НА ВИДЕО
# ============================================================
# Этот скрипт НЕ обучает и НЕ докалибровывает модель.
# Он только:
#   1) загружает p07_repetition_rf_v4.joblib;
#   2) берёт START/END из calibration внутри joblib;
#   3) прогоняет TARGET_VIDEO;
#   4) сохраняет overlay-video, events CSV, scores CSV,
#      results JSON и timeline graph.
#
# Для честной проверки лучше использовать видео, которое НЕ входило
# в обучение P07 V5.
# ============================================================

# откалиброван на interview
# VS CODE / ЛОКАЛЬНЫЙ ЗАПУСК:
# 1) Создай виртуальное окружение и установи зависимости:
#    python -m venv .venv
#    .venv\\Scripts\\activate
#    python -m pip install -r requirements.txt
#
# 2) Структура проекта:
#    models/pose_landmarker_heavy.task
#    models/hand_landmarker.task
#    models/p07_repetition_rf_v5.joblib
#    models/p07_repetition_rf_v5_calibration.json
#    videos/Nikolaev(melkie_dvizheniya)3.mp4
#    videos/My_Interview.mp4
#    videos/video6044313067708227319.mp4
#    output/
#
# 3) Запуск:
#    python p07_v5_vscode.py
#
# Если RETRAIN = False, калибровочные видео не обрабатываются основным main().
# Логика P07, признаки, Random Forest, thresholds и state machine НЕ изменены.

import csv
import json
import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# VS CODE / ЛОКАЛЬНАЯ СРЕДА
# ============================================================

# Переменная оставлена для совместимости с исходной логикой preview.
# В локальном VS Code она всегда False.
IN_GOOGLE_COLAB = False

from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

from PIL import Image, ImageDraw, ImageFont

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:
    raise ImportError(
        "Нужен scikit-learn. Установи: pip install scikit-learn joblib"
    ) from exc



# ============================================================
# 1. ПУТИ — VS CODE / WINDOWS
# ============================================================

# Все пути строятся относительно папки, где лежит этот .py-файл.
# Это единственное существенное отличие от Colab-версии:
# вычислительная логика P07 ниже НЕ изменена.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_VIDEOS = {
    str(VIDEOS_DIR / "Nikolaev(melkie_dvizheniya)3.mp4"): [
        (10.0, 50.0)
    ],
    str(VIDEOS_DIR / "My_Interview.mp4"): [
        (13.0, 18.0)
    ],
}

# Видео для инференса.
# Чтобы проверить другое видео, положи его в videos/
# и поменяй только имя файла в следующей строке.
TARGET_VIDEO = str(
    VIDEOS_DIR / "video6044313067708227319.mp4"
)

POSE_MODEL_PATH = str(
    MODELS_DIR / "pose_landmarker_heavy.task"
)

HAND_MODEL_PATH = str(
    MODELS_DIR / "hand_landmarker.task"
)

RF_MODEL_PATH = str(
    MODELS_DIR / "p07_repetition_rf_v5.joblib"
)

CALIBRATION_JSON = str(
    MODELS_DIR / "p07_repetition_rf_v5_calibration.json"
)

OUTPUT_VIDEO = str(
    OUTPUT_DIR / "P07_V5_video6044313067708227319.mp4"
)

OUTPUT_CSV = str(
    OUTPUT_DIR / "P07_V5_video6044313067708227319.csv"
)

# Оставлено ровно как в исходном ноутбуке:
# OUTPUT_SCORE_CSV указывает на тот же файл, что и OUTPUT_CSV.
OUTPUT_SCORE_CSV = str(
    OUTPUT_DIR / "P07_V5_video6044313067708227319.csv"
)


# JSON в той же папке, что и итоговое видео.
OUTPUT_JSON = str(
    Path(
        OUTPUT_VIDEO
    ).with_name(
        Path(
            OUTPUT_VIDEO
        ).stem
        +
        "_results.json"
    )
)

# График P07: вероятность + START/END каждого события.
OUTPUT_GRAPH = str(
    Path(
        OUTPUT_VIDEO
    ).with_name(
        Path(
            OUTPUT_VIDEO
        ).stem
        +
        "_p07_timeline.png"
    )
)

# Первый запуск: True.
# Когда .joblib уже обучен, можно поставить False.
RETRAIN = False


# ============================================================
# HARD RECALIBRATION — My_Interview
# ============================================================
#
# ВАЖНО:
# этот режим НЕ переобучает RandomForest.
# Он использует уже обученный RF и заново подбирает только
# START / END thresholds на явно размеченных FN/FP.
#
# Участки вне перечисленных POSITIVE / NEGATIVE интервалов
# НЕ считаются отрицательными и НЕ участвуют в оценке.
# ============================================================

HARD_CALIBRATION_VIDEO = str(
    VIDEOS_DIR / "My_Interview.mp4"
)

HARD_POSITIVE_INTERVALS = [
    # 01:12–01:19 — одна рука -> две руки
    (72.0, 79.0),

    # 01:26–01:38 — две руки
    (86.0, 98.0),

    # 01:52–01:58 — две руки
    (112.0, 118.0),

    # 05:01–05:03 — правая рука
    (301.0, 303.0),
]

HARD_NEGATIVE_INTERVALS = [
    # Ложные срабатывания старой модели
    (245.0, 248.0),  # 04:05–04:08
    (296.0, 298.0),  # 04:56–04:58
    (326.0, 327.0),  # 05:26–05:27
]

HARD_OUTPUT_MODEL = str(
    MODELS_DIR / "p07_repetition_rf_v3_myinterview_calibrated.joblib"
)

HARD_OUTPUT_CALIBRATION_JSON = str(
    MODELS_DIR / "p07_repetition_rf_v3_myinterview_calibration.json"
)

HARD_OUTPUT_REPORT_CSV = str(
    OUTPUT_DIR / "p07_myinterview_hard_recalibration_report.csv"
)

HARD_OUTPUT_GRAPH = str(
    OUTPUT_DIR / "p07_myinterview_hard_recalibration_graph.png"
)

# Используем ТЕ ЖЕ grids, что и в исходном P07 V3:
# START 0.45..0.90 с шагом 0.05
# END   0.20..0.70 с шагом 0.05
#
# ВАЖНО: исходная логика Random Forest и START/END не менялась.


# ============================================================
# 2. ОКНО АНАЛИЗА
# ============================================================

# Окно достаточно длинное, чтобы увидеть 2–4 повторения,
# но не настолько длинное, чтобы склеивать весь ролик.
WINDOW_SECONDS = 2.40

# Шаг прогноза.
HOP_SECONDS = 0.10

# Периоды повторяющихся движений, которые считаем разумными.
# 0.18 сек = быстрые пальцы.
# 2.00 сек = медленное повторное движение рукой.
MIN_PERIOD_SECONDS = 0.18
MAX_PERIOD_SECONDS = 2.00

# Минимум валидных кадров в окне.
MIN_VALID_RATIO = 0.70

# Короткие пропуски landmark можно интерполировать.
MAX_INTERP_GAP_SECONDS = 0.20

# EMA сглаживание.
EMA_ALPHA = 0.32


# ============================================================
# 3. РАЗМЕТКА ОБУЧАЮЩИХ ОКОН
# ============================================================

# Окно считается positive, если такая доля окна лежит
# внутри размеченного P07.
POSITIVE_OVERLAP = 0.55

# Окно считается negative, если пересечение с P07 не больше.
NEGATIVE_OVERLAP = 0.08

# Остальные пограничные окна не используем для обучения.
# Это помогает не учить модель на спорных границах.


# ============================================================
# 4. RANDOM FOREST
# ============================================================

RF_TREES = 500
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 3
RANDOM_STATE = 42


# ============================================================
# 5. EVENT STATE MACHINE
# ============================================================

# Вероятность дополнительно сглаживается медианой.
PROBABILITY_SMOOTH_SECONDS = 0.50

# При калибровке эти thresholds подбираются автоматически.
START_THRESHOLD_GRID = np.arange(
    0.45,
    0.91,
    0.05,
)

END_THRESHOLD_GRID = np.arange(
    0.20,
    0.71,
    0.05,
)

START_CONFIRM_POINTS = 2
END_CONFIRM_POINTS = 3

MIN_EVENT_SECONDS = 0.60
MERGE_GAP_SECONDS = 0.25


# ============================================================
# 6. MEDIAPIPE
# ============================================================

MIN_POSE_VISIBILITY = 0.45

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

H_WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20


# ============================================================
# 7. OVERLAY
# ============================================================

PANEL_ALPHA = 0.50
SHOW_PREVIEW = False  # В VS Code можно поставить True для окна preview


# ============================================================
# 8. HELPERS
# ============================================================

def find_font():
    """
    Надёжный поиск шрифта с кириллицей для:
    - Windows;
    - Linux;
    - matplotlib bundled fonts.

    В VS Code на Windows обычно будет выбран Segoe UI / Arial.
    """

    candidates = [
        # Linux fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",

        # Windows
        r"C:/Windows/Fonts/segoeui.ttf",
        r"C:/Windows/Fonts/arial.ttf",
        r"C:/Windows/Fonts/calibri.ttf",
        r"C:/Windows/Fonts/tahoma.ttf",
    ]

    for path in candidates:
        if Path(path).exists():
            return str(path)

    # --------------------------------------------------------
    # Matplotlib обычно содержит собственный DejaVuSans.ttf.
    # Используется как кроссплатформенный fallback.
    # --------------------------------------------------------
    try:
        import matplotlib

        mpl_font = (
            Path(
                matplotlib.get_data_path()
            )
            / "fonts"
            / "ttf"
            / "DejaVuSans.ttf"
        )

        if mpl_font.exists():
            return str(
                mpl_font
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Резервный поиск в стандартных Linux-каталогах.
    # --------------------------------------------------------
    font_roots = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("/root/.fonts"),
    ]

    preferred_names = [
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "NotoSans-Regular.ttf",
        "FreeSans.ttf",
    ]

    for root in font_roots:

        if not root.exists():
            continue

        for font_name in preferred_names:

            try:
                matches = list(
                    root.rglob(
                        font_name
                    )
                )

                if matches:
                    return str(
                        matches[0]
                    )

            except Exception:
                pass

    # Не останавливаем всю обработку из-за overlay.
    return None


FONT_PATH = find_font()


def load_overlay_font(
    size,
):
    """
    Загружает шрифт для overlay.

    Если кириллический TTF найти не удалось, обработка видео,
    P07, JSON и график НЕ останавливаются: используем
    ImageFont.load_default() как аварийный fallback.
    """

    size = max(
        10,
        int(
            size
        )
    )

    if (
        FONT_PATH is not None
        and Path(
            FONT_PATH
        ).exists()
    ):
        try:
            return ImageFont.truetype(
                FONT_PATH,
                size,
            )

        except Exception:
            pass

    # Иногда Pillow/FreeType может найти DejaVuSans по имени.
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            size,
        )

    except Exception:
        pass

    print(
        "WARNING: кириллический TTF не найден. "
        "Используется fallback-шрифт; "
        "P07/JSON/график продолжат работать."
    )

    return ImageFont.load_default()


def fmt_time(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "--:--.-"

    seconds = max(
        0.0,
        float(seconds),
    )

    minutes = int(
        seconds // 60
    )

    sec = (
        seconds
        - 60 * minutes
    )

    return (
        f"{minutes:02d}:"
        f"{sec:04.1f}"
    )


def xy(lm):
    return np.array(
        [
            float(lm.x),
            float(lm.y),
        ],
        dtype=np.float64,
    )


def d2(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a)
            - np.asarray(b)
        )
    )


def visibility(lm):
    value = getattr(
        lm,
        "visibility",
        1.0,
    )

    if value is None:
        return 1.0

    return float(value)


def interpolate_short_gaps(
    x,
    max_gap,
):
    y = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    n = len(y)
    i = 0

    while i < n:

        if np.isfinite(
            y[i]
        ):
            i += 1
            continue

        start = i

        while (
            i < n
            and not np.isfinite(
                y[i]
            )
        ):
            i += 1

        end = i - 1
        gap = (
            end - start + 1
        )

        left = (
            start - 1
        )

        right = i

        if (
            gap <= max_gap
            and left >= 0
            and right < n
            and np.isfinite(
                y[left]
            )
            and np.isfinite(
                y[right]
            )
        ):
            y[
                start:right
            ] = np.linspace(
                y[left],
                y[right],
                gap + 2,
            )[1:-1]

    return y


def ema_nan(
    x,
    alpha,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    out = np.full_like(
        x,
        np.nan,
    )

    state = None

    for i, value in enumerate(
        x
    ):
        if not np.isfinite(
            value
        ):
            continue

        if state is None:
            state = float(
                value
            )

        else:
            state = (
                alpha
                * float(value)
                + (
                    1.0
                    - alpha
                )
                * state
            )

        out[i] = state

    return out


def fill_remaining_nans(
    x,
):
    """
    Для расчёта временных признаков:
    если валидных точек достаточно, оставшиеся NaN заполняем
    линейной интерполяцией + ближайшими краями.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    mask = np.isfinite(
        x
    )

    if mask.sum() < 3:
        return None

    indices = np.arange(
        len(x)
    )

    x[~mask] = np.interp(
        indices[~mask],
        indices[mask],
        x[mask],
    )

    return x


def overlap_seconds(
    a0,
    a1,
    b0,
    b1,
):
    return max(
        0.0,
        min(
            a1,
            b1,
        )
        - max(
            a0,
            b0,
        ),
    )


# ============================================================
# 9. FRAME-LEVEL FEATURES
# ============================================================

POSE_DIM = 8
HAND_DIM = 14
INTERACTION_DIM = 6


@dataclass
class VideoFeatures:
    fps: float
    width: int
    height: int
    frame_count: int
    times: np.ndarray

    pose: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    interaction: np.ndarray


def pose_features(
    result,
):
    out = np.full(
        POSE_DIM,
        np.nan,
        dtype=np.float64,
    )

    if not result.pose_landmarks:
        return out

    lm = result.pose_landmarks[0]

    ls = lm[
        LEFT_SHOULDER
    ]

    rs = lm[
        RIGHT_SHOULDER
    ]

    if (
        visibility(ls)
        < MIN_POSE_VISIBILITY
        or visibility(rs)
        < MIN_POSE_VISIBILITY
    ):
        return out

    ls_xy = xy(ls)
    rs_xy = xy(rs)

    center = (
        ls_xy + rs_xy
    ) / 2.0

    shoulder_width = d2(
        ls_xy,
        rs_xy,
    )

    if shoulder_width < 1e-6:
        return out

    points = [
        lm[LEFT_WRIST],
        lm[LEFT_ELBOW],
        lm[RIGHT_WRIST],
        lm[RIGHT_ELBOW],
    ]

    values = []

    for point in points:

        if (
            visibility(point)
            < MIN_POSE_VISIBILITY
        ):
            values.extend(
                [
                    np.nan,
                    np.nan,
                ]
            )

        else:
            q = (
                xy(point)
                - center
            ) / shoulder_width

            values.extend(
                [
                    float(
                        q[0]
                    ),
                    float(
                        q[1]
                    ),
                ]
            )

    return np.asarray(
        values,
        dtype=np.float64,
    )


@dataclass
class HandFrame:
    features: np.ndarray
    landmarks_xy: np.ndarray
    palm_scale: float


def hand_frame(
    landmarks,
):
    pts = np.asarray(
        [
            xy(lm)
            for lm in landmarks
        ],
        dtype=np.float64,
    )

    wrist = pts[
        H_WRIST
    ]

    scale = max(
        d2(
            pts[INDEX_MCP],
            pts[PINKY_MCP],
        ),
        d2(
            wrist,
            pts[MIDDLE_MCP],
        ),
    )

    if scale < 1e-6:
        return None

    thumb = pts[
        THUMB_TIP
    ]

    index = pts[
        INDEX_TIP
    ]

    middle = pts[
        MIDDLE_TIP
    ]

    ring = pts[
        RING_TIP
    ]

    pinky = pts[
        PINKY_TIP
    ]

    def local(
        point,
    ):
        return (
            point - wrist
        ) / scale

    tl = local(
        thumb
    )

    il = local(
        index
    )

    ml = local(
        middle
    )

    rl = local(
        ring
    )

    pl = local(
        pinky
    )

    spread = float(
        np.mean(
            [
                d2(
                    thumb,
                    index,
                ),
                d2(
                    index,
                    middle,
                ),
                d2(
                    middle,
                    ring,
                ),
                d2(
                    ring,
                    pinky,
                ),
            ]
        )
        / scale
    )

    features = np.asarray(
        [
            d2(
                thumb,
                index,
            )
            / scale,

            d2(
                thumb,
                middle,
            )
            / scale,

            d2(
                index,
                middle,
            )
            / scale,

            tl[0],
            tl[1],

            il[0],
            il[1],

            ml[0],
            ml[1],

            rl[0],
            rl[1],

            pl[0],
            pl[1],

            spread,
        ],
        dtype=np.float64,
    )

    return HandFrame(
        features=features,
        landmarks_xy=pts,
        palm_scale=float(
            scale
        ),
    )


def interaction_features(
    left: Optional[HandFrame],
    right: Optional[HandFrame],
):
    out = np.full(
        INTERACTION_DIM,
        np.nan,
        dtype=np.float64,
    )

    if (
        left is None
        or right is None
    ):
        return out

    scale = max(
        1e-6,
        0.5
        * (
            left.palm_scale
            + right.palm_scale
        ),
    )

    lp = left.landmarks_xy
    rp = right.landmarks_xy

    # Симметричные/межкистевые признаки.
    # Полезны для потирания пальцев двух рук друг о друга.
    values = [
        d2(
            lp[THUMB_TIP],
            rp[THUMB_TIP],
        )
        / scale,

        d2(
            lp[INDEX_TIP],
            rp[INDEX_TIP],
        )
        / scale,

        d2(
            lp[THUMB_TIP],
            rp[INDEX_TIP],
        )
        / scale,

        d2(
            rp[THUMB_TIP],
            lp[INDEX_TIP],
        )
        / scale,

        d2(
            lp[H_WRIST],
            rp[H_WRIST],
        )
        / scale,

        d2(
            (
                lp[THUMB_TIP]
                + lp[INDEX_TIP]
            )
            / 2.0,

            (
                rp[THUMB_TIP]
                + rp[INDEX_TIP]
            )
            / 2.0,
        )
        / scale,
    ]

    return np.asarray(
        values,
        dtype=np.float64,
    )


# ============================================================
# 10. VIDEO EXTRACTION
# ============================================================

def extract_video(
    path,
):
    path = str(
        path
    )

    cap = cv2.VideoCapture(
        path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть видео:\n{path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if (
        not np.isfinite(fps)
        or fps <= 0
    ):
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    pose_arr = np.full(
        (
            frame_count,
            POSE_DIM,
        ),
        np.nan,
    )

    left_arr = np.full(
        (
            frame_count,
            HAND_DIM,
        ),
        np.nan,
    )

    right_arr = np.full(
        (
            frame_count,
            HAND_DIM,
        ),
        np.nan,
    )

    interaction_arr = np.full(
        (
            frame_count,
            INTERACTION_DIM,
        ),
        np.nan,
    )

    times = (
        np.arange(
            frame_count,
            dtype=np.float64,
        )
        / fps
    )

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    RM = (
        mp.tasks.vision.RunningMode
    )

    pose_options = (
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=
                BaseOptions(
                    model_asset_path=
                        POSE_MODEL_PATH
                ),

            running_mode=
                RM.VIDEO,

            num_poses=1,

            min_pose_detection_confidence=
                0.5,

            min_pose_presence_confidence=
                0.5,

            min_tracking_confidence=
                0.5,

            output_segmentation_masks=
                False,
        )
    )

    hand_options = (
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=
                BaseOptions(
                    model_asset_path=
                        HAND_MODEL_PATH
                ),

            running_mode=
                RM.VIDEO,

            num_hands=2,

            min_hand_detection_confidence=
                0.45,

            min_hand_presence_confidence=
                0.45,

            min_tracking_confidence=
                0.45,
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "Pose + Hands:"
    )
    print(
        path
    )
    print(
        "=" * 72
    )

    PoseLandmarker = (
        mp.tasks.vision.PoseLandmarker
    )

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    with (
        PoseLandmarker.create_from_options(
            pose_options
        ) as pose_model,

        HandLandmarker.create_from_options(
            hand_options
        ) as hand_model
    ):

        i = 0

        while i < frame_count:

            ok, frame = (
                cap.read()
            )

            if not ok:
                break

            ts_ms = int(
                round(
                    i
                    * 1000.0
                    / fps
                )
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = mp.Image(
                image_format=
                    mp.ImageFormat.SRGB,
                data=rgb,
            )

            pose_result = (
                pose_model
                .detect_for_video(
                    image,
                    ts_ms,
                )
            )

            hand_result = (
                hand_model
                .detect_for_video(
                    image,
                    ts_ms,
                )
            )

            pose_arr[i] = (
                pose_features(
                    pose_result
                )
            )

            left = None
            right = None

            if (
                hand_result.hand_landmarks
            ):

                for k, landmarks in enumerate(
                    hand_result.hand_landmarks
                ):

                    hf = hand_frame(
                        landmarks
                    )

                    if hf is None:
                        continue

                    side = None

                    if (
                        k
                        < len(
                            hand_result.handedness
                        )
                        and hand_result.handedness[k]
                    ):
                        side = (
                            hand_result
                            .handedness[k][0]
                            .category_name
                        )

                    if side == "Left":
                        left = hf

                    elif side == "Right":
                        right = hf

                    elif left is None:
                        left = hf

                    else:
                        right = hf

            if left is not None:
                left_arr[i] = (
                    left.features
                )

            if right is not None:
                right_arr[i] = (
                    right.features
                )

            interaction_arr[i] = (
                interaction_features(
                    left,
                    right,
                )
            )

            i += 1

            if (
                i % 300
                == 0
            ):
                print(
                    f"  {i}/"
                    f"{frame_count}"
                )

    cap.release()

    if i < frame_count:

        pose_arr = (
            pose_arr[:i]
        )

        left_arr = (
            left_arr[:i]
        )

        right_arr = (
            right_arr[:i]
        )

        interaction_arr = (
            interaction_arr[:i]
        )

        times = (
            times[:i]
        )

        frame_count = i

    return VideoFeatures(
        fps=float(
            fps
        ),

        width=width,
        height=height,

        frame_count=
            frame_count,

        times=times,

        pose=pose_arr,
        left_hand=left_arr,
        right_hand=right_arr,
        interaction=interaction_arr,
    )


# ============================================================
# 11. PREPROCESS TEMPORAL WINDOW
# ============================================================

def prepare_window(
    segment,
    fps,
):
    """
    segment: frames x channels
    """

    x = np.asarray(
        segment,
        dtype=np.float64,
    ).copy()

    valid_ratio = float(
        np.isfinite(
            x
        ).mean()
    )

    if (
        valid_ratio
        < MIN_VALID_RATIO
    ):
        return None

    max_gap = max(
        1,
        int(
            round(
                MAX_INTERP_GAP_SECONDS
                * fps
            )
        ),
    )

    channels = []

    for j in range(
        x.shape[1]
    ):
        col = interpolate_short_gaps(
            x[:, j],
            max_gap,
        )

        col = fill_remaining_nans(
            col
        )

        if col is None:
            continue

        col = ema_nan(
            col,
            EMA_ALPHA,
        )

        col = fill_remaining_nans(
            col
        )

        if col is None:
            continue

        channels.append(
            col
        )

    if not channels:
        return None

    return (
        np.column_stack(
            channels
        ),
        valid_ratio,
    )


# ============================================================
# 12. ПРИЗНАКИ ПОВТОРЯЕМОСТИ ОДНОГО КАНАЛА
# ============================================================

def linear_detrend(
    x,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    t = np.linspace(
        -1.0,
        1.0,
        len(x),
    )

    coef = np.polyfit(
        t,
        x,
        1,
    )

    trend = (
        coef[0]
        * t
        + coef[1]
    )

    return (
        x - trend
    )


def normalized_autocorr(
    x,
    fps,
):
    """
    Возвращает лучший положительный autocorrelation
    и соответствующий period.
    """

    y = linear_detrend(
        x
    )

    y = (
        y - np.mean(
            y
        )
    )

    std = float(
        np.std(
            y
        )
    )

    if std < 1e-8:
        return (
            0.0,
            0.0,
        )

    y = y / std

    min_lag = max(
        2,
        int(
            round(
                MIN_PERIOD_SECONDS
                * fps
            )
        ),
    )

    max_lag = min(
        len(y) - 3,
        int(
            round(
                MAX_PERIOD_SECONDS
                * fps
            )
        ),
    )

    if max_lag <= min_lag:
        return (
            0.0,
            0.0,
        )

    best = 0.0
    best_lag = 0

    for lag in range(
        min_lag,
        max_lag + 1,
    ):
        a = y[
            :-lag
        ]

        b = y[
            lag:
        ]

        if len(a) < 6:
            continue

        score = float(
            np.mean(
                a * b
            )
        )

        if score > best:
            best = score
            best_lag = lag

    period = (
        best_lag / fps
        if best_lag > 0
        else 0.0
    )

    return (
        max(
            0.0,
            best,
        ),
        period,
    )


def spectral_peak_ratio(
    x,
    fps,
):
    """
    Насколько спектр похож на периодический:
    одна частота должна заметно доминировать.
    """

    y = linear_detrend(
        x
    )

    y = (
        y - np.mean(
            y
        )
    )

    if len(y) < 8:
        return (
            0.0,
            0.0,
        )

    window = np.hanning(
        len(y)
    )

    spec = np.fft.rfft(
        y * window
    )

    power = (
        np.abs(
            spec
        )
        ** 2
    )

    freqs = np.fft.rfftfreq(
        len(y),
        d=1.0 / fps,
    )

    low_f = (
        1.0
        / MAX_PERIOD_SECONDS
    )

    high_f = min(
        fps / 2.0 - 1e-6,
        1.0
        / MIN_PERIOD_SECONDS,
    )

    mask = (
        (freqs >= low_f)
        & (freqs <= high_f)
    )

    band = power[
        mask
    ]

    band_freqs = freqs[
        mask
    ]

    if (
        len(band) == 0
        or np.sum(
            band
        )
        <= 1e-12
    ):
        return (
            0.0,
            0.0,
        )

    idx = int(
        np.argmax(
            band
        )
    )

    peak = float(
        band[idx]
    )

    total = float(
        np.sum(
            band
        )
    )

    ratio = (
        peak
        / max(
            1e-12,
            total,
        )
    )

    frequency = float(
        band_freqs[idx]
    )

    return (
        ratio,
        frequency,
    )


def turning_period_features(
    x,
    fps,
):
    """
    Реальные развороты + регулярность полного цикла.
    """

    y = linear_detrend(
        x
    )

    dy = np.diff(
        y
    )

    if len(dy) < 4:
        return (
            0.0,
            0.0,
            1.0,
        )

    noise = float(
        np.median(
            np.abs(
                dy
                - np.median(
                    dy
                )
            )
        )
    )

    eps = max(
        1e-6,
        0.40 * noise,
    )

    sign = np.zeros(
        len(dy),
        dtype=np.int8,
    )

    sign[
        dy > eps
    ] = 1

    sign[
        dy < -eps
    ] = -1

    # Протягиваем направление через короткие остановки.
    last = 0

    for i in range(
        len(sign)
    ):
        if sign[i] != 0:
            last = sign[i]

        elif last != 0:
            sign[i] = last

    turns = []

    for i in range(
        1,
        len(sign),
    ):
        if (
            sign[i - 1]
            * sign[i]
            < 0
        ):
            turns.append(
                i
            )

    duration = (
        len(x)
        / fps
    )

    reversal_rate = (
        len(turns)
        / max(
            1e-6,
            duration,
        )
    )

    periods = []

    for i in range(
        len(turns) - 2
    ):
        period = (
            turns[i + 2]
            - turns[i]
        ) / fps

        if (
            MIN_PERIOD_SECONDS
            <= period
            <= MAX_PERIOD_SECONDS
        ):
            periods.append(
                period
            )

    if periods:

        mean_period = float(
            np.mean(
                periods
            )
        )

        if len(periods) >= 2:
            cv = float(
                np.std(
                    periods
                )
                / max(
                    1e-6,
                    mean_period,
                )
            )

        else:
            cv = 1.0

        cycle_count = (
            duration
            / max(
                1e-6,
                mean_period,
            )
        )

    else:
        cycle_count = 0.0
        mean_period = 0.0
        cv = 1.0

    return (
        reversal_rate,
        cycle_count,
        cv,
    )


def channel_repetition_features(
    x,
    fps,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    amplitude = float(
        np.percentile(
            x,
            95,
        )
        - np.percentile(
            x,
            5,
        )
    )

    velocity = (
        np.diff(
            x
        )
        * fps
    )

    median_speed = float(
        np.median(
            np.abs(
                velocity
            )
        )
    )

    p90_speed = float(
        np.percentile(
            np.abs(
                velocity
            ),
            90,
        )
    )

    ac, ac_period = (
        normalized_autocorr(
            x,
            fps,
        )
    )

    spectral, spectral_freq = (
        spectral_peak_ratio(
            x,
            fps,
        )
    )

    reversal_rate, cycles, period_cv = (
        turning_period_features(
            x,
            fps,
        )
    )

    # Универсальный periodic score.
    # Никакого конкретного жеста здесь нет.
    regularity = max(
        0.0,
        1.0
        - min(
            1.0,
            period_cv,
        ),
    )

    periodic_score = (
        ac
        * spectral
        * (
            0.60
            + 0.40
            * min(
                1.0,
                cycles / 3.0,
            )
        )
        * (
            0.60
            + 0.40
            * regularity
        )
    )

    return {
        "amplitude":
            amplitude,

        "median_speed":
            median_speed,

        "p90_speed":
            p90_speed,

        "autocorr":
            ac,

        "autocorr_period":
            ac_period,

        "spectral_ratio":
            spectral,

        "spectral_frequency":
            spectral_freq,

        "reversal_rate":
            reversal_rate,

        "cycles":
            cycles,

        "period_cv":
            period_cv,

        "periodic_score":
            periodic_score,
    }


# ============================================================
# 13. BRANCH-LEVEL FEATURES
# ============================================================

BRANCH_FEATURE_COUNT = 18


def branch_features(
    segment,
    fps,
):
    prepared = prepare_window(
        segment,
        fps,
    )

    if prepared is None:
        return np.zeros(
            BRANCH_FEATURE_COUNT,
            dtype=np.float64,
        )

    x, valid_ratio = (
        prepared
    )

    channel_stats = [
        channel_repetition_features(
            x[:, j],
            fps,
        )
        for j in range(
            x.shape[1]
        )
    ]

    def arr(name):
        return np.asarray(
            [
                item[
                    name
                ]
                for item
                in channel_stats
            ],
            dtype=np.float64,
        )

    amplitude = arr(
        "amplitude"
    )

    med_speed = arr(
        "median_speed"
    )

    p90_speed = arr(
        "p90_speed"
    )

    autocorr = arr(
        "autocorr"
    )

    spectral = arr(
        "spectral_ratio"
    )

    cycles = arr(
        "cycles"
    )

    period_cv = arr(
        "period_cv"
    )

    periodic = arr(
        "periodic_score"
    )

    reversal = arr(
        "reversal_rate"
    )

    ac_period = arr(
        "autocorr_period"
    )

    order = np.argsort(
        periodic
    )[::-1]

    top = order[
        :min(
            3,
            len(order),
        )
    ]

    top_periodic = (
        periodic[top]
        if len(top)
        else np.array(
            [0.0]
        )
    )

    top_autocorr = (
        autocorr[top]
        if len(top)
        else np.array(
            [0.0]
        )
    )

    top_spectral = (
        spectral[top]
        if len(top)
        else np.array(
            [0.0]
        )
    )

    top_cycles = (
        cycles[top]
        if len(top)
        else np.array(
            [0.0]
        )
    )

    top_cv = (
        period_cv[top]
        if len(top)
        else np.array(
            [1.0]
        )
    )

    top_period = (
        ac_period[top]
        if len(top)
        else np.array(
            [0.0]
        )
    )

    # Канал считаем действительно периодическим только если
    # есть и циклы, и autocorrelation, и spectral concentration.
    periodic_channels = (
        (
            autocorr >= 0.25
        )
        & (
            spectral >= 0.18
        )
        & (
            cycles >= 1.8
        )
    )

    values = np.asarray(
        [
            valid_ratio,

            float(
                np.median(
                    amplitude
                )
            ),

            float(
                np.max(
                    amplitude
                )
            ),

            float(
                np.median(
                    med_speed
                )
            ),

            float(
                np.max(
                    p90_speed
                )
            ),

            float(
                np.max(
                    periodic
                )
            ),

            float(
                np.mean(
                    top_periodic
                )
            ),

            float(
                np.max(
                    autocorr
                )
            ),

            float(
                np.mean(
                    top_autocorr
                )
            ),

            float(
                np.max(
                    spectral
                )
            ),

            float(
                np.mean(
                    top_spectral
                )
            ),

            float(
                np.max(
                    cycles
                )
            ),

            float(
                np.mean(
                    top_cycles
                )
            ),

            float(
                np.min(
                    top_cv
                )
            ),

            float(
                np.mean(
                    top_cv
                )
            ),

            float(
                np.max(
                    reversal
                )
            ),

            float(
                np.mean(
                    periodic_channels.astype(
                        np.float64
                    )
                )
            ),

            float(
                np.median(
                    top_period[
                        top_period > 0
                    ]
                )
                if np.any(
                    top_period > 0
                )
                else 0.0
            ),
        ],
        dtype=np.float64,
    )

    values[
        ~np.isfinite(
            values
        )
    ] = 0.0

    return values


# ============================================================
# 14. WINDOW FEATURES — ALL SOURCES
# ============================================================

def feature_vector(
    video,
    center_time,
):
    fps = (
        video.fps
    )

    half = (
        WINDOW_SECONDS
        / 2.0
    )

    start_time = (
        center_time
        - half
    )

    end_time = (
        center_time
        + half
    )

    start = max(
        0,
        int(
            math.floor(
                start_time
                * fps
            )
        ),
    )

    end = min(
        video.frame_count,
        int(
            math.ceil(
                end_time
                * fps
            )
        ),
    )

    if (
        end - start
        < max(
            8,
            int(
                0.60
                * WINDOW_SECONDS
                * fps
            ),
        )
    ):
        return None

    pose_f = branch_features(
        video.pose[
            start:end
        ],
        fps,
    )

    left_f = branch_features(
        video.left_hand[
            start:end
        ],
        fps,
    )

    right_f = branch_features(
        video.right_hand[
            start:end
        ],
        fps,
    )

    interaction_f = branch_features(
        video.interaction[
            start:end
        ],
        fps,
    )

    # Дополнительные "max across source" признаки.
    sources = np.vstack(
        [
            pose_f,
            left_f,
            right_f,
            interaction_f,
        ]
    )

    max_source = np.max(
        sources,
        axis=0,
    )

    mean_source = np.mean(
        sources,
        axis=0,
    )

    return np.concatenate(
        [
            pose_f,
            left_f,
            right_f,
            interaction_f,

            max_source,
            mean_source,
        ]
    )


def all_window_features(
    video,
):
    half = (
        WINDOW_SECONDS
        / 2.0
    )

    duration = (
        video.frame_count
        / video.fps
    )

    times = np.arange(
        half,
        max(
            half,
            duration - half
        )
        + 1e-9,
        HOP_SECONDS,
    )

    X = []
    good_times = []

    for center_time in times:

        features = (
            feature_vector(
                video,
                center_time,
            )
        )

        if features is None:
            continue

        X.append(
            features
        )

        good_times.append(
            center_time
        )

    if not X:
        return (
            np.empty(
                (
                    0,
                    BRANCH_FEATURE_COUNT
                    * 6,
                )
            ),
            np.empty(
                0
            ),
        )

    return (
        np.vstack(
            X
        ),
        np.asarray(
            good_times,
            dtype=np.float64,
        ),
    )


# ============================================================
# 15. WINDOW LABELS
# ============================================================

def window_overlap_ratio(
    center_time,
    intervals,
):
    half = (
        WINDOW_SECONDS
        / 2.0
    )

    a0 = (
        center_time
        - half
    )

    a1 = (
        center_time
        + half
    )

    best = 0.0

    for b0, b1 in intervals:

        overlap = overlap_seconds(
            a0,
            a1,
            b0,
            b1,
        )

        ratio = (
            overlap
            / WINDOW_SECONDS
        )

        best = max(
            best,
            ratio,
        )

    return best


def make_training_labels(
    times,
    intervals,
):
    labels = np.full(
        len(times),
        -1,
        dtype=np.int8,
    )

    for i, t in enumerate(
        times
    ):
        ratio = (
            window_overlap_ratio(
                t,
                intervals,
            )
        )

        if (
            ratio
            >= POSITIVE_OVERLAP
        ):
            labels[i] = 1

        elif (
            ratio
            <= NEGATIVE_OVERLAP
        ):
            labels[i] = 0

    return labels


# ============================================================
# 16. PROBABILITY + EVENTS
# ============================================================

def smooth_probability(
    probabilities,
):
    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    points = max(
        1,
        int(
            round(
                PROBABILITY_SMOOTH_SECONDS
                / HOP_SECONDS
            )
        ),
    )

    if points <= 1:
        return p.copy()

    out = np.empty_like(
        p
    )

    radius = (
        points // 2
    )

    for i in range(
        len(p)
    ):
        a = max(
            0,
            i - radius,
        )

        b = min(
            len(p),
            i + radius + 1,
        )

        out[i] = float(
            np.median(
                p[a:b]
            )
        )

    return out


def probabilities_to_events(
    times,
    probabilities,
    start_threshold,
    end_threshold,
):
    times = np.asarray(
        times,
        dtype=np.float64,
    )

    p = smooth_probability(
        probabilities
    )

    events = []

    active = False

    start_count = 0
    end_count = 0

    candidate_start_index = None

    current_start = None
    last_high_time = None

    for i, (
        t,
        prob,
    ) in enumerate(
        zip(
            times,
            p,
        )
    ):

        if not active:

            if (
                prob
                >= start_threshold
            ):
                if (
                    start_count
                    == 0
                ):
                    candidate_start_index = i

                start_count += 1

            else:
                start_count = 0
                candidate_start_index = None

            if (
                start_count
                >= START_CONFIRM_POINTS
            ):
                active = True

                start_i = (
                    candidate_start_index
                    if candidate_start_index
                    is not None
                    else i
                )

                current_start = float(
                    times[
                        start_i
                    ]
                )

                last_high_time = float(
                    t
                )

                start_count = 0
                end_count = 0

        else:

            if (
                prob
                >= end_threshold
            ):
                end_count = 0

                # Только устойчивые вероятности продлевают событие.
                if (
                    prob
                    >= start_threshold
                ):
                    last_high_time = float(
                        t
                    )

            else:
                end_count += 1

            if (
                end_count
                >= END_CONFIRM_POINTS
            ):

                first_low_i = max(
                    0,
                    i
                    - END_CONFIRM_POINTS
                    + 1,
                )

                end_time = float(
                    times[
                        first_low_i
                    ]
                )

                if (
                    last_high_time
                    is not None
                ):
                    end_time = max(
                        last_high_time,
                        end_time,
                    )

                if (
                    current_start
                    is not None
                    and (
                        end_time
                        - current_start
                    )
                    >= MIN_EVENT_SECONDS
                ):
                    events.append(
                        {
                            "start":
                                current_start,

                            "end":
                                end_time,
                        }
                    )

                active = False

                start_count = 0
                end_count = 0

                candidate_start_index = None
                current_start = None
                last_high_time = None

    if (
        active
        and current_start
        is not None
    ):
        end_time = float(
            last_high_time
            if last_high_time
            is not None
            else times[-1]
        )

        if (
            end_time
            - current_start
            >= MIN_EVENT_SECONDS
        ):
            events.append(
                {
                    "start":
                        current_start,

                    "end":
                        end_time,
                }
            )

    # Merge only tiny gaps.
    merged = []

    for event in events:

        if (
            merged
            and event["start"]
            - merged[-1]["end"]
            <= MERGE_GAP_SECONDS
        ):
            merged[-1]["end"] = max(
                merged[-1]["end"],
                event["end"],
            )

        else:
            merged.append(
                dict(event)
            )

    for i, event in enumerate(
        merged,
        1,
    ):
        event["event"] = i

        event["duration"] = (
            event["end"]
            - event["start"]
        )

    return merged


# ============================================================
# 17. CALIBRATION METRICS
# ============================================================

def intervals_mask(
    times,
    intervals,
):
    mask = np.zeros(
        len(times),
        dtype=bool,
    )

    for start, end in intervals:
        mask |= (
            (times >= start)
            & (times <= end)
        )

    return mask


def events_mask(
    times,
    events,
):
    mask = np.zeros(
        len(times),
        dtype=bool,
    )

    for event in events:
        mask |= (
            (
                times
                >= event["start"]
            )
            & (
                times
                <= event["end"]
            )
        )

    return mask


def binary_metrics(
    true,
    pred,
):
    true = np.asarray(
        true,
        dtype=bool,
    )

    pred = np.asarray(
        pred,
        dtype=bool,
    )

    tp = int(
        np.sum(
            true & pred
        )
    )

    fp = int(
        np.sum(
            ~true & pred
        )
    )

    fn = int(
        np.sum(
            true & ~pred
        )
    )

    precision = (
        tp
        / max(
            1,
            tp + fp,
        )
    )

    recall = (
        tp
        / max(
            1,
            tp + fn,
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / max(
            1e-12,
            precision + recall,
        )
    )

    union = int(
        np.sum(
            true | pred
        )
    )

    iou = (
        tp
        / max(
            1,
            union,
        )
    )

    return {
        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "iou":
            iou,
    }


def boundary_score(
    predicted,
    true_intervals,
):
    """
    Оценивает не только overlap,
    но и точность начала/конца.
    """

    if not true_intervals:
        return 1.0

    if not predicted:
        return 0.0

    scores = []

    for gt_start, gt_end in (
        true_intervals
    ):
        best = None

        for event in predicted:

            overlap = overlap_seconds(
                gt_start,
                gt_end,
                event["start"],
                event["end"],
            )

            if (
                best is None
                or overlap
                > best[0]
            ):
                best = (
                    overlap,
                    event,
                )

        if (
            best is None
            or best[0] <= 0
        ):
            scores.append(
                0.0
            )

            continue

        event = best[1]

        start_error = abs(
            event["start"]
            - gt_start
        )

        end_error = abs(
            event["end"]
            - gt_end
        )

        # Ошибка 3 сек на границе уже считается большой.
        score = max(
            0.0,
            1.0
            - (
                start_error
                + end_error
            )
            / 6.0,
        )

        scores.append(
            score
        )

    return float(
        np.mean(
            scores
        )
    )


# ============================================================
# 18. TRAIN
# ============================================================

def train_and_calibrate():
    datasets = []

    X_parts = []
    y_parts = []

    sample_groups = []

    print()
    print(
        "=" * 72
    )
    print(
        "P07 V3 — ОБУЧЕНИЕ"
    )
    print(
        "=" * 72
    )

    for video_index, (
        path,
        intervals,
    ) in enumerate(
        CALIBRATION_VIDEOS.items()
    ):

        if not Path(
            path
        ).exists():
            raise FileNotFoundError(
                f"Нет калибровочного видео:\n{path}"
            )

        video = extract_video(
            path
        )

        X_all, times_all = (
            all_window_features(
                video
            )
        )

        labels_all = (
            make_training_labels(
                times_all,
                intervals,
            )
        )

        use = (
            labels_all >= 0
        )

        X_train = (
            X_all[
                use
            ]
        )

        y_train = (
            labels_all[
                use
            ]
        )

        print()
        print(
            Path(
                path
            ).name
        )

        print(
            "  training positive:",
            int(
                np.sum(
                    y_train == 1
                )
            ),
        )

        print(
            "  training negative:",
            int(
                np.sum(
                    y_train == 0
                )
            ),
        )

        X_parts.append(
            X_train
        )

        y_parts.append(
            y_train
        )

        sample_groups.append(
            np.full(
                len(
                    y_train
                ),
                video_index,
                dtype=np.int32,
            )
        )

        datasets.append(
            {
                "path":
                    path,

                "name":
                    Path(
                        path
                    ).name,

                "intervals":
                    intervals,

                "video":
                    video,

                "X_all":
                    X_all,

                "times_all":
                    times_all,
            }
        )

    X = np.vstack(
        X_parts
    )

    y = np.concatenate(
        y_parts
    )

    groups = np.concatenate(
        sample_groups
    )

    if (
        len(
            np.unique(
                y
            )
        )
        < 2
    ):
        raise RuntimeError(
            "Для обучения нужны и positive, и negative окна."
        )

    # Вес каждой пары (video, class) делаем примерно одинаковым.
    sample_weight = np.ones(
        len(y),
        dtype=np.float64,
    )

    unique_groups = np.unique(
        groups
    )

    for group in unique_groups:

        for label in [
            0,
            1,
        ]:
            mask = (
                (groups == group)
                & (y == label)
            )

            count = int(
                np.sum(
                    mask
                )
            )

            if count > 0:
                sample_weight[
                    mask
                ] = (
                    1.0
                    / count
                )

    sample_weight *= (
        len(
            sample_weight
        )
        / np.sum(
            sample_weight
        )
    )

    model = (
        RandomForestClassifier(
            n_estimators=
                RF_TREES,

            max_depth=
                RF_MAX_DEPTH,

            min_samples_leaf=
                RF_MIN_SAMPLES_LEAF,

            max_features=
                "sqrt",

            class_weight=
                "balanced_subsample",

            bootstrap=
                True,

            oob_score=
                True,

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1,
        )
    )

    model.fit(
        X,
        y,
        sample_weight=
            sample_weight,
    )

    print()
    print(
        f"OOB accuracy: "
        f"{model.oob_score_:.3f}"
    )

    # --------------------------------------------------------
    # Event threshold calibration on annotated videos.
    # --------------------------------------------------------

    prediction_sets = []

    for data in datasets:

        probability = (
            model.predict_proba(
                data[
                    "X_all"
                ]
            )[:, 1]
        )

        prediction_sets.append(
            {
                **data,

                "probability":
                    probability,
            }
        )

    best = None

    for start_th in (
        START_THRESHOLD_GRID
    ):

        for end_th in (
            END_THRESHOLD_GRID
        ):

            if (
                end_th
                >= start_th
                - 0.05
            ):
                continue

            per_video = []

            for data in (
                prediction_sets
            ):

                events = (
                    probabilities_to_events(
                        data[
                            "times_all"
                        ],

                        data[
                            "probability"
                        ],

                        float(
                            start_th
                        ),

                        float(
                            end_th
                        ),
                    )
                )

                true_mask = intervals_mask(
                    data[
                        "times_all"
                    ],

                    data[
                        "intervals"
                    ],
                )

                pred_mask = events_mask(
                    data[
                        "times_all"
                    ],
                    events,
                )

                metrics = binary_metrics(
                    true_mask,
                    pred_mask,
                )

                bscore = boundary_score(
                    events,
                    data[
                        "intervals"
                    ],
                )

                extra_events = max(
                    0,
                    len(events)
                    - len(
                        data[
                            "intervals"
                        ]
                    ),
                )

                # Boundary и IoU имеют высокий вес:
                # "событие на весь ролик" здесь проигрывает.
                objective = (
                    0.38
                    * metrics[
                        "iou"
                    ]
                    + 0.27
                    * metrics[
                        "f1"
                    ]
                    + 0.25
                    * bscore
                    + 0.10
                    * metrics[
                        "precision"
                    ]
                    - 0.05
                    * extra_events
                )

                per_video.append(
                    {
                        "events":
                            events,

                        "metrics":
                            metrics,

                        "boundary":
                            bscore,

                        "objective":
                            objective,
                    }
                )

            macro_objective = float(
                np.mean(
                    [
                        item[
                            "objective"
                        ]
                        for item
                        in per_video
                    ]
                )
            )

            macro_iou = float(
                np.mean(
                    [
                        item[
                            "metrics"
                        ][
                            "iou"
                        ]
                        for item
                        in per_video
                    ]
                )
            )

            macro_boundary = float(
                np.mean(
                    [
                        item[
                            "boundary"
                        ]
                        for item
                        in per_video
                    ]
                )
            )

            key = (
                macro_objective,
                macro_iou,
                macro_boundary,
                float(
                    start_th
                ),
            )

            if (
                best is None
                or key
                > best[
                    "key"
                ]
            ):
                best = {
                    "key":
                        key,

                    "start_threshold":
                        float(
                            start_th
                        ),

                    "end_threshold":
                        float(
                            end_th
                        ),

                    "per_video":
                        per_video,

                    "objective":
                        macro_objective,

                    "macro_iou":
                        macro_iou,

                    "macro_boundary":
                        macro_boundary,
                }

    if best is None:
        raise RuntimeError(
            "Не удалось откалибровать event thresholds."
        )

    calibration = {
        "version":
            "p07_repetition_rf_v3",

        "window_seconds":
            WINDOW_SECONDS,

        "hop_seconds":
            HOP_SECONDS,

        "start_threshold":
            best[
                "start_threshold"
            ],

        "end_threshold":
            best[
                "end_threshold"
            ],

        "macro_iou":
            best[
                "macro_iou"
            ],

        "macro_boundary_score":
            best[
                "macro_boundary"
            ],

        "oob_accuracy":
            float(
                model.oob_score_
            ),

        "calibration_videos":
            {
                Path(
                    path
                ).name:
                    intervals

                for path, intervals
                in CALIBRATION_VIDEOS.items()
            },
    }

    Path(
        RF_MODEL_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model":
                model,

            "calibration":
                calibration,
        },
        RF_MODEL_PATH,
    )

    with open(
        CALIBRATION_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            calibration,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "=" * 72
    )
    print(
        "КАЛИБРОВКА ГРАНИЦ"
    )
    print(
        "=" * 72
    )

    print(
        "START threshold:",
        f"{best['start_threshold']:.2f}",
    )

    print(
        "END threshold:  ",
        f"{best['end_threshold']:.2f}",
    )

    print(
        "Macro IoU:      ",
        f"{best['macro_iou']:.3f}",
    )

    print(
        "Boundary score: ",
        f"{best['macro_boundary']:.3f}",
    )

    for data, result in zip(
        prediction_sets,
        best[
            "per_video"
        ],
    ):

        print()
        print(
            data[
                "name"
            ]
        )

        print(
            "  Эталон:",
            ", ".join(
                [
                    (
                        f"{fmt_time(a)}"
                        f"–"
                        f"{fmt_time(b)}"
                    )
                    for a, b
                    in data[
                        "intervals"
                    ]
                ]
            ),
        )

        if not result[
            "events"
        ]:
            print(
                "  Найдено: нет"
            )

        else:
            for event in result[
                "events"
            ]:
                print(
                    "  Найдено:",
                    f"{fmt_time(event['start'])}"
                    f"–"
                    f"{fmt_time(event['end'])}",
                )

        print(
            "  IoU:",
            f"{result['metrics']['iou']:.3f}",
            "| F1:",
            f"{result['metrics']['f1']:.3f}",
            "| Boundary:",
            f"{result['boundary']:.3f}",
        )

    print()
    print(
        "Модель:",
        RF_MODEL_PATH,
    )

    print(
        "Калибровка:",
        CALIBRATION_JSON,
    )

    print(
        "=" * 72
    )

    return {
        "model":
            model,

        "calibration":
            calibration,
    }


# ============================================================
# 19. LOAD
# ============================================================

def load_bundle():
    if not Path(
        RF_MODEL_PATH
    ).exists():
        raise FileNotFoundError(
            "Нет обученной модели:\n"
            f"{RF_MODEL_PATH}\n"
            "Поставь RETRAIN = False."
        )

    bundle = joblib.load(
        RF_MODEL_PATH
    )

    return bundle


# ============================================================
# 20. OVERLAY
# ============================================================

def state_at_time(
    events,
    t,
):
    for event in events:

        if (
            event["start"]
            <= t
            <= event["end"]
        ):
            return {
                "active":
                    True,

                "start":
                    event[
                        "start"
                    ],

                "end":
                    None,
            }

    previous = [
        event
        for event in events
        if event["end"] < t
    ]

    if previous:

        event = previous[
            -1
        ]

        return {
            "active":
                False,

            "start":
                event[
                    "start"
                ],

            "end":
                event[
                    "end"
                ],
        }

    return None


def draw_overlay(
    frame,
    state,
):
    if state is None:
        return frame

    h, w = (
        frame.shape[
            :2
        ]
    )

    scale = float(
        np.clip(
            np.sqrt(
                min(
                    h,
                    w,
                )
                / 720.0
            ),
            0.80,
            1.35,
        )
    )

    title_font = (
        load_overlay_font(
            max(
                15,
                int(
                    round(
                        18
                        * scale
                    )
                ),
            )
        )
    )

    body_font = (
        load_overlay_font(
            max(
                14,
                int(
                    round(
                        16
                        * scale
                    )
                ),
            )
        )
    )

    line_h = max(
        22,
        int(
            round(
                25
                * scale
            )
        ),
    )

    margin = max(
        10,
        int(
            round(
                15
                * scale
            )
        ),
    )

    pad_x = max(
        12,
        int(
            round(
                14
                * scale
            )
        ),
    )

    pad_y = max(
        10,
        int(
            round(
                12
                * scale
            )
        ),
    )

    lines = [
        "P07 — ПОВТОРЯЮЩИЕСЯ ДВИЖЕНИЯ",

        (
            "Начало: "
            + fmt_time(
                state[
                    "start"
                ]
            )
        ),

        (
            "Конец: "
            + (
                "—"
                if state[
                    "active"
                ]
                else fmt_time(
                    state[
                        "end"
                    ]
                )
            )
        ),
    ]

    dummy = Image.new(
        "RGB",
        (
            10,
            10,
        ),
    )

    md = ImageDraw.Draw(
        dummy
    )

    widths = []

    for i, line in enumerate(
        lines
    ):
        font = (
            title_font
            if i == 0
            else body_font
        )

        box = md.textbbox(
            (
                0,
                0,
            ),
            line,
            font=font,
        )

        widths.append(
            box[2]
            - box[0]
        )

    panel_w = (
        max(
            widths
        )
        + 2
        * pad_x
    )

    panel_h = (
        len(
            lines
        )
        * line_h
        + 2
        * pad_y
    )

    x1 = (
        w
        - margin
        - panel_w
    )

    y1 = margin

    x2 = (
        w
        - margin
    )

    y2 = (
        y1
        + panel_h
    )

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        PANEL_ALPHA,
        frame,
        1.0 - PANEL_ALPHA,
        0,
    )

    image = Image.fromarray(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )
    )

    draw = ImageDraw.Draw(
        image
    )

    for i, line in enumerate(
        lines
    ):
        font = (
            title_font
            if i == 0
            else body_font
        )

        draw.text(
            (
                x1 + pad_x,
                y1
                + pad_y
                + i
                * line_h,
            ),
            line,
            font=font,
            fill=(
                255,
                255,
                255,
            ),
        )

    return cv2.cvtColor(
        np.asarray(
            image
        ),
        cv2.COLOR_RGB2BGR,
    )



# ============================================================
# P07 -> JSON / ГРАФИК
# ============================================================
#
# ВАЖНО:
# ниже НЕТ новой логики детекции.
#
# JSON и график строятся ТОЛЬКО из уже готовых:
# - events
# - probability
# - smoothed
# - start_threshold
# - end_threshold
#
# То есть probabilities_to_events() остаётся единственным местом,
# где решается START / END P07.
# ============================================================


def format_axis_time(
    seconds,
):
    """
    Формат оси X:
    00:00, 00:30, 01:00, 01:30 ...
    """

    seconds = max(
        0.0,
        float(
            seconds
        ),
    )

    total = int(
        round(
            seconds
        )
    )

    minutes = (
        total // 60
    )

    sec = (
        total % 60
    )

    return (
        f"{minutes:02d}:"
        f"{sec:02d}"
    )


def human_duration(
    seconds,
):
    seconds = max(
        0.0,
        float(
            seconds
        ),
    )

    total = int(
        round(
            seconds
        )
    )

    hours = (
        total // 3600
    )

    minutes = (
        (
            total % 3600
        )
        // 60
    )

    sec = (
        total % 60
    )

    if hours > 0:
        return (
            f"{hours}ч "
            f"{minutes}м "
            f"{sec}с"
        )

    if minutes > 0:
        return (
            f"{minutes}м "
            f"{sec}с"
        )

    return (
        f"{sec}с"
    )


def safe_float(
    value,
):
    try:
        value = float(
            value
        )

        if np.isfinite(
            value
        ):
            return value

    except Exception:
        pass

    return None


def build_p07_cuts(
    events,
    times,
    probability,
    smoothed,
    start_threshold,
    end_threshold,
):
    """
    Переводит УЖЕ готовые events в структуру cuts.
    Детекцию не меняет.
    """

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    smoothed = np.asarray(
        smoothed,
        dtype=np.float64,
    )

    cuts = []

    for event in events:

        start = float(
            event[
                "start"
            ]
        )

        end = float(
            event[
                "end"
            ]
        )

        duration = float(
            event[
                "duration"
            ]
        )

        mask = (
            (
                times >= start
            )
            &
            (
                times <= end
            )
        )

        raw_values = (
            probability[
                mask
            ]
        )

        smooth_values = (
            smoothed[
                mask
            ]
        )

        mean_probability = (
            float(
                np.nanmean(
                    raw_values
                )
            )
            if
            raw_values.size > 0
            else
            None
        )

        max_probability = (
            float(
                np.nanmax(
                    raw_values
                )
            )
            if
            raw_values.size > 0
            else
            None
        )

        mean_smoothed_probability = (
            float(
                np.nanmean(
                    smooth_values
                )
            )
            if
            smooth_values.size > 0
            else
            None
        )

        max_smoothed_probability = (
            float(
                np.nanmax(
                    smooth_values
                )
            )
            if
            smooth_values.size > 0
            else
            None
        )

        cuts.append(
            {
                "source":
                    "mediapipe+random_forest",

                "pattern":
                    "P07",

                "name":
                    "повторные мелкие движения",

                "event_number":
                    int(
                        event[
                            "event"
                        ]
                    ),

                "t_start":
                    round(
                        start,
                        3
                    ),

                "t_end":
                    round(
                        end,
                        3
                    ),

                "start_sec":
                    round(
                        start,
                        3
                    ),

                "end_sec":
                    round(
                        end,
                        3
                    ),

                "duration_sec":
                    round(
                        duration,
                        3
                    ),

                # Диагностика того же Random Forest,
                # уже рассчитанная исходным скриптом.
                "mean_probability":
                    (
                        round(
                            mean_probability,
                            6
                        )
                        if
                        mean_probability
                        is not None
                        else
                        None
                    ),

                "max_probability":
                    (
                        round(
                            max_probability,
                            6
                        )
                        if
                        max_probability
                        is not None
                        else
                        None
                    ),

                "mean_smoothed_probability":
                    (
                        round(
                            mean_smoothed_probability,
                            6
                        )
                        if
                        mean_smoothed_probability
                        is not None
                        else
                        None
                    ),

                "max_smoothed_probability":
                    (
                        round(
                            max_smoothed_probability,
                            6
                        )
                        if
                        max_smoothed_probability
                        is not None
                        else
                        None
                    ),

                "start_threshold":
                    round(
                        float(
                            start_threshold
                        ),
                        6
                    ),

                "end_threshold":
                    round(
                        float(
                            end_threshold
                        ),
                        6
                    ),

                # Совместимость с Alyona_interview_results.json.
                "clip_pad_sec":
                    0.0,

                "clip_file":
                    None,

                "clip":
                    None,

                "scenario_label":
                    None,

                "scenario_kind":
                    None,
            }
        )

    return cuts


def save_p07_timeline_graph(
    times,
    probability,
    smoothed,
    events,
    start_threshold,
    end_threshold,
    output_path,
):
    """
    График напрямую показывает:
    - вероятность Random Forest;
    - сглаженную вероятность;
    - START threshold;
    - END threshold;
    - интервалы P07;
    - START и END каждого P07.

    Ось времени — MM:SS, шаг 30 секунд.
    """

    try:
        import matplotlib.pyplot as plt

    except ImportError:
        print(
            "Предупреждение: matplotlib не установлен. "
            "JSON будет сохранён, но PNG-график не создан.\n"
            "Установка: pip install matplotlib"
        )

        return False


    times = np.asarray(
        times,
        dtype=np.float64,
    )

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    smoothed = np.asarray(
        smoothed,
        dtype=np.float64,
    )


    fig, ax = plt.subplots(
        figsize=(
            16,
            8
        ),
        dpi=140,
    )


    # Исходная вероятность — тоньше.
    ax.plot(
        times,
        probability,
        linewidth=1.0,
        alpha=0.30,
        label="Вероятность RF",
    )

    # Сглаженная вероятность — основная линия.
    ax.plot(
        times,
        smoothed,
        linewidth=2.7,
        label="Сглаженная вероятность",
    )


    ax.axhline(
        start_threshold,
        linewidth=1.6,
        linestyle="--",
        label=(
            "START threshold "
            f"{start_threshold:.2f}"
        ),
    )

    ax.axhline(
        end_threshold,
        linewidth=1.4,
        linestyle=":",
        label=(
            "END threshold "
            f"{end_threshold:.2f}"
        ),
    )


    # Реальные интервалы, уже найденные исходной state machine.
    first_interval = True
    first_start = True
    first_end = True

    for event in events:

        start = float(
            event[
                "start"
            ]
        )

        end = float(
            event[
                "end"
            ]
        )

        ax.axvspan(
            start,
            end,
            alpha=0.10,
            label=(
                "P07 активен"
                if first_interval
                else None
            ),
        )

        ax.axvline(
            start,
            linewidth=1.2,
            linestyle="--",
            alpha=0.55,
            label=(
                "START P07"
                if first_start
                else None
            ),
        )

        ax.axvline(
            end,
            linewidth=1.2,
            linestyle=":",
            alpha=0.55,
            label=(
                "END P07"
                if first_end
                else None
            ),
        )

        first_interval = False
        first_start = False
        first_end = False


    max_time = (
        float(
            np.nanmax(
                times
            )
        )
        if
        len(
            times
        ) > 0
        else
        0.0
    )

    tick_step = 30.0

    x_ticks = np.arange(
        0.0,
        max_time
        +
        tick_step,
        tick_step,
        dtype=np.float64,
    )

    if len(
        x_ticks
    ) == 0:
        x_ticks = np.asarray(
            [
                0.0
            ],
            dtype=np.float64,
        )


    ax.set_xticks(
        x_ticks
    )

    ax.set_xticklabels(
        [
            format_axis_time(
                value
            )
            for value
            in x_ticks
        ],
        fontsize=10,
    )


    ax.set_ylim(
        0.0,
        1.02,
    )


    ax.set_title(
        "P07 — повторные мелкие движения во времени",
        fontsize=16,
        pad=18,
    )

    ax.set_xlabel(
        "Время видео",
        fontsize=12,
    )

    ax.set_ylabel(
        "Вероятность P07",
        fontsize=12,
    )


    fig.text(
        0.125,
        0.94,
        (
            "Закрашенная область = P07 активен | "
            "вертикальные линии = START / END"
        ),
        fontsize=10,
    )


    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=0.28,
        linestyle="--",
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    legend.get_frame().set_alpha(
        0.90
    )


    fig.text(
        0.125,
        0.02,
        (
            "График не участвует в детекции: "
            "он только визуализирует уже рассчитанные "
            "Random Forest probability и P07 events."
        ),
        fontsize=9,
    )


    plt.tight_layout(
        rect=[
            0.03,
            0.05,
            0.99,
            0.92,
        ]
    )


    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return True


def build_p07_results_json(
    cuts,
    video,
    frames_processed,
    run_duration_sec,
    start_threshold,
    end_threshold,
    graph_created,
):
    p07_count = len(
        cuts
    )

    video_duration_sec = (
        video.frame_count
        /
        video.fps
        if
        video.fps > 0
        else
        0.0
    )

    processed_duration_sec = (
        frames_processed
        /
        video.fps
        if
        video.fps > 0
        else
        video_duration_sec
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
            "code":
                code,

            "name":
                name,

            "count":
                (
                    p07_count
                    if
                    code
                    ==
                    "P07"
                    else
                    0
                ),
        }
        for
        code,
        name
        in
        pattern_names
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
        "schema":
            "bs_profiling_cut_v1",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "tracker":
            "MediaPipe Pose + Hands + Random Forest P07 V3",

        "video":
            str(
                TARGET_VIDEO
            ),

        "video_name":
            Path(
                TARGET_VIDEO
            ).name,

        "timing": {
            "video_duration_sec":
                round(
                    video_duration_sec,
                    3
                ),

            "video_duration_human":
                human_duration(
                    video_duration_sec
                ),

            "video_duration_min":
                round(
                    video_duration_sec
                    /
                    60.0,
                    2
                ),

            "run_duration_sec":
                round(
                    run_duration_sec,
                    2
                ),

            "run_duration_human":
                human_duration(
                    run_duration_sec
                ),

            "run_duration_min":
                round(
                    run_duration_sec
                    /
                    60.0,
                    2
                ),

            "processed_duration_sec":
                round(
                    processed_duration_sec,
                    3
                ),

            "processed_duration_human":
                human_duration(
                    processed_duration_sec
                ),

            "processed_full_video":
                bool(
                    frames_processed
                    >=
                    video.frame_count
                ),

            "speed_ratio":
                round(
                    speed_ratio,
                    3
                ),

            "wall_sec_per_video_min":
                round(
                    wall_sec_per_video_min,
                    1
                ),

            "frames_total":
                int(
                    video.frame_count
                ),

            "frames_processed":
                int(
                    frames_processed
                ),

            "video_fps":
                round(
                    float(
                        video.fps
                    ),
                    3
                ),

            "effective_fps":
                round(
                    effective_fps,
                    3
                ),

            "run_finished_local":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "note":
                (
                    "P07 detector unchanged; JSON and graph "
                    "are generated from existing P07 events."
                ),
        },

        "duration_processed_sec":
            round(
                processed_duration_sec,
                3
            ),

        "video_file_duration_sec":
            round(
                video_duration_sec,
                3
            ),

        "timeline_file":
            None,

        "timeline_segments":
            [],

        "patterns_hud":
            patterns_hud,

        "cuts":
            cuts,

        "p07_model": {
            "model_type":
                "RandomForestClassifier",

            "rf_model_path":
                str(
                    RF_MODEL_PATH
                ),

            "start_threshold":
                round(
                    float(
                        start_threshold
                    ),
                    6
                ),

            "end_threshold":
                round(
                    float(
                        end_threshold
                    ),
                    6
                ),

            "window_seconds":
                float(
                    WINDOW_SECONDS
                ),

            "hop_seconds":
                float(
                    HOP_SECONDS
                ),

            "probability_smooth_seconds":
                float(
                    PROBABILITY_SMOOTH_SECONDS
                ),

            "start_confirm_points":
                int(
                    START_CONFIRM_POINTS
                ),

            "end_confirm_points":
                int(
                    END_CONFIRM_POINTS
                ),

            "min_event_seconds":
                float(
                    MIN_EVENT_SECONDS
                ),

            "merge_gap_seconds":
                float(
                    MERGE_GAP_SECONDS
                ),
        },

        "p07_timeline_graph":
            str(
                OUTPUT_GRAPH
            ),

        "p07_timeline_graph_created":
            bool(
                graph_created
            ),

        "overlay_timeline_mp4":
            str(
                OUTPUT_VIDEO
            ),

        "event_csv":
            str(
                OUTPUT_CSV
            ),

        "score_csv":
            str(
                OUTPUT_SCORE_CSV
            ),

        "notes": [
            (
                "P07 = повторные мелкие движения."
            ),
            (
                "START/END/duration берутся напрямую из "
                "исходной probabilities_to_events()."
            ),
            (
                "JSON и график не меняют Random Forest, "
                "thresholds, smoothing или event state machine."
            ),
            (
                "График показывает вероятность модели и "
                "фактические START/END уже найденных P07."
            ),
            (
                "clip_file/clip = null, потому что этот скрипт "
                "не вырезает отдельные event clips."
            ),
        ],
    }


# ============================================================
# 21. INFERENCE
# ============================================================

def process_target(
    bundle,
):
    run_started_perf = (
        time.perf_counter()
    )

    model = (
        bundle[
            "model"
        ]
    )

    calibration = (
        bundle[
            "calibration"
        ]
    )

    if not Path(
        TARGET_VIDEO
    ).exists():
        raise FileNotFoundError(
            f"Нет TARGET_VIDEO:\n{TARGET_VIDEO}"
        )

    video = extract_video(
        TARGET_VIDEO
    )

    X, times = (
        all_window_features(
            video
        )
    )

    if len(X) == 0:
        raise RuntimeError(
            "Не удалось получить окна признаков."
        )

    probability = (
        model.predict_proba(
            X
        )[:, 1]
    )

    smoothed = (
        smooth_probability(
            probability
        )
    )

    start_threshold = float(
        calibration[
            "start_threshold"
        ]
    )

    end_threshold = float(
        calibration[
            "end_threshold"
        ]
    )

    events = (
        probabilities_to_events(
            times,
            probability,
            start_threshold,
            end_threshold,
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "P07 V4 — НАЙДЕННЫЕ СОБЫТИЯ"
    )
    print(
        "=" * 72
    )

    if not events:
        print(
            "P07 не обнаружен"
        )

    for event in events:
        print(
            f"#{event['event']}: "
            f"{fmt_time(event['start'])}"
            f" -> "
            f"{fmt_time(event['end'])}"
            f" | "
            f"{event['duration']:.2f} сек"
        )

    # --------------------------------------------------------
    # Save event CSV
    # --------------------------------------------------------

    Path(
        OUTPUT_CSV
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    Path(
        OUTPUT_JSON
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        OUTPUT_GRAPH
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "event",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "start_time",
                "end_time",
            ]
        )

        for event in events:

            writer.writerow(
                [
                    event[
                        "event"
                    ],

                    round(
                        event[
                            "start"
                        ],
                        3,
                    ),

                    round(
                        event[
                            "end"
                        ],
                        3,
                    ),

                    round(
                        event[
                            "duration"
                        ],
                        3,
                    ),

                    fmt_time(
                        event[
                            "start"
                        ]
                    ),

                    fmt_time(
                        event[
                            "end"
                        ]
                    ),
                ]
            )

    # --------------------------------------------------------
    # Save probability diagnostics
    # --------------------------------------------------------

    with open(
        OUTPUT_SCORE_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "time_seconds",
                "probability",
                "smoothed_probability",
                "start_threshold",
                "end_threshold",
            ]
        )

        for t, p, ps in zip(
            times,
            probability,
            smoothed,
        ):
            writer.writerow(
                [
                    round(
                        float(t),
                        3,
                    ),

                    round(
                        float(p),
                        5,
                    ),

                    round(
                        float(ps),
                        5,
                    ),

                    start_threshold,
                    end_threshold,
                ]
            )

    # --------------------------------------------------------
    # Render video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        TARGET_VIDEO
    )

    # В Colab гарантируем существование папок результатов.
    for output_path in [
        OUTPUT_VIDEO,
        OUTPUT_CSV,
        OUTPUT_SCORE_CSV,
        OUTPUT_JSON,
        OUTPUT_GRAPH,
    ]:
        Path(
            output_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        video.fps,
        (
            video.width,
            video.height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Не удалось создать:\n{OUTPUT_VIDEO}"
        )

    i = 0

    while True:

        ok, frame = (
            cap.read()
        )

        if not ok:
            break

        t = (
            i
            / video.fps
        )

        state = (
            state_at_time(
                events,
                t,
            )
        )

        frame = draw_overlay(
            frame,
            state,
        )

        writer.write(
            frame
        )

        if SHOW_PREVIEW and not IN_GOOGLE_COLAB:

            preview_width = min(
                1280,
                video.width,
            )

            preview_scale = (
                preview_width
                / video.width
            )

            preview = cv2.resize(
                frame,
                (
                    preview_width,
                    int(
                        video.height
                        * preview_scale
                    ),
                ),
            )

            cv2.imshow(
                "P07 V3",
                preview,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if (
                key == ord("q")
                or key == 27
            ):
                break

        i += 1

    cap.release()
    writer.release()

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    # ========================================================
    # P07 JSON + TIMELINE GRAPH
    # ========================================================
    #
    # Детекция к этому моменту уже полностью завершена.
    # Ниже только сохранение/визуализация её результатов.

    p07_cuts = (
        build_p07_cuts(
            events=
                events,

            times=
                times,

            probability=
                probability,

            smoothed=
                smoothed,

            start_threshold=
                start_threshold,

            end_threshold=
                end_threshold,
        )
    )


    graph_created = (
        save_p07_timeline_graph(
            times=
                times,

            probability=
                probability,

            smoothed=
                smoothed,

            events=
                events,

            start_threshold=
                start_threshold,

            end_threshold=
                end_threshold,

            output_path=
                OUTPUT_GRAPH,
        )
    )


    run_duration_sec = (
        time.perf_counter()
        -
        run_started_perf
    )


    results_json = (
        build_p07_results_json(
            cuts=
                p07_cuts,

            video=
                video,

            frames_processed=
                i,

            run_duration_sec=
                run_duration_sec,

            start_threshold=
                start_threshold,

            end_threshold=
                end_threshold,

            graph_created=
                graph_created,
        )
    )


    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            results_json,
            json_file,
            ensure_ascii=False,
            indent=2,
        )


    print()
    print(
        "Видео:",
        OUTPUT_VIDEO,
    )

    print(
        "Events CSV:",
        OUTPUT_CSV,
    )

    print(
        "Scores CSV:",
        OUTPUT_SCORE_CSV,
    )

    print(
        "JSON:",
        OUTPUT_JSON,
    )

    if graph_created:

        print(
            "График P07:",
            OUTPUT_GRAPH,
        )

    else:

        print(
            "График P07: не создан "
            "(установи matplotlib: pip install matplotlib)"
        )

    print(
        "P07 событий в JSON:",
        len(
            p07_cuts
        ),
    )

    print(
        "=" * 72
    )



# ============================================================
# HARD RECALIBRATION HELPERS
# ============================================================


def interval_union_mask(
    times,
    intervals,
):
    """
    Маска только явно размеченных участков.
    Всё остальное видео игнорируется при докалибровке.
    """

    mask = np.zeros(
        len(times),
        dtype=bool,
    )

    for start, end in intervals:
        mask |= (
            (times >= float(start))
            &
            (times <= float(end))
        )

    return mask


def restricted_binary_metrics(
    true_mask,
    pred_mask,
    eval_mask,
):
    """
    Те же binary metrics, но только внутри размеченных
    positive/negative интервалов.
    """

    if not np.any(
        eval_mask
    ):
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
        }

    return binary_metrics(
        true_mask[
            eval_mask
        ],
        pred_mask[
            eval_mask
        ],
    )


def event_overlaps_interval(
    event,
    interval,
):
    start, end = interval

    return (
        overlap_seconds(
            float(event["start"]),
            float(event["end"]),
            float(start),
            float(end),
        )
        > 0.0
    )


def hard_extra_events(
    events,
    positive_intervals,
    negative_intervals,
):
    """
    Считаем лишними только события, которые заходят
    в явно размеченные NEGATIVE участки.

    События вне размеченных hard intervals не штрафуем,
    потому что пользователь не говорил, что там обязательно negative.
    """

    count = 0

    for event in events:
        if any(
            event_overlaps_interval(
                event,
                interval,
            )
            for interval in negative_intervals
        ):
            count += 1

    return count


def positive_interval_stats(
    times,
    probability,
    smoothed,
    intervals,
):
    rows = []

    for index, (
        start,
        end,
    ) in enumerate(
        intervals,
        1,
    ):
        mask = (
            (times >= start)
            &
            (times <= end)
        )

        raw = probability[
            mask
        ]

        smooth = smoothed[
            mask
        ]

        rows.append(
            {
                "kind": "POSITIVE_FN",
                "interval_index": index,
                "start_sec": float(start),
                "end_sec": float(end),
                "duration_sec": float(
                    end - start
                ),

                "raw_mean": (
                    float(
                        np.mean(raw)
                    )
                    if len(raw)
                    else np.nan
                ),

                "raw_max": (
                    float(
                        np.max(raw)
                    )
                    if len(raw)
                    else np.nan
                ),

                "smooth_mean": (
                    float(
                        np.mean(smooth)
                    )
                    if len(smooth)
                    else np.nan
                ),

                "smooth_max": (
                    float(
                        np.max(smooth)
                    )
                    if len(smooth)
                    else np.nan
                ),

                "smooth_min": (
                    float(
                        np.min(smooth)
                    )
                    if len(smooth)
                    else np.nan
                ),
            }
        )

    return rows


def negative_interval_stats(
    times,
    probability,
    smoothed,
    intervals,
):
    rows = []

    for index, (
        start,
        end,
    ) in enumerate(
        intervals,
        1,
    ):
        mask = (
            (times >= start)
            &
            (times <= end)
        )

        raw = probability[
            mask
        ]

        smooth = smoothed[
            mask
        ]

        rows.append(
            {
                "kind": "NEGATIVE_FP",
                "interval_index": index,
                "start_sec": float(start),
                "end_sec": float(end),
                "duration_sec": float(
                    end - start
                ),

                "raw_mean": (
                    float(
                        np.mean(raw)
                    )
                    if len(raw)
                    else np.nan
                ),

                "raw_max": (
                    float(
                        np.max(raw)
                    )
                    if len(raw)
                    else np.nan
                ),

                "smooth_mean": (
                    float(
                        np.mean(smooth)
                    )
                    if len(smooth)
                    else np.nan
                ),

                "smooth_max": (
                    float(
                        np.max(smooth)
                    )
                    if len(smooth)
                    else np.nan
                ),

                "smooth_min": (
                    float(
                        np.min(smooth)
                    )
                    if len(smooth)
                    else np.nan
                ),
            }
        )

    return rows


def evaluate_hard_thresholds(
    times,
    probability,
    positive_intervals,
    negative_intervals,
    start_threshold,
    end_threshold,
):
    """
    Используем ту же state machine probabilities_to_events().
    Меняются только thresholds, которые подбираются.
    """

    events = probabilities_to_events(
        times,
        probability,
        float(
            start_threshold
        ),
        float(
            end_threshold
        ),
    )

    true_mask = interval_union_mask(
        times,
        positive_intervals,
    )

    negative_mask = interval_union_mask(
        times,
        negative_intervals,
    )

    eval_mask = (
        true_mask
        |
        negative_mask
    )

    pred_mask = events_mask(
        times,
        events,
    )

    metrics = restricted_binary_metrics(
        true_mask,
        pred_mask,
        eval_mask,
    )

    bscore = boundary_score(
        events,
        positive_intervals,
    )

    extra_events = hard_extra_events(
        events,
        positive_intervals,
        negative_intervals,
    )

    # Сохраняем исходную формулу objective из P07 V3.
    objective = (
        0.38
        * metrics[
            "iou"
        ]
        + 0.27
        * metrics[
            "f1"
        ]
        + 0.25
        * bscore
        + 0.10
        * metrics[
            "precision"
        ]
        - 0.05
        * extra_events
    )

    positive_hits = []

    for interval in positive_intervals:
        positive_hits.append(
            any(
                event_overlaps_interval(
                    event,
                    interval,
                )
                for event in events
            )
        )

    negative_hits = []

    for interval in negative_intervals:
        negative_hits.append(
            any(
                event_overlaps_interval(
                    event,
                    interval,
                )
                for event in events
            )
        )

    return {
        "events": events,
        "metrics": metrics,
        "boundary": float(
            bscore
        ),
        "extra_events": int(
            extra_events
        ),
        "objective": float(
            objective
        ),
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
    }


def save_hard_report_csv(
    path,
    interval_rows,
    old_result,
    new_result,
    old_start,
    old_end,
    new_start,
    new_end,
):
    Path(
        path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "kind",
            "interval_index",
            "start_sec",
            "end_sec",
            "duration_sec",
            "raw_mean",
            "raw_max",
            "smooth_mean",
            "smooth_max",
            "smooth_min",
            "old_start_threshold",
            "old_end_threshold",
            "new_start_threshold",
            "new_end_threshold",
            "old_event_overlap",
            "new_event_overlap",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        pos_old = old_result[
            "positive_hits"
        ]
        pos_new = new_result[
            "positive_hits"
        ]

        neg_old = old_result[
            "negative_hits"
        ]
        neg_new = new_result[
            "negative_hits"
        ]

        for row in interval_rows:

            out = dict(
                row
            )

            idx = int(
                row[
                    "interval_index"
                ]
            ) - 1

            if row[
                "kind"
            ] == "POSITIVE_FN":
                old_overlap = bool(
                    pos_old[
                        idx
                    ]
                )
                new_overlap = bool(
                    pos_new[
                        idx
                    ]
                )

            else:
                old_overlap = bool(
                    neg_old[
                        idx
                    ]
                )
                new_overlap = bool(
                    neg_new[
                        idx
                    ]
                )

            out.update(
                {
                    "old_start_threshold":
                        old_start,

                    "old_end_threshold":
                        old_end,

                    "new_start_threshold":
                        new_start,

                    "new_end_threshold":
                        new_end,

                    "old_event_overlap":
                        int(
                            old_overlap
                        ),

                    "new_event_overlap":
                        int(
                            new_overlap
                        ),
                }
            )

            writer.writerow(
                out
            )


def hard_recalibrate():
    print()
    print(
        "=" * 72
    )
    print(
        "P07 V3 — HARD RECALIBRATION / My_Interview"
    )
    print(
        "=" * 72
    )

    for path in [
        POSE_MODEL_PATH,
        HAND_MODEL_PATH,
        RF_MODEL_PATH,
        HARD_CALIBRATION_VIDEO,
    ]:
        if not Path(
            path
        ).exists():
            raise FileNotFoundError(
                f"Не найден файл:\n{path}"
            )

    bundle = load_bundle()

    model = bundle[
        "model"
    ]

    old_calibration = dict(
        bundle[
            "calibration"
        ]
    )

    old_start = float(
        old_calibration[
            "start_threshold"
        ]
    )

    old_end = float(
        old_calibration[
            "end_threshold"
        ]
    )

    print(
        "Текущие thresholds:"
    )
    print(
        "  START:",
        old_start,
    )
    print(
        "  END:  ",
        old_end,
    )

    print()
    print(
        "Positive hard intervals:"
    )
    for start, end in (
        HARD_POSITIVE_INTERVALS
    ):
        print(
            f"  {fmt_time(start)}–"
            f"{fmt_time(end)}"
        )

    print()
    print(
        "Negative hard intervals:"
    )
    for start, end in (
        HARD_NEGATIVE_INTERVALS
    ):
        print(
            f"  {fmt_time(start)}–"
            f"{fmt_time(end)}"
        )

    # --------------------------------------------------------
    # 1. Extract same features as normal P07 inference
    # --------------------------------------------------------

    video = extract_video(
        HARD_CALIBRATION_VIDEO
    )

    X, times = (
        all_window_features(
            video
        )
    )

    if len(
        X
    ) == 0:
        raise RuntimeError(
            "Не удалось получить окна признаков "
            "из My_Interview."
        )

    if (
        hasattr(
            model,
            "n_features_in_"
        )
        and int(
            model.n_features_in_
        )
        != int(
            X.shape[1]
        )
    ):
        raise RuntimeError(
            "Несовпадение числа признаков:\n"
            f"joblib ожидает {model.n_features_in_}, "
            f"скрипт сформировал {X.shape[1]}."
        )

    probability = (
        model.predict_proba(
            X
        )[:, 1]
    )

    smoothed = smooth_probability(
        probability
    )

    # --------------------------------------------------------
    # 2. Current model on new hard labels
    # --------------------------------------------------------

    old_result = (
        evaluate_hard_thresholds(
            times=
                times,

            probability=
                probability,

            positive_intervals=
                HARD_POSITIVE_INTERVALS,

            negative_intervals=
                HARD_NEGATIVE_INTERVALS,

            start_threshold=
                old_start,

            end_threshold=
                old_end,
        )
    )

    print()
    print(
        "ТЕКУЩАЯ КАЛИБРОВКА НА HARD-ПРИМЕРАХ"
    )
    print(
        f"  objective: "
        f'{old_result["objective"]:.4f}'
    )
    print(
        f"  IoU:       "
        f'{old_result["metrics"]["iou"]:.4f}'
    )
    print(
        f"  F1:        "
        f'{old_result["metrics"]["f1"]:.4f}'
    )
    print(
        f"  precision: "
        f'{old_result["metrics"]["precision"]:.4f}'
    )
    print(
        f"  recall:    "
        f'{old_result["metrics"]["recall"]:.4f}'
    )
    print(
        f"  boundary:  "
        f'{old_result["boundary"]:.4f}'
    )

    # --------------------------------------------------------
    # 3. Search using original P07 V3 threshold grids
    # --------------------------------------------------------

    best = None

    for start_th in (
        START_THRESHOLD_GRID
    ):

        for end_th in (
            END_THRESHOLD_GRID
        ):

            if (
                end_th
                >=
                start_th
                - 0.05
            ):
                continue

            result = evaluate_hard_thresholds(
                times=
                    times,

                probability=
                    probability,

                positive_intervals=
                    HARD_POSITIVE_INTERVALS,

                negative_intervals=
                    HARD_NEGATIVE_INTERVALS,

                start_threshold=
                    float(
                        start_th
                    ),

                end_threshold=
                    float(
                        end_th
                    ),
            )

            # Главный критерий — тот же objective.
            #
            # Если objective одинаковый, предпочитаем:
            # 1) выше IoU;
            # 2) лучше boundary;
            # 3) thresholds ближе к старым значениям.
            drift = (
                abs(
                    float(
                        start_th
                    )
                    -
                    old_start
                )
                +
                abs(
                    float(
                        end_th
                    )
                    -
                    old_end
                )
            )

            key = (
                result[
                    "objective"
                ],

                result[
                    "metrics"
                ][
                    "iou"
                ],

                result[
                    "boundary"
                ],

                -float(
                    drift
                ),
            )

            if (
                best is None
                or key
                >
                best[
                    "key"
                ]
            ):
                best = {
                    "key":
                        key,

                    "start_threshold":
                        float(
                            start_th
                        ),

                    "end_threshold":
                        float(
                            end_th
                        ),

                    "result":
                        result,
                }

    if best is None:
        raise RuntimeError(
            "Не удалось подобрать hard thresholds."
        )

    new_start = best[
        "start_threshold"
    ]

    new_end = best[
        "end_threshold"
    ]

    new_result = best[
        "result"
    ]

    print()
    print(
        "=" * 72
    )
    print(
        "НОВАЯ HARD-КАЛИБРОВКА"
    )
    print(
        "=" * 72
    )
    print(
        f"START: {old_start:.2f} "
        f"-> {new_start:.2f}"
    )
    print(
        f"END:   {old_end:.2f} "
        f"-> {new_end:.2f}"
    )
    print(
        f"objective: "
        f'{old_result["objective"]:.4f} '
        f'-> {new_result["objective"]:.4f}'
    )
    print(
        f"IoU:       "
        f'{old_result["metrics"]["iou"]:.4f} '
        f'-> {new_result["metrics"]["iou"]:.4f}'
    )
    print(
        f"F1:        "
        f'{old_result["metrics"]["f1"]:.4f} '
        f'-> {new_result["metrics"]["f1"]:.4f}'
    )
    print(
        f"precision: "
        f'{old_result["metrics"]["precision"]:.4f} '
        f'-> {new_result["metrics"]["precision"]:.4f}'
    )
    print(
        f"recall:    "
        f'{old_result["metrics"]["recall"]:.4f} '
        f'-> {new_result["metrics"]["recall"]:.4f}'
    )
    print(
        f"boundary:  "
        f'{old_result["boundary"]:.4f} '
        f'-> {new_result["boundary"]:.4f}'
    )

    # --------------------------------------------------------
    # 4. Interval probability diagnostics
    # --------------------------------------------------------

    interval_rows = (
        positive_interval_stats(
            times,
            probability,
            smoothed,
            HARD_POSITIVE_INTERVALS,
        )
        +
        negative_interval_stats(
            times,
            probability,
            smoothed,
            HARD_NEGATIVE_INTERVALS,
        )
    )

    save_hard_report_csv(
        HARD_OUTPUT_REPORT_CSV,
        interval_rows,
        old_result,
        new_result,
        old_start,
        old_end,
        new_start,
        new_end,
    )

    # --------------------------------------------------------
    # 5. Save NEW bundle — old file is NOT overwritten
    # --------------------------------------------------------

    new_calibration = dict(
        old_calibration
    )

    new_calibration.update(
        {
            "version":
                "p07_repetition_rf_v3_myinterview_hard_calibrated",

            "start_threshold":
                float(
                    new_start
                ),

            "end_threshold":
                float(
                    new_end
                ),

            "hard_recalibration": {
                "video":
                    Path(
                        HARD_CALIBRATION_VIDEO
                    ).name,

                "positive_intervals":
                    [
                        [
                            float(a),
                            float(b),
                        ]
                        for a, b
                        in HARD_POSITIVE_INTERVALS
                    ],

                "negative_intervals":
                    [
                        [
                            float(a),
                            float(b),
                        ]
                        for a, b
                        in HARD_NEGATIVE_INTERVALS
                    ],

                "old_start_threshold":
                    old_start,

                "old_end_threshold":
                    old_end,

                "old_metrics":
                    {
                        "objective":
                            old_result[
                                "objective"
                            ],

                        **old_result[
                            "metrics"
                        ],

                        "boundary":
                            old_result[
                                "boundary"
                            ],
                    },

                "new_metrics":
                    {
                        "objective":
                            new_result[
                                "objective"
                            ],

                        **new_result[
                            "metrics"
                        ],

                        "boundary":
                            new_result[
                                "boundary"
                            ],
                    },

                "note":
                    (
                        "RandomForest не переобучался. "
                        "Изменены только event START/END thresholds "
                        "по явно размеченным FN/FP hard intervals."
                    ),
            },
        }
    )

    new_bundle = {
        "model":
            model,

        "calibration":
            new_calibration,
    }

    Path(
        HARD_OUTPUT_MODEL
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        new_bundle,
        HARD_OUTPUT_MODEL,
    )

    with open(
        HARD_OUTPUT_CALIBRATION_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            new_calibration,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 6. Graph with NEW thresholds/events
    # --------------------------------------------------------

    graph_created = save_p07_timeline_graph(
        times=
            times,

        probability=
            probability,

        smoothed=
            smoothed,

        events=
            new_result[
                "events"
            ],

        start_threshold=
            new_start,

        end_threshold=
            new_end,

        output_path=
            HARD_OUTPUT_GRAPH,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "СОХРАНЕНО"
    )
    print(
        "=" * 72
    )
    print(
        "Новый joblib:",
        HARD_OUTPUT_MODEL,
    )
    print(
        "Calibration JSON:",
        HARD_OUTPUT_CALIBRATION_JSON,
    )
    print(
        "Hard report CSV:",
        HARD_OUTPUT_REPORT_CSV,
    )
    print(
        "Graph:",
        (
            HARD_OUTPUT_GRAPH
            if graph_created
            else "не создан"
        ),
    )

    print()
    print(
        "Исходный p07_repetition_rf_v3.joblib "
        "НЕ перезаписывался."
    )

    return new_bundle


# ============================================================
# 22. MAIN
# ============================================================

def main():
    for path in [
        POSE_MODEL_PATH,
        HAND_MODEL_PATH,
        RF_MODEL_PATH,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Не найден обязательный файл:\n{path}"
            )

    bundle = load_bundle()

    calibration = bundle["calibration"]

    print()
    print("=" * 72)
    print("P07 V4 — INFERENCE CHECK")
    print("=" * 72)
    print("MODEL:", RF_MODEL_PATH)
    print("VERSION:", calibration.get("version", "unknown"))
    print("START:", calibration.get("start_threshold"))
    print("END:", calibration.get("end_threshold"))
    print("TARGET:", TARGET_VIDEO)
    print("=" * 72)

    process_target(bundle)


if __name__ == "__main__":
    main()
