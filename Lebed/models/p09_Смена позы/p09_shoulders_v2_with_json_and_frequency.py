# ============================================================
# SHOULDERS V2 — INFERENCE
#
# Главные изменения против V1:
#
# POSE CHANGE:
# - Random Forest/multiclass больше НЕ решает событие;
# - используется симметричная величина изменения:
#       pose_magnitude = norm(standardized shoulders)
# - знак движения не важен:
#       вперёд и назад считаются одинаково;
# - одновременно нужен настоящий локальный transition_score;
# - thresholds адаптивно вычисляются из neutral НОВОГО человека.
#
# ROCKING:
# - отдельный независимый detector;
# - не конкурирует с pose_change.
#
# ML validators из .joblib записываются в CSV для диагностики,
# но НЕ имеют права отменить сильное геометрическое событие.
#
# pip install mediapipe opencv-python numpy pandas joblib tqdm matplotlib
# ============================================================

from pathlib import Path

import json
import time
from datetime import datetime, timezone

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm


# ============================================================
# 1. ПУТИ
# ============================================================

TARGET_VIDEO_PATH = (
    r"C:\interview\openface\my_pose(!!!).mp4"
)

POSE_LANDMARKER_PATH = (
    r"C:\interview\openface\pose_landmarker_heavy.task"
)

V2_MODEL_PATH = (
    r"C:\interview\openface\shoulders_v3_clean.joblib"
)

OUTPUT_VIDEO = (
    r"D:\Стажировка Братские сердца\Итог обработки golden_video\p09_my_pose(!!!)(p09_v3).mp4"
)

OUTPUT_CSV = (
    r"D:\Стажировка Братские сердца\Итог обработки golden_video\p09_my_pose(!!!)(p09_v3).csv"
)


# P09 JSON в той же папке, что и CSV.
OUTPUT_JSON = str(
    Path(
        OUTPUT_CSV
    ).with_name(
        Path(
            OUTPUT_CSV
        ).stem
        +
        "_results.json"
    )
)

# График частоты смен позы во времени.
OUTPUT_FREQUENCY_GRAPH = str(
    Path(
        OUTPUT_CSV
    ).with_name(
        Path(
            OUTPUT_CSV
        ).stem
        +
        "_p09_frequency.png"
    )
)

# Аналитика частоты P09.
# Каждую секунду смотрим, сколько P09-событий началось
# за предыдущие 20 секунд, затем пересчитываем в события/мин.
# На детекцию P09 это НЕ влияет.
FREQUENCY_WINDOW_SEC = 20.0
FREQUENCY_STEP_SEC = 1.0

RUN_STARTED_PERF = time.perf_counter()


# ============================================================
# 2. NEUTRAL НОВОГО ЧЕЛОВЕКА
# ============================================================

# ВАЖНО:
# здесь должен быть ЧИСТЫЙ участок спокойного сидения.
#
# Для контрольного my_pose:
TARGET_NEUTRAL_INTERVALS = [
    (0.0, 1.0),
]

# По умолчанию rocking использует тот же baseline.
# Если у тебя есть отдельная чистая нейтраль перед rocking,
# можешь указать её отдельно.
TARGET_ROCKING_NEUTRAL_INTERVALS = None


# ============================================================
# 3. VISUALIZATION
# ============================================================

DRAW_SKELETON = True
PANEL_ALPHA = 0.55


# ============================================================
# 4. LANDMARKS
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14


def visibility(lm):
    value = lm.visibility

    return (
        1.0
        if value is None
        else float(value)
    )


def point3(lm):
    return np.array(
        [
            float(lm.x),
            float(lm.y),
            float(lm.z),
        ],
        dtype=np.float64,
    )


def robust_sigma(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return 0.0

    median = np.median(arr)

    mad = np.median(
        np.abs(
            arr - median
        )
    )

    return float(
        1.4826 * mad
    )


def ema_smooth(
    values,
    valid,
    alpha,
):
    output = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    current = None

    for i, (value, ok) in enumerate(
        zip(
            values,
            valid,
        )
    ):
        if (
            not ok
            or not np.isfinite(value)
        ):
            continue

        if current is None:
            current = float(value)

        else:
            current = (
                alpha
                * float(value)
                + (
                    1.0 - alpha
                )
                * current
            )

        output[i] = current

    return output


def interval_mask(
    timestamps,
    intervals,
):
    mask = np.zeros(
        len(timestamps),
        dtype=bool,
    )

    for start, end in intervals:
        mask |= (
            (timestamps >= start)
            & (timestamps <= end)
        )

    return mask


def get_upper_body_features(
    result,
    min_shoulder_visibility,
    min_elbow_visibility,
):
    if (
        not result.pose_world_landmarks
        or not result.pose_landmarks
    ):
        return None

    world = (
        result.pose_world_landmarks[0]
    )

    image = (
        result.pose_landmarks[0]
    )

    v_ls = visibility(
        world[
            LEFT_SHOULDER
        ]
    )

    v_rs = visibility(
        world[
            RIGHT_SHOULDER
        ]
    )

    if (
        v_ls
        < min_shoulder_visibility
        or v_rs
        < min_shoulder_visibility
    ):
        return None

    left_shoulder = point3(
        world[
            LEFT_SHOULDER
        ]
    )

    right_shoulder = point3(
        world[
            RIGHT_SHOULDER
        ]
    )

    shoulder_width_world = float(
        np.linalg.norm(
            left_shoulder
            - right_shoulder
        )
    )

    if (
        not np.isfinite(
            shoulder_width_world
        )
        or shoulder_width_world
        < 1e-5
    ):
        return None

    shoulder_center_z = (
        left_shoulder[2]
        + right_shoulder[2]
    ) / 2.0

    depth_norm = (
        shoulder_center_z
        / shoulder_width_world
    )

    img_ls = image[
        LEFT_SHOULDER
    ]

    img_rs = image[
        RIGHT_SHOULDER
    ]

    width_image = abs(
        float(img_rs.x)
        - float(img_ls.x)
    )

    center_y = (
        float(img_ls.y)
        + float(img_rs.y)
    ) / 2.0

    center_x = (
        float(img_ls.x)
        + float(img_rs.x)
    ) / 2.0

    rotation_asymmetry = (
        abs(
            float(
                left_shoulder[2]
                - right_shoulder[2]
            )
        )
        / shoulder_width_world
    )

    v_le = visibility(
        world[
            LEFT_ELBOW
        ]
    )

    v_re = visibility(
        world[
            RIGHT_ELBOW
        ]
    )

    quality = (
        "GOOD"
        if (
            v_le
            >= min_elbow_visibility
            and v_re
            >= min_elbow_visibility
        )
        else
        "SHOULDERS_ONLY"
    )

    draw_landmarks = [
        (
            float(item.x),
            float(item.y),
            visibility(item),
        )
        for item in image
    ]

    return (
        np.array(
            [
                depth_norm,
                width_image,
                center_y,
                center_x,
            ],
            dtype=np.float64,
        ),

        float(
            rotation_asymmetry
        ),

        quality,

        draw_landmarks,
    )


def build_transition_score(
    standardized,
    timestamps,
    current_window,
    previous_window,
    gap,
):
    scores = np.full(
        len(timestamps),
        np.nan,
        dtype=np.float64,
    )

    for i, t in enumerate(
        timestamps
    ):

        current_start_t = (
            t - current_window
        )

        previous_end_t = (
            t
            - current_window
            - gap
        )

        previous_start_t = (
            previous_end_t
            - previous_window
        )

        current_start = int(
            np.searchsorted(
                timestamps,
                current_start_t,
                side="left",
            )
        )

        previous_end = int(
            np.searchsorted(
                timestamps,
                previous_end_t,
                side="left",
            )
        )

        previous_start = int(
            np.searchsorted(
                timestamps,
                previous_start_t,
                side="left",
            )
        )

        current_values = (
            standardized[
                current_start:i + 1
            ]
        )

        previous_values = (
            standardized[
                previous_start:
                previous_end
            ]
        )

        current_values = (
            current_values[
                np.all(
                    np.isfinite(
                        current_values
                    ),
                    axis=1,
                )
            ]
        )

        previous_values = (
            previous_values[
                np.all(
                    np.isfinite(
                        previous_values
                    ),
                    axis=1,
                )
            ]
        )

        if (
            len(current_values) < 5
            or len(previous_values) < 5
        ):
            continue

        scores[i] = float(
            np.linalg.norm(
                np.nanmedian(
                    current_values,
                    axis=0,
                )
                - np.nanmedian(
                    previous_values,
                    axis=0,
                )
            )
        )

    return scores


class PoseCounterV2:

    def __init__(
        self,
        start_threshold,
        transition_threshold,
        return_threshold,
        hold_frames,
        return_hold_frames,
        max_rotation_asymmetry,
        analysis_start_time,
    ):
        self.start_threshold = float(
            start_threshold
        )

        self.transition_threshold = float(
            transition_threshold
        )

        self.return_threshold = float(
            return_threshold
        )

        self.hold_frames = int(
            hold_frames
        )

        self.return_hold_frames = int(
            return_hold_frames
        )

        self.max_rotation_asymmetry = float(
            max_rotation_asymmetry
        )

        self.analysis_start_time = float(
            analysis_start_time
        )

        self.count = 0

        self.state = "WARMUP"

        self.event_active = False

        self.start_hold = 0
        self.return_hold = 0

    def update(
        self,
        timestamp,
        magnitude,
        transition_score,
        rotation_asymmetry,
        pose_valid,
    ):
        event = False

        if (
            timestamp
            < self.analysis_start_time
        ):
            self.state = "WARMUP"
            return event

        if (
            not pose_valid
            or not np.isfinite(
                magnitude
            )
        ):
            self.state = "NO_POSE"
            self.start_hold = 0
            self.return_hold = 0
            return event

        if (
            np.isfinite(
                rotation_asymmetry
            )
            and rotation_asymmetry
            > self.max_rotation_asymmetry
        ):
            self.state = "ROTATED"
            self.start_hold = 0
            self.return_hold = 0
            return event

        # ----------------------------------------------------
        # Главное отличие V2:
        # magnitude = длина вектора отклонения.
        # ЗНАКА НЕТ -> вперёд и назад симметричны.
        # ----------------------------------------------------

        if not self.event_active:

            candidate = (
                magnitude
                >= self.start_threshold
                and np.isfinite(
                    transition_score
                )
                and transition_score
                >= self.transition_threshold
            )

            if candidate:
                self.start_hold += 1
                self.state = "STARTING"

            else:
                self.start_hold = 0
                self.state = "NEUTRAL"

            if (
                self.start_hold
                >= self.hold_frames
            ):
                self.count += 1
                self.event_active = True

                self.start_hold = 0
                self.return_hold = 0

                self.state = "CHANGED"
                event = True

            return event

        # ----------------------------------------------------
        # Event already active -> wait for neutral.
        # ----------------------------------------------------

        if (
            magnitude
            <= self.return_threshold
        ):
            self.return_hold += 1
            self.state = "RETURNING"

        else:
            self.return_hold = 0
            self.state = "CHANGED"

        if (
            self.return_hold
            >= self.return_hold_frames
        ):
            self.event_active = False

            self.return_hold = 0
            self.start_hold = 0

            self.state = "NEUTRAL"

        return event


class RockingCounterV2:

    def __init__(
        self,
        start_threshold,
        return_threshold,
        start_hold_frames,
        return_hold_frames,
    ):
        self.start_threshold = float(
            start_threshold
        )

        self.return_threshold = float(
            return_threshold
        )

        self.start_hold_frames = int(
            start_hold_frames
        )

        self.return_hold_frames = int(
            return_hold_frames
        )

        self.count = 0
        self.state = "READY"

        self.event_active = False

        self.start_hold = 0
        self.return_hold = 0

    def update(
        self,
        signed_offset,
        pose_valid,
    ):
        event = False

        if (
            not pose_valid
            or not np.isfinite(
                signed_offset
            )
        ):
            self.state = "NO_POSE"
            self.start_hold = 0
            self.return_hold = 0
            return event

        if not self.event_active:

            if (
                signed_offset
                >= self.start_threshold
            ):
                self.start_hold += 1
                self.state = "OUTWARD"

            else:
                self.start_hold = 0
                self.state = "READY"

            if (
                self.start_hold
                >= self.start_hold_frames
            ):
                self.event_active = True

                self.start_hold = 0
                self.return_hold = 0

            return event

        if (
            abs(
                signed_offset
            )
            <= self.return_threshold
        ):
            self.return_hold += 1
            self.state = "RETURNING"

        else:
            self.return_hold = 0
            self.state = "OUTWARD"

        if (
            self.return_hold
            >= self.return_hold_frames
        ):
            self.count += 1
            event = True

            self.event_active = False
            self.return_hold = 0
            self.start_hold = 0
            self.state = "READY"

        return event


def draw_skeleton(
    frame,
    landmarks,
):
    if (
        not DRAW_SKELETON
        or landmarks is None
    ):
        return frame

    h, w = frame.shape[:2]

    def pixel(index):
        x, y, _ = landmarks[
            index
        ]

        return (
            int(x * w),
            int(y * h),
        )

    for a, b in [
        (11, 12),
        (11, 13),
        (12, 14),
    ]:
        cv2.line(
            frame,
            pixel(a),
            pixel(b),
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    for index in [
        11,
        12,
        13,
        14,
    ]:
        cv2.circle(
            frame,
            pixel(index),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

    return frame


def draw_panel(
    frame,
    pose_counter,
    rocking_counter,
    combined_count,
    magnitude,
    transition,
):
    overlay = frame.copy()

    h, w = frame.shape[:2]

    x1 = 15
    y1 = h - 145
    x2 = min(
        w - 15,
        460,
    )
    y2 = h - 15

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        PANEL_ALPHA,
        frame,
        1.0 - PANEL_ALPHA,
        0,
    )

    lines = [
        "MediaPipe Shoulders V2",

        (
            "Izmenenie polozheniya: "
            f"{combined_count}"
        ),

        (
            "Pose / rocking: "
            f"{pose_counter.count} / "
            f"{rocking_counter.count}"
        ),

        (
            "State: "
            f"{pose_counter.state}"
        ),

        (
            "Magnitude / transition: "
            f"{magnitude:.2f} / "
            f"{transition:.2f}"
            if (
                np.isfinite(magnitude)
                and np.isfinite(transition)
            )
            else
            "Magnitude / transition: --"
        ),
    ]

    y = y1 + 24

    for line in lines:
        cv2.putText(
            frame,
            line,
            (
                x1 + 10,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 23

    return frame



# ============================================================
# P09 -> JSON / ГРАФИК
# ============================================================
#
# ВАЖНО:
# этот блок НЕ меняет существующую детекцию.
#
# PoseCounterV2, RockingCounterV2, adaptive thresholds,
# combined_merge_seconds и вся геометрическая логика остаются
# прежними.
#
# P09JsonTracker только наблюдает за уже готовыми:
# - pose_event
# - rocking_event
# - combined_event
# - pose_counter.event_active
# - rocking_counter.event_active
#
# Поэтому решение "есть смена позы / нет смены позы"
# по-прежнему принимает исходный код.
# ============================================================


def finite_or_none(
    value
):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


def format_axis_time(
    seconds
):
    """
    Формат времени для графика:
    00:00, 00:30, 01:00, 01:30 ...
    """

    seconds = max(
        0.0,
        float(seconds)
    )

    total = int(
        round(seconds)
    )

    minutes = total // 60
    secs = total % 60

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def human_duration(
    seconds
):
    seconds = max(
        0.0,
        float(seconds)
    )

    total = int(
        round(seconds)
    )

    hours = total // 3600
    minutes = (
        total % 3600
    ) // 60
    secs = total % 60

    if hours > 0:
        return (
            f"{hours}ч "
            f"{minutes}м "
            f"{secs}с"
        )

    if minutes > 0:
        return (
            f"{minutes}м "
            f"{secs}с"
        )

    return (
        f"{secs}с"
    )


class P09JsonTracker:
    """
    Пассивный регистратор финальных combined P09 events.

    Почему не создаём новый detector:
    combined_event уже является итоговым событием исходного
    Shoulders V2. Мы лишь добавляем ему START / END / duration.

    Для pose_change:
        START = момент существующего pose_event.
        END   = момент, когда PoseCounterV2 сам вернулся из
                event_active в neutral.

    Для rocking:
        START = момент, когда существующий RockingCounterV2
                вошёл в event_active.
        END   = существующий rocking_event, то есть подтверждённый
                возврат.

    Таким образом число cuts соответствует combined_count.
    """

    def __init__(
        self
    ):
        self.previous_pose_active = False
        self.previous_rocking_active = False

        self.pose_active_start = None
        self.rocking_active_start = None

        self.rocking_peak_magnitude = None
        self.rocking_peak_transition = None

        # У PoseCounter одновременно может быть только одно
        # активное событие.
        self.pending_pose_cut = None

        self.cuts = []


    @staticmethod
    def _max_optional(
        old_value,
        new_value
    ):
        new_value = finite_or_none(
            new_value
        )

        if new_value is None:
            return old_value

        if old_value is None:
            return new_value

        return max(
            float(old_value),
            float(new_value)
        )


    def _make_cut(
        self,
        event_type,
        start_sec,
        end_sec,
        trigger_sec,
        pose_magnitude_trigger,
        transition_trigger,
        rocking_offset_trigger,
        peak_pose_magnitude=None,
        peak_transition_score=None,
        status="completed"
    ):
        start_sec = max(
            0.0,
            float(start_sec)
        )

        end_sec = max(
            start_sec,
            float(end_sec)
        )

        duration_sec = max(
            0.0,
            end_sec
            -
            start_sec
        )

        return {
            "source":
                "mediapipe",

            "pattern":
                "P09",

            "name":
                "смена позы",

            # Диагностический тип исходного детектора.
            "event_type":
                event_type,

            "t_start":
                round(
                    start_sec,
                    3
                ),

            "t_end":
                round(
                    end_sec,
                    3
                ),

            "start_sec":
                round(
                    start_sec,
                    3
                ),

            "end_sec":
                round(
                    end_sec,
                    3
                ),

            "duration_sec":
                round(
                    duration_sec,
                    3
                ),

            "trigger_sec":
                round(
                    float(trigger_sec),
                    3
                ),

            "pose_magnitude_at_trigger":
                finite_or_none(
                    pose_magnitude_trigger
                ),

            "transition_score_at_trigger":
                finite_or_none(
                    transition_trigger
                ),

            "rocking_offset_at_trigger":
                finite_or_none(
                    rocking_offset_trigger
                ),

            "peak_pose_magnitude":
                finite_or_none(
                    peak_pose_magnitude
                ),

            "peak_transition_score":
                finite_or_none(
                    peak_transition_score
                ),

            "status":
                status,

            # Поля совместимости со структурой
            # Alyona_interview_results.json.
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


    def update(
        self,
        timestamp,
        pose_event,
        rocking_event,
        combined_event,
        pose_active,
        rocking_active,
        pose_magnitude,
        transition_score,
        rocking_offset
    ):
        timestamp = float(
            timestamp
        )

        pose_active = bool(
            pose_active
        )

        rocking_active = bool(
            rocking_active
        )

        # ----------------------------------------------------
        # Запоминаем реальный START активного pose/rocking
        # из уже существующих state machines.
        # ----------------------------------------------------

        if (
            pose_active
            and
            not self.previous_pose_active
        ):
            self.pose_active_start = (
                timestamp
            )


        if (
            rocking_active
            and
            not self.previous_rocking_active
        ):
            self.rocking_active_start = (
                timestamp
            )

            self.rocking_peak_magnitude = (
                finite_or_none(
                    pose_magnitude
                )
            )

            self.rocking_peak_transition = (
                finite_or_none(
                    transition_score
                )
            )


        # Во время rocking сохраняем только диагностические пики.
        if (
            rocking_active
            or self.previous_rocking_active
        ):
            self.rocking_peak_magnitude = (
                self._max_optional(
                    self.rocking_peak_magnitude,
                    pose_magnitude
                )
            )

            self.rocking_peak_transition = (
                self._max_optional(
                    self.rocking_peak_transition,
                    transition_score
                )
            )


        # Если уже открыт финальный pose P09 cut,
        # пассивно обновляем его диагностические пики.
        if (
            self.pending_pose_cut
            is not None
        ):
            self.pending_pose_cut[
                "peak_pose_magnitude"
            ] = self._max_optional(
                self.pending_pose_cut.get(
                    "peak_pose_magnitude"
                ),
                pose_magnitude
            )

            self.pending_pose_cut[
                "peak_transition_score"
            ] = self._max_optional(
                self.pending_pose_cut.get(
                    "peak_transition_score"
                ),
                transition_score
            )


        # ----------------------------------------------------
        # FINAL combined_event:
        # именно он увеличивает combined_count в исходном коде.
        # На каждый такой trigger создаём ровно один P09.
        # ----------------------------------------------------

        if combined_event:

            # Pose change начинается сейчас и может продолжаться
            # до возвращения в neutral.
            if pose_event:

                pose_start = (
                    self.pose_active_start
                    if
                    self.pose_active_start
                    is not None
                    else
                    timestamp
                )

                event_type = (
                    "pose_change+rocking"
                    if
                    rocking_event
                    else
                    "pose_change"
                )

                # Если в тот же момент завершился rocking,
                # объединённый P09 может начинаться раньше —
                # с фактического START rocking.
                if (
                    rocking_event
                    and
                    self.rocking_active_start
                    is not None
                ):
                    pose_start = min(
                        pose_start,
                        self.rocking_active_start
                    )

                self.pending_pose_cut = {
                    "event_type":
                        event_type,

                    "start_sec":
                        float(
                            pose_start
                        ),

                    "trigger_sec":
                        timestamp,

                    "pose_magnitude_at_trigger":
                        finite_or_none(
                            pose_magnitude
                        ),

                    "transition_score_at_trigger":
                        finite_or_none(
                            transition_score
                        ),

                    "rocking_offset_at_trigger":
                        finite_or_none(
                            rocking_offset
                        ),

                    "peak_pose_magnitude":
                        finite_or_none(
                            pose_magnitude
                        ),

                    "peak_transition_score":
                        finite_or_none(
                            transition_score
                        ),
                }


            # Rocking-only combined event завершается на текущем
            # кадре — RockingCounter уже подтвердил возврат.
            elif rocking_event:

                rocking_start = (
                    self.rocking_active_start
                    if
                    self.rocking_active_start
                    is not None
                    else
                    timestamp
                )

                cut = self._make_cut(
                    event_type=
                        "rocking",

                    start_sec=
                        rocking_start,

                    end_sec=
                        timestamp,

                    trigger_sec=
                        timestamp,

                    pose_magnitude_trigger=
                        pose_magnitude,

                    transition_trigger=
                        transition_score,

                    rocking_offset_trigger=
                        rocking_offset,

                    peak_pose_magnitude=
                        self.rocking_peak_magnitude,

                    peak_transition_score=
                        self.rocking_peak_transition,
                )

                self.cuts.append(
                    cut
                )


        # ----------------------------------------------------
        # END pose_change:
        # исходный PoseCounter сам вышел из event_active.
        # ----------------------------------------------------

        pose_just_ended = (
            self.previous_pose_active
            and
            not pose_active
        )

        if (
            pose_just_ended
            and
            self.pending_pose_cut
            is not None
        ):
            item = self.pending_pose_cut

            cut = self._make_cut(
                event_type=
                    item[
                        "event_type"
                    ],

                start_sec=
                    item[
                        "start_sec"
                    ],

                end_sec=
                    timestamp,

                trigger_sec=
                    item[
                        "trigger_sec"
                    ],

                pose_magnitude_trigger=
                    item[
                        "pose_magnitude_at_trigger"
                    ],

                transition_trigger=
                    item[
                        "transition_score_at_trigger"
                    ],

                rocking_offset_trigger=
                    item[
                        "rocking_offset_at_trigger"
                    ],

                peak_pose_magnitude=
                    item[
                        "peak_pose_magnitude"
                    ],

                peak_transition_score=
                    item[
                        "peak_transition_score"
                    ],
            )

            self.cuts.append(
                cut
            )

            self.pending_pose_cut = None


        # После завершения rocking очищаем его START,
        # но только после того, как возможный rocking_event
        # уже был записан.
        rocking_just_ended = (
            self.previous_rocking_active
            and
            not rocking_active
        )

        if rocking_just_ended:
            self.rocking_active_start = None
            self.rocking_peak_magnitude = None
            self.rocking_peak_transition = None


        if (
            self.previous_pose_active
            and
            not pose_active
        ):
            self.pose_active_start = None


        self.previous_pose_active = (
            pose_active
        )

        self.previous_rocking_active = (
            rocking_active
        )


    def finalize(
        self,
        video_end_sec
    ):
        # Если видео закончилось в состоянии CHANGED,
        # не теряем уже подтверждённый final combined P09.
        if (
            self.pending_pose_cut
            is not None
        ):
            item = self.pending_pose_cut

            cut = self._make_cut(
                event_type=
                    item[
                        "event_type"
                    ],

                start_sec=
                    item[
                        "start_sec"
                    ],

                end_sec=
                    video_end_sec,

                trigger_sec=
                    item[
                        "trigger_sec"
                    ],

                pose_magnitude_trigger=
                    item[
                        "pose_magnitude_at_trigger"
                    ],

                transition_trigger=
                    item[
                        "transition_score_at_trigger"
                    ],

                rocking_offset_trigger=
                    item[
                        "rocking_offset_at_trigger"
                    ],

                peak_pose_magnitude=
                    item[
                        "peak_pose_magnitude"
                    ],

                peak_transition_score=
                    item[
                        "peak_transition_score"
                    ],

                status=
                    "open_at_video_end",
            )

            self.cuts.append(
                cut
            )

            self.pending_pose_cut = None


        self.cuts.sort(
            key=lambda item:
                (
                    item[
                        "start_sec"
                    ],
                    item[
                        "event_type"
                    ]
                )
        )


def build_posture_change_frequency(
    cuts,
    duration_sec,
    window_sec=
        FREQUENCY_WINDOW_SEC,
    step_sec=
        FREQUENCY_STEP_SEC
):
    """
    Частота P09 во времени.

    На каждой секунде t:
    сколько FINAL P09 cuts началось за предыдущие 20 сек,
    пересчитанное в события в минуту.

    Отдельно сохраняются:
    - total
    - pose_change
    - rocking

    Эта аналитика НЕ влияет на detector.
    """

    duration_sec = max(
        0.0,
        float(duration_sec)
    )

    total_starts = np.asarray(
        [
            cut[
                "start_sec"
            ]
            for cut
            in cuts
        ],
        dtype=np.float64
    )

    pose_starts = np.asarray(
        [
            cut[
                "start_sec"
            ]
            for cut
            in cuts
            if
            "pose_change"
            in cut[
                "event_type"
            ]
        ],
        dtype=np.float64
    )

    rocking_starts = np.asarray(
        [
            cut[
                "start_sec"
            ]
            for cut
            in cuts
            if
            "rocking"
            in cut[
                "event_type"
            ]
        ],
        dtype=np.float64
    )


    sample_times = np.arange(
        0.0,
        duration_sec
        +
        step_sec,
        step_sec,
        dtype=np.float64
    )

    if len(
        sample_times
    ) == 0:
        sample_times = np.asarray(
            [
                0.0
            ],
            dtype=np.float64
        )


    points = []


    for t in sample_times:

        window_start = (
            t
            -
            window_sec
        )

        def count_in_window(
            values
        ):
            return int(
                np.sum(
                    (
                        values
                        >
                        window_start
                    )
                    &
                    (
                        values
                        <=
                        t
                    )
                )
            )

        total_count = count_in_window(
            total_starts
        )

        pose_count = count_in_window(
            pose_starts
        )

        rocking_count = count_in_window(
            rocking_starts
        )


        points.append(
            {
                "time_sec":
                    round(
                        float(t),
                        3
                    ),

                "time_human":
                    format_axis_time(
                        t
                    ),

                "total_changes_per_min":
                    round(
                        total_count
                        *
                        60.0
                        /
                        window_sec,
                        3
                    ),

                "pose_changes_per_min":
                    round(
                        pose_count
                        *
                        60.0
                        /
                        window_sec,
                        3
                    ),

                "rocking_changes_per_min":
                    round(
                        rocking_count
                        *
                        60.0
                        /
                        window_sec,
                        3
                    ),
            }
        )


    return {
        "method":
            "trailing_rolling_window",

        "window_sec":
            float(
                window_sec
            ),

        "step_sec":
            float(
                step_sec
            ),

        "unit":
            "posture_changes_per_minute",

        "meaning":
            (
                "Число финальных P09 событий, начавшихся "
                f"за предыдущие {window_sec:.0f} секунд, "
                "пересчитанное в смены позы/мин."
            ),

        "points":
            points,
    }


def save_posture_change_graph(
    frequency_data,
    cuts,
    output_path
):
    """
    Красивый график в том же стиле, что и P05:
    - X: MM:SS
    - подписи через 30 секунд
    - Y: смен позы/мин
    - общая линия + pose + rocking
    - вертикальные метки реальных START P09.
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


    points = frequency_data[
        "points"
    ]

    if not points:
        return False


    x = np.asarray(
        [
            item[
                "time_sec"
            ]
            for item
            in points
        ],
        dtype=np.float64
    )

    total = np.asarray(
        [
            item[
                "total_changes_per_min"
            ]
            for item
            in points
        ],
        dtype=np.float64
    )

    pose = np.asarray(
        [
            item[
                "pose_changes_per_min"
            ]
            for item
            in points
        ],
        dtype=np.float64
    )

    rocking = np.asarray(
        [
            item[
                "rocking_changes_per_min"
            ]
            for item
            in points
        ],
        dtype=np.float64
    )


    fig, ax = plt.subplots(
        figsize=(
            16,
            8
        ),
        dpi=140
    )


    ax.plot(
        x,
        total,
        linewidth=3.2,
        label="Всего P09"
    )

    ax.plot(
        x,
        pose,
        linewidth=2.0,
        label="Pose change"
    )

    ax.plot(
        x,
        rocking,
        linewidth=2.0,
        label="Rocking"
    )

    ax.fill_between(
        x,
        total,
        alpha=0.12
    )


    # Тонкая вертикальная метка = фактический START P09.
    for cut in cuts:
        ax.axvline(
            cut[
                "start_sec"
            ],
            alpha=0.10,
            linewidth=1.0
        )


    ax.set_title(
        "P09 — частота смены позы во времени",
        fontsize=16,
        pad=18
    )

    ax.set_xlabel(
        "Время видео",
        fontsize=12
    )

    ax.set_ylabel(
        "Смен позы/мин",
        fontsize=12
    )


    subtitle = (
        "Скользящее окно: "
        f'{frequency_data["window_sec"]:.0f} с'
        " | Шаг расчёта: "
        f'{frequency_data["step_sec"]:.0f} с'
        " | Вертикальные метки = START P09"
    )

    fig.text(
        0.125,
        0.94,
        subtitle,
        fontsize=10
    )


    max_time = (
        float(
            np.max(x)
        )
        if
        len(x) > 0
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
        dtype=np.float64
    )

    if len(
        x_ticks
    ) == 0:
        x_ticks = np.asarray(
            [
                0.0
            ]
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
        fontsize=10
    )


    y_max = max(
        1.0,
        float(
            np.nanmax(
                total
            )
        )
        if
        len(total) > 0
        else
        1.0
    )

    y_upper = max(
        5.0,
        np.ceil(
            y_max
            /
            5.0
        )
        *
        5.0
    )

    ax.set_ylim(
        0,
        y_upper
    )


    ax.grid(
        True,
        which="major",
        axis="both",
        alpha=0.28,
        linestyle="--"
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
        fontsize=10
    )

    legend.get_frame().set_alpha(
        0.9
    )


    fig.text(
        0.125,
        0.02,
        (
            "Чем выше линия, тем чаще модель фиксировала "
            "смену позы в предыдущие 20 секунд."
        ),
        fontsize=9
    )


    plt.tight_layout(
        rect=[
            0.03,
            0.05,
            0.99,
            0.92
        ]
    )


    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    fig.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return True


def build_p09_results_json(
    p09_tracker,
    frequency_data,
    fps,
    total_frames,
    frames_processed,
    video_duration_sec,
    processed_duration_sec,
    run_duration_sec,
    pose_counter,
    rocking_counter,
    combined_count,
    pose_start_threshold,
    transition_threshold,
    pose_return_threshold
):
    p09_count = len(
        p09_tracker.cuts
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
                    p09_count
                    if
                    code
                    ==
                    "P09"
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
            "MediaPipe Shoulders V2 — P09",

        "video":
            str(
                TARGET_VIDEO_PATH
            ),

        "video_name":
            Path(
                TARGET_VIDEO_PATH
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
                    total_frames
                    <=
                    0
                    or
                    frames_processed
                    >=
                    total_frames
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
                    total_frames
                ),

            "frames_processed":
                int(
                    frames_processed
                ),

            "video_fps":
                round(
                    float(
                        fps
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
                    "P09 detection logic is unchanged; "
                    "JSON observes existing Shoulders V2 states."
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
            p09_tracker.cuts,

        # Сохраняем исходные счётчики модели отдельно,
        # чтобы ничего из текущего Shoulders V2 не потерять.
        "detector_counts": {
            "pose_changes":
                int(
                    pose_counter.count
                ),

            "rocking_events":
                int(
                    rocking_counter.count
                ),

            "combined_position_changes":
                int(
                    combined_count
                ),

            "p09_json_cuts":
                int(
                    p09_count
                ),
        },

        "adaptive_thresholds": {
            "pose_magnitude_start":
                float(
                    pose_start_threshold
                ),

            "transition_start":
                float(
                    transition_threshold
                ),

            "pose_return":
                float(
                    pose_return_threshold
                ),
        },

        "posture_change_frequency":
            frequency_data,

        "posture_change_frequency_graph":
            str(
                OUTPUT_FREQUENCY_GRAPH
            ),

        "overlay_timeline_mp4":
            str(
                OUTPUT_VIDEO
            ),

        "diagnostic_csv":
            str(
                OUTPUT_CSV
            ),

        "notes": [
            (
                "P09 = смена позы. Исходные PoseCounterV2, "
                "RockingCounterV2 и combined_event не изменены."
            ),
            (
                "Каждый P09 cut создаётся только из уже существующего "
                "final combined_event и содержит START/END/duration."
            ),
            (
                "Для pose_change END — выход исходного PoseCounterV2 "
                "из event_active; для rocking END — исходный rocking_event."
            ),
            (
                "График — скользящая частота P09 за предыдущие "
                f"{FREQUENCY_WINDOW_SEC:.0f} секунд; "
                "на detector он не влияет."
            ),
            (
                "clip_file/clip = null, потому что этот скрипт "
                "не вырезает отдельные clips."
            ),
        ],
    }


def main():

    bundle = joblib.load(
        V2_MODEL_PATH
    )

    pose_cfg = bundle[
        "pose_config"
    ]

    rocking_cfg = bundle[
        "rocking_config"
    ]

    norm_cfg = bundle[
        "normalization_config"
    ]

    transition_cfg = bundle[
        "transition_config"
    ]

    quality_cfg = bundle[
        "quality_config"
    ]

    pose_validator = bundle.get(
        "pose_validator"
    )

    rocking_validator = bundle.get(
        "rocking_validator"
    )

    cap = cv2.VideoCapture(
        TARGET_VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Не удалось открыть видео:\n"
            f"{TARGET_VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    timestamps = (
        np.arange(
            total_frames,
            dtype=np.float64,
        )
        / fps
    )

    raw = np.full(
        (
            total_frames,
            4,
        ),
        np.nan,
        dtype=np.float64,
    )

    rotation = np.full(
        total_frames,
        np.nan,
        dtype=np.float64,
    )

    valid = np.zeros(
        total_frames,
        dtype=bool,
    )

    quality = np.asarray(
        [
            "NO_POSE"
        ] * total_frames,
        dtype=object,
    )

    draw_landmarks = [
        None
    ] * total_frames

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    PoseLandmarker = (
        mp.tasks.vision.PoseLandmarker
    )

    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=
                POSE_LANDMARKER_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    print(
        "PASS 1/2: MediaPipe"
    )

    progress = tqdm(
        total=total_frames,
        desc="Shoulders V2",
        unit="frame",
        dynamic_ncols=True,
    )

    frame_index = 0

    with PoseLandmarker.create_from_options(
        options
    ) as landmarker:

        while frame_index < total_frames:

            ok, frame = cap.read()

            if not ok:
                break

            timestamp_ms = int(
                round(
                    timestamps[
                        frame_index
                    ]
                    * 1000.0
                )
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=
                    mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            item = get_upper_body_features(
                result,
                quality_cfg[
                    "min_shoulder_visibility"
                ],
                quality_cfg[
                    "min_elbow_visibility"
                ],
            )

            if item is not None:

                (
                    feature,
                    asymmetry,
                    q_name,
                    landmarks,
                ) = item

                raw[
                    frame_index
                ] = feature

                rotation[
                    frame_index
                ] = asymmetry

                valid[
                    frame_index
                ] = True

                quality[
                    frame_index
                ] = q_name

                draw_landmarks[
                    frame_index
                ] = landmarks

            frame_index += 1
            progress.update(1)

    progress.close()
    cap.release()

    # ========================================================
    # SMOOTH
    # ========================================================

    smooth = np.column_stack([
        ema_smooth(
            raw[:, i],
            valid,
            norm_cfg[
                "ema_alpha"
            ],
        )
        for i in range(4)
    ])

    # ========================================================
    # TARGET NEUTRAL
    # ========================================================

    neutral_mask = (
        interval_mask(
            timestamps,
            TARGET_NEUTRAL_INTERVALS,
        )
        & valid
        & np.all(
            np.isfinite(smooth),
            axis=1,
        )
    )

    neutral_count = int(
        np.sum(
            neutral_mask
        )
    )

    if neutral_count < 20:
        raise RuntimeError(
            "Недостаточно neutral кадров: "
            f"{neutral_count}"
        )

    neutral_center = np.nanmedian(
        smooth[
            neutral_mask,
            :3,
        ],
        axis=0,
    )

    neutral_delta = (
        smooth[
            neutral_mask,
            :3,
        ]
        - neutral_center
    )

    scales = np.array(
        [
            max(
                norm_cfg[
                    "depth_scale_floor"
                ],

                norm_cfg[
                    "noise_scale_multiplier"
                ]
                * robust_sigma(
                    neutral_delta[:, 0]
                ),
            ),

            max(
                norm_cfg[
                    "width_scale_floor"
                ],

                norm_cfg[
                    "noise_scale_multiplier"
                ]
                * robust_sigma(
                    neutral_delta[:, 1]
                ),
            ),

            max(
                norm_cfg[
                    "center_y_scale_floor"
                ],

                norm_cfg[
                    "noise_scale_multiplier"
                ]
                * robust_sigma(
                    neutral_delta[:, 2]
                ),
            ),
        ],
        dtype=np.float64,
    )

    standardized = (
        smooth[:, :3]
        - neutral_center
    ) / scales

    pose_magnitude = np.linalg.norm(
        standardized,
        axis=1,
    )

    transition_score = (
        build_transition_score(
            standardized,
            timestamps,

            transition_cfg[
                "current_window_seconds"
            ],

            transition_cfg[
                "previous_window_seconds"
            ],

            transition_cfg[
                "gap_seconds"
            ],
        )
    )

    threshold_mask = (
        neutral_mask
        & np.isfinite(
            pose_magnitude
        )
        & np.isfinite(
            transition_score
        )
    )

    if int(
        np.sum(
            threshold_mask
        )
    ) < 10:
        raise RuntimeError(
            "Недостаточно neutral кадров "
            "для adaptive thresholds."
        )

    neutral_mag_p95 = float(
        np.percentile(
            pose_magnitude[
                threshold_mask
            ],
            95.0,
        )
    )

    neutral_transition_p95 = float(
        np.percentile(
            transition_score[
                threshold_mask
            ],
            95.0,
        )
    )

    pose_start_threshold = max(
        pose_cfg[
            "min_magnitude_threshold"
        ],

        pose_cfg[
            "magnitude_neutral_p95_multiplier"
        ]
        * neutral_mag_p95,
    )

    transition_threshold = max(
        pose_cfg[
            "min_transition_threshold"
        ],

        pose_cfg[
            "transition_neutral_p95_multiplier"
        ]
        * neutral_transition_p95,
    )

    pose_return_threshold = max(
        pose_cfg[
            "min_return_threshold"
        ],

        pose_cfg[
            "return_neutral_p95_multiplier"
        ]
        * neutral_mag_p95,
    )

    analysis_start_time = (
        max(
            end
            for _, end
            in TARGET_NEUTRAL_INTERVALS
        )
        + pose_cfg[
            "post_calibration_warmup_seconds"
        ]
    )

    print()
    print("=" * 72)
    print("V2 ADAPTIVE THRESHOLDS")
    print("=" * 72)

    print(
        f"Neutral frames:      "
        f"{neutral_count}"
    )

    print(
        f"Neutral magnitude p95:"
        f" {neutral_mag_p95:.4f}"
    )

    print(
        f"Neutral transition p95:"
        f" {neutral_transition_p95:.4f}"
    )

    print(
        f"POSE magnitude start:"
        f" {pose_start_threshold:.4f}"
    )

    print(
        f"POSE transition start:"
        f" {transition_threshold:.4f}"
    )

    print(
        f"POSE return threshold:"
        f" {pose_return_threshold:.4f}"
    )

    print("=" * 72)

    # ========================================================
    # ROCKING BASELINE
    # ========================================================

    rocking_intervals = (
        TARGET_NEUTRAL_INTERVALS
        if (
            TARGET_ROCKING_NEUTRAL_INTERVALS
            is None
        )
        else
        TARGET_ROCKING_NEUTRAL_INTERVALS
    )

    rocking_neutral_mask = (
        interval_mask(
            timestamps,
            rocking_intervals,
        )
        & valid
        & np.isfinite(
            smooth[:, 3]
        )
        & np.isfinite(
            smooth[:, 1]
        )
    )

    baseline_x = float(
        np.nanmedian(
            smooth[
                rocking_neutral_mask,
                3,
            ]
        )
    )

    baseline_width = float(
        np.nanmedian(
            smooth[
                rocking_neutral_mask,
                1,
            ]
        )
    )

    raw_center_x_offset = (
        smooth[:, 3]
        - baseline_x
    ) / baseline_width

    signed_rocking_offset = (
        raw_center_x_offset
        * rocking_cfg[
            "direction_sign"
        ]
    )

    # ========================================================
    # DIAGNOSTIC ML PROBABILITIES
    # ========================================================

    velocity = np.full(
        standardized.shape,
        np.nan,
        dtype=np.float64,
    )

    center_x_velocity = np.full(
        total_frames,
        np.nan,
        dtype=np.float64,
    )

    for i, t in enumerate(
        timestamps
    ):

        j = int(
            np.searchsorted(
                timestamps,
                t - 0.20,
                side="left",
            )
        )

        if j >= i:
            continue

        dt = (
            timestamps[i]
            - timestamps[j]
        )

        if dt <= 0:
            continue

        if np.all(
            np.isfinite(
                standardized[
                    [i, j]
                ]
            )
        ):
            velocity[i] = (
                standardized[i]
                - standardized[j]
            ) / dt

        if (
            np.isfinite(
                raw_center_x_offset[i]
            )
            and np.isfinite(
                raw_center_x_offset[j]
            )
        ):
            center_x_velocity[i] = (
                raw_center_x_offset[i]
                - raw_center_x_offset[j]
            ) / dt

    pose_features = np.column_stack([
        np.abs(
            standardized[:, 0]
        ),
        np.abs(
            standardized[:, 1]
        ),
        np.abs(
            standardized[:, 2]
        ),
        pose_magnitude,
        np.abs(
            velocity[:, 0]
        ),
        np.abs(
            velocity[:, 1]
        ),
        np.abs(
            velocity[:, 2]
        ),
        transition_score,
        rotation,
    ])

    rocking_features = np.column_stack([
        np.abs(
            raw_center_x_offset
        ),
        np.abs(
            center_x_velocity
        ),
        pose_magnitude,
        transition_score,
        np.abs(
            standardized[:, 1]
        ),
        np.abs(
            standardized[:, 0]
        ),
    ])

    pose_validator_probability = np.full(
        total_frames,
        np.nan,
        dtype=np.float64,
    )

    rocking_validator_probability = np.full(
        total_frames,
        np.nan,
        dtype=np.float64,
    )

    if pose_validator is not None:

        finite = np.all(
            np.isfinite(
                pose_features
            ),
            axis=1,
        )

        pose_validator_probability[
            finite
        ] = (
            pose_validator.predict_proba(
                pose_features[
                    finite
                ]
            )[:, 1]
        )

    if rocking_validator is not None:

        finite = np.all(
            np.isfinite(
                rocking_features
            ),
            axis=1,
        )

        rocking_validator_probability[
            finite
        ] = (
            rocking_validator.predict_proba(
                rocking_features[
                    finite
                ]
            )[:, 1]
        )

    # ========================================================
    # COUNTERS
    # ========================================================

    pose_counter = PoseCounterV2(
        pose_start_threshold,
        transition_threshold,
        pose_return_threshold,

        pose_cfg[
            "hold_frames"
        ],

        pose_cfg[
            "return_hold_frames"
        ],

        quality_cfg[
            "max_shoulder_depth_asymmetry"
        ],

        analysis_start_time,
    )

    rocking_counter = RockingCounterV2(
        rocking_cfg[
            "start_threshold"
        ],

        rocking_cfg[
            "return_threshold"
        ],

        rocking_cfg[
            "start_hold_frames"
        ],

        rocking_cfg[
            "return_hold_frames"
        ],
    )

    combined_count = 0

    # Пассивный P09 logger.
    # Он НЕ участвует в решении detector.
    p09_json_tracker = (
        P09JsonTracker()
    )

    last_combined_frame = (
        -10**9
    )

    combined_merge_frames = max(
        1,

        int(
            round(
                fps
                * bundle[
                    "combined_merge_seconds"
                ]
            )
        ),
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    Path(
        OUTPUT_VIDEO
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        OUTPUT_FREQUENCY_GRAPH
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height,
        ),
    )

    rows = []

    cap = cv2.VideoCapture(
        TARGET_VIDEO_PATH
    )

    print()
    print(
        "PASS 2/2: V2 events"
    )

    progress = tqdm(
        total=total_frames,
        desc="V2 inference",
        unit="frame",
        dynamic_ncols=True,
    )

    for i in range(
        total_frames
    ):

        ok, frame = cap.read()

        if not ok:
            break

        pose_event = (
            pose_counter.update(
                timestamps[i],
                pose_magnitude[i],
                transition_score[i],
                rotation[i],
                bool(valid[i]),
            )
        )

        rocking_event = (
            rocking_counter.update(
                signed_rocking_offset[i],
                bool(valid[i]),
            )
        )

        combined_event = False

        if (
            pose_event
            or rocking_event
        ):
            if (
                i
                - last_combined_frame
                >= combined_merge_frames
            ):
                combined_count += 1

                last_combined_frame = i

                combined_event = True

        # ====================================================
        # P09 JSON LOGGER
        # ====================================================
        #
        # Здесь НЕТ новой классификации.
        # Передаём только уже рассчитанные исходным кодом
        # pose_event / rocking_event / combined_event и states.

        p09_json_tracker.update(
            timestamp=
                timestamps[i],

            pose_event=
                pose_event,

            rocking_event=
                rocking_event,

            combined_event=
                combined_event,

            pose_active=
                pose_counter.event_active,

            rocking_active=
                rocking_counter.event_active,

            pose_magnitude=
                pose_magnitude[i],

            transition_score=
                transition_score[i],

            rocking_offset=
                signed_rocking_offset[i],
        )

        frame = draw_skeleton(
            frame,
            draw_landmarks[i],
        )

        frame = draw_panel(
            frame,
            pose_counter,
            rocking_counter,
            combined_count,
            pose_magnitude[i],
            transition_score[i],
        )

        writer.write(
            frame
        )

        rows.append({
            "frame":
                i + 1,

            "timestamp":
                round(
                    timestamps[i],
                    3,
                ),

            "shoulders_valid":
                int(
                    valid[i]
                ),

            "quality":
                quality[i],

            "raw_depth":
                raw[i, 0],

            "raw_width":
                raw[i, 1],

            "raw_center_y":
                raw[i, 2],

            "raw_center_x":
                raw[i, 3],

            "rotation_asymmetry":
                rotation[i],

            "pose_magnitude":
                pose_magnitude[i],

            "pose_start_threshold":
                pose_start_threshold,

            "transition_score":
                transition_score[i],

            "transition_threshold":
                transition_threshold,

            "pose_return_threshold":
                pose_return_threshold,

            "pose_validator_probability":
                pose_validator_probability[i],

            "pose_state":
                pose_counter.state,

            "pose_change_event":
                int(
                    pose_event
                ),

            "total_pose_changes":
                pose_counter.count,

            "rocking_offset":
                signed_rocking_offset[i],

            "rocking_validator_probability":
                rocking_validator_probability[i],

            "rocking_state":
                rocking_counter.state,

            "rocking_event":
                int(
                    rocking_event
                ),

            "total_rocking_events":
                rocking_counter.count,

            "combined_position_event":
                int(
                    combined_event
                ),

            "total_position_changes":
                combined_count,

            # Только диагностические поля для JSON/аналитики.
            # На detector не влияют.
            "pose_event_active":
                int(
                    pose_counter.event_active
                ),

            "rocking_event_active":
                int(
                    rocking_counter.event_active
                ),

            "p09_active":
                int(
                    pose_counter.event_active
                    or
                    rocking_counter.event_active
                ),
        })

        progress.update(1)

    progress.close()

    cap.release()
    writer.release()

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # P09 JSON + FREQUENCY GRAPH
    # ========================================================

    frames_processed = len(
        rows
    )

    processed_duration_sec = (
        frames_processed
        /
        fps
        if
        fps > 0
        else
        0.0
    )

    video_duration_sec = (
        total_frames
        /
        fps
        if
        (
            total_frames > 0
            and
            fps > 0
        )
        else
        processed_duration_sec
    )


    p09_json_tracker.finalize(
        processed_duration_sec
    )


    posture_change_frequency = (
        build_posture_change_frequency(
            cuts=
                p09_json_tracker.cuts,

            duration_sec=
                processed_duration_sec,

            window_sec=
                FREQUENCY_WINDOW_SEC,

            step_sec=
                FREQUENCY_STEP_SEC,
        )
    )


    graph_created = (
        save_posture_change_graph(
            frequency_data=
                posture_change_frequency,

            cuts=
                p09_json_tracker.cuts,

            output_path=
                OUTPUT_FREQUENCY_GRAPH,
        )
    )


    run_duration_sec = (
        time.perf_counter()
        -
        RUN_STARTED_PERF
    )


    results_json = (
        build_p09_results_json(
            p09_tracker=
                p09_json_tracker,

            frequency_data=
                posture_change_frequency,

            fps=
                fps,

            total_frames=
                total_frames,

            frames_processed=
                frames_processed,

            video_duration_sec=
                video_duration_sec,

            processed_duration_sec=
                processed_duration_sec,

            run_duration_sec=
                run_duration_sec,

            pose_counter=
                pose_counter,

            rocking_counter=
                rocking_counter,

            combined_count=
                combined_count,

            pose_start_threshold=
                pose_start_threshold,

            transition_threshold=
                transition_threshold,

            pose_return_threshold=
                pose_return_threshold,
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


    print()
    print("=" * 72)
    print("V2 DONE")
    print("=" * 72)

    print(
        f"Pose changes: "
        f"{pose_counter.count}"
    )

    print(
        f"Rocking events: "
        f"{rocking_counter.count}"
    )

    print(
        f"Combined: "
        f"{combined_count}"
    )

    print(
        f"Video:\n"
        f"{OUTPUT_VIDEO}"
    )

    print(
        f"CSV:\n"
        f"{OUTPUT_CSV}"
    )


    print(
        f"JSON:\n"
        f"{OUTPUT_JSON}"
    )

    if graph_created:

        print(
            f"График P09:\n"
            f"{OUTPUT_FREQUENCY_GRAPH}"
        )

    else:

        print(
            "График P09: не создан "
            "(установи matplotlib: pip install matplotlib)"
        )

    print(
        f"P09 cuts in JSON: "
        f"{len(p09_json_tracker.cuts)}"
    )


if __name__ == "__main__":
    main()
