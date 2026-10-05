"""
Local inference script for UniGaze Large (unigaze_l16_joint).

Adapted from the original Colab notebook for local execution in VS Code.
The gaze inference, normalization, direction classification, event counter,
CSV fields and video overlay logic are intentionally preserved.

Local-only changes:
- project-relative paths instead of /content;
- cross-platform Cyrillic font discovery (Windows/Linux/macOS);
- output directory creation;
- local clone path for the official UniGaze repository.
"""

import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

import face_alignment
import unigaze


# ============================================================
# ПУТИ — ЛОКАЛЬНЫЙ ЗАПУСК
# ============================================================

# Корень этого репозитория.
PROJECT_ROOT = Path(__file__).resolve().parent

# Входное видео.
# Для другого файла достаточно изменить только это имя
# или положить нужное видео в папку input под именем input.mp4.
VIDEO_PATH = str(
    PROJECT_ROOT
    / "input"
    / "input.mp4"
)

# Результаты.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO = str(
    OUTPUT_DIR
    / "input_large_unigaze.mp4"
)

OUTPUT_CSV = str(
    OUTPUT_DIR
    / "input_large_unigaze.csv"
)

# Официальный репозиторий UniGaze.
# setup_windows.ps1 клонирует его сюда автоматически.
UNIGAZE_REPO = str(
    PROJECT_ROOT
    / "third_party"
    / "UniGaze"
)


# ============================================================
# UNIGAZE
# ============================================================

MODEL_NAME = "unigaze_l16_joint"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Устройство:", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# ОСНОВНЫЕ НАСТРОЙКИ
# ============================================================

FACE_DETECTION_RESIZE = 0.5

# Временный порог классификации направления.
DIRECTION_THRESHOLD_DEG = 8.0


# ============================================================
# ЛЕВО / ПРАВО
# ============================================================

# True:
# направления считаются относительно самого участника.
#
# Например:
# участник смотрит вправо вниз ->
# RIGHT_DOWN / "Вправо вниз".
MIRROR_HORIZONTAL_LABELS = True


# ============================================================
# НОРМАЛИЗАЦИЯ UNIGAZE
# ============================================================

FOCAL_NORM = 960
DISTANCE_NORM = 600

ROI_SIZE = (
    224,
    224
)

MAX_HEAD_ANGLE_DEG = 80.0

ARROW_LENGTH_SCALE = 1.5


# ============================================================
# СОБЫТИЯ ВЗГЛЯДА
# ============================================================

EVENT_DIRECTIONS = (
    "CENTER",
    "RIGHT_UP",
    "RIGHT_DOWN",
    "LEFT_UP",
    "LEFT_DOWN",
)


HOLD_FRAMES = {
    "CENTER": 6,

    "RIGHT_UP": 4,
    "RIGHT_DOWN": 4,

    "LEFT_UP": 4,
    "LEFT_DOWN": 4,
}


RELEASE_FRAMES = 5


SAME_DIRECTION_REARM_FRAMES = {
    "CENTER": 8,

    "RIGHT_UP": 60,
    "RIGHT_DOWN": 60,

    "LEFT_UP": 60,
    "LEFT_DOWN": 60,
}


# ============================================================
# ОВЕРЛЕЙ — СЛЕВА ВНИЗУ, УВЕЛИЧЕН ЕЩЁ НА 20%
# ============================================================

# 128 / 255 = примерно 50% непрозрачности.
PANEL_ALPHA_255 = 128

# Было 25
PANEL_MARGIN = 12

# Было 31
LINE_HEIGHT = 29

# Было 23 / 22
TITLE_FONT_SIZE = 22
BODY_FONT_SIZE = 18


# ============================================================
# ПОИСК ШРИФТА С КИРИЛЛИЦЕЙ
# ============================================================

def find_cyrillic_font():

    candidates = []

    # Windows.
    windir = os.environ.get(
        "WINDIR",
        r"C:\Windows"
    )

    candidates.extend(
        [
            os.path.join(
                windir,
                "Fonts",
                "arial.ttf"
            ),
            os.path.join(
                windir,
                "Fonts",
                "calibri.ttf"
            ),
            os.path.join(
                windir,
                "Fonts",
                "segoeui.ttf"
            ),
        ]
    )

    # Linux / WSL.
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )

    # macOS.
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    )

    for path in candidates:

        if os.path.exists(path):

            print(
                "Русский шрифт:",
                path
            )

            return path


    search_roots = [
        os.path.join(
            windir,
            "Fonts"
        ),
        "/usr/share/fonts",
        "/Library/Fonts",
        "/System/Library/Fonts",
    ]


    preferred_names = {
        "arial.ttf",
        "dejavusans.ttf",
        "liberationsans-regular.ttf",
        "segoeui.ttf",
        "calibri.ttf",
    }


    for root in search_roots:

        if not os.path.isdir(root):

            continue

        for current_root, dirs, files in os.walk(
            root
        ):

            for filename in files:

                if (
                    filename.lower()
                    in
                    preferred_names
                ):

                    path = os.path.join(
                        current_root,
                        filename
                    )

                    print(
                        "Русский шрифт:",
                        path
                    )

                    return path


    raise RuntimeError(
        "\nНе найден системный шрифт с поддержкой кириллицы.\n"
        "Windows: проверь наличие Arial / Segoe UI / Calibri в C:\\Windows\\Fonts.\n"
        "Linux: установи DejaVu Sans или Liberation Sans.\n"
    )


RUSSIAN_FONT_PATH = (
    find_cyrillic_font()
)


TITLE_FONT = ImageFont.truetype(
    RUSSIAN_FONT_PATH,
    TITLE_FONT_SIZE
)


BODY_FONT = ImageFont.truetype(
    RUSSIAN_FONT_PATH,
    BODY_FONT_SIZE
)


# ============================================================
# МОДУЛИ UNIGAZE
# ============================================================

UNIGAZE_CODE_DIR = (
    Path(UNIGAZE_REPO)
    / "unigaze"
)


if not UNIGAZE_CODE_DIR.exists():

    raise RuntimeError(
        "Не найдена локальная копия UniGaze:\n"
        f"{UNIGAZE_CODE_DIR}\n\n"
        "Запусти setup_windows.ps1 или выполни:\n"
        "git clone https://github.com/ut-vision/UniGaze.git "
        "third_party/UniGaze"
    )


sys.path.insert(
    0,
    str(UNIGAZE_CODE_DIR)
)


from gazelib.gaze.gaze_utils import (
    pitchyaw_to_vector,
    vector_to_pitchyaw,
)

from gazelib.gaze.normalize import (
    estimateHeadPose,
    normalize,
)

from gazelib.label_transform import (
    get_face_center_by_nose,
)

from datasets.helper.image_transform import (
    wrap_transforms,
)


FACE_MODEL_PATH = (
    UNIGAZE_CODE_DIR
    / "data"
    / "face_model.txt"
)


if not FACE_MODEL_PATH.exists():

    raise RuntimeError(
        "Не найден face_model.txt:\n"
        f"{FACE_MODEL_PATH}"
    )


# ============================================================
# FACE ALIGNMENT
# ============================================================

def create_face_alignment():

    try:

        landmark_type = (
            face_alignment
            .LandmarksType
            .TWO_D
        )

    except AttributeError:

        landmark_type = (
            face_alignment
            .LandmarksType
            ._2D
        )


    return face_alignment.FaceAlignment(
        landmark_type,

        flip_input=False,

        device=DEVICE,

        # Не используем torch.compile
        compile=False,
    )


# ============================================================
# CAMERA MODEL
# ============================================================

def set_dummy_camera_model(image):

    h, w = image.shape[:2]

    focal_length = w * 4

    center = (
        w // 2,
        h // 2
    )


    camera_matrix = np.array(
        [
            [
                focal_length,
                0,
                center[0]
            ],

            [
                0,
                focal_length,
                center[1]
            ],

            [
                0,
                0,
                1
            ],
        ],
        dtype=np.float64
    )


    camera_distortion = np.zeros(
        (1, 5),
        dtype=np.float64
    )


    return (
        camera_matrix,
        camera_distortion
    )


# ============================================================
# DENORMALIZE
# ============================================================

def denormalize_predicted_gaze(
    gaze_pitch_yaw,
    R_inv
):

    gaze_vector = (
        pitchyaw_to_vector(
            np.asarray(
                gaze_pitch_yaw,
                dtype=np.float64
            ).reshape(
                1,
                2
            )
        ).reshape(
            3,
            1
        )
    )


    gaze_vector = (
        R_inv
        @ gaze_vector
    )


    norm = np.linalg.norm(
        gaze_vector
    )


    if norm > 0:

        gaze_vector = (
            gaze_vector
            / norm
        )


    pitch_yaw = (
        vector_to_pitchyaw(
            gaze_vector.reshape(
                1,
                3
            )
        ).reshape(
            2
        )
    )


    return (
        gaze_vector,
        pitch_yaw
    )


# ============================================================
# НАПРАВЛЕНИЕ
# ============================================================

def classify_direction(
    pitch_deg,
    yaw_deg,
    arrow_dx,
    arrow_dy
):

    horizontal = "CENTER"
    vertical = "CENTER"


    # --------------------------------------------------------
    # ГОРИЗОНТАЛЬ
    # --------------------------------------------------------

    if (
        abs(yaw_deg)
        >=
        DIRECTION_THRESHOLD_DEG
    ):

        # Сначала направление относительно изображения.

        if arrow_dx < 0:

            horizontal = "LEFT"

        else:

            horizontal = "RIGHT"


    # --------------------------------------------------------
    # ВЕРТИКАЛЬ
    # --------------------------------------------------------

    if (
        abs(pitch_deg)
        >=
        DIRECTION_THRESHOLD_DEG
    ):

        if arrow_dy < 0:

            vertical = "UP"

        else:

            vertical = "DOWN"


    # --------------------------------------------------------
    # СИСТЕМА КООРДИНАТ УЧАСТНИКА
    # --------------------------------------------------------

    if MIRROR_HORIZONTAL_LABELS:

        if horizontal == "LEFT":

            horizontal = "RIGHT"

        elif horizontal == "RIGHT":

            horizontal = "LEFT"


    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    if (
        horizontal == "CENTER"
        and
        vertical == "CENTER"
    ):

        return "CENTER"


    if vertical == "CENTER":

        return horizontal


    if horizontal == "CENTER":

        return vertical


    return (
        horizontal
        + "_"
        + vertical
    )


# ============================================================
# RAW -> EVENT
# ============================================================

def to_event_direction(
    raw_direction
):

    if raw_direction in EVENT_DIRECTIONS:

        return raw_direction


    # LEFT / RIGHT / UP / DOWN
    # считаются переходными кадрами.
    return "OTHER"


# ============================================================
# СЧЁТЧИК СОБЫТИЙ
# ============================================================

class GazeEventCounter:

    def __init__(self):

        self.counts = {
            "CENTER": 0,

            "RIGHT_UP": 0,
            "RIGHT_DOWN": 0,

            "LEFT_UP": 0,
            "LEFT_DOWN": 0,
        }


        self.stable_direction = "OTHER"

        self.candidate_direction = None
        self.candidate_frames = 0

        self.release_frames = 0

        self.frame_index = -1

        self.last_counted_direction = None
        self.last_count_frame = -1000000


    def update(
        self,
        raw_direction
    ):

        self.frame_index += 1


        support_direction = (
            to_event_direction(
                raw_direction
            )
        )


        event = ""


        # ----------------------------------------------------
        # OTHER
        # ----------------------------------------------------

        if support_direction == "OTHER":

            self.release_frames += 1

            self.candidate_direction = None
            self.candidate_frames = 0


            if (
                self.release_frames
                >=
                RELEASE_FRAMES
            ):

                self.stable_direction = (
                    "OTHER"
                )


            return (
                support_direction,
                self.stable_direction,
                event
            )


        # ----------------------------------------------------
        # VALID DIRECTION
        # ----------------------------------------------------

        self.release_frames = 0


        if (
            support_direction
            ==
            self.stable_direction
        ):

            self.candidate_direction = None
            self.candidate_frames = 0


            return (
                support_direction,
                self.stable_direction,
                event
            )


        # ----------------------------------------------------
        # CANDIDATE
        # ----------------------------------------------------

        if (
            self.candidate_direction
            ==
            support_direction
        ):

            self.candidate_frames += 1

        else:

            self.candidate_direction = (
                support_direction
            )

            self.candidate_frames = 1


        required_frames = (
            HOLD_FRAMES[
                support_direction
            ]
        )


        if (
            self.candidate_frames
            <
            required_frames
        ):

            return (
                support_direction,
                self.stable_direction,
                event
            )


        # ----------------------------------------------------
        # NEW STABLE
        # ----------------------------------------------------

        previous_stable = (
            self.stable_direction
        )


        new_direction = (
            support_direction
        )


        self.stable_direction = (
            new_direction
        )


        self.candidate_direction = None
        self.candidate_frames = 0


        # ----------------------------------------------------
        # ANTI-FRAGMENTATION
        # ----------------------------------------------------

        allow_count = True


        if (
            previous_stable == "OTHER"
            and
            self.last_counted_direction
            ==
            new_direction
        ):

            frames_since_last = (
                self.frame_index
                -
                self.last_count_frame
            )


            if (
                frames_since_last
                <
                SAME_DIRECTION_REARM_FRAMES[
                    new_direction
                ]
            ):

                allow_count = False


        # ----------------------------------------------------
        # +1
        # ----------------------------------------------------

        if allow_count:

            self.counts[
                new_direction
            ] += 1


            event = (
                new_direction
            )


            self.last_counted_direction = (
                new_direction
            )


            self.last_count_frame = (
                self.frame_index
            )


        return (
            support_direction,
            self.stable_direction,
            event
        )


# ============================================================
# САМОЕ БОЛЬШОЕ ЛИЦО
# ============================================================

def select_largest_face(
    landmarks_list,
    resize_factor
):

    best_landmarks = None
    best_area = -1


    for landmarks_small in landmarks_list:

        landmarks = (
            np.asarray(
                landmarks_small,
                dtype=np.float64
            )
            /
            resize_factor
        )


        x_min = landmarks[:, 0].min()
        x_max = landmarks[:, 0].max()

        y_min = landmarks[:, 1].min()
        y_max = landmarks[:, 1].max()


        area = (
            (x_max - x_min)
            *
            (y_max - y_min)
        )


        if area > best_area:

            best_area = area
            best_landmarks = landmarks


    return best_landmarks


# ============================================================
# РУССКИЕ НАЗВАНИЯ
# ============================================================

def pretty_direction(
    direction
):

    names = {
        "CENTER":
            "Центр",

        "RIGHT_UP":
            "Вправо вверх",

        "RIGHT_DOWN":
            "Вправо вниз",

        "LEFT_UP":
            "Влево вверх",

        "LEFT_DOWN":
            "Влево вниз",

        "OTHER":
            "Переход",

        "LEFT":
            "Влево",

        "RIGHT":
            "Вправо",

        "UP":
            "Вверх",

        "DOWN":
            "Вниз",

        "NO_DATA":
            "Нет данных",
    }


    return names.get(
        direction,
        direction
    )


# ============================================================
# КОМПАКТНЫЙ РУССКИЙ ОВЕРЛЕЙ
# СЛЕВА ВНИЗУ
# ============================================================

def draw_counter_panel(
    frame,
    stable_direction,
    event,
    counts
):

    h, w = frame.shape[:2]


    # --------------------------------------------------------
    # СТРОКИ
    # --------------------------------------------------------

    lines = [

        (
            "ВЗГЛЯД — UniGaze L16",
            TITLE_FONT
        ),

        (
            "Направление: "
            +
            pretty_direction(
                stable_direction
            ),
            BODY_FONT
        ),

        (
            "",
            BODY_FONT
        ),

        (
            "Центр: "
            +
            str(
                counts["CENTER"]
            ),
            BODY_FONT
        ),

        (
            "Вправо вверх: "
            +
            str(
                counts["RIGHT_UP"]
            ),
            BODY_FONT
        ),

        (
            "Вправо вниз: "
            +
            str(
                counts["RIGHT_DOWN"]
            ),
            BODY_FONT
        ),

        (
            "Влево вверх: "
            +
            str(
                counts["LEFT_UP"]
            ),
            BODY_FONT
        ),

        (
            "Влево вниз: "
            +
            str(
                counts["LEFT_DOWN"]
            ),
            BODY_FONT
        ),
    ]


    if event:

        lines.append(
            (
                "",
                BODY_FONT
            )
        )

        lines.append(
            (
                "Событие +1: "
                +
                pretty_direction(
                    event
                ),
                BODY_FONT
            )
        )


    # --------------------------------------------------------
    # ВЫЧИСЛЯЕМ РАЗМЕР ПАНЕЛИ
    # --------------------------------------------------------

    temp_image = Image.new(
        "RGB",
        (
            10,
            10
        )
    )


    temp_draw = ImageDraw.Draw(
        temp_image
    )


    max_width = 0


    for text_line, font in lines:

        if text_line == "":

            continue


        bbox = temp_draw.textbbox(
            (
                0,
                0
            ),
            text_line,
            font=font
        )


        text_width = (
            bbox[2]
            -
            bbox[0]
        )


        max_width = max(
            max_width,
            text_width
        )


    # Было +42
    panel_width = (
        max_width
        +
        35
    )


    # Было +28
    panel_height = (
        len(lines)
        *
        LINE_HEIGHT
        +
        25
    )


    # ========================================================
    # ЛЕВЫЙ НИЖНИЙ УГОЛ
    # ========================================================

    x1 = max(
        0,
        PANEL_MARGIN
    )


    y1 = max(
        0,
        h
        -
        PANEL_MARGIN
        -
        panel_height
    )


    x2 = min(
        w,
        x1
        +
        panel_width
    )


    y2 = min(
        h,
        y1
        +
        panel_height
    )


    # --------------------------------------------------------
    # OpenCV -> PIL RGBA
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    base_image = (
        Image.fromarray(
            frame_rgb
        )
        .convert(
            "RGBA"
        )
    )


    overlay = Image.new(
        "RGBA",
        base_image.size,
        (
            0,
            0,
            0,
            0
        )
    )


    overlay_draw = ImageDraw.Draw(
        overlay
    )


    # --------------------------------------------------------
    # 50% ПРОЗРАЧНЫЙ ЧЁРНЫЙ ФОН
    # --------------------------------------------------------

    overlay_draw.rectangle(
        (
            x1,
            y1,
            x2,
            y2
        ),
        fill=(
            0,
            0,
            0,
            PANEL_ALPHA_255
        )
    )


    # --------------------------------------------------------
    # КОМПАКТНЫЕ ОТСТУПЫ ТЕКСТА
    # --------------------------------------------------------

    # Было +18
    text_x = (
        x1 + 14
    )


    # Было +13
    text_y = (
        y1 + 11
    )


    # --------------------------------------------------------
    # РУССКИЙ ТЕКСТ
    # --------------------------------------------------------

    for index, (
        text_line,
        font
    ) in enumerate(
        lines
    ):

        if text_line == "":

            continue


        overlay_draw.text(
            (
                text_x,
                text_y
                +
                index
                *
                LINE_HEIGHT
            ),
            text_line,

            font=font,

            fill=(
                255,
                255,
                255,
                255
            )
        )


    # --------------------------------------------------------
    # СОЕДИНЯЕМ С КАДРОМ
    # --------------------------------------------------------

    composed = (
        Image.alpha_composite(
            base_image,
            overlay
        )
        .convert(
            "RGB"
        )
    )


    result = cv2.cvtColor(
        np.asarray(
            composed
        ),
        cv2.COLOR_RGB2BGR
    )


    frame[:] = result


# ============================================================
# SAFE CSV
# ============================================================

def safe(value):

    try:

        if np.isfinite(value):

            return value

    except Exception:

        pass


    return ""


# ============================================================
# LOAD UNIGAZE
# ============================================================

print()
print("=" * 72)

print(
    "UniGaze L16 + русский счётчик взгляда"
)

print("=" * 72)


model = unigaze.load(
    MODEL_NAME,
    device=DEVICE
)


model.eval()


image_transform = (
    wrap_transforms(
        "basic_imagenet",
        image_size=224
    )
)


# ============================================================
# FACE ALIGNMENT
# ============================================================

print(
    "Загрузка face_alignment..."
)


fa = create_face_alignment()


print(
    "face_alignment готов"
)


# ============================================================
# 3D FACE MODEL
# ============================================================

face_model_load = np.loadtxt(
    str(
        FACE_MODEL_PATH
    )
)


face_model = (
    face_model_load[
        [
            20,
            23,
            26,
            29,
            15,
            19
        ],
        :
    ]
)


face_pts = (
    face_model.reshape(
        6,
        1,
        3
    )
)


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        "Не удалось открыть видео:\n"
        f"{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)


if (
    not np.isfinite(fps)
    or
    fps <= 0
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


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print(
    f"Видео: {width}x{height}"
)

print(
    f"FPS: {fps:.2f}"
)

print(
    f"Кадров: {total_frames}"
)


# ============================================================
# VIDEO WRITER
# ============================================================

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,

    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),

    fps,

    (
        width,
        height
    )
)


if not writer.isOpened():

    raise RuntimeError(
        "Не удалось создать итоговое видео"
    )


# ============================================================
# CSV
# ============================================================

csv_file = open(
    OUTPUT_CSV,

    "w",

    newline="",

    encoding="utf-8-sig"
)


csv_writer = csv.writer(
    csv_file
)


csv_writer.writerow(
    [
        "frame",
        "timestamp",

        "face_found",

        "pitch_rad",
        "yaw_rad",

        "pitch_deg",
        "yaw_deg",

        "raw_direction",
        "support_direction",
        "stable_direction",

        "event",

        "center_count",

        "right_up_count",
        "right_down_count",

        "left_up_count",
        "left_down_count",

        "face_x1",
        "face_y1",
        "face_x2",
        "face_y2",

        "gaze_start_x",
        "gaze_start_y",

        "gaze_end_x",
        "gaze_end_y",
    ]
)


# ============================================================
# EVENT COUNTER
# ============================================================

event_counter = (
    GazeEventCounter()
)


# ============================================================
# PROCESS VIDEO
# ============================================================

progress = tqdm(
    total=total_frames,

    desc="UniGaze L16",

    unit="frame"
)


frame_index = 0


while True:

    ok, frame = cap.read()


    if not ok:

        break


    timestamp = (
        frame_index
        /
        fps
    )


    face_found = False

    raw_direction = (
        "NO_DATA"
    )


    pitch_rad = np.nan
    yaw_rad = np.nan

    pitch_deg = np.nan
    yaw_deg = np.nan


    face_box = [
        np.nan
    ] * 4


    gaze_start = [
        np.nan
    ] * 2


    gaze_end = [
        np.nan
    ] * 2


    # ========================================================
    # UNIGAZE
    # ========================================================

    try:

        # ----------------------------------------------------
        # УМЕНЬШАЕМ КАДР ДЛЯ FACE ALIGNMENT
        # ----------------------------------------------------

        if (
            FACE_DETECTION_RESIZE
            <
            1.0
        ):

            small = cv2.resize(
                frame,

                None,

                fx=
                    FACE_DETECTION_RESIZE,

                fy=
                    FACE_DETECTION_RESIZE,

                interpolation=
                    cv2.INTER_AREA
            )

        else:

            small = (
                frame.copy()
            )


        small_rgb = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB
        )


        landmarks_list = (
            fa.get_landmarks(
                small_rgb
            )
        )


        # ----------------------------------------------------
        # ЛИЦО
        # ----------------------------------------------------

        if (
            landmarks_list is not None
            and
            len(
                landmarks_list
            ) > 0
        ):

            landmarks_original = (
                select_largest_face(
                    landmarks_list,
                    FACE_DETECTION_RESIZE
                )
            )


            x_min0 = int(
                landmarks_original[
                    :,
                    0
                ].min()
            )


            x_max0 = int(
                landmarks_original[
                    :,
                    0
                ].max()
            )


            y_min0 = int(
                landmarks_original[
                    :,
                    1
                ].min()
            )


            y_max0 = int(
                landmarks_original[
                    :,
                    1
                ].max()
            )


            bbox_width = max(
                1,
                x_max0
                -
                x_min0
            )


            bbox_height = max(
                1,
                y_max0
                -
                y_min0
            )


            center_x = (
                x_min0 + x_max0
            ) // 2


            center_y = (
                y_min0 + y_max0
            ) // 2


            # =================================================
            # FACE BOX
            # =================================================

            draw_scale = 1.2


            draw_x1 = max(
                0,

                center_x
                -
                int(
                    bbox_width
                    *
                    draw_scale
                    /
                    2
                )
            )


            draw_x2 = min(
                width - 1,

                center_x
                +
                int(
                    bbox_width
                    *
                    draw_scale
                    /
                    2
                )
            )


            draw_y1 = max(
                0,

                center_y
                -
                int(
                    bbox_height
                    *
                    draw_scale
                    /
                    2
                )
            )


            draw_y2 = min(
                height - 1,

                center_y
                +
                int(
                    bbox_height
                    *
                    draw_scale
                    /
                    2
                )
            )


            face_box = [
                draw_x1,
                draw_y1,

                draw_x2,
                draw_y2
            ]


            # =================================================
            # CROP
            # =================================================

            crop_scale = 2.0


            crop_x1 = max(
                0,

                center_x
                -
                int(
                    bbox_width
                    *
                    crop_scale
                    /
                    2
                )
            )


            crop_x2 = min(
                width,

                center_x
                +
                int(
                    bbox_width
                    *
                    crop_scale
                    /
                    2
                )
            )


            crop_y1 = max(
                0,

                center_y
                -
                int(
                    bbox_height
                    *
                    crop_scale
                    /
                    2
                )
            )


            crop_y2 = min(
                height,

                center_y
                +
                int(
                    bbox_height
                    *
                    crop_scale
                    /
                    2
                )
            )


            face_image = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]


            if (
                face_image.size
                > 0
            ):

                landmarks = (
                    landmarks_original

                    -

                    np.array(
                        [
                            crop_x1,
                            crop_y1
                        ]
                    )
                )


                # =============================================
                # CAMERA
                # =============================================

                (
                    camera_matrix,
                    camera_distortion
                ) = set_dummy_camera_model(
                    face_image
                )


                # =============================================
                # HEAD POSE
                # =============================================

                landmarks_sub = (
                    landmarks[
                        [
                            36,
                            39,
                            42,
                            45,
                            31,
                            35
                        ],
                        :
                    ]
                    .astype(float)
                    .reshape(
                        6,
                        1,
                        2
                    )
                )


                hr, ht = estimateHeadPose(
                    landmarks_sub,

                    face_pts,

                    camera_matrix,

                    camera_distortion
                )


                hR = cv2.Rodrigues(
                    hr
                )[0]


                (
                    face_center_camera,
                    _
                ) = get_face_center_by_nose(
                    hR=hR,

                    ht=ht,

                    face_model_load=
                        face_model_load
                )


                # =============================================
                # NORMALIZATION
                # =============================================

                (
                    normalized_image,
                    R,
                    hR_norm,
                    _,
                    _,
                    _
                ) = normalize(
                    face_image,

                    landmarks,

                    FOCAL_NORM,

                    DISTANCE_NORM,

                    ROI_SIZE,

                    face_center_camera,

                    hr,

                    ht,

                    camera_matrix,

                    gc=None
                )


                head_angles = np.array(
                    [
                        np.arcsin(
                            np.clip(
                                hR_norm[
                                    1,
                                    2
                                ],
                                -1,
                                1
                            )
                        ),

                        np.arctan2(
                            hR_norm[
                                0,
                                2
                            ],

                            hR_norm[
                                2,
                                2
                            ]
                        )
                    ]
                )


                if (
                    np.linalg.norm(
                        head_angles
                    )
                    <=
                    np.deg2rad(
                        MAX_HEAD_ANGLE_DEG
                    )
                ):

                    # =========================================
                    # INPUT
                    # =========================================

                    model_input = (
                        normalized_image[
                            :,
                            :,
                            [
                                2,
                                1,
                                0
                            ]
                        ]
                    )


                    model_input = (
                        image_transform(
                            model_input
                        )
                    )


                    model_input = (
                        model_input

                        .float()

                        .to(
                            DEVICE
                        )

                        .unsqueeze(
                            0
                        )
                    )


                    # =========================================
                    # INFERENCE
                    # =========================================

                    with torch.inference_mode():

                        result = model(
                            model_input
                        )


                    predicted_gaze = (
                        result[
                            "pred_gaze"
                        ][0]

                        .detach()

                        .cpu()

                        .numpy()
                    )


                    (
                        gaze_vector,
                        pitch_yaw
                    ) = denormalize_predicted_gaze(
                        predicted_gaze,

                        np.linalg.inv(
                            R
                        )
                    )


                    pitch_rad = float(
                        pitch_yaw[
                            0
                        ]
                    )


                    yaw_rad = float(
                        pitch_yaw[
                            1
                        ]
                    )


                    pitch_deg = float(
                        np.degrees(
                            pitch_rad
                        )
                    )


                    yaw_deg = float(
                        np.degrees(
                            yaw_rad
                        )
                    )


                    # =========================================
                    # СТРЕЛКА
                    # =========================================

                    vector_length = (
                        gaze_vector

                        * -112

                        *
                        ARROW_LENGTH_SCALE
                    )


                    gaze_ray = np.concatenate(
                        (
                            face_center_camera
                            .reshape(
                                1,
                                3
                            ),

                            (
                                face_center_camera
                                +
                                vector_length
                            )
                            .reshape(
                                1,
                                3
                            )
                        ),
                        axis=0
                    )


                    projected = (
                        cv2.projectPoints(
                            gaze_ray,

                            np.zeros(
                                (
                                    3,
                                    1
                                )
                            ),

                            np.zeros(
                                (
                                    3,
                                    1
                                )
                            ),

                            camera_matrix,

                            camera_distortion

                        )[0]

                        .reshape(
                            2,
                            2
                        )
                    )


                    projected += np.array(
                        [
                            crop_x1,
                            crop_y1
                        ]
                    )


                    gaze_start = [
                        int(
                            projected[
                                0,
                                0
                            ]
                        ),

                        int(
                            projected[
                                0,
                                1
                            ]
                        )
                    ]


                    gaze_end = [
                        int(
                            projected[
                                1,
                                0
                            ]
                        ),

                        int(
                            projected[
                                1,
                                1
                            ]
                        )
                    ]


                    # =========================================
                    # CLASSIFICATION
                    # =========================================

                    raw_direction = (
                        classify_direction(
                            pitch_deg,

                            yaw_deg,

                            gaze_end[0]
                            -
                            gaze_start[0],

                            gaze_end[1]
                            -
                            gaze_start[1]
                        )
                    )


                    face_found = True


    except Exception as error:

        if frame_index < 10:

            print(
                "\nОшибка кадра "
                f"{frame_index + 1}: "
                f"{type(error).__name__}: "
                f"{error}"
            )


    # ========================================================
    # EVENT COUNTER
    # ========================================================

    (
        support_direction,
        stable_direction,
        event
    ) = event_counter.update(
        raw_direction
    )


    # ========================================================
    # FACE + ARROW
    # ========================================================

    if face_found:

        cv2.rectangle(
            frame,

            (
                face_box[0],
                face_box[1]
            ),

            (
                face_box[2],
                face_box[3]
            ),

            (
                0,
                0,
                240
            ),

            2
        )


        # Стрелку не зеркалим.
        # Зеркалится только логическое LEFT/RIGHT.

        cv2.arrowedLine(
            frame,

            tuple(
                gaze_start
            ),

            tuple(
                gaze_end
            ),

            (
                47,
                255,
                173
            ),

            4,

            cv2.LINE_AA,

            tipLength=0.2
        )


    # ========================================================
    # РУССКИЙ КОМПАКТНЫЙ ОВЕРЛЕЙ
    # ========================================================

    draw_counter_panel(
        frame,

        stable_direction,

        event,

        event_counter.counts
    )


    writer.write(
        frame
    )


    # ========================================================
    # CSV
    # ========================================================

    csv_writer.writerow(
        [
            frame_index + 1,

            round(
                timestamp,
                6
            ),

            int(
                face_found
            ),

            safe(
                pitch_rad
            ),

            safe(
                yaw_rad
            ),

            safe(
                pitch_deg
            ),

            safe(
                yaw_deg
            ),

            raw_direction,

            support_direction,

            stable_direction,

            event,

            event_counter.counts[
                "CENTER"
            ],

            event_counter.counts[
                "RIGHT_UP"
            ],

            event_counter.counts[
                "RIGHT_DOWN"
            ],

            event_counter.counts[
                "LEFT_UP"
            ],

            event_counter.counts[
                "LEFT_DOWN"
            ],

            *[
                safe(value)
                for value
                in face_box
            ],

            *[
                safe(value)
                for value
                in gaze_start
            ],

            *[
                safe(value)
                for value
                in gaze_end
            ],
        ]
    )


    frame_index += 1

    progress.update(
        1
    )


# ============================================================
# FINISH
# ============================================================

progress.close()

cap.release()
writer.release()
csv_file.close()


print()
print("=" * 72)
print("ГОТОВО")
print("=" * 72)


print(
    "Видео:",
    OUTPUT_VIDEO
)


print(
    "CSV:",
    OUTPUT_CSV
)


print()
print("ИТОГ:")


print(
    "Центр:",
    event_counter.counts[
        "CENTER"
    ]
)


print(
    "Вправо вверх:",
    event_counter.counts[
        "RIGHT_UP"
    ]
)


print(
    "Вправо вниз:",
    event_counter.counts[
        "RIGHT_DOWN"
    ]
)


print(
    "Влево вверх:",
    event_counter.counts[
        "LEFT_UP"
    ]
)


print(
    "Влево вниз:",
    event_counter.counts[
        "LEFT_DOWN"
    ]
)