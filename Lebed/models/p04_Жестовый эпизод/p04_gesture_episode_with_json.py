
import csv
import json
import time
from datetime import datetime, timezone
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# P04 — ЖЕСТОВЫЙ ЭПИЗОД
# покой -> движение кисти/локтя -> окончание
# ============================================================

VIDEO_PATH = r"D:\Стажировка Братские сердца\Нарезки для p04\video_cut(p04_1).MP4"
POSE_MODEL_PATH = r"C:\interview\openface\pose_landmarker_heavy.task"

OUTPUT_VIDEO = r"D:\Стажировка Братские сердца\Нарезки для p04/P04_1_video_cut_gesture.mp4"
OUTPUT_CSV = r"D:\Стажировка Братские сердца\Нарезки для p04/P04_1_video_cut_gesture.csv"

# JSON в формате bs_profiling_cut_v1 — как в приложенном примере.
OUTPUT_JSON = r"D:\Стажировка Братские сердца\Нарезки для p04/P04_1_video_cut_gesture_results.json"

# В эталонном JSON для будущих нарезок используется padding 1.2 сек.
# Сам этот P04-скрипт клипы НЕ создаёт, поэтому clip_file/clip будут пустыми.
JSON_CLIP_PAD_SEC = 1.2

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

MIN_VISIBILITY = 0.50

POSITION_ALPHA = 0.35
SPEED_ALPHA = 0.30

# Скорость в "ширинах плеч / секунду"
GESTURE_START_SPEED = 0.18
GESTURE_END_SPEED = 0.075

START_HOLD_FRAMES = 3
END_HOLD_FRAMES = 7

MIN_EPISODE_SECONDS = 0.35
MIN_EPISODE_PATH = 0.08

MAX_MISSING_FRAMES = 5

WRIST_WEIGHT = 0.70
ELBOW_WEIGHT = 0.30

PANEL_ALPHA = 0.50
SHOW_PREVIEW = False


def find_font():
    candidates = [
        r"C:/Windows/Fonts/segoeui.ttf",
        r"C:/Windows/Fonts/arial.ttf",
        r"C:/Windows/Fonts/calibri.ttf",
        r"C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise RuntimeError("Не найден шрифт с поддержкой кириллицы.")


FONT_PATH = find_font()


def fmt_time(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "--:--.-"
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:04.1f}"


def lm_xy(lm):
    return np.array([float(lm.x), float(lm.y)], dtype=np.float64)


def lm_vis(lm):
    v = getattr(lm, "visibility", 1.0)
    return 1.0 if v is None else float(v)


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


class PointTracker:
    def __init__(self):
        self.position = None
        self.speed = 0.0
        self.missing = 0

    def update(self, point, dt):
        if point is None:
            self.missing += 1
            if self.missing > MAX_MISSING_FRAMES:
                self.position = None
                self.speed = 0.0
            return self.speed

        self.missing = 0
        point = np.asarray(point, dtype=np.float64)

        if self.position is None:
            self.position = point
            self.speed = 0.0
            return self.speed

        new_pos = (
            POSITION_ALPHA * point
            + (1.0 - POSITION_ALPHA) * self.position
        )

        raw_speed = dist(new_pos, self.position) / max(dt, 1e-6)

        self.position = new_pos

        self.speed = (
            SPEED_ALPHA * raw_speed
            + (1.0 - SPEED_ALPHA) * self.speed
        )

        return self.speed


class ArmTracker:
    def __init__(self):
        self.wrist = PointTracker()
        self.elbow = PointTracker()

    def update(self, wrist_point, elbow_point, dt):
        ws = self.wrist.update(wrist_point, dt)
        es = self.elbow.update(elbow_point, dt)

        wrist_ok = self.wrist.position is not None
        elbow_ok = self.elbow.position is not None

        if wrist_ok and elbow_ok:
            activity = WRIST_WEIGHT * ws + ELBOW_WEIGHT * es
        elif wrist_ok:
            activity = ws
        elif elbow_ok:
            activity = es
        else:
            activity = 0.0

        return float(activity)


class GestureEpisodeState:
    def __init__(self):
        self.active = False

        self.start_hold = 0
        self.end_hold = 0

        self.candidate_start = None
        self.current_start = None

        self.last_start = None
        self.last_end = None

        self.last_active_time = None

        self.current_path = 0.0
        self.event_count = 0
        self.events = []

    def update(self, activity, timestamp, dt):
        if not self.active:
            if activity >= GESTURE_START_SPEED:
                if self.start_hold == 0:
                    self.candidate_start = timestamp
                self.start_hold += 1
            else:
                self.start_hold = 0
                self.candidate_start = None

            if self.start_hold >= START_HOLD_FRAMES:
                self.active = True
                self.current_start = (
                    self.candidate_start
                    if self.candidate_start is not None
                    else timestamp
                )
                self.last_start = self.current_start
                self.last_end = None
                self.last_active_time = timestamp
                self.current_path = 0.0
                self.start_hold = 0
                self.end_hold = 0

        else:
            self.current_path += max(0.0, activity) * max(dt, 0.0)

            if activity >= GESTURE_END_SPEED:
                self.end_hold = 0
                self.last_active_time = timestamp
            else:
                self.end_hold += 1

            if self.end_hold >= END_HOLD_FRAMES:
                end_time = (
                    self.last_active_time
                    if self.last_active_time is not None
                    else timestamp
                )

                duration = (
                    end_time - self.current_start
                    if self.current_start is not None
                    else 0.0
                )

                if (
                    self.current_start is not None
                    and duration >= MIN_EPISODE_SECONDS
                    and self.current_path >= MIN_EPISODE_PATH
                ):
                    self.event_count += 1
                    self.last_start = self.current_start
                    self.last_end = end_time

                    self.events.append({
                        "event": self.event_count,
                        "start": float(self.current_start),
                        "end": float(end_time),
                        "duration": float(duration),
                        "path": float(self.current_path),
                    })

                self.active = False
                self.start_hold = 0
                self.end_hold = 0
                self.candidate_start = None
                self.current_start = None
                self.last_active_time = None
                self.current_path = 0.0

        return self.state()

    def state(self):
        return {
            "active": self.active,
            "event_count": self.event_count,
            "start_time": self.current_start if self.active else self.last_start,
            "end_time": None if self.active else self.last_end,
        }

    def close_at_end(self, timestamp):
        if not self.active or self.current_start is None:
            return

        end_time = (
            self.last_active_time
            if self.last_active_time is not None
            else timestamp
        )
        duration = end_time - self.current_start

        if (
            duration >= MIN_EPISODE_SECONDS
            and self.current_path >= MIN_EPISODE_PATH
        ):
            self.event_count += 1
            self.last_start = self.current_start
            self.last_end = end_time
            self.events.append({
                "event": self.event_count,
                "start": float(self.current_start),
                "end": float(end_time),
                "duration": float(duration),
                "path": float(self.current_path),
            })

        self.active = False


def extract_points(pose_result):
    if not pose_result.pose_landmarks:
        return None

    lm = pose_result.pose_landmarks[0]

    ls = lm[LEFT_SHOULDER]
    rs = lm[RIGHT_SHOULDER]

    if lm_vis(ls) < MIN_VISIBILITY or lm_vis(rs) < MIN_VISIBILITY:
        return None

    ls_xy = lm_xy(ls)
    rs_xy = lm_xy(rs)

    center = (ls_xy + rs_xy) / 2.0
    shoulder_width = dist(ls_xy, rs_xy)

    if shoulder_width < 1e-6:
        return None

    def norm_point(landmark):
        if lm_vis(landmark) < MIN_VISIBILITY:
            return None
        return (lm_xy(landmark) - center) / shoulder_width

    return {
        "left_wrist": norm_point(lm[LEFT_WRIST]),
        "left_elbow": norm_point(lm[LEFT_ELBOW]),
        "right_wrist": norm_point(lm[RIGHT_WRIST]),
        "right_elbow": norm_point(lm[RIGHT_ELBOW]),
    }


def draw_pose(frame, pose_result):
    if not pose_result.pose_landmarks:
        return

    h, w = frame.shape[:2]
    lm = pose_result.pose_landmarks[0]

    ids = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
    ]

    pts = {
        idx: (int(lm[idx].x * w), int(lm[idx].y * h))
        for idx in ids
    }

    links = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
    ]

    for a, b in links:
        cv2.line(
            frame,
            pts[a],
            pts[b],
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    for p in pts.values():
        cv2.circle(
            frame,
            p,
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_overlay(frame, state):
    h, w = frame.shape[:2]

    scale = float(
        np.clip(
            np.sqrt(min(h, w) / 720.0),
            0.80,
            1.35,
        )
    )

    title_font = ImageFont.truetype(
        FONT_PATH,
        max(15, int(round(18 * scale))),
    )
    body_font = ImageFont.truetype(
        FONT_PATH,
        max(14, int(round(16 * scale))),
    )

    line_h = max(22, int(round(25 * scale)))
    margin = max(10, int(round(15 * scale)))
    pad_x = max(12, int(round(14 * scale)))
    pad_y = max(10, int(round(12 * scale)))

    lines = [
        "P04 — ЖЕСТОВЫЙ ЭПИЗОД",
        f"Эпизодов: {state['event_count']}",
        f"Начало: {fmt_time(state['start_time'])}",
        (
            "Конец: —"
            if state["active"]
            else f"Конец: {fmt_time(state['end_time'])}"
        ),
    ]

    dummy = Image.new("RGB", (10, 10))
    md = ImageDraw.Draw(dummy)

    widths = []

    for i, line in enumerate(lines):
        font = title_font if i == 0 else body_font
        box = md.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])

    panel_w = max(widths) + 2 * pad_x
    panel_h = len(lines) * line_h + 2 * pad_y

    x1 = w - margin - panel_w
    y1 = margin
    x2 = w - margin
    y2 = y1 + panel_h

    overlay = frame.copy()

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

    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(image)

    for i, line in enumerate(lines):
        font = title_font if i == 0 else body_font

        draw.text(
            (
                x1 + pad_x,
                y1 + pad_y + i * line_h,
            ),
            line,
            font=font,
            fill=(255, 255, 255),
        )

    return cv2.cvtColor(
        np.asarray(image),
        cv2.COLOR_RGB2BGR,
    )



# ============================================================
# 13B. JSON — ФОРМАТ bs_profiling_cut_v1
# ============================================================

def human_duration(seconds):
    """
    Формат как в приложенном JSON:
    6м 1с
    1ч 12м 58с
    """
    seconds = max(0.0, float(seconds))

    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    parts = []

    if hours > 0:
        parts.append(f"{hours}ч")

    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}м")

    parts.append(f"{secs}с")

    return " ".join(parts)


def build_p04_json(
    events,
    fps,
    total_frames,
    frames_processed,
    run_duration_sec,
):
    """
    Создаёт структуру того же типа, что приложенный
    Alyona_interview_results(4).json.

    Этот скрипт анализирует ТОЛЬКО P04, поэтому:
      - patterns_hud содержит P04;
      - cuts содержит только P04;
      - timeline_segments пустой, т.к. Q/A timeline сюда не передан;
      - clip_file/clip пустые, т.к. этот скрипт не режет отдельные клипы.
    """

    fps = float(fps)

    if fps <= 0:
        fps = 30.0

    video_duration_sec = (
        float(total_frames) / fps
        if total_frames > 0
        else 0.0
    )

    processed_duration_sec = (
        float(frames_processed) / fps
        if frames_processed > 0
        else 0.0
    )

    processed_full_video = (
        total_frames <= 0
        or frames_processed >= total_frames
    )

    effective_fps = (
        float(frames_processed) / run_duration_sec
        if run_duration_sec > 1e-9
        else 0.0
    )

    speed_ratio = (
        processed_duration_sec / run_duration_sec
        if run_duration_sec > 1e-9
        else 0.0
    )

    wall_sec_per_video_min = (
        run_duration_sec / (processed_duration_sec / 60.0)
        if processed_duration_sec > 1e-9
        else 0.0
    )

    cuts = []

    for event in events:
        start = round(float(event["start"]), 3)
        end = round(float(event["end"]), 3)
        duration = round(float(event["duration"]), 3)

        cuts.append(
            {
                "source": "mediapipe",
                "pattern": "P04",
                "name": "жестовый эпизод",

                "t_start": start,
                "t_end": end,

                "start_sec": start,
                "end_sec": end,

                "duration_sec": duration,

                "clip_pad_sec": JSON_CLIP_PAD_SEC,

                # Отдельные clips этим скриптом не создаются.
                "clip_file": "",
                "clip": "",

                # Timeline Q/A этому скрипту не передан.
                "scenario_label": "",
                "scenario_kind": "",
            }
        )

    result = {
        "schema": "bs_profiling_cut_v1",

        "created_utc": (
            datetime.now(timezone.utc).isoformat()
        ),

        "tracker": (
            "MediaPipe Pose Landmarker (P04 gesture episode only)"
        ),

        "video": str(VIDEO_PATH),

        "video_name": Path(VIDEO_PATH).name,

        "timing": {
            "video_duration_sec": round(video_duration_sec, 3),
            "video_duration_human": human_duration(video_duration_sec),
            "video_duration_min": round(video_duration_sec / 60.0, 2),

            "run_duration_sec": round(run_duration_sec, 3),
            "run_duration_human": human_duration(run_duration_sec),
            "run_duration_min": round(run_duration_sec / 60.0, 2),

            "processed_duration_sec": round(processed_duration_sec, 3),
            "processed_duration_human": human_duration(processed_duration_sec),
            "processed_full_video": bool(processed_full_video),

            "speed_ratio": round(speed_ratio, 3),
            "wall_sec_per_video_min": round(wall_sec_per_video_min, 1),

            "frames_total": int(total_frames),
            "frames_processed": int(frames_processed),

            "video_fps": round(fps, 3),
            "effective_fps": round(effective_fps, 3),

            "run_finished_local": datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M"
            ),

            "note": (
                "speed_ratio 1.0 = realtime; "
                "P04 only: gesture episode from wrist/elbow motion"
            ),
        },

        "duration_processed_sec": round(processed_duration_sec, 3),
        "video_file_duration_sec": round(video_duration_sec, 3),

        "timeline_file": "",
        "timeline_segments": [],

        "patterns_hud": [
            {
                "code": "P04",
                "name": "жестовый эпизод",
                "count": int(len(events)),
            }
        ],

        "cuts": cuts,

        "overlay_timeline_mp4": str(OUTPUT_VIDEO),

        "notes": [
            (
                "Детектор P04: MediaPipe Pose Landmarker; "
                "событие = покой → движение кисти/локтя → окончание."
            ),
            (
                "Координаты кистей/локтей нормализуются по ширине плеч."
            ),
            (
                "timeline_segments, scenario_label и scenario_kind "
                "не заполняются, потому что отдельный timeline-файл "
                "этому скрипту не передан."
            ),
            (
                "clip_file и clip пустые: этот скрипт создаёт overlay-видео, "
                "но не вырезает отдельные event-клипы."
            ),
        ],
    }

    return result


def save_p04_json(
    events,
    fps,
    total_frames,
    frames_processed,
    run_duration_sec,
):
    data = build_p04_json(
        events=events,
        fps=fps,
        total_frames=total_frames,
        frames_processed=frames_processed,
        run_duration_sec=run_duration_sec,
    )

    Path(OUTPUT_JSON).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    run_started = time.perf_counter()

    for name, path in [
        ("VIDEO_PATH", VIDEO_PATH),
        ("POSE_MODEL_PATH", POSE_MODEL_PATH),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} не найден:\n{path}"
            )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть видео:\n{VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(OUTPUT_VIDEO).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(OUTPUT_CSV).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(OUTPUT_JSON).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Не удалось создать:\n{OUTPUT_VIDEO}"
        )

    BaseOptions = mp.tasks.BaseOptions
    RunningMode = mp.tasks.vision.RunningMode
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=POSE_MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    left_arm = ArmTracker()
    right_arm = ArmTracker()
    event_state = GestureEpisodeState()

    frame_index = 0

    print("=" * 72)
    print("P04 — ЖЕСТОВЫЙ ЭПИЗОД")
    print(f"FPS: {fps:.2f}")
    print(f"Кадров: {total_frames}")
    print("=" * 72)

    with PoseLandmarker.create_from_options(options) as pose_model:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp = frame_index / fps
            dt = 1.0 / fps

            timestamp_ms = int(round(timestamp * 1000.0))

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            pose_result = pose_model.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            points = extract_points(
                pose_result
            )

            if points is None:
                left_activity = left_arm.update(
                    None,
                    None,
                    dt,
                )
                right_activity = right_arm.update(
                    None,
                    None,
                    dt,
                )
            else:
                left_activity = left_arm.update(
                    points["left_wrist"],
                    points["left_elbow"],
                    dt,
                )

                right_activity = right_arm.update(
                    points["right_wrist"],
                    points["right_elbow"],
                    dt,
                )

            # Берём более активную руку.
            activity = max(
                left_activity,
                right_activity,
            )

            state = event_state.update(
                activity,
                timestamp,
                dt,
            )

            draw_pose(
                frame,
                pose_result,
            )

            frame = draw_overlay(
                frame,
                state,
            )

            writer.write(
                frame
            )

            if SHOW_PREVIEW:
                preview_w = min(1280, width)
                scale = preview_w / width

                preview = cv2.resize(
                    frame,
                    (
                        preview_w,
                        int(height * scale),
                    ),
                )

                cv2.imshow(
                    "P04",
                    preview,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

            frame_index += 1

            if frame_index % 300 == 0:
                print(
                    f"Обработано: "
                    f"{frame_index}/{total_frames}"
                )

    last_timestamp = max(
        0.0,
        (frame_index - 1) / fps,
    )

    event_state.close_at_end(
        last_timestamp
    )

    cap.release()
    writer.release()

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer_csv = csv.writer(f)

        writer_csv.writerow([
            "event",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "start_time",
            "end_time",
            "path_shoulder_widths",
        ])

        for event in event_state.events:
            writer_csv.writerow([
                event["event"],
                round(event["start"], 3),
                round(event["end"], 3),
                round(event["duration"], 3),
                fmt_time(event["start"]),
                fmt_time(event["end"]),
                round(event["path"], 4),
            ])

    # --------------------------------------------------------
    # JSON в формате bs_profiling_cut_v1
    # --------------------------------------------------------

    run_duration_sec = (
        time.perf_counter()
        - run_started
    )

    save_p04_json(
        events=event_state.events,
        fps=fps,
        total_frames=total_frames,
        frames_processed=frame_index,
        run_duration_sec=run_duration_sec,
    )

    print()
    print("=" * 72)
    print("ГОТОВО")
    print("=" * 72)
    print(
        "Количество жестовых эпизодов:",
        event_state.event_count,
    )

    for event in event_state.events:
        print(
            f"#{event['event']}: "
            f"{fmt_time(event['start'])}"
            f" -> "
            f"{fmt_time(event['end'])}"
            f" | "
            f"{event['duration']:.2f} сек"
        )

    print("Видео:", OUTPUT_VIDEO)
    print("CSV:", OUTPUT_CSV)
    print("JSON:", OUTPUT_JSON)
    print("=" * 72)


if __name__ == "__main__":
    main()
