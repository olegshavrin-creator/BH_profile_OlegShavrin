"""Plain-text journal of the web service: who opened the page and when, which file was analysed and the outcome.

One UTF-8 text file, entries appended in order and never rewritten: ~/bs3_data/logs/journal.txt (BS3_JOURNAL
overrides). A start entry names the model chosen for the analysis (BS Profiler 3.1: one model per analysis, Russian
speech only). A result entry carries the clean scores of that model, its MBTI type and the paragraph «Коротко» of the
characterization, the same as the page shows them. Visitor details come from the reverse proxy: X-Forwarded-For
(client address) and X-Remote-User (login name from Caddy basic_auth); opened directly on the machine, the address is
marked «локально». Writing the journal never breaks the page: every failure is logged and swallowed.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path

from . import MODEL_TITLES, settings
from .labels import model_title
from .textfmt import clock

log = logging.getLogger("bs3.journal")

PATH = settings.JOURNAL_PATH          # a module attribute: tests and compare_baseline point it elsewhere
_lock = threading.Lock()

TRAITS = (("openness", "открытость"), ("conscientiousness", "добросовестность"), ("extraversion", "экстраверсия"),
          ("agreeableness", "доброжелательность"), ("emotional_stability", "эмоциональная стабильность"))
_DEVICES = (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"), ("Windows", "Windows"),
            ("Macintosh", "Mac"), ("Linux", "Linux"))
_BROWSERS = (("YaBrowser", "Яндекс Браузер"), ("Edg", "Edge"), ("OPR/", "Opera"), ("Firefox", "Firefox"),
             ("FxiOS", "Firefox"), ("CriOS", "Chrome"), ("Chrome", "Chrome"), ("Safari", "Safari"))


def _agent(ua: str) -> str:
    ua = ua or ""
    dev = next((name for key, name in _DEVICES if key in ua), "")
    br = next((name for key, name in _BROWSERS if key in ua), "")
    return ", ".join(x for x in (dev, br) if x) or (ua[:60] or "браузер не указан")


def _who(request) -> str:
    """«сеанс 3f9a1c  логин  адрес  устройство» for the head line of an entry."""
    try:
        headers = request.headers
        fwd = (headers.get("x-forwarded-for") or "").split(",")[0].strip()
        ip = fwd or "локально"
        user = (headers.get("x-remote-user") or "").strip() or "без логина"
        sess = (getattr(request, "session_hash", "") or "")[:6] or "------"
        return f"сеанс {sess}  {user}  {ip}  {_agent(headers.get('user-agent'))}"
    except Exception:  # noqa: BLE001
        return "сеанс ?"


def _write(kind: str, request, tail: str = "", body: list[str] | tuple = ()) -> None:
    head = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {kind:<10} {_who(request)}" + (f"  {tail}" if tail else "")
    text = head + "\n" + "".join(f"    {line}\n" for line in body) + ("\n" if body else "")
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(PATH, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:  # noqa: BLE001
        log.exception("journal write failed (%s)", PATH)


def _file(video) -> str:
    name = Path(str(video)).name
    try:
        return f"файл «{name}», {os.path.getsize(video) / 1e6:.0f} МБ"
    except OSError:
        return f"файл «{name}»"


def _scores(scores: dict) -> str:
    return ", ".join(f"{ru} {float(scores[k]):.2f}" for k, ru in TRAITS if k in scores)


def result_lines(rep: dict) -> list[str]:
    """Body of a result entry (design 10.6; one model since 3.1): the clean scores of the model the view shows
    (scores.clean_view), its MBTI type, the paragraph «Коротко» of the characterization and the job folder. The view,
    the type and the characterization are built here from `rep` (jobview.from_report); nothing is written back."""
    from . import jobview
    return _lines(jobview.from_report(rep))


def _lines(jv) -> list[str]:
    """result_lines of a JobView (jobview): its view, MBTI section and characterization as they are."""
    from .mbti import journal_lines
    from .scores import shown_model

    view = jv.view
    main = {k: (v.get("score") if isinstance(v, dict) else v) for k, v in (view.get("traits") or {}).items()}
    line = f"Итог ({model_title(shown_model(view))}): {_scores(main)}"
    iv = view.get("interview")
    iv = iv.get("score") if isinstance(iv, dict) else iv
    if iv is not None:
        line += f"; «пригласил бы на собеседование» {float(iv):.2f}"
    lines = [line]
    lines += journal_lines(jv.mb)
    short = re.sub(r"\s+", " ", jv.character.short_plain() or "").strip()
    if short:
        lines.append(f"Характеристика (коротко): {short}")
    if jv.rep.get("job_dir"):
        lines.append(f"Папка: {jv.rep['job_dir']}")
    return lines


def visit(request) -> None:
    _write("ВХОД", request)


def start(request, video, member: str) -> None:
    """`member`: the model chosen for the analysis ("oceanai" | "mm"), written by its title. Explanations follow the
    model (AMLAI 1.0 only, 3.1), so the entry does not mention them."""
    _write("СТАРТ", request, f"{_file(video)}, модель {MODEL_TITLES.get(member, member)}")


def result(request, rep: dict, wall_sec: float, jv=None) -> None:
    """`jv`: the JobView the page of this analysis was built from (jobview.for_page); the entry then takes its view,
    type and characterization instead of building them again. Without it they are built from `rep` (result_lines)."""
    name = rep.get("original_file_name") or Path(str(rep.get("input", ""))).name
    tail = f"файл «{name}», ролик {clock(rep.get('duration_sec', 0))}, обработка {clock(wall_sec)}"
    try:
        body = result_lines(rep) if jv is None else _lines(jv)
    except Exception:  # noqa: BLE001
        log.exception("journal: could not format the result")
        body = ["(не удалось оформить итог, см. result.json в папке задачи)"]
    _write("РЕЗУЛЬТАТ", request, tail, body)


def failed(request, video, message: str, stopped: bool = False) -> None:
    _write("ОСТАНОВЛЕНО" if stopped else "ОШИБКА", request, f"{Path(str(video)).name}: {message}")
