"""The journal entries of BS Profiler 3.1 (design 10.6; task T19) on the numeric copy of sample B: the start entry
names the chosen model (and nothing about explanations: they follow the model), the result entry carries the clean
scores of one model, its MBTI line and the paragraph «Коротко»; no second opinion, no language, no «Краткие выводы».
The entries are written into a temporary file, never into the journal of the service."""
from __future__ import annotations

import contextlib
import logging
import re
import tempfile
import types
from pathlib import Path

from samples import rep

from bs3 import journal


class _Req:
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120"}
    session_hash = "abcdef123"


@contextlib.contextmanager
def _at(path: Path):
    """journal.PATH pointed at `path` while the block runs."""
    old = journal.PATH
    journal.PATH = path
    try:
        yield
    finally:
        journal.PATH = old


@contextlib.contextmanager
def _records(name: str = "bs3.journal"):
    """Capture the log records of `name` (and, as a handler is present, keep them off stderr)."""
    lg = logging.getLogger(name)
    recs: list = []
    h = logging.Handler()
    h.emit = recs.append
    lg.addHandler(h)
    try:
        yield recs
    finally:
        lg.removeHandler(h)


def test_result_lines_b():
    r = rep("B")
    r["job_dir"] = "/tmp/job"
    lines = journal.result_lines(r)
    assert lines[0].startswith("Итог (OCEAN-AI, веса MuPTA): ") and "экстраверсия 0.73" in lines[0]
    assert lines[1] == "Тип MBTI (OCEAN-AI): ENFJ «Наставник»; нейротизм — средний уровень"
    short = [ln for ln in lines if ln.startswith("Характеристика (коротко): ")]
    assert len(short) == 1 and "ENFJ" in short[0] and "Вторая система" not in short[0]
    for ln in lines:
        for w in ("предварительн", "типичн", "опорн", "русских роликов", "Второе мнение", "второе мнение",
                  "своя модель", "Участник", "ISXX", "ISTP"):
            assert w not in ln, (w, ln)
    assert lines[-1] == "Папка: /tmp/job"
    assert len(lines) == 4                                        # scores, type, «Коротко», folder: one model
    assert not any("Краткие выводы" in ln for ln in lines)
    assert "mbti" not in r                                        # the entry never stores the computed section


def test_result_lines_own_model():
    """A job of AMLAI 1.0 (3.1) or an old job where OCEAN-AI gave nothing: the lines name AMLAI 1.0."""
    r = rep("B")
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    lines = journal.result_lines(r)
    assert lines[0].startswith("Итог (AMLAI 1.0): ") and "MuPTA" not in lines[0]
    assert lines[1].startswith("Тип MBTI (AMLAI 1.0): ISXX, ближайший ISTP")
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    line = journal.result_lines(r)[0]
    assert line.startswith("Итог (AMLAI 1.0): открытость 0.") and "экстраверсия 0.25" in line


def test_start_and_result_write_entries():
    old = journal.PATH
    with tempfile.TemporaryDirectory() as d:
        journal.PATH = Path(d) / "journal.txt"
        try:
            journal.start(_Req(), "/tmp/video.mp4", "mm")
            journal.start(_Req(), "/tmp/video.mp4", "oceanai")
            journal.result(_Req(), rep("B"), 65.0)
            text = journal.PATH.read_text(encoding="utf-8")
        finally:
            journal.PATH = old
    assert "СТАРТ" in text and "модель AMLAI 1.0\n" in text and "модель OCEAN-AI\n" in text
    assert "язык" not in text.lower() and "объяснени" not in text.lower()      # no checkbox, no language (3.1)
    assert "РЕЗУЛЬТАТ" in text and "обработка 1:05" in text
    assert "    Тип MBTI (OCEAN-AI): ENFJ" in text
    assert "Характеристика (коротко): " in text and "Краткие выводы" not in text and "торое мнение" not in text


def _no_time(line: str) -> str:
    return re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", "", line)


def test_failed_head_lines():
    """The exact head line of an ОШИБКА and an ОСТАНОВЛЕНО entry (T5): the padded kind column, the «who» head and the
    «<file>: <message>» tail. The message is already the calm Russian one from bs3.errors; the journal quotes it."""
    with tempfile.TemporaryDirectory() as d, _at(Path(d) / "journal.txt"):
        journal.failed(_Req(), "/tmp/Интервью.mp4", "Не хватило памяти видеокарты. Подождите минуту.")
        journal.failed(_Req(), "/tmp/Интервью.mp4", "остановлено пользователем", stopped=True)
        lines = journal.PATH.read_text(encoding="utf-8").splitlines()
    who = "сеанс abcdef  без логина  локально  Windows, Chrome"
    err = _no_time(next(ln for ln in lines if "ОШИБКА" in ln))
    stop = _no_time(next(ln for ln in lines if "ОСТАНОВЛЕНО" in ln))
    assert err == f"  ОШИБКА     {who}  Интервью.mp4: Не хватило памяти видеокарты. Подождите минуту."
    assert stop == f"  ОСТАНОВЛЕНО {who}  Интервью.mp4: остановлено пользователем"


def test_result_falls_back_when_formatting_fails():
    """result() never breaks the page: if the lines cannot be built it writes a note pointing at result.json (T5)."""
    def boom(rep):
        raise ValueError("bad view")

    old = journal.result_lines
    journal.result_lines = boom
    try:
        with tempfile.TemporaryDirectory() as d, _at(Path(d) / "journal.txt"), _records() as recs:
            journal.result(_Req(), {"duration_sec": 20.0}, 65.0)      # must not raise
            text = journal.PATH.read_text(encoding="utf-8")
    finally:
        journal.result_lines = old
    assert "РЕЗУЛЬТАТ" in text
    assert "    (не удалось оформить итог, см. result.json в папке задачи)" in text
    assert len(recs) == 1 and "could not format the result" in recs[0].getMessage()


def test_write_never_raises_when_path_is_a_directory():
    """A misconfigured PATH does not break the page: the write is swallowed and logged once (T5)."""
    with tempfile.TemporaryDirectory() as d:
        as_dir = Path(d) / "journal_is_a_dir"
        as_dir.mkdir()
        with _at(as_dir), _records() as recs:
            journal.visit(_Req())                                     # must not raise
    assert len(recs) == 1 and "journal write failed" in recs[0].getMessage()


def test_who_reads_the_proxy_headers_and_degrades():
    """_who: the proxy's client address and login, a short session id; and it never raises — a bare request and an
    object without headers both give a safe head (T5)."""
    proxied = types.SimpleNamespace(
        headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1", "x-remote-user": "tester",
                 "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
                               "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"},
        session_hash="0123456789")
    assert journal._who(proxied) == "сеанс 012345  tester  203.0.113.7  iPhone, Safari"
    bare = types.SimpleNamespace(headers={}, session_hash="")
    assert journal._who(bare) == "сеанс ------  без логина  локально  браузер не указан"
    assert journal._who(object()) == "сеанс ?"                        # no .headers at all


def test_visit_writes_an_enter_line():
    with tempfile.TemporaryDirectory() as d, _at(Path(d) / "journal.txt"):
        journal.visit(_Req())
        text = journal.PATH.read_text(encoding="utf-8")
    assert "ВХОД" in text and "сеанс abcdef" in text
