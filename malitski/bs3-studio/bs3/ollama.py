"""The one client of the local Ollama server (standard library only): where it listens (`url`), whether it answers
(`available`, checked at most once per settings.OLLAMA_PROBE_TTL), one HTTP call (`post_json`) and the model of this
process (`configure` / `model`: the --ollama-model of `bs3 web`, which the translations use too). The callers keep
their payloads and retry policies: backend_mm (behaviour descriptions, frame phrases) and translate."""
from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request

from . import settings

_lock = threading.Lock()
_found: str | None = None                  # the address found by url(), once per process
_state = {"t": 0.0, "ok": False}           # the last availability check: when, and whether the server answered
_model: str | None = None                  # configure(); None -> settings.OLLAMA_MODEL


def _answers(base: str) -> bool:
    """GET <base>/api/tags answers 200 within settings.OLLAMA_PROBE_TIMEOUT seconds."""
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=settings.OLLAMA_PROBE_TIMEOUT) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _candidates() -> list[str]:
    """Windows Ollama listens on localhost; from WSL (NAT mode) it is reachable through the default gateway."""
    cands = [f"http://localhost:{settings.OLLAMA_PORT}"]
    try:
        gw = subprocess.run(["sh", "-c", "ip route show default | awk '{print $3}'"], capture_output=True, text=True).stdout.strip()
        if gw:
            cands.append(f"http://{gw}:{settings.OLLAMA_PORT}")
    except Exception:  # noqa: BLE001
        pass
    return cands


def url() -> str:
    """settings.OLLAMA_URL (BS3_OLLAMA_URL) when it is set, without a probe. Otherwise the first of localhost and the
    WSL default gateway whose /api/tags answers, found once per process. When neither answers, the last candidate is
    returned and not remembered: the next call looks again (Ollama may be started later)."""
    global _found
    if settings.OLLAMA_URL:
        return settings.OLLAMA_URL
    with _lock:
        if _found is None:
            cands = _candidates()
            hit = next((c for c in cands if _answers(c)), None)
            if hit is None:
                return cands[-1]
            _found = hit
        return _found


def available(ttl: float = settings.OLLAMA_PROBE_TTL) -> bool:
    """The Ollama server answers. Checked at most once per `ttl` seconds, so a page render does not wait for a dead
    server again and again; a failed call (mark_down) counts as a check that said no."""
    now = time.time()
    if now - _state["t"] < ttl:
        return _state["ok"]
    try:
        ok = _answers(url())
    except Exception:  # noqa: BLE001
        ok = False
    _state.update(t=now, ok=ok)
    return ok


def mark_down() -> None:
    """A call failed: available() says no for the next settings.OLLAMA_PROBE_TTL seconds without probing, so the next
    texts of a run fall back at once instead of waiting for the dead server again."""
    _state.update(t=time.time(), ok=False)


def post_json(path: str, payload: dict, timeout: float, base: str | None = None) -> dict:
    """POST `payload` as JSON to `path` of the server (`base`, default url()) and return the decoded answer. Errors come
    out as they are (urllib.error.URLError, TimeoutError, OSError, a JSON error): retrying is the caller's business."""
    req = urllib.request.Request(f"{base or url()}{path}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def status(model_name: str | None = None) -> str:
    """A fresh /api/tags probe for the analysis preflight (bs3.pipeline, AMLAI 1.0 needs Ollama): "ok" when the server
    answers and lists `model_name` (or its :latest tag), "no_model" when it answers with valid JSON that lists neither,
    and "down" when it does not answer. `model_name` defaults to model(). Unlike available(), this does not use the
    TTL cache: the person is about to wait minutes for an analysis, so the check is worth one fresh request."""
    want = model_name or model()
    try:
        with urllib.request.urlopen(f"{url()}/api/tags", timeout=settings.OLLAMA_PROBE_TIMEOUT) as r:
            if r.status != 200:
                return "down"
            data = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return "down"
    names = {m.get("name") or m.get("model") for m in (data.get("models") or []) if isinstance(m, dict)}
    return "ok" if (want in names or f"{want}:latest" in names) else "no_model"


def configure(model: str | None = None) -> None:
    """The Ollama model of this process (pipeline.Studio: the --ollama-model of `bs3 web`); None -> the default."""
    global _model
    _model = model or None


def model() -> str:
    """The Ollama model of this process: the configured one, else settings.OLLAMA_MODEL."""
    return _model or settings.OLLAMA_MODEL
