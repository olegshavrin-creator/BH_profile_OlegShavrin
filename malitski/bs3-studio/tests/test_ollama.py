"""bs3/ollama.py, the one client of the local Ollama server: BS3_OLLAMA_URL replaces the search, the search runs once
per process, a failed call makes the server «unavailable» for the probe interval without a probe, and the model given
to the Studio (`bs3 web --ollama-model`) reaches the translation requests. No request leaves the process: the module's
urllib and the gateway lookup are replaced while a test runs, and everything is put back afterwards."""
from __future__ import annotations

import contextlib
import inspect
import json
import subprocess
import types
import urllib.error
import urllib.request

import bs3.ollama as ollama
import bs3.translate as translate
from bs3 import pipeline, settings

GATEWAY = "172.20.0.1"
LOCAL = f"http://localhost:{settings.OLLAMA_PORT}"
VIA_GATEWAY = f"http://{GATEWAY}:{settings.OLLAMA_PORT}"


class _Answer:
    status = 200

    def __init__(self, body: bytes = b"{}"):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@contextlib.contextmanager
def _isolated(answer=None, **setting_values):
    """bs3.ollama with a fresh state. `answer(url, timeout)` stands in for urlopen (default: a request fails the
    test); the gateway lookup says GATEWAY; `setting_values` replace settings for the test. Yields the log
    {"urlopen": [(url, timeout, request)], "route": number of gateway lookups}."""
    log = {"urlopen": [], "route": 0}

    def urlopen(req, timeout=None):
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        log["urlopen"].append((url, timeout, req))
        if answer is None:
            raise AssertionError(f"a network request in a test: {url}")
        return answer(url, timeout)

    def run(*_a, **_k):
        log["route"] += 1
        return subprocess.CompletedProcess(args=_a, returncode=0, stdout=f"{GATEWAY}\n", stderr="")

    saved = (dict(ollama._state), ollama._found, ollama._model, ollama.urllib, ollama.subprocess)
    saved_settings = {k: getattr(settings, k) for k in setting_values}
    ollama.urllib = types.SimpleNamespace(request=types.SimpleNamespace(urlopen=urlopen,
                                                                        Request=urllib.request.Request))
    ollama.subprocess = types.SimpleNamespace(run=run)
    ollama._state.update(t=0.0, ok=False)
    ollama._found = None
    for k, v in setting_values.items():
        setattr(settings, k, v)
    try:
        yield log
    finally:
        state, ollama._found, ollama._model, ollama.urllib, ollama.subprocess = saved
        ollama._state.clear()
        ollama._state.update(state)
        for k, v in saved_settings.items():
            setattr(settings, k, v)


def test_env_url_replaces_the_search():
    with _isolated(OLLAMA_URL="http://10.0.0.5:11500") as log:
        assert ollama.url() == "http://10.0.0.5:11500"
        assert log == {"urlopen": [], "route": 0}                  # no probe, no gateway lookup
    got = []
    with _isolated(lambda url, timeout: got.append(url) or _Answer(b'{"ok": true}'),
                   OLLAMA_URL="http://10.0.0.5:11500") as log:
        assert ollama.post_json("/api/generate", {"a": 1}, 7) == {"ok": True}
        assert got == ["http://10.0.0.5:11500/api/generate"] and log["route"] == 0


def test_url_is_found_once_per_process():
    def gateway_only(url, timeout):
        if url.startswith(LOCAL):
            raise urllib.error.URLError("connection refused")
        return _Answer()

    with _isolated(gateway_only, OLLAMA_URL="") as log:
        assert ollama.url() == VIA_GATEWAY
        assert [u for u, _t, _r in log["urlopen"]] == [f"{LOCAL}/api/tags", f"{VIA_GATEWAY}/api/tags"]
        assert all(t == settings.OLLAMA_PROBE_TIMEOUT for _u, t, _r in log["urlopen"]) and log["route"] == 1
        for _ in range(3):
            assert ollama.url() == VIA_GATEWAY
        assert len(log["urlopen"]) == 2 and log["route"] == 1         # found once, then remembered
    with _isolated(lambda url, timeout: _Answer(), OLLAMA_URL="") as log:
        assert ollama.url() == LOCAL and len(log["urlopen"]) == 1    # localhost first


def test_url_not_remembered_while_nothing_answers():
    def down(url, timeout):
        raise urllib.error.URLError("connection refused")

    with _isolated(down, OLLAMA_URL="") as log:
        assert ollama.url() == VIA_GATEWAY                           # the last candidate, as before
        assert ollama._found is None and len(log["urlopen"]) == 2
        ollama.url()
        assert len(log["urlopen"]) == 4 and log["route"] == 2         # looked again: Ollama may start later


def test_mark_down_says_no_for_the_ttl_without_a_probe():
    with _isolated(lambda url, timeout: _Answer(), OLLAMA_URL="http://h:1") as log:
        assert ollama.available() is True and len(log["urlopen"]) == 1
        assert ollama.available() is True and len(log["urlopen"]) == 1   # within the TTL: no second probe
        ollama._state["t"] -= settings.OLLAMA_PROBE_TTL + 1                # the last probe is an old one
        ollama.mark_down()
        for _ in range(3):
            assert ollama.available() is False
        assert len(log["urlopen"]) == 1                                  # no probe after the failed call
        ollama._state["t"] -= settings.OLLAMA_PROBE_TTL + 1                # the TTL has passed: probed again
        assert ollama.available() is True and len(log["urlopen"]) == 2


def test_failed_translation_request_marks_ollama_down():
    """FP2(d): after a failed request the next texts of the run go to Marian at once, instead of probing again and
    waiting up to 3×300 s per text."""
    def refuse(url, timeout):
        if url.endswith("/api/generate"):
            raise urllib.error.URLError("connection reset")
        return _Answer()

    with _isolated(refuse, OLLAMA_URL="http://h:1") as log:
        assert ollama.available() is True
        ollama._state["t"] -= settings.OLLAMA_PROBE_TTL + 1        # the probe was long ago: the server looks fine
        try:
            translate._ollama_json("prompt", 10)
        except urllib.error.URLError:
            pass
        else:
            raise AssertionError("the failed request was swallowed")
        n = len(log["urlopen"])
        assert ollama.available() is False and len(log["urlopen"]) == n


def test_post_json_sends_json_and_decodes_the_answer():
    with _isolated(lambda url, timeout: _Answer(json.dumps({"response": "да"}).encode("utf-8"))) as log:
        assert ollama.post_json("/api/generate", {"model": "m", "prompt": "п"}, 12, base="http://h:2") == \
            {"response": "да"}
        (url, timeout, req), = log["urlopen"]
        assert url == "http://h:2/api/generate" and timeout == 12 and req.get_method() == "POST"
        assert req.data == json.dumps({"model": "m", "prompt": "п"}).encode("utf-8")
        assert req.get_header("Content-type") == "application/json"


def test_studio_model_reaches_the_translation_requests():
    """S2: `bs3 web --ollama-model x` -> Studio(ollama_model="x") -> the model of the translation requests; the payload
    is otherwise the one of 3.1 before this module (same keys in the same order)."""
    sent = []

    def post_json(path, payload, timeout, base=None):
        sent.append((path, payload, timeout, base))
        return {"response": json.dumps({"calm": {"tr": "спокойный", "name": False}})}

    with _isolated():
        real = ollama.post_json
        ollama.post_json = post_json
        try:
            pipeline.Studio(ollama_model="x")
            assert ollama.model() == "x"
            translate._ollama_json("prompt", 10)
            assert translate._ollama_dictionary(["calm"], "ru") == {"calm": "спокойный"}
            pipeline.Studio()
            assert ollama.model() == settings.OLLAMA_MODEL
            translate._ollama_json("prompt", 10, model="given")
        finally:
            ollama.post_json = real
    assert [p["model"] for _path, p, _t, _b in sent] == ["x", "x", "given"]
    path, payload, timeout, base = sent[0]
    assert path == "/api/generate" and base is None and timeout == settings.OLLAMA_TRANSLATE_TIMEOUT
    assert list(payload) == ["model", "prompt", "stream", "format", "think", "keep_alive", "options"]
    assert payload["keep_alive"] == settings.OLLAMA_KEEP_ALIVE and payload["stream"] is False
    assert payload["options"] == {"temperature": 0.0, "seed": 0, "num_predict": 10}
    assert sent[1][2] == settings.OLLAMA_DICTIONARY_TIMEOUT


def test_translate_keeps_its_names():
    """ru_texts, tests/run.py, compare_baseline and scripts/check_behavior_translation replace
    translate.ollama_available; translate calls it through that name, and the old private copies are gone."""
    assert "ollama_available" in vars(translate)
    for gone in ("_OLLAMA_STATE", "OLLAMA_MODEL", "LLM_PARALLEL"):
        assert gone not in vars(translate), gone
    src = inspect.getsource(translate)
    assert "default_ollama_url" not in src and "urlopen" not in src
    assert src.count("ollama_available()") >= 3
