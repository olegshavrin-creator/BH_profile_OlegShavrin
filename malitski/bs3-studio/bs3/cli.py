"""bs3 - Big Five (OCEAN) apparent personality scores from video (BS Profiler 3.1).

  bs3 web [--port 7880]                       the web page: one model per analysis (OCEAN-AI or AMLAI 1.0), Russian speech
  bs3 infer VIDEO [VIDEO ...] --out out.json  score videos (ASR on by default)
  bs3 explain VIDEO --out DIR                 AMLAI 1.0 only: scores, modality/frame/word attributions, key frames

Models: --backend mm (default, own model AMLAI 1.0; `bs3 explain` is the command that writes its explanations,
`bs3 infer` does not) | oceanai (OCEAN-AI, all weights public; MuPTA weights for Russian speech).
The speech is Russian (bs3.LANG); neither the page nor `web`, `infer` and `explain` have a language option.
The accuracy on FIV2 clips is research code outside the package: `python -m training.eval_fiv2` from bs3-studio.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from . import DEFAULT_MODEL, LANG, MODEL_TITLES, PRODUCT, __version__, settings
from .errors import UserFacingError, user_message
from .norms import TRAIT_KEYS
from .report import build_report


def _add_models(p, backend: bool = True, models_dir: bool = True):
    """The model options of a subcommand: which model (`backend`), AMLAI 1.0 checkpoints and its Ollama model, the
    OCEAN-AI weights cache (`models_dir`), Whisper, and -v."""
    if backend:
        # the default follows bs3.DEFAULT_MODEL, the model the page offers first: a run without --backend uses AMLAI 1.0
        p.add_argument("--backend", default=DEFAULT_MODEL, choices=list(MODEL_TITLES),
                       help="mm = own model AMLAI 1.0 (MM-PSYCHE recipe), the default; oceanai = OCEAN-AI, all public "
                            "weights")
    p.add_argument("--mm-ckpt", default=None,
                   help="mm: checkpoint path, comma-separated list or glob; several checkpoints are averaged "
                        f"(default {settings.shown(settings.MM_CHECKPOINTS)}, 5 seeds)")
    p.add_argument("--ollama-model", default=settings.OLLAMA_MODEL,
                   help="mm: Ollama vision model for behaviour descriptions (qwen3-vl:30b gives the same accuracy, 3x heavier)")
    if models_dir:
        p.add_argument("--models-dir", default=None,
                       help=f"oceanai: weights cache (default {settings.shown(settings.OCEANAI_MODELS_DIR)})")
    p.add_argument("--asr-model", default=settings.ASR_MODEL, help="HF Whisper id for transcription")
    p.add_argument("-v", "--verbose", action="store_true")


def _backend(a, lang: str = LANG, corpus: str | None = None):
    """The one model the command runs, loaded: AMLAI 1.0 (`--backend mm`) or OCEAN-AI (`--backend oceanai`). `lang`
    and `corpus` differ from the Russian defaults only for the FIV2 evaluation (training/eval_fiv2.py)."""
    if a.backend == "mm":
        from .backend_mm import MMBackend, MMConfig
        mm_kw = dict(lang=lang, asr_model=a.asr_model, ollama_model=a.ollama_model)
        if a.mm_ckpt:
            mm_kw["checkpoint"] = a.mm_ckpt
        return MMBackend(MMConfig(**mm_kw)).load()
    from .backend_oceanai import BackendConfig, OceanAIBackend
    oa_kw = dict(lang=lang, corpus=corpus, asr_model=a.asr_model)
    if a.models_dir:
        oa_kw["models_dir"] = a.models_dir
    return OceanAIBackend(BackendConfig(**oa_kw)).load()


def cmd_infer(a):
    from .media import check_upload
    work = Path(a.out).parent if a.out else Path.cwd()
    for v in a.video:
        if not Path(v).is_file():
            raise UserFacingError(f"Файл не найден: {v}")
        check_upload(v, work)                # unreadable / no sound / shorter than 2 s -> a calm refusal, no model built
    be = _backend(a)
    reports = []
    analyzer = None
    for v in a.video:
        transcript = Path(a.transcript).read_text(encoding="utf-8") if a.transcript else None
        if a.segment > 0 and transcript is None and not a.no_asr:
            # long videos: 20-s segments analysed in full, duration-weighted mean + timeline
            from .longvideo import LongVideoAnalyzer, video_duration
            if analyzer is None:
                analyzer = LongVideoAnalyzer(be, lang=LANG, seg_len=a.segment, asr_model=a.asr_model)
            work = Path(a.out).with_suffix("") if a.out else Path(tempfile.mkdtemp(prefix="bs_seg_"))
            res = analyzer.analyze(v, work / "segments") if video_duration(v) > analyzer.single_max else be.predict_video(v, asr=True)
        else:
            res = be.predict_video(v, asr=not a.no_asr, transcript=transcript)
        mods = {"oceanai": ("audio", "video", "text"),
                "mm": tuple(getattr(be, "modalities", ("face", "audio", "text", "behavior")))}[a.backend]
        primary = res.get("primary")
        rep = build_report(v, res, backend=a.backend, corpus=be.cfg.corpus, lang=be.cfg.lang,
                           asr_model=None if (a.no_asr or transcript is not None) else a.asr_model,
                           modalities=mods, primary=primary)
        rep["timings_sec"]["model_load"] = round(be.load_seconds, 1)
        if "timings" in res:
            rep["timings_sec"].update(res["timings"])
        for key in ("duration_sec", "segments", "timeline", "scores_std", "representative_segment", "transcript_en"):
            if res.get(key) is not None:
                rep[key] = res[key]
        reports.append(rep)
        line = "  ".join(f"{k[:5]}={rep['traits'][k]['score']:.3f}" for k in TRAIT_KEYS)
        if "interview" in rep:
            line += f"  interview={rep['interview']['score']:.3f}"
        print(f"{Path(v).name}: {line}  ({res['seconds']}s)")
    payload = reports[0] if len(reports) == 1 else reports
    if a.out:
        Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {a.out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_explain(a):
    from .media import check_upload
    if not Path(a.video).is_file():
        raise UserFacingError(f"Файл не найден: {a.video}")
    check_upload(a.video, Path(a.out))       # same early refusal as infer / the web page (2-second floor included)
    a.backend = "mm"
    be = _backend(a)
    transcript = Path(a.transcript).read_text(encoding="utf-8") if a.transcript else None
    behavior = Path(a.behavior).read_text(encoding="utf-8") if a.behavior else None
    res = be.explain_video(a.video, a.out, asr=not a.no_asr, transcript=transcript, behavior=behavior, top_k=a.top_k)
    print("scores:", res["scores"])
    for k, v in res["modalities"]["input_x_gradient"].items():
        print(f"  {k:20s} " + "  ".join(f"{m}={d['share']:.2f}" for m, d in v.items()))
    if "frames" in res:
        print("key frames:", res["frames"]["top_frames_overall"], "->", len(res["frames"]["key_frame_files"]), "jpeg files")
    for key in ("transcript_words", "behavior_words"):
        if key in res:
            first = next(iter(res[key]["per_output"].values()))
            print(f"{key} (openness top words):", [w["word"] for w in first["top_words"]])
    print(f"[ok] wrote {Path(a.out) / 'explanation.json'} ({res['seconds']}s)")


def cmd_web(a):
    # the page chooses the model per analysis (OCEAN-AI or AMLAI 1.0), so `web` has no --backend
    from .web.app import main as web_main
    web_main(port=a.port, work_dir=a.work_dir, share=a.share, asr_model=a.asr_model, ollama_model=a.ollama_model,
             mm_ckpt=a.mm_ckpt, host=a.host, models_dir=a.models_dir)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bs3", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{PRODUCT} ({__version__})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer", help="score one or more videos")
    _add_models(p)
    p.add_argument("video", nargs="+")
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--no-asr", action="store_true", help="skip speech recognition (text branch then needs <stem>.txt or --transcript)")
    p.add_argument("--transcript", default=None, help="text file with the transcript (disables ASR)")
    p.add_argument("--segment", type=float, default=settings.SEGMENT_SEC,
                   help=f"videos longer than {settings.SINGLE_CLIP_MAX_SEC:g} s are analysed in segments of this many "
                        "seconds (0 = never segment)")
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("explain", help="own model only: scores + modality/frame/word attributions, key frames")
    _add_models(p, backend=False, models_dir=False)       # always AMLAI 1.0: no model choice, no OCEAN-AI weights
    p.add_argument("video")
    p.add_argument("--out", required=True, help="output folder (explanation.json + key frame JPEGs)")
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--transcript", default=None)
    p.add_argument("--behavior", default=None, help="text file with a behaviour description (skips Ollama)")
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("web", help="Gradio web UI: upload a video, choose the model, get the characterization")
    p.add_argument("--port", type=int, default=settings.PORT)
    p.add_argument("--host", default=settings.HOST, help="bind address (0.0.0.0 = reachable from Windows via localhost)")
    p.add_argument("--work-dir", default=None,
                   help=f"where uploads and results are stored (default {settings.shown(settings.JOBS_DIR)})")
    p.add_argument("--share", action="store_true", help="also create a public gradio.live link")
    _add_models(p, backend=False)                         # the page chooses the model per analysis
    p.set_defaults(fn=cmd_web)
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.verbose:
        logging.getLogger("bs").setLevel(logging.DEBUG)   # keep numba/urllib3 quiet
    for noisy in ("numba", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        print("Прервано.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001  — one calm Russian line; the full traceback only with -v
        if a.verbose:
            raise
        first = (str(e).splitlines() or [""])[0][:120]
        print(f"Ошибка: {user_message(e)} ({first}; подробности — с ключом -v)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
