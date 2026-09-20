"""bs - Big Five (OCEAN) apparent personality scores from video.

  bs setup-weights [--lang en|ru|all]        download/cache OCEAN-AI weights
  bs infer VIDEO [VIDEO ...] --out out.json  score videos (ASR on by default)
  bs infer-dir DIR --out results.csv         score every media file in a folder
  bs eval-fiv2 --dir DIR --out eval.json     mACC/CCC on FIV2 clips (DIR has <stem>.mp4, <stem>.txt, labels.csv)

Backends: --backend oceanai (default, all weights public) | sslmepr (benchmark, scene+audio+text only).
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .norms import TRAIT_KEYS
from .report import build_report


def _add_common(p, lang_choices=("en", "ru")):
    p.add_argument("--backend", default="oceanai", choices=["oceanai", "sslmepr", "ensemble", "mm"],
                   help="oceanai = all public weights; sslmepr = benchmark on scene+audio+text; "
                        "ensemble = mean of oceanai and the sslmepr scene branch; mm = own MM-PSYCHE-style model")
    p.add_argument("--mm-ckpt", default=None, help="mm: checkpoint (default ~/bs/mm_runs_full/all4_interview/best.pt)")
    p.add_argument("--ensemble-members", default="oceanai,mm",
                   help="ensemble: comma-separated subset of oceanai,mm,scene (default oceanai,mm = best on FIV2 test)")
    p.add_argument("--primary", default="auto",
                   help="ensemble: member giving the main score (auto = oceanai for --lang ru, mean otherwise; "
                        "mean | oceanai | mm | scene). With a primary member, percentiles are taken against the pool "
                        "of processed videos of that language (~/bs/pool), not FIV2")
    p.add_argument("--ollama-model", default="qwen2.5vl:7b",
                   help="mm: Ollama vision model for behaviour descriptions (qwen3-vl:30b gives the same accuracy, 3x heavier)")
    p.add_argument("--lang", default="en", choices=list(lang_choices),
                   help="language of speech (oceanai: en -> FIV2 weights, ru -> MuPTA weights)")
    p.add_argument("--corpus", default=None, choices=["fi", "mupta"], help="oceanai: override the weight set")
    p.add_argument("--models-dir", default=None, help="oceanai: weights cache (default ~/bs/models)")
    p.add_argument("--asr-model", default="openai/whisper-large-v3-turbo", help="HF Whisper id for transcription")
    p.add_argument("--sslmepr-ckpt", default=None, help="sslmepr: checkpoint root (default ~/bs/ssl_mepr_ckpt)")
    p.add_argument("--sslmepr-modalities", default="scene,audio,text",
                   help="sslmepr: which branches to run, comma-separated subset of scene,audio,text")
    p.add_argument("-v", "--verbose", action="store_true")


def _backend(a):
    from .backend_oceanai import BackendConfig, OceanAIBackend
    from .backend_sslmepr import SSLMEPRBackend, SSLMEPRConfig

    sm_kw = dict(lang=a.lang, asr_model=a.asr_model)
    if a.sslmepr_ckpt:
        sm_kw["ckpt_root"] = Path(a.sslmepr_ckpt)
    mods = tuple(m.strip() for m in a.sslmepr_modalities.split(",") if m.strip())
    if "scene" not in mods:
        raise SystemExit("--sslmepr-modalities must include scene")
    oa_kw = dict(lang=a.lang, corpus=a.corpus, asr_model=a.asr_model)
    if a.models_dir:
        oa_kw["models_dir"] = a.models_dir

    if a.backend == "sslmepr":
        return SSLMEPRBackend(SSLMEPRConfig(modalities=mods, **sm_kw)).load()
    from .backend_mm import MMBackend, MMConfig
    mm_kw = dict(lang=a.lang, asr_model=a.asr_model, ollama_model=a.ollama_model)
    if a.mm_ckpt:
        mm_kw["checkpoint"] = a.mm_ckpt
    if a.backend == "mm":
        return MMBackend(MMConfig(**mm_kw)).load()
    if a.backend == "ensemble":
        from .backend_ensemble import EnsembleBackend, EnsembleConfig
        members = tuple(m.strip() for m in a.ensemble_members.split(",") if m.strip())
        return EnsembleBackend(EnsembleConfig(members=members, lang=a.lang, primary=getattr(a, "primary", "auto"),
                                              oceanai_cfg=BackendConfig(**oa_kw), mm_cfg=MMConfig(**mm_kw),
                                              sslmepr_cfg=SSLMEPRConfig(modalities=("scene",), **sm_kw))).load()
    return OceanAIBackend(BackendConfig(**oa_kw)).load()


def cmd_setup(a):
    langs = ["en", "ru"] if a.lang == "all" else [a.lang]
    for lang in langs:
        a.lang = lang
        be = _backend(a)
        print(f"[ok] weights for lang={lang} corpus={be.cfg.corpus} ready ({be.load_seconds:.1f}s)")


def cmd_infer(a):
    be = _backend(a)
    reports = []
    analyzer = None
    for v in a.video:
        transcript = Path(a.transcript).read_text(encoding="utf-8") if a.transcript else None
        if a.segment > 0 and transcript is None and not a.no_asr:
            # long videos: 20-s segments analysed in full, duration-weighted mean + timeline
            from .longvideo import LongVideoAnalyzer, video_duration
            if analyzer is None:
                analyzer = LongVideoAnalyzer(be, lang=a.lang, seg_len=a.segment, asr_model=a.asr_model)
            work = Path(a.out).with_suffix("") if a.out else Path(tempfile.mkdtemp(prefix="bs_seg_"))
            res = analyzer.analyze(v, work / "segments") if video_duration(v) > analyzer.single_max else be.predict_video(v, asr=True)
        else:
            res = be.predict_video(v, asr=not a.no_asr, transcript=transcript)
        mods = {"oceanai": ("audio", "video", "text"), "sslmepr": ("scene", "audio", "text"),
                "ensemble": tuple(getattr(be.cfg, "members", ())),
                "mm": tuple(getattr(be, "modalities", ("face", "audio", "text", "behavior")))}[a.backend]
        primary = res.get("primary")
        if primary:
            from .pool import add as pool_add
            pool_add(v, res["scores"], a.lang, primary)
        rep = build_report(v, res, backend=a.backend, corpus=be.cfg.corpus, lang=be.cfg.lang,
                           asr_model=None if (a.no_asr or transcript is not None) else a.asr_model,
                           modalities=mods, pool_lang=a.lang if primary else None, primary=primary)
        rep["timings_sec"]["model_load"] = round(be.load_seconds, 1)
        if "timings" in res:
            rep["timings_sec"].update(res["timings"])
        for key in ("duration_sec", "segments", "timeline", "scores_std", "representative_segment", "transcript_en"):
            if res.get(key) is not None:
                rep[key] = res[key]
        if "variants" in res:
            rep["variant_scores"] = res["variants"]     # sslmepr: MCDM fusion + per-modality predictions
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
    from .webapp import main as web_main
    web_main(port=a.port, members=a.ensemble_members, work_dir=a.work_dir, share=a.share,
             asr_model=a.asr_model, ollama_model=a.ollama_model, mm_ckpt=a.mm_ckpt, host=a.host)


def cmd_infer_dir(a):
    be = _backend(a)
    t0 = time.time()
    df = be.predict_dir(a.dir, asr=not a.no_asr)
    df.to_csv(a.out, index=False)
    print(f"[ok] {len(df)} files scored in {time.time() - t0:.1f}s -> {a.out}")
    print(df.head(10).to_string(index=False))


def cmd_eval(a):
    import pandas as pd
    from .evaluate import evaluate

    d = Path(a.dir)
    labels = pd.read_csv(a.labels or (d / "labels.csv"))
    be = _backend(a)
    t0 = time.time()
    if a.limit:
        # score a subset through a temporary folder of links
        tmp = Path(tempfile.mkdtemp(prefix="bs_eval_"))
        for n in labels["video_name"].tolist()[: a.limit]:
            for ext in (".mp4", ".txt"):
                src = d / (n + ext)
                if src.exists():
                    os.symlink(src.resolve(), tmp / (n + ext))
        pred = be.predict_dir(tmp, asr=a.asr)
    else:
        pred = be.predict_dir(d, asr=a.asr)
    secs = time.time() - t0
    res = evaluate(pred, labels)
    res["failed_files"] = list(pred.attrs.get("failed", []))
    # variant score sets (e.g. sslmepr: fusion / scene / audio / text) stored as "<variant>:<Trait>" columns
    from .norms import OCEANAI_COLUMNS
    prefixes = sorted({c.split(":", 1)[0] for c in pred.columns if ":" in c})
    if prefixes:
        res["variants"] = {}
        for v in prefixes:
            sub = pred[["Path"] + [f"{v}:{c}" for c in OCEANAI_COLUMNS]].rename(columns={f"{v}:{c}": c for c in OCEANAI_COLUMNS})
            r = evaluate(sub, labels)
            res["variants"][v] = {"mACC": r["mACC"], "mCCC": r["mCCC"],
                                  "acc": {k: t["acc"] for k, t in r["per_trait"].items()},
                                  "ccc": {k: t["ccc"] for k, t in r["per_trait"].items()}}
    res.update({
        "seconds_total": round(secs, 1), "seconds_per_clip": round(secs / max(1, len(pred)), 2),
        "backend": a.backend, "corpus": be.cfg.corpus, "lang": be.cfg.lang, "asr": a.asr,
        "dir": str(d), "bs_version": __version__,
    })
    out = Path(a.out)
    pred.to_csv(out.with_suffix(".pred.csv"), index=False)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_trait", "variants")}, indent=2))
    print(pd.DataFrame(res["per_trait"]).T.to_string())
    if "variants" in res:
        print("variants:", json.dumps({v: {"mACC": d["mACC"], "mCCC": d["mCCC"]} for v, d in res["variants"].items()}, indent=1))
    print(f"[ok] wrote {out} and {out.with_suffix('.pred.csv')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bs2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"bs-bigfive {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup-weights", help="download and cache weights")
    _add_common(p, ("en", "ru", "all"))
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("infer", help="score one or more videos")
    _add_common(p)
    p.add_argument("video", nargs="+")
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--no-asr", action="store_true", help="skip speech recognition (text branch then needs <stem>.txt or --transcript)")
    p.add_argument("--transcript", default=None, help="text file with the transcript (disables ASR)")
    p.add_argument("--segment", type=float, default=20.0,
                   help="videos longer than 30 s are analysed in segments of this many seconds (0 = never segment)")
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("explain", help="own model only: scores + modality/frame/word attributions, key frames")
    _add_common(p)
    p.add_argument("video")
    p.add_argument("--out", required=True, help="output folder (explanation.json + key frame JPEGs)")
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--transcript", default=None)
    p.add_argument("--behavior", default=None, help="text file with a behaviour description (skips Ollama)")
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("web", help="Gradio web UI: upload a video, get scores + explanations")
    _add_common(p)
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 = reachable from Windows via localhost)")
    p.add_argument("--work-dir", default=None, help="where uploads and results are stored (default ~/bs2_data/web_jobs)")
    p.add_argument("--share", action="store_true", help="also create a public gradio.live link")
    p.set_defaults(fn=cmd_web)

    p = sub.add_parser("infer-dir", help="score every media file in a folder")
    _add_common(p)
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--no-asr", action="store_true")
    p.set_defaults(fn=cmd_infer_dir)

    p = sub.add_parser("eval-fiv2", help="accuracy on FIV2 clips")
    _add_common(p)
    p.add_argument("--dir", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--asr", action="store_true", help="use Whisper instead of the .txt transcripts")
    p.set_defaults(fn=cmd_eval)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.verbose:
        logging.getLogger("bs").setLevel(logging.DEBUG)   # keep numba/urllib3 quiet
    for noisy in ("numba", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
