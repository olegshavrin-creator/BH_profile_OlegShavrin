"""Accuracy of one model on FIV2 clips: MAE, ACC = 1 - MAE, mACC, CCC and Pearson per trait (was `bs3 eval-fiv2`).

  python -m training.eval_fiv2 --dir DIR --out eval.json [--backend mm|oceanai] [--lang en] [--limit N] [--asr]

Run from bs3-studio. DIR holds <stem>.mp4 with <stem>.txt (the transcript; --asr uses Whisper instead) and labels.csv
(video_name and the five FIV2 label columns); AMLAI 1.0 also reads <stem>.behavior.txt when present. FIV2 speech is
English, so pass --lang en (the default is Russian, bs3.LANG). Writes eval.json and eval.pred.csv next to it.
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

import pandas as pd

from bs3 import LANG, __version__
from bs3.cli import _add_models, _backend
from bs3.norms import OCEANAI_COLUMNS, TRAIT_KEYS

from .evaluate import evaluate

log = logging.getLogger("bs.eval_fiv2")


def predict_dir(backend, directory: str | Path, asr: bool = True,
                exts=(".mp4", ".mov", ".mkv", ".avi", ".webm")) -> pd.DataFrame:
    """AMLAI 1.0 (`backend` is a bs3.backend_mm.MMBackend): the scores of every clip in `directory` as
    DataFrame[Path, 5 traits]. <stem>.txt is the transcript unless `asr`; <stem>.behavior.txt, when present, replaces
    the Ollama description. A clip that fails is logged and left out. OCEAN-AI has its own
    OceanAIBackend.predict_dir, which its predict_video uses too."""
    backend.load()
    rows = []
    files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in exts)
    for i, p in enumerate(files, 1):
        txt = p.with_suffix(".txt")
        transcript = txt.read_text(encoding="utf-8") if (not asr and txt.exists()) else None
        beh_file = p.with_suffix(".behavior.txt")
        behavior = beh_file.read_text(encoding="utf-8") if beh_file.exists() else None
        try:
            r = backend.predict_video(p, asr=asr, transcript=transcript, behavior=behavior)
            rows.append({"Path": p.name, **{c: r["scores"][k] for k, c in zip(TRAIT_KEYS, OCEANAI_COLUMNS)}})
        except Exception as e:
            log.warning("%s failed: %s", p.name, e)
        if i % 20 == 0:
            log.info("scored %d/%d", i, len(files))
    return pd.DataFrame(rows)


def _score_dir(be, backend: str, directory: Path, asr: bool) -> pd.DataFrame:
    """The predictions of the chosen model (`--backend`) for every clip of `directory`."""
    if backend == "mm":
        return predict_dir(be, directory, asr=asr)
    return be.predict_dir(directory, asr=asr)


def cmd_eval(a):
    d = Path(a.dir)
    labels = pd.read_csv(a.labels or (d / "labels.csv"))
    be = _backend(a, lang=a.lang, corpus=a.corpus)
    t0 = time.time()
    if a.limit:
        # score a subset through a temporary folder of links
        tmp = Path(tempfile.mkdtemp(prefix="bs_eval_"))
        for n in labels["video_name"].tolist()[: a.limit]:
            for ext in (".mp4", ".txt"):
                src = d / (n + ext)
                if src.exists():
                    os.symlink(src.resolve(), tmp / (n + ext))
        pred = _score_dir(be, a.backend, tmp, a.asr)
    else:
        pred = _score_dir(be, a.backend, d, a.asr)
    secs = time.time() - t0
    res = evaluate(pred, labels)
    res["failed_files"] = list(pred.attrs.get("failed", []))
    res.update({
        "seconds_total": round(secs, 1), "seconds_per_clip": round(secs / max(1, len(pred)), 2),
        "backend": a.backend, "corpus": be.cfg.corpus, "lang": be.cfg.lang, "asr": a.asr,
        "dir": str(d), "bs_version": __version__,
    })
    out = Path(a.out)
    pred.to_csv(out.with_suffix(".pred.csv"), index=False)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_trait"}, indent=2))
    print(pd.DataFrame(res["per_trait"]).T.to_string())
    print(f"[ok] wrote {out} and {out.with_suffix('.pred.csv')}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m training.eval_fiv2", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_models(ap)                                       # the model options of `bs3 infer`
    ap.add_argument("--lang", default=LANG, choices=["ru", "en"],
                    help="language of speech (oceanai: ru -> MuPTA weights, en -> FIV2 weights)")
    ap.add_argument("--corpus", default=None, choices=["fi", "mupta"], help="oceanai: override the weight set")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--asr", action="store_true", help="use Whisper instead of the .txt transcripts")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.verbose:
        logging.getLogger("bs").setLevel(logging.DEBUG)   # keep numba/urllib3 quiet
    for noisy in ("numba", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return cmd_eval(a)


if __name__ == "__main__":
    sys.exit(main())
