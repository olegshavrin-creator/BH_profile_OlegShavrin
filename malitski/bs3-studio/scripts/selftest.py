"""End-to-end smoke test of BS Profiler 3.1 without Gradio: analyse one video with one model on the GPU, build the PDF,
print a summary.

Usage: selftest.py VIDEO [mm|oceanai] [--work-dir DIR]

The model defaults to bs3.DEFAULT_MODEL (AMLAI 1.0); explanations are asked for from AMLAI 1.0 only, as on the page.
The job folder goes into a new temporary folder (bs3_selftest_*) unless --work-dir is given, so a smoke run does not
add a job to the folder of the page (~/bs3_data/web_jobs).
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # bs3-studio/: `import bs3` is this working tree
from bs3 import DEFAULT_MODEL, MODEL_TITLES  # noqa: E402
from bs3.pipeline import Studio, run_analysis  # noqa: E402
from bs3.pdf import export_pdf  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("member", nargs="?", default=DEFAULT_MODEL, choices=list(MODEL_TITLES))
    ap.add_argument("--work-dir", default=None, help="where the job folder is created (default: a new temp folder)")
    a = ap.parse_args(argv)
    work_dir = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="bs3_selftest_"))
    m = a.member
    t0 = time.time()
    studio = Studio()
    rep = run_analysis(studio, work_dir, a.video, member=m, explain=(m == "mm"),
                       progress=lambda f, d: print(f"  {f * 100:5.1f}% {d}", flush=True))
    print(f"analysis done in {time.time() - t0:.0f} s -> {rep['job_dir']}")
    an = rep.get("analyses", {})
    print("model.selected:", (rep.get("model") or {}).get("selected"))
    print("traits:", {k: round(v["score"], 2) for k, v in rep["traits"].items()})
    print("mbti.type:", (rep.get("mbti") or {}).get("type"))
    print("emotions_text:", an.get("emotions_text", {}).get("mean"))
    print("face:", {k: v for k, v in (an.get("face") or {}).items() if k != "mean"}, (an.get("face") or {}).get("mean"))
    print("voice:", an.get("voice", {}).get("mean"))
    print("speech:", {k: v for k, v in (an.get("speech") or {}).items() if k not in ("vocabulary", "description")})
    t1 = time.time()
    pdf = export_pdf(rep["job_dir"])
    print(f"pdf: {pdf} ({Path(pdf).stat().st_size // 1024} KB, {time.time() - t1:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
