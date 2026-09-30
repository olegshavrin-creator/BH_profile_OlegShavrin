"""Copy a finished job of BS 2.0 into the work dir of BS Profiler 3.1, so 3.1 can show it without re-analysis
(design 13.3, item 2; task T16).

    ~/bs/venv/bin/python bs3-studio/scripts/import_job.py ~/bs2_data/web_jobs/<job id> [...] [--force]

Why a copy: showing a job writes into its folder (Russian texts are added to result.json and explanation.json, the PDF
export writes charts/ and the PDF), and 3.1 must never write into the 2.0 work dir.

What is copied into ~/bs3_data/web_jobs/<job id>/: result.json and explain/ (explanation.json and the key frames).
Not copied: input.* (the video), segments/, seg*.mp4 (hundreds of MB; the PDF takes `media` from result.json), the
2.0 PDF and charts/. In the copy `job_dir`, `key_frames` (result.json) and `frames.key_frame_files`
(explanation.json) point into the copy; `input` and `timeline[].file` keep naming the 2.0 files (only their names are
read, never the files).

The source is only read: the script hashes every file of the source job (except segments/ and seg*.mp4) before and
after the copy and fails if anything changed. It refuses to write anywhere outside ~/bs3_data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # bs3-studio/: `import bs3` is this working tree
from bs3 import settings  # noqa: E402

BS3_ROOT = settings.DATA_DIR
DEFAULT_DEST = settings.JOBS_DIR
SOURCE_ROOT = Path.home() / "bs2_data" / "web_jobs"          # where 2.0 keeps its jobs (read only)


class ImportRefused(RuntimeError):
    pass


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def tree_sha256(job: Path) -> dict[str, str]:
    """{relative path: sha256} of every file of a job folder except the segment videos (segments/, seg*.mp4)."""
    out = {}
    for p in sorted(Path(job).rglob("*")):
        rel = p.relative_to(job)
        if not p.is_file() or rel.parts[0] == "segments" or (p.name.startswith("seg") and p.suffix == ".mp4"):
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[rel.as_posix()] = h.hexdigest()
    return out


def _moved(value, src: Path, dest: Path):
    """A path string under the source job moved to the same place under the copy; anything else unchanged."""
    if not isinstance(value, str):
        return value
    s = str(src)
    if value == s or value.startswith(s + "/"):
        return str(dest) + value[len(s):]
    return value


def import_job(src, dest_root=DEFAULT_DEST, *, root: Path = BS3_ROOT, force: bool = False) -> Path:
    """Copy one job (see the module docstring); returns the folder of the copy."""
    src = Path(src).expanduser().resolve()
    dest_root = Path(dest_root).expanduser()
    if not _inside(dest_root, root):
        raise ImportRefused(f"refused: {dest_root} is outside {root}")
    if _inside(src, root):
        raise ImportRefused(f"refused: {src} is already inside {root}")
    res_path = src / "result.json"
    if not res_path.is_file():
        raise ImportRefused(f"not a finished job (no result.json): {src}")
    dest_root = dest_root.resolve()
    dest = dest_root / src.name
    if dest.exists() and not force:
        raise ImportRefused(f"{dest} exists (use --force to replace it)")

    before = tree_sha256(src)
    rep = json.loads(res_path.read_text(encoding="utf-8"))
    rep["job_dir"] = str(dest)
    if isinstance(rep.get("key_frames"), list):
        rep["key_frames"] = [_moved(p, src, dest) for p in rep["key_frames"]]

    dest_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".import_{src.name}_", dir=dest_root))
    tmp.chmod(0o755)                                   # like a job folder made by the web app, not mkdtemp's 0700
    try:
        (tmp / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        ex_src = src / "explain"
        if ex_src.is_dir():
            (tmp / "explain").mkdir()
            for p in sorted(ex_src.iterdir()):
                if not p.is_file():
                    continue
                if p.name == "explanation.json":
                    expl = json.loads(p.read_text(encoding="utf-8"))
                    fr = expl.get("frames")
                    if isinstance(fr, dict) and isinstance(fr.get("key_frame_files"), list):
                        fr["key_frame_files"] = [_moved(x, src, dest) for x in fr["key_frame_files"]]
                    (tmp / "explain" / p.name).write_text(json.dumps(expl, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
                else:
                    shutil.copy2(p, tmp / "explain" / p.name)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    after = tree_sha256(src)
    if after != before:
        raise RuntimeError(f"the source job changed while it was copied: {src}")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", nargs="+", help=f"job folders of 2.0 (usually under {SOURCE_ROOT})")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="where the copies go (must be inside ~/bs3_data)")
    ap.add_argument("--force", action="store_true", help="replace an existing copy")
    args = ap.parse_args(argv)
    rc = 0
    for job in args.jobs:
        try:
            before = tree_sha256(Path(job).expanduser())
            dest = import_job(job, args.dest, force=args.force)
            same = tree_sha256(Path(job).expanduser()) == before
            print(f"{Path(job).name}: copied to {dest}; source files unchanged (sha256 of {len(before)} files): {same}")
            if not same:
                rc = 1
        except (ImportRefused, OSError, ValueError) as e:
            print(f"{Path(job).name}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
