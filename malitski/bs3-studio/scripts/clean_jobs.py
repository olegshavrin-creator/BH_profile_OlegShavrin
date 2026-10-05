"""Reclaim disk space in the work dir of BS Profiler 3.1 without touching what a job needs (owner decision of
2026-09-27; refactoring stage 22a). New analyses already keep only the source video and the result, but jobs that were
made before that stage still hold their segment clips, and a crash or a stop can leave a folder that never finished.

    ~/bs/venv/bin/python bs3-studio/scripts/clean_jobs.py [--work-dir DIR] [--apply]

Without --apply it only lists what it would remove and how much space that frees (a dry run); --apply removes it.
Inside the work dir (default: settings.JOBS_DIR, one folder per analysis of the page) it removes only:
  (a) segments/ of a finished job (a folder with result.json): the segment clips were cut from input.*, and nothing
      reads them once result.json exists. input.* stays, so the job can still be re-analysed and the PDF has its media.
  (b) a whole folder without result.json whose newest file is older than 24 hours: a run that never finished. The
      24-hour floor means a running analysis is never touched.
It never touches input.*, result.json, explain/, charts/ or the PDF of a finished job, and it works only inside the
work dir: it stays with the direct subfolders, never follows a symlink (a symlink is skipped, so a link pointing
outside the work dir can never be followed), and refuses a path that resolves outside the work dir. Existing
jobs keep opening; this only frees disk space.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # bs3-studio/: `import bs3` is this working tree
from bs3 import jobfiles, settings  # noqa: E402

STALE_SEC = 24 * 3600            # a folder without result.json older than this is a run that never finished

Removal = namedtuple("Removal", "path size reason")


def _inside(path: Path, root: Path) -> bool:
    """True when `path`, resolved, is `root` (already resolved) or below it; False if it cannot be resolved."""
    try:
        rp = path.resolve()
    except OSError:
        return False
    return rp == root or root in rp.parents


def _dir_size(path: Path) -> int:
    """Total size of the regular files in a folder tree, in bytes; symlinks are counted as nothing (never followed)."""
    total = 0
    for p in path.rglob("*"):
        if p.is_symlink():
            continue
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _newest_mtime(path: Path) -> float | None:
    """The most recent mtime in a folder tree (the folder and every entry, symlinks by their own mtime); None on error."""
    try:
        newest = path.stat().st_mtime
    except OSError:
        return None
    for p in path.rglob("*"):
        try:
            newest = max(newest, p.lstat().st_mtime)
        except OSError:
            continue
    return newest


def plan(work_dir: Path) -> list[Removal]:
    """What clean_jobs would remove inside `work_dir`: the segments/ of finished jobs and the folders of runs that
    never finished. Never follows a symlink and never leaves the work dir."""
    work_dir = Path(work_dir).resolve()
    out: list[Removal] = []
    if not work_dir.is_dir():
        return out
    for child in sorted(work_dir.iterdir()):
        # only a real subfolder is a job; a symlink (even to a folder) is skipped, so a link out of the work dir is
        # never followed, and a file at the top level is not a job either
        if child.is_symlink() or not child.is_dir() or not _inside(child, work_dir):
            continue
        if (child / jobfiles.RESULT).is_file():
            seg = child / jobfiles.SEGMENTS_DIR
            if seg.is_dir() and not seg.is_symlink() and _inside(seg, work_dir):
                out.append(Removal(seg, _dir_size(seg), "segment clips of a finished job"))
        else:
            newest = _newest_mtime(child)
            if newest is not None and (age := time.time() - newest) > STALE_SEC:
                out.append(Removal(child, _dir_size(child),
                                   f"unfinished run (no {jobfiles.RESULT}), quiet for {age / 3600:.0f} h"))
    return out


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work-dir", default=str(settings.JOBS_DIR),
                    help=f"the folder that holds the page's jobs (default: {settings.shown(settings.JOBS_DIR)})")
    ap.add_argument("--apply", action="store_true",
                    help="remove what is listed; without it this is a dry run that only prints what it would remove")
    a = ap.parse_args(argv)
    work_dir = Path(a.work_dir).expanduser()
    if not work_dir.is_dir():
        print(f"clean_jobs: {work_dir} is not a folder", file=sys.stderr)
        return 2
    work_dir = work_dir.resolve()
    removals = plan(work_dir)
    total = 0
    for r in removals:
        rel = r.path.relative_to(work_dir)
        if a.apply:
            try:
                shutil.rmtree(r.path)
            except Exception as e:  # noqa: BLE001
                print(f"could not remove {rel}: {e}", file=sys.stderr)
                continue
            total += r.size
            print(f"removed {rel} ({_human(r.size)}): {r.reason}")
        else:
            total += r.size
            print(f"would remove {rel} ({_human(r.size)}): {r.reason}")
    verb = "removed" if a.apply else "would free"
    hint = "" if a.apply else " — re-run with --apply to remove them"
    print(f"clean_jobs: {verb} {_human(total)} in {len(removals)} item(s) under {settings.shown(work_dir)}{hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
