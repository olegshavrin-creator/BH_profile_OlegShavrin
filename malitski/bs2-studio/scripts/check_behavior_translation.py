"""Regression check of the Russian behaviour descriptions: FIV2 .behavior.txt files through
bs2.translate.translate_description, searched for the known mistranslations of the stock phrases.

Usage (WSL, Ollama running):  python scripts/check_behavior_translation.py [N=60] [--marian]
  --marian   check the fallback path (Marian with fix_marian_ru), as when Ollama is unreachable
Exit code 1 when any description still contains one of the patterns below.
"""
import re
import sys
import time
from pathlib import Path

import bs2.translate as tr

CLIPS = Path.home() / "bs" / "eval" / "fi_test200"
# Marian's errors seen in real reports («спокойной и спокойной» for 'calm and composed', «извращению» for
# 'extraversion', «разбитые губы» for 'parted lips' …), English words left in, other scripts, «они» about one person
PATTERNS = [r"спокойн\w* и спокойн", r"состоятельн", r"помолвлен", r"скомпрометир", r"созидательн", r"утешени",
            r"государств", r"разбит\w* губ", r"разорван", r"приемлем\w* личност", r"извращени", r"непристойн",
            r"торчат", r"[A-Za-z]{3,}", r"\bони\b", r"индивидуум", r"ныя\b"]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 60
    if "--marian" in sys.argv:
        tr.ollama_available = lambda ttl=60.0: False
    files = sorted(CLIPS.glob("*.behavior.txt"))[:n]
    t0, bad, by = time.time(), 0, {}
    for f in files:
        ru, who = tr.translate_description(f.read_text(encoding="utf-8", errors="ignore").strip())
        by[who] = by.get(who, 0) + 1
        hits = [p for p in PATTERNS if re.search(p, ru, re.I)]
        if hits:
            bad += 1
            print(f"{f.name}: {hits}\n  {ru[:300]}")
    print(f"{len(files)} descriptions in {time.time() - t0:.0f} s, translated by {by}; with problems: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
