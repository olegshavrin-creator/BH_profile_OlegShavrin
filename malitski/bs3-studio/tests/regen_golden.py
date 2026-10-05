"""Rebuild the golden «Характеристика личности» texts (tests/golden/char_A.txt, char_B.txt) with the same builders as
tests/test_characterization.py, printing a unified diff of every change before writing.

Run it only in a commit that deliberately changes the characterization lexicon (bs3/config/lexicon_ru.json) or the
characterization templates:

    ~/bs/venv/bin/python tests/regen_golden.py            # show the diff and rewrite the files
    ~/bs/venv/bin/python tests/regen_golden.py --check    # show the diff and exit 1 if anything differs; write nothing

The text is byte-for-byte what test_characterization.test_golden_texts compares against (scores.clean_view ->
mbti.get_mbti -> characterization.build -> Character.plain()), so with no lexicon change this rewrites the files
identically and prints no diff. This is a manual tool: it is not a test and tests/run.py does not collect it."""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # bs3-studio/: `import bs3` resolves to this working tree
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# the very builders and inputs test_characterization uses, so the output cannot drift from the test
from test_characterization import GOLDEN, _build, _with_analyses, rep  # noqa: E402

NAMES = ("A", "B")


def golden_text(name: str) -> str:
    """The expected characterization of sample `name`, exactly as test_characterization builds and compares it."""
    _, _, ch = _build(_with_analyses(rep(name), name))
    return ch.plain()


def main(argv: list[str]) -> int:
    check = "--check" in argv
    changed = 0
    for name in NAMES:
        path = GOLDEN / f"char_{name}.txt"
        want = golden_text(name)
        have = path.read_text(encoding="utf-8") if path.is_file() else ""
        if have == want:
            print(f"char_{name}.txt: unchanged")
            continue
        changed += 1
        sys.stdout.writelines(difflib.unified_diff(
            have.splitlines(keepends=True), want.splitlines(keepends=True),
            fromfile=f"a/char_{name}.txt", tofile=f"b/char_{name}.txt"))
        if check:
            print(f"char_{name}.txt: WOULD be rewritten")
        else:
            path.write_bytes(want.encode("utf-8"))        # write bytes: keep the LF line endings on every platform
            print(f"char_{name}.txt: rewritten")
    if check and changed:
        print(f"{changed} golden file(s) differ from the current builders")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
