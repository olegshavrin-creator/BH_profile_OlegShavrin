# Test fixtures

## `samples.json`

Two synthetic analysis records, `A` and `B`, are the numeric backbone of most of the page, PDF and
characterization tests. They are **numbers only**, derived from the two reference interview recordings
that the report design was tuned on:

- **A** — 18 segments; OCEAN-AI is missing on one segment (16).
- **B** — 33 segments; OCEAN-AI is missing on seven segments (10, 11, 12, 14, 15, 26, 33).

Each record keeps only the fields the render code reads: the model block (spoken language and primary
system), the per-trait scores, the per-system means, the spread, and the per-segment rows (time,
members used, and the scores rounded to four digits). Nothing is read from anyone's working directory;
the file is self-contained.

`tests/samples.py` loads this file and dresses the same numbers as each job generation (`rep`,
`english`, `own`, `oceanai31`); `tests/jobs.py` writes them out as whole job folders (`make_job`) for
every generation the page and the PDF must read.

## The rule for fixtures

Fixtures carry **numbers only**. Never add a real person's name, a file name, a job id, a transcript
or a behaviour description to `samples.json`, or anywhere under `tests/`. A test that needs text uses
short invented strings written in the test itself, never a fragment copied from a real analysis.

## `golden/char_A.txt`, `golden/char_B.txt`

The verbatim expected «Характеристика личности» for samples A and B, compared byte for byte by
`test_characterization.test_golden_texts`. They are produced by the same builders the tests use:
`scores.clean_view` → `mbti.get_mbti` → `characterization.build`, then `Character.plain()`.

Do not edit them by hand. When a change to the lexicon (`bs3/config/lexicon_ru.json`) or the
characterization templates deliberately changes the wording, regenerate them:

```
~/bs/venv/bin/python tests/regen_golden.py            # print the diff of every change, then rewrite the files
~/bs/venv/bin/python tests/regen_golden.py --check    # print the diff and fail without writing (nothing changed -> exit 0)
```

Review the printed unified diff, and regenerate only in the same commit that makes the lexicon or
template change on purpose. With no such change the script rewrites the files identically (no diff).
