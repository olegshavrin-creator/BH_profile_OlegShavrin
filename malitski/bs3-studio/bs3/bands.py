"""The thresholds behind the words BS Profiler 3.1 uses for speech and head motion, in one place: the sentence about
the manner of speech (analyses/speech_stats.describe, stored in result.json), the paragraph «Что видно в поведении на
записи» of the characterization, and the words of the head motion on the page and in the PDF (facts.head_motion_word).

The speech tempo has no band here: its words («медленный», «спокойный», «быстрый») and the colour of the tempo card of
«Ключевые факты» follow one band, scores.TEMPO_BAND (100…160 words per minute), through one classifier,
scores.tempo_state (owner, 2026-09-27).

Each pair is (lower edge, upper edge) of the middle band:
- PAUSE_WORDS: the share of the time in pauses; «пауз мало» below 0.10, «паузы умеренные» below 0.25, else «много
  пауз» (the lower edge belongs to the middle band, the upper one to the band above it; the callers compare the share
  in percent, 10 and 25).
- FILLER_WORDS: filler words per 100 words; «почти нет» below 2, «встречаются» below 6, else «много».
- HEAD_MOTION_WORDS: the head shift between frames as a share of the face width (analyses/face_expr); «слабое» below
  0.05, «умеренное» below 0.15, else «активное».

A leaf of the import layers (tests/test_layers.py) with no imports at all: face_expr, which measures the head motion,
imports torch, and the texts must not load it for two numbers.
"""
from __future__ import annotations

PAUSE_WORDS = (0.10, 0.25)
FILLER_WORDS = (2, 6)
HEAD_MOTION_WORDS = (0.05, 0.15)
