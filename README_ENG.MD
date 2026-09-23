# Brotherly Hearts Profiling

**BS 2.0** — a multimodal profiling pipeline that analyses a short video of a person and produces an interpretable report on their apparent personality, emotions, voice, and speech.

The system is designed for research and demonstration purposes: given a video clip, it returns a structured assessment in the spirit of First Impressions — how a person is perceived by observers — rather than a psychological diagnosis.

## What It Does

Given a video file, the pipeline:

1. **Extracts audio and video** and splits the clip into short segments (~20 s each).
2. **Transcribes speech** using Whisper.
3. **Analyses the face** across frames — facial expression, head movement, key frames.
4. **Analyses the voice** — arousal, confidence, positivity.
5. **Analyses speech content** — dominant emotion by text, speech rate, pauses, vocabulary.
6. **Builds a Big Five profile** (openness, conscientiousness, extraversion, agreeableness, emotional stability) plus an "interview" impression score.
7. **Produces a structured report** — with per-segment values, dynamics across the clip, modality contribution, and explanations for the final assessment.

The final output is a presentation-ready report (PDF) with charts, key frames, and an appendix of raw values per segment.

## How It Works

The system combines two assessment layers:

- **Primary assessment:** OCEAN-AI — a model trained on the MuPTA corpus for Russian speech.
- **Secondary assessment:** MM-PSYCHE — a custom model based on First Impressions V2 weights, used as a second opinion.

| Component | Purpose |
| --- | --- |
| Whisper large-v3-turbo | Speech recognition |
| OCEAN-AI (MuPTA) | Primary Big Five assessment for Russian speech |
| MM-PSYCHE (First Impressions V2) | Secondary assessment |
| FER-2013-based model | Facial expression recognition |
| Voice emotion model | Arousal, confidence, positivity |
| Video-language model | Behavior description per segment |

The pipeline uses four modalities: **audio, video, text, and behavior description**. Final scores are duration-weighted averages across segments.

## How to Use

### Input

A short video file (`.mp4` / `.mov`). Recommended length: **1–2 minutes**, single person in frame, stable lighting, clear speech.

### Processing

Run the pipeline on the video file. Processing time depends on clip length — for a 1 min 12 s clip, approximately **23 minutes** on a standard CPU setup.

### Output

A structured report containing:

- **Key facts card** — dominant speech emotion, dominant facial expression, voice scores, speech rate, interview impression, segment count.
- **Big Five profile** — radar chart with primary and secondary scores, plus per-trait comparison.
- **Dynamics across the clip** — how each trait changes over time.
- **Emotions and facial expression** — per-emotion distribution by speech and by face.
- **Voice and speech** — voice scores over time, speech rate, pause share, vocabulary metrics.
- **Explanation section** — key frames, modality contribution per trait, interpretation of what influenced the final score.
- **Appendices** — file parameters, per-segment values, behavior descriptions, full transcript.

### How to Read the Output

- **Apparent personality** — how observers perceive the person on first impression. Not a psychological diagnosis.
- **"Interview" label** — how often annotators said they would invite the person to an interview after a 15-second first impression. Not a hiring recommendation.
- **Emotions, voice, facial expression** — signals from models trained on English-language corpora and photographs. These are behavioral observations, not clinical judgments.

## Example Output

A sample report built on a 1 min 12 s clip is included in this repository. It demonstrates:

- a full Big Five profile with primary and secondary scores;
- per-segment dynamics across four segments;
- modality contribution analysis (face 91%, voice 7–8%, speech <1%, behavior <1%);
- explanation based on the segment closest to the average profile;
- five key frames that most influenced the custom model's assessment.

## Limitations

- The system estimates **apparent** personality, not actual psychological traits.
- Emotion, voice, and facial expression models are trained on **English-language corpora** — values for Russian speech should be treated as **relative**, not absolute.
- The facial expression model (FER-2013) tends to read "sadness" and "fear" in calm faces — compare changes across the clip rather than absolute shares.
- The "interview" label is a research metric and must not be used for decisions affecting people.
- This is a research and demonstration tool, not a diagnostic or hiring system.

## Tech Stack

- **Python**
- **Whisper** (speech recognition)
- **OCEAN-AI** / MuPTA (Russian speech Big Five)
- **First Impressions V2** / MM-PSYCHE (secondary assessment)
- **FER-2013** (facial expression)
- Video and audio processing (FFmpeg, OpenCV)
- Report generation (charts, PDF)
