# My Contribution to Brotherly Hearts Profiling (BS 2.0)

**Author:** Oleg Shavrin
**Role:** Research & experimentation — behavioral pattern analysis, anomaly detection pipeline, and emotional-context mapping.

---

## Summary

My work in this project focused on building an experimental CPU-only pipeline for behavioral analysis of video. The goal was not to guess "emotions" directly, but to find measurable behavioral deviations from a person's individual baseline, and to interpret them with the help of emotional and Big Five context.

The main methodological result: **separating the detection of behavioral deviation from its interpretation.** The pipeline does not claim that a specific pattern means a specific emotion — it reports what was observed, when, and against what baseline.

---

## 1. From 14 Emotions to 12 Behavioral Patterns

The project started from a manually labeled set of **14 emotional intervals** on a reference video (surprise, embarrassment, interest, joy/laughter, distress/sadness, fear, strong fright, anger/irritation, indignation, tenderness, playfulness/flirt, strict important look, joy/smile, neutral state/completion).

Clustering on behavioral features P01–P12 showed that emotional categories and behavioral states are **not the same object**. A more rigorous task was formulated instead:

> Determine how much behavior at a given moment deviates from the person's usual behavior and from the population distribution.

This led to the architecture:

**behavior → anomalous deviation → emotional context**

instead of:

**behavior → automatically determined emotion**

### The 12 behavioral patterns (P01–P12)

| **CodePatternWhat it measures** |                            |                                                             |
| ------------------------------- | -------------------------- | ----------------------------------------------------------- |
| P01                             | Frequent blinking          | Blink rate                                                  |
| P02                             | Nod                        | Vertical head movements (later excluded from anomaly score) |
| P03                             | Head movement              | Angular velocity of head motion                             |
| P04                             | Gesture episode            | Episodes of active gestures                                 |
| P05                             | Gesture amplitude / speed  | Amplitude and speed of gestures                             |
| P06                             | Hand near face             | Hand position near the face                                 |
| P07                             | Repetitive micro-movements | Repeated small movements                                    |
| P08                             | Crossed / center hands     | Hand position in the center / crossing                      |
| P09                             | Posture shift              | Changes in body position                                    |
| P10                             | Lip corner raise           | Raise of lip corners                                        |
| P11                             | Brow / lip tension         | Tension of brows and lips                                   |
| P12                             | Gaze direction             | Direction of gaze (descriptive feature)                     |

P02 is excluded from the final anomaly score; P12 is kept as a descriptive feature.

---

## 2. Video Data Extraction

Processing was done with **MediaPipe** in single-image mode on CPU (Google Colab, CPU-only).

Videos were converted into sequences of frames with face / body / hand coordinates, and then into **temporal windows**. For each window, values of P01–P12 were computed together with a time coordinate.

Example extractions:

| **VideoFramesFPSApprox. duration** |      |       |          |
| ---------------------------------- | ---- | ----- | -------- |
| Визитка1.mp4                       | 2305 | 30    | \~76.8 s |
| Визитка2.mp4                       | 2992 | 25    | —        |
| Визитка3.mp4                       | 4869 | 29.97 | —        |

This shifted the pipeline from a sequence of individual frames to a **temporal series of behavior**:

video → frames → behavioral features → temporal windows → P01–P12 → comparison with reference.

---

## 3. Population Baseline and Anomaly Score

A **frozen population reference** was constructed as a baseline for anomaly detection.

- Each person is treated as a separate subject.
- Multiple videos of the same person do not automatically give that person more weight.
- Individual behavior structure is evaluated first, then deviations are compared with the distribution across people.
- Manual labeling is used for verification, not as the main source of training for the anomaly detector.

Current reference version: **`v51_population_v1`** — built on three people. Active patterns: P01, P03–P12 (P02 excluded from score; P12 descriptive).

Population quantile thresholds were preserved for numeric patterns (q95 / q99), for example:

| **Patternq95q99** |       |       |
| ----------------- | ----- | ----- |
| P01               | 6.000 | 6.000 |
| P03               | 2.295 | 3.960 |
| P04               | 6.000 | 6.000 |
| P05               | 3.705 | 6.000 |
| P10               | 1.416 | 4.047 |
| P11               | 2.698 | 6.000 |

For P12 a separate descriptive range was kept: q50 ≈ 0.289, q95 ≈ 2.539, q99 ≈ 4.173.

### Known limitation of the score

For some patterns **q95 = q99** (notably P01 and P04). As a result, any value above q95 immediately gets the maximum normalized anomaly score of 100. This means:

- the anomaly score **cannot** be interpreted as a linear measure of deviation strength;
- for visual analysis, raw P01–P11 values were returned alongside the score.

This is especially important for P04.

---

## 4. Personal Baseline and Local Deviations

A second-stage experiment moved from a population reference to an **individual baseline**:

- Primary baseline: `interview.mp4`
- Supplemental (stability check): `Визитка2.mp4`, `Визитка3.mp4`
- Test-only: `Визитка1.mp4`, `3.mp4`
- Sequential review: `video.mp4`, `11.mp4`–`16.mp4`

Goal: find **1–3 strong, non-repeating deviations** in an interval, each with a frame and explanation.

### Calibration anchors (interview\.mp4)

| **StrengthPatterns**         |                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------- |
| Strongly confirmed           | P01 (frequent blinking), P03 (head movement), P04 (gesture episode)             |
| Weakly / partially confirmed | P05 (gesture amplitude / speed), P11 (brow / lip tension), P12 (gaze direction) |
| No separate anchor           | P06 (hand near face), P08 (hand position)                                       |

P07, P09, P10 had `anchor_signal = 0` in the provided calibration.

Reliability weighting was used to set each pattern's contribution to the integral anomaly score — not to manually correct the top-3 results.

### Test results

| **VideoTop-1 timeTop-1 scoreNotes** |         |       |                    |
| ----------------------------------- | ------- | ----- | ------------------ |
| Визитка1.mp4                        | 14.25 s | 4.749 | 5 flagged features |
| 3.mp4                               | 63.45 s | 1.343 | noticeably calmer  |

Top-1 of `3.mp4` is roughly **3.5× weaker** than Top-1 of `Визитка1.mp4` — consistent with the idea that staged/test videos produce more pronounced deviations.

After reliability weighting, the signal on `Визитка1.mp4` shifted from 14.25 s to 16.15 s under a different baseline configuration — showing the result's sensitivity to baseline composition.

---

## 5. Task-Specific Interval Analysis

A user-facing mode was added: the user specifies a video and a time interval, and the model evaluates **only that interval**.

Workflow:

1. Enter the video filename.
2. Enter start and end in seconds.
3. Receive **1–4 distinct strong deviations** (fewer if fewer are present).

For each deviation:

- one evidence frame with an overlay (what deviated + deviation score);
- a timeline of the interval with 3–5 most noticeable patterns.

Example — `16.mp4`, interval 20–30 s:

| **TimePattern** |                |
| --------------- | -------------- |
| 20.44 s         | Gaze direction |
| 28.96 s         | Hand near face |

Each deviation is shown as a frame with overlay, plus a combined timeline of behavioral deviations for the interval.

---

## 6. Emotional Context from Audio

Automatic emotion classification was **not** used as a source of truth. Instead:

- an audio track was extracted for the target interval (`emotion_audio.wav`);
- speech recognition was done with **faster-whisper** in CPU mode;
- explicitly spoken emotion names were extracted (joy, sadness, fear, anger, surprise, disgust, neutrality).

### Rule of emotional context inheritance

- If an emotion name is spoken **at the moment** of an anomalous episode — that name is used.
- If not spoken at that moment — the **last previously spoken emotion** is used.
- If no emotion has been spoken before the episode — the state is marked as `no_previous_emotion`.

This is important because the absence of a spoken emotion word does not mean the absence of emotion.

### Result for Визитка1.mp4

| **TimePatternEmotional contextRule** |     |                     |                         |
| ------------------------------------ | --- | ------------------- | ----------------------- |
| 30.95 s                              | P04 | no previous emotion | no emotion named before |
| 34.85 s                              | P01 | joy                 | inherited from previous |
| 38.45 s                              | P11 | joy                 | inherited from previous |
| 47.85 s                              | P04 | anger               | inherited from previous |

The system does **not** claim that P04 means anger or P01 means joy. It records only temporal adjacency: behavioral episode + last explicitly spoken emotion word.

---

## 7. Big Five as a Context Layer

Big Five was added as a separate **mapping / context layer** — not as a set of individual psychometric scores.

Mapping used:

| **Big Five factorBehavioral indicators** |                     |
| ---------------------------------------- | ------------------- |
| Extraversion                             | P03, P04, P05       |
| Agreeableness                            | P03, P10            |
| Conscientiousness                        | no direct indicator |
| Neuroticism                              | P01, P06, P07, P11  |
| Openness                                 | P03                 |

For example, P04 can be linked to **Extraversion context** — but this does not mean that Extraversion level of a specific person is computed from a single P04.

This is a fundamental limit of interpretation.

---

## 8. Actor Group Stress Test

An independent group of eight videos was processed as a stress test:

`11.mp4`, `12.mp4`, `13.mp4`, `14.mp4`, `15.mp4`, `16.mp4`, `Визитка2.mp4`, `Визитка3.mp4`.

These videos were treated as a separate validation group and were **not added** to the frozen training population.

Outputs:

- `actors_results.csv`
- `actors_profiles.csv`
- `actors_summary.csv`

The group showed significant behavioral heterogeneity — in some videos the dominant pattern was P04, in others P01. This confirmed the need to compare behavior not only by the absolute value of a single feature, but against **both individual and population context**.

---

## 9. Experiment on a New Unseen Video

After freezing the population reference, a new person not used in building the reference was processed: `Визитка1.mp4`, target interval **30–50 seconds**.

Extraction results:

- 2305 frames, 30 FPS, \~76.8 s duration
- 764 behavioral windows
- 200 windows inside the target interval

### Selected anomalous episodes

| **TimePatternRaw valueAnomaly score** |                          |     |     |
| ------------------------------------- | ------------------------ | --- | --- |
| 30.95 s                               | P04 — Gesture episode    | 360 | 100 |
| 34.85 s                               | P01 — Frequent blinking  | 360 | 100 |
| 38.45 s                               | P11 — Brow / lip tension | 360 | 100 |
| 47.85 s                               | P04 — Gesture episode    | 600 | 100 |

Here the score limitation is especially visible: despite different raw P04 values, the final score is 100 in both cases. For further interpretation, raw P04 / P01 / P11 curves were used.

Each selected episode produced a separate frame with an English overlay: `ANOMALY / Pattern / Time / Score / Context`.

---

## 10. Separate Verification of P04

After the analysis, it became clear that **P04 requires separate verification** because its anomaly score saturates at q95 = q99 = 6.

A dedicated visualization was built: P04 across the full video plus the target interval 30–50 s, with q95, q99, the P04 maximum inside the target, and the global P04 maximum across the whole video.

This verification is critical because it separates two different statements:

1. P04 is high **relative to the frozen population reference**.
2. P04 is unusually high **for this particular person** relative to the rest of their video.

These two statements are not the same.

---

## 11. What Changed Compared to the Initial Setup

The work moved from the task:

> "determine emotion from behavior"

to a more rigorous task:

> "find pronounced behavioral deviations and separately show their emotional context."

This avoided mixing different levels of analysis. Currently:

- P01–P12 describe **observed behavior**;
- anomaly score describes **deviation** relative to the frozen reference;
- raw Pxx shows the **actual magnitude** of the pattern;
- audio gives **additional emotional context**;
- Big Five gives **additional behavioral context**;
- neither audio nor Big Five is used to artificially inflate the anomaly score.

---

## 12. What Cannot Yet Be Claimed

The current system is an **experimental prototype**, not a validated universal diagnostic tool.

In particular:

- the current population reference is built on only three people, so it cannot be considered representative of the whole population;
- it cannot be claimed that a specific P04 indicates a specific emotion;
- it cannot be claimed that P01 means stress;
- it cannot be claimed that P04 by itself measures Extraversion;
- Big Five factors are not computed as individual psychometric scores;
- the anomaly score is not a probability of a psychological state;
- a detected deviation is not a cause of an emotional state.

---

## 13. Saved Results

**Population / actors**

- `actors_results.csv`
- `actors_profiles.csv`
- `actors_summary.csv`

**New unseen-video experiment**

- `unknown_results.csv`
- `unknown_scores.csv`
- `unknown_visuals.csv`

**Graphs**

- `unknown_anomaly_patterns_combined.png`
- `unknown_emotions_patterns.png`

**Audio**

- `emotion_audio.wav`

**Big Five**

- `BIG5_LABELS.csv`

---

## 14. Outcome of This Stage

A working **CPU-only pipeline** was built that allows:

1. process a new video;
2. obtain P01–P12;
3. exclude P02 from the anomaly score;
4. keep P12 as descriptive;
5. compare P01–P11 against the frozen population reference;
6. find local pronounced behavioral episodes;
7. automatically select 3–4 strongest visual examples;
8. output these episodes on frames;
9. build a behavioral timeline graph;
10. add emotional context from audio;
11. apply the rule of inheriting the last previously spoken emotion;
12. add Big Five as a separate context layer;
13. separately verify a specific pattern across the whole video.

**Main methodological result:** separating the detection of behavioral deviation from its interpretation.

**Most promising next step:** expand the frozen population reference with more independent people, and run a separate experiment on truly new, unfamiliar subjects. Only then can the method's robustness on new people be quantitatively evaluated.

---

## Contact

- LinkedIn: [www.linkedin.com/in/olegshavrin](https://www.linkedin.com/in/olegshavrin)
- Email: olegshavrin\@gmail.com