# Папка задания и `result.json`

Одно задание — одна папка под рабочим каталогом (`settings.JOBS_DIR`, обычно `~/bs3_data/web_jobs`). Всё, что показывает
страница и PDF, читается из этой папки; исходные файлы по прямой ссылке сервер не отдаёт. Этот файл описывает устройство
папки и полей `result.json` для тех, кто читает старые задания или добавляет поля. Настройки — в [config.md](config.md).

Имена полей и ключей JSON даны по-английски (так они лежат в файле); пояснения — по-русски. Функции указаны по имени
(`модуль.функция`), без номеров строк.

## 1. Устройство папки и состояния

| Файл / папка | Имя в `jobfiles` | Что это |
|---|---|---|
| `result.json` | `RESULT` | результат анализа (`pipeline.run_analysis`); **обязателен** |
| `input.<ext>` | `INPUT_GLOB` | загруженное видео (нужно для повторного анализа и как запасной источник для PDF) |
| `segments/` | `SEGMENTS_DIR` | отрезки длинного ролика; существуют **только во время анализа**, после записи `result.json` удаляются (`BS3_KEEP_SEGMENTS=1` их оставляет) |
| `explain/explanation.json` | `EXPLAIN_DIR`, `EXPLANATION` | объяснения; **только AMLAI 1.0** |
| `explain/key_*.jpg` | `KEY_FRAME_GLOB` | ключевые кадры (AMLAI 1.0) |
| `charts/` | `CHARTS_DIR` | картинки графиков для PDF; пересобираются при каждом экспорте |
| `BS_Profiler_3_report_<stem>.pdf` | — | PDF-отчёт; пересобирается при каждом экспорте (`pdf/build.export_pdf`) |

Папку всегда открывают по ней самой (`jobfiles.load_job`), а не по абсолютным путям, которые хранит `result.json`
(`job_dir`, `key_frames`, `input`): перенесённое или скопированное задание (импортированное 2.0, копия в
`scripts/compare_baseline.py`) открывается так же. Записи атомарны (`jobfiles.write_json`).

**Состояния.** Папка без `result.json` — незавершённое задание, читатели её пропускают. Неудачный анализ папки не
оставляет вовсе (`BS3_KEEP_FAILED_JOBS=1` оставляет для отладки).

**Имена папок.** Старые задания: `ГГГГММДД_ЧЧММСС`. Новые: `ГГГГММДД_ЧЧММСС_<8 hex>` — по имени нельзя угадать время
загрузки, и два анализа в одну секунду получают разные папки (`pipeline.run_analysis`).

## 2. Поколения и как их различать

| Поколение | Как определяется |
|---|---|
| **3.1** | в `model` есть `selected` |
| **3.0** | `selected` нет, но есть `model.product` |
| **2.0** (импортированное) | нет `model.product` |

Правило: `model.selected` → 3.1; иначе `model.product` → 3.0; иначе 2.0. **По `model.version` не ветвиться** — это
номер сборки, а не признак поколения. Модель для показа берёт `scores.recorded_model` (`selected`, затем `primary`,
затем `backend`); 3.1 всегда показывает одну модель.

## 3. Поля `result.json` по разделам

Столбец «Есть в» — 2.0 / 3.0 / 3.1 / короткий ролик (ролик ≤ `SINGLE_CLIP_MAX_SEC`, анализируется целиком); «~» —
зависит от модели или задания. Старые задания по форме разнятся; читатели берут значения через `rep.get(k) or …` и
не падают на отсутствующих (`jobfiles.read_json`, `scores.num`).

### Верхний уровень

| Поле | Тип | Пишет | Читают | Есть в | Нужно для показа |
|---|---|---|---|---|---|
| `input` | str | `report.build_report` | — (папку открывают по ней самой) | все | нет (путь на сервере не показывается) |
| `created_at` | str (ISO) | `report.build_report` | PDF, `check_job` | все | нет |
| `traits` | object | `build_report` / `scores.clean_view` | страница, PDF, `mbti` | все | да |
| `interview` | object | `build_report` | страница, PDF, `mbti` | ~ (только AMLAI 1.0) | нет |
| `behavior_description` | str | `build_report` (из результата) | страница, PDF | ~ (AMLAI 1.0) | нет |
| `transcript` | str | `longvideo` → `build_report` | страница, PDF, анализы речи | все | нет |
| `modalities_used` | list[str] | `build_report` (`bs3.MODALITIES`) | страница, PDF (фильтр имён моделей) | все | да |
| `model` | object | `build_report` | `scores.recorded_model`, страница, PDF, журнал | все | да |
| `variant_scores` | object | `run_analysis` / `longvideo` | `scores.clean_view`, `mbti` | 3.0 (2 модели), 3.1/кор. (1) | да |
| `duration_sec` | float | `longvideo` | страница, PDF | 3.x | нет |
| `segments` | int | `longvideo` | страница, PDF, `mbti` | 3.x (кор. = 1) | нет |
| `timeline` | list | `longvideo` | страница, PDF, `scores`, `mbti` | 3.x (кор. = []) | для таблицы отрезков |
| `representative_segment` | int (с 1) | `longvideo` | объяснения, страница | 3.x (длинные) | нет |
| `chunks` | list[[start,end,text]] | `longvideo` | анализы речи (`analyses`) | 3.x (длинные) | нет |
| `scores_std_across_segments` | object | `run_analysis` (`res.scores_std`) | страница, PDF | 3.x | нет |
| `analyses` | object | `pipeline.run_extra_analyses` | страница, PDF | 3.x | нет |
| `media` | object | `media.probe_media` | страница, PDF (приложение) | 3.x | нет |
| `original_file_name` | str | `run_analysis` | журнал, PDF | 3.x | нет |
| `key_frames` | list[str] | `run_analysis` | `jobfiles.key_frame_paths`, страница, PDF | ~ (AMLAI 1.0) | для кадров |
| `timings_sec` | object | `build_report` / `run_analysis` | PDF, `check_job` | все | нет |
| `mbti` | object (schema 3) | `mbti.build_section` (`run_analysis`) | `mbti.get_mbti`, страница, PDF, журнал | 3.1/кор. (старые пересчитываются) | да |
| `job_dir` | str | `run_analysis`; переустанавливается `jobfiles.load_job` | — (папку открывают по ней самой) | 3.x | нет |
| `disclaimer`, `disclaimer_ru` | str | `build_report` | — (write-only) | все | нет |

`view_meta` в файле не хранится: его добавляет `scores.clean_view` при показе (какая модель показана и какие отрезки
выпали).

### `traits[k]` (пять черт: `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `emotional_stability`)

| Поле | Тип | Пишет | Смысл |
|---|---|---|---|
| `score` | float 0…1 | `build_report` / `clean_view` | оценка модели (в файле округлена до 4 знаков; на показе — до 2, `scores.shown`) |
| `name_ru` | str | `build_report` | русское название черты |
| `alias` | str | `build_report` | только у `emotional_stability`: `"non-neuroticism"` |
| `percentile*` | float | (старые задания) | перцентили (`RELATIVE_KEYS`): только старые **английские** задания; для русской речи `clean_view`/`data_json` их убирают (legacy) |

### `interview` (метка «собеседование», только AMLAI 1.0)

`score` (float), `name_ru` (str), `disclaimer` (str, write-only). Для показа берётся, только когда есть числовой
`score` (`scores.scored`).

### `model`

`name` (`"bs-bigfive"`), `version` (`__version__`, **для ветвления не используется**), `product` (`bs3.PRODUCT`),
`backend` / `primary` / `selected` (`"mm"`|`"oceanai"`), `selected_title`, `corpus`, `lang` (`"ru"`), `asr_model`,
`trained_on`, `scale`.

### `variant_scores[model]`

Средние по всему ролику для каждой модели: пять черт (float), у AMLAI 1.0 — `interview`, и служебное `_segments`
(int, сколько отрезков дали оценку; **write-only**). `clean_view` оставляет в задании только показанную модель.

### `timeline[]` (один элемент — один отрезок)

| Поле | Тип | Смысл |
|---|---|---|
| `segment` | int (с 1) | номер отрезка |
| `start`, `end` | float (округл. до 0.1) | границы, секунды |
| `scores` | object \| null | оценки модели на отрезке; `null` — модель отрезок не оценила (или пропуск `clean_view`) |
| `transcript` | str | речь на отрезке |
| `behavior_description` | str | описание поведения (AMLAI 1.0) |
| `members_used` | list \| null | какие модели дали оценку (старые задания) |
| `variants` | object \| null | оценки по моделям на отрезке |
| `primary_used` | object \| null | оценки основной модели на отрезке (3.x) |
| `file` | str | путь к клипу отрезка (`segments/`; после анализа удаляется, читателям не нужен) |
| `error` | str | только у пропущенного отрезка (без лица или речи) |
| `no_primary` | bool | добавляет `clean_view` отрезку-пропуску |

### `analyses.*` (`pipeline.run_extra_analyses`)

- `per_segment[]` — по отрезку: `segment`, `start`, `end`, `text_en`, `emotions_text` (7 эмоций, `labels.EMOTION_ORDER`),
  `voice` (`arousal`, `dominance`, `valence`), `face` (`expressions` — 7 выражений `labels.EXPR_ORDER`, `face_share`,
  `head_motion`, …), `speech` (см. ниже).
- `emotions_text` — `mean` (7 эмоций), `dominant`, `dominant_per_segment`.
- `voice` — `mean` и `std` по `arousal`/`dominance`/`valence`.
- `face` — `mean` (7 выражений), `dominant`, `face_share`, `head_motion`.
- `speech` — `words`, `speech_sec`, `words_per_min_speech`, `words_per_min_wall`, `pause_share`, `long_pauses`,
  `fillers`, `fillers_per_100`, `mean_sentence`, `ttr`, `unique_words` (`analyses/speech_stats.stats_for`), плюс
  `description` (`speech_stats.describe`) и `vocabulary` (`words.vocabulary`).

### `media` (`media.probe_media`)

`file_name`, `size_bytes`, `size_mb`, `modified`, `container`, `duration_sec`, `bitrate_kbps`, `tag_*` (creation_time,
encoder, make/model, title), `video_codec`, `width`, `height`, `fps`, `frames`, `rotation`, `audio_codec`,
`sample_rate`, `channels`, `sha256`. При сбое ffprobe — `ffprobe_error`, при иной ошибке — `error`.

### `mbti` (schema 3, `mbti.build_section`)

`schema_version` (3), `computed_by`, `computed_at`, `method` (`"raw"`), `borderline` (0.15), `source`
(`"ocean_ai"`|`"own_model"`), `model`, `model_title`, `role` (`"main"`), `primary_missing`, `type`, `type_strict`,
`type_name`, `alternatives`, `x_count`, `axes` (`EI`/`SN`/`TF`/`JP` → `trait`, `value`, `threshold`, `letter`,
`confidence`, `borderline`, `word`, при необходимости `missing`/`clipped`), `reliability`, `reliability_r`,
`reliability_basis`, `neuroticism` (`value`, `level`, `note` \| null), `neuroticism_note`, `segments_total`,
`segments_used`, `stability`, `modal_types`, `timeline` (тип по каждому отрезку), `llm` (**всегда `null`**, зарезервировано).
Раздел, посчитанный на показе, помечается `computed_on_render: true` и на диск не пишется.

### `timings_sec`

`total` (из `res.seconds` анализатора), `total_wall` (полное время `run_analysis`).

## 4. `explanation.json` (только AMLAI 1.0, `backend_mm.explain_video`)

`input`, `transcript`, `behavior_description`, `scores` (5 черт), `modalities` (`mm/explain.modality_attribution`:
по каждой модальности `signed` и `share`, плюс `input_x_gradient`, `leave_one_out_delta`), `frames`
(`mm/explain.frame_attribution`: `per_output` с `importance`/`signed`/`top_frames`, `top_frames_overall`, `n_frames`,
`key_frame_files`, `key_frame_info`), `transcript_words` и `behavior_words` (`token_attribution`: `per_output.top_words`,
`n_tokens`), `seconds`.

Есть только у более новых заданий:

- `clip_fps` — частота кадров клипа, на котором считалось объяснение: подпись переводит номер кадра в момент видео;
- `frames.per_frame_effect` — для каждого кадра две черты, которые он сдвинул сильнее всего, с направлением (это печатает подпись под кадром);
- `frames.key_frame_info` — по каждому сохранённому кадру: `file`, `frame` (номер в выборке), при наличии `expressions`
  (две сильнейшие мимики) и `phrase` (короткая фраза о видимом);
- `transcript_en` — только если перевод отличается от исходного текста.

## 5. Записи после анализа

`result.json` и `explain/explanation.json` пишет один раз `pipeline.run_analysis` (через `jobfiles.write_json`).
После этого файл могут менять только:

- `scripts/add_mbti.py` — дописывает раздел `mbti` в старое задание, где его нет (атомарно; ничего другого не трогает);
- экспорт PDF (`pdf/build.export_pdf`) — пересобирает `charts/` и файл PDF; `result.json` не меняет;
- `scripts/import_job.py` — создаёт `result.json` при переносе задания 2.0 (исходная папка 2.0 остаётся нетронутой).

Читают `result.json`, не меняя его: страница (`bs3/web`, через `jobview`/`scores`/`facts`/`segments`/`mbti`), PDF
(`bs3/pdf`), журнал (`bs3/journal`), `scripts/check_job.py`, `scripts/rerender_samples.py`,
`scripts/compare_baseline.py`.

## 6. Совместимые чтения (что оставлено ради старых заданий)

Ни один этап не удаляет это — старым заданиям оно нужно (полный список — раздел «Guard list» плана рефакторинга):

- `scores.clean_view` (шаги 1–3a и 5) и `recorded_model`, `main_system`, `segment_ok`, `RELATIVE_KEYS`, `LEGACY_KEYS`,
  `READ_FALLBACK`, `data_json`: двухмодельные `variant_scores`, перцентили пула и датасета, старый `narrative`;
- `mbti.get_mbti` пересчитывает раздел схемы 1 (позиция в группе роликов) или 2 (второе мнение и согласие двух систем):
  ни то, ни другое не показывается;
- фильтр имён моделей в `modalities_used` (PDF);
- ограничение объяснений и кадров моделью AMLAI 1.0 (`own`) в `page_outputs`, `_frames_html`, PDF: часть заданий помечена
  как OCEAN-AI, но несёт файлы AMLAI 1.0;
- английский показ только на чтение: `ru_texts.transcript_shown`/`vocabulary_shown` с `TRANSCRIPT_NOTE`,
  `WRONG_LANGUAGE_NOTE`, `NO_TRANSCRIPT_TRANSLATION`; английская ветка `clean_view`; `_pct_phrase`/`_ref_ru`/тик в
  `_bar_html`; `FILLERS["en"]` и `words.STOP_EN`.

**Пишутся, но не читаются** (write-only): `model.version` (для ветвления не используется), `disclaimer`,
`disclaimer_ru`, `interview.disclaimer`, `variant_scores[model]._segments`, `mbti.llm` (всегда `null`), старый
`narrative` (его убирает `scores.data_json`).

## 7. Правила для новых полей

- **Добавлять поля, не переименовывать** старые: старые задания не переписываются, читатели рассчитывают на прежние имена.
- **Читать через `rep.get(k) or default`**, а числа — через `scores.num` (конечное число или `None`; ни `NaN`, ни
  бесконечность, ни строка).
- **Числа конечны или `null`**.
- **Не заводить поле «версии схемы» на верхнем уровне**: поколение определяется по `model.selected`/`model.product`
  (раздел 2), а `mbti` несёт своё `schema_version` само.
