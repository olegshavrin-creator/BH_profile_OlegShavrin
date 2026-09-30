# training/ — research code, not part of the app

This folder holds the code that built and measured AMLAI 1.0, the own Big Five model (MM-PSYCHE recipe): FIV2
feature extraction, training, and the accuracy of a model on FIV2 clips. The web page and the `bs3` command never
import it. It imports the runtime package `bs3` (encoders, fusion model, trait keys, the model options of `bs3 infer`),
so run it from `bs3-studio` with the project venv:

    cd bs3-studio
    ~/bs/venv/bin/python -m training.mm_train --help

| File | What it does |
|---|---|
| `mm_data.py` | FIV2 index from the MM-PSYCHE csv files (`BS/MM-PSYCHE/data/fiv2`), audio extraction, per-modality feature cache under `~/data/fiv2` |
| `mm_extract.py` | extracts and caches the face, audio, text and behaviour features of one split |
| `mm_train.py` | trains the fusion model on the cached features; writes `best.pt`, `result.json` and `test_pred.csv` |
| `evaluate.py` | MAE, ACC = 1 - MAE, CCC and Pearson per trait of predictions against FIV2 labels |
| `eval_fiv2.py` | scores a folder of FIV2 clips with AMLAI 1.0 or OCEAN-AI and evaluates them (formerly `bs3 eval-fiv2`) |
| `interview_labels.csv` | ChaLearn job-interview label per FIV2 clip (HF mirror), for `mm_train --targets big5+interview` |

## Typical runs

    python -m training.mm_extract --split train --modalities audio,text,behavior
    python -m training.mm_extract --split train --modalities face --shard 0/6    # one of 6 parallel shards
    python -m training.mm_extract --split train --modalities face --merge
    python -m training.mm_train --modalities face,audio,text,behavior --out ~/bs/mm_runs/all
    python -m training.eval_fiv2 --dir DIR --out eval.json --backend mm --lang en

The app loads its checkpoints from `~/bs/mm_runs_seeds/seed*/best.pt` (5 seeds, averaged). A training run writes to
its `--out` folder and does not replace them.

`eval_fiv2` expects in `DIR`: `<stem>.mp4`, `<stem>.txt` (the transcript; `--asr` uses Whisper instead) and
`labels.csv` (`video_name` and the five FIV2 label columns). AMLAI 1.0 also reads `<stem>.behavior.txt` when present;
without it the behaviour description comes from Ollama. FIV2 speech is English, so pass `--lang en`: the default is
Russian, and OCEAN-AI then uses its MuPTA weights instead of the FIV2 ones.
