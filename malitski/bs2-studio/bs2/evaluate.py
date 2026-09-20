"""Accuracy of predictions against FIV2 labels: per-trait MAE, ACC = 1 - MAE, mACC, CCC, Pearson."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .norms import FIV2_COLUMNS, OCEANAI_COLUMNS, TRAIT_KEYS


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    return float(2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-12))


def evaluate(pred: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """pred: DataFrame with Path (file name) + OCEANAI_COLUMNS; labels: video_name + FIV2_COLUMNS."""
    p = pred.copy()
    p["video_name"] = p["Path"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    m = p.merge(labels, on="video_name", how="inner", suffixes=("_pred", "_true"))
    out = {"n": int(len(m)), "per_trait": {}}
    accs, cccs = [], []
    for k, pc, tc in zip(TRAIT_KEYS, OCEANAI_COLUMNS, FIV2_COLUMNS):
        yp = m[pc if pc in m else pc + "_pred"].to_numpy(dtype=float)
        yt = m[tc if tc in m else tc + "_true"].to_numpy(dtype=float)
        mae = float(np.abs(yp - yt).mean())
        c = ccc(yt, yp)
        r = float(np.corrcoef(yt, yp)[0, 1]) if len(m) > 2 else float("nan")
        out["per_trait"][k] = {
            "mae": round(mae, 4), "acc": round(1 - mae, 4), "ccc": round(c, 4), "pearson": round(r, 4),
            "pred_mean": round(float(yp.mean()), 4), "true_mean": round(float(yt.mean()), 4),
            "pred_std": round(float(yp.std()), 4), "true_std": round(float(yt.std()), 4),
        }
        accs.append(1 - mae)
        cccs.append(c)
    out["mACC"] = round(float(np.mean(accs)), 4)
    out["mCCC"] = round(float(np.mean(cccs)), 4)
    return out
