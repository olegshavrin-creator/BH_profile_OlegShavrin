"""Train / evaluate the personality fusion model on cached FIV2 features.

Recipe (MM-PSYCHE config.toml): MAE loss, Adam lr 1e-4 wd 1e-5, batch 32, up to 100 epochs, early stopping
patience 15, ReduceLROnPlateau on the dev metric, selection metric = mean(mACC, CCC) on dev, seed 42.

Usage:
  python -m bs2.mm.train --modalities face,audio,text,behavior --out ~/bs/mm_runs/all
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULT_ROOT, FeatureTable
from .model import ModelConfig, PersonalityFusionModel
from ..norms import TRAIT_KEYS

log = logging.getLogger("bs.mm.train")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ccc_1d(t: np.ndarray, p: np.ndarray) -> float:
    mt, mp = t.mean(), p.mean()
    return float(2 * ((t - mt) * (p - mp)).mean() / (t.var() + p.var() + (mt - mp) ** 2 + 1e-12))


OUTPUT_KEYS = TRAIT_KEYS + ["interview"]


def metrics(t: np.ndarray, p: np.ndarray) -> dict:
    """Big Five metrics over the first five columns: mACC = mean(1 - MAE), CCC per trait (mean) and CCC over the
    flattened matrix (the variant MM-PSYCHE train.py uses for dev/test). A sixth column (interview) is reported
    separately and does not enter mACC / mCCC, so runs stay comparable."""
    mae = np.abs(t - p).mean(axis=0)
    cccs = [ccc_1d(t[:, i], p[:, i]) for i in range(t.shape[1])]
    keys = OUTPUT_KEYS[: t.shape[1]]
    out = {"mACC": float((1 - mae[:5]).mean()), "mCCC": float(np.mean(cccs[:5])),
           "CCC_flat": ccc_1d(t[:, :5].ravel(), p[:, :5].ravel()),
           "acc": {k: float(1 - mae[i]) for i, k in enumerate(keys)},
           "ccc": {k: cccs[i] for i, k in enumerate(keys)}, "n": int(len(t))}
    if t.shape[1] > 5:
        out["interview"] = {"acc": float(1 - mae[5]), "ccc": cccs[5]}
    return out


@torch.no_grad()
def predict(model, table: FeatureTable, device, batch_size=256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(table), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(table)))
        b = table.batch(idx)
        out.append(model({m: v.to(device) for m, v in b["features"].items()}).cpu())
    return torch.cat(out).numpy()


def evaluate(model, table, device) -> dict:
    return metrics(table.labels.numpy(), predict(model, table, device))


def pick_device(require_cuda: bool = True, attempts: int = 6) -> str:
    """CUDA can be transiently 'busy or unavailable' when other processes initialise it; retry before giving up.
    Silent CPU fallback made one run 10x slower, so by default a missing GPU is an error."""
    for i in range(1, attempts + 1):
        try:
            if torch.cuda.is_available():
                torch.zeros(1, device="cuda")
                return "cuda"
            raise RuntimeError("torch.cuda.is_available() is False")
        except Exception as e:  # noqa: BLE001
            log.warning("CUDA attempt %d/%d failed: %s", i, attempts, str(e).splitlines()[0][:100])
            time.sleep(5 * i)
    if require_cuda:
        raise SystemExit("no usable CUDA device; pass --allow-cpu to train on CPU")
    return "cpu"


def train(args):
    seed_everything(args.seed)
    device = pick_device(require_cuda=not args.allow_cpu)
    root = Path(args.root)
    mods = [m for m in args.modalities.split(",") if m]
    tr, dv, te = (FeatureTable(root, s, mods, targets=args.targets) for s in ("train", "dev", "test"))
    cfg = ModelConfig(modality_dims={m: int(tr.x[m].shape[1]) for m in mods}, hidden_dim=args.hidden_dim,
                      num_heads=args.heads, out_dim=args.out_dim, dropout=args.dropout, n_traits=len(tr.target_columns),
                      use_graph=not args.no_graph, use_attention=not args.no_attention,
                      use_guidebank=not args.no_guidebank, use_task_projectors=not args.no_task_projectors)
    model = PersonalityFusionModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("modalities=%s params=%d train/dev/test=%d/%d/%d device=%s", mods, n_params, len(tr), len(dv), len(te), device)

    best, best_epoch, patience, history = -1.0, -1, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr))
        total, nb = 0.0, 0
        for i in range(0, len(tr), args.batch_size):
            b = tr.batch(perm[i:i + args.batch_size])
            pred = model({m: v.to(device) for m, v in b["features"].items()})
            loss = torch.mean(torch.abs(pred - b["labels"].to(device)))     # MAE
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        dev_m = evaluate(model, dv, device)
        score = (dev_m["mACC"] + dev_m["mCCC"]) / 2
        sched.step(score)
        history.append({"epoch": epoch, "train_loss": total / max(1, nb), "dev_mACC": dev_m["mACC"],
                        "dev_mCCC": dev_m["mCCC"], "dev_score": score, "lr": opt.param_groups[0]["lr"]})
        improved = score > best
        log.info("epoch %3d loss %.4f | dev mACC %.4f mCCC %.4f score %.4f %s", epoch, total / max(1, nb),
                 dev_m["mACC"], dev_m["mCCC"], score, "*" if improved else "")
        if improved:
            best, best_epoch, patience = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "modalities": mods,
                        "targets": list(tr.target_columns), "epoch": epoch, "dev": dev_m}, out_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.patience:
                log.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    dev_m, test_m = evaluate(model, dv, device), evaluate(model, te, device)
    result = {"modalities": mods, "targets": list(tr.target_columns), "params": n_params, "best_epoch": best_epoch,
              "epochs_run": len(history), "seconds": round(time.time() - t0, 1), "dev": dev_m, "test": test_m,
              "args": vars(args), "history": history}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # per-clip test predictions for later comparison / ensembling
    p = predict(model, te, device)
    import pandas as pd
    from ..norms import OCEANAI_COLUMNS
    cols = (OCEANAI_COLUMNS + ["Interview"])[: p.shape[1]]
    df = pd.DataFrame(p, columns=cols)
    df.insert(0, "Path", [n + ".mp4" for n in te.names])
    df.to_csv(out_dir / "test_pred.csv", index=False)
    extra = f" | interview ACC {test_m['interview']['acc']:.4f} CCC {test_m['interview']['ccc']:.4f}" if "interview" in test_m else ""
    log.info("DONE %s | best epoch %d | dev mACC %.4f mCCC %.4f | TEST mACC %.4f mCCC %.4f CCC_flat %.4f%s",
             mods, best_epoch, dev_m["mACC"], dev_m["mCCC"], test_m["mACC"], test_m["mCCC"], test_m["CCC_flat"], extra)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--modalities", default="face,audio,text,behavior")
    ap.add_argument("--targets", default="big5", choices=["big5", "big5+interview"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--no-guidebank", action="store_true")
    ap.add_argument("--no-task-projectors", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="fall back to CPU if CUDA is unavailable (slow)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args)


if __name__ == "__main__":
    main()
