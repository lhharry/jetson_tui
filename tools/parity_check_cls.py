"""Offline parity check: vendored CLS classifier vs the original LIMU-BERT BERT path.

Proves the copy in ``jetson_imu_tui.cls`` (nn.py + preprocess.py) loads the same checkpoint
and produces the SAME argmax as ``inference/jetson_run16_test.predict_bert`` on real
jetson_leg windows — i.e. the vendored model, normalization, and feature/axis order are
faithful.

Run from the LIMU-BERT-Public repo root, in that repo's venv (needs torch):

    cd D:\\01_Code\\LIMU-BERT-Public
    python D:\\01_Code\\jetson_tui\\tools\\parity_check_cls.py \
        --ckpt saved\\history\\<...>\\bert_classifier_base_gru_jetson_leg_10_20_both_xyz\\<file>.pt

Optionally --npy points at a data_*.npy (default: the pocket jetson_leg set).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make both repos importable: this repo (LIMU-BERT root == cwd) and jetson_tui/src.
LIMU_ROOT = Path.cwd()
JETSON_SRC = Path(r"D:\01_Code\jetson_tui\src")
sys.path.insert(0, str(LIMU_ROOT))
sys.path.insert(0, str(JETSON_SRC))

from config import load_model_config          # noqa: E402  (LIMU-BERT)
from models import BERTClassifier, fetch_classifier  # noqa: E402
from utils import Preprocess4Normalization    # noqa: E402

from jetson_imu_tui.cls.classifier import ActivityClassifier  # noqa: E402
from jetson_imu_tui.cls.model import CLASSES   # noqa: E402


def original_preds(npy: np.ndarray, ckpt: Path, device) -> np.ndarray:
    bert_cfg = load_model_config("pretrain_base", "base", "v3")
    classifier_cfg = load_model_config("bench_gru", "gru", "v3")
    inner = fetch_classifier("gru", classifier_cfg, input=bert_cfg.hidden, output=len(CLASSES))
    model = BERTClassifier(bert_cfg, classifier=inner, frozen_bert=False)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model = model.to(device).eval()
    norm = np.stack([Preprocess4Normalization(6)(s) for s in npy], axis=0).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(norm).to(device), False)
        return torch.argmax(logits, dim=-1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--npy", type=Path,
                    default=Path("dataset/jetson_leg/data_10_20_both_xyz_pocket.npy"))
    ap.add_argument("--n", type=int, default=64, help="number of windows to compare")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.npy).astype(np.float32)
    if args.n and args.n < data.shape[0]:
        data = data[: args.n]
    print(f"Windows: {data.shape}  device={device}\nCheckpoint: {args.ckpt}")

    orig = original_preds(data, args.ckpt, device)

    clf = ActivityClassifier(args.ckpt, device=str(device))
    vend = np.array([CLASSES.index(clf.predict(w)[0]) for w in data])

    mism = int(np.sum(orig != vend))
    print(f"\nMismatches: {mism}/{len(orig)}")
    if mism:
        bad = np.nonzero(orig != vend)[0][:10]
        for i in bad:
            print(f"  window {i}: original={CLASSES[orig[i]]}  vendored={CLASSES[vend[i]]}")
        sys.exit(1)
    print("PASS — vendored CLS matches the original BERT path exactly.")


if __name__ == "__main__":
    main()
