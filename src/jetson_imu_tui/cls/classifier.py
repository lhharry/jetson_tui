"""ActivityClassifier — load a BERT-finetune checkpoint and classify one 20x6 window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from jetson_imu_tui.cls.model import CLASSES
from jetson_imu_tui.cls.nn import BASE_V3, GRU_V3, BERTClassifier, ClassifierGRU
from jetson_imu_tui.cls.preprocess import Preprocess4Normalization


class ActivityClassifier:
    """Wraps the vendored BERTClassifier: window (seq_len, 6) float -> (class, confidence)."""

    def __init__(self, model_path: Path | str, device: str | None = None) -> None:
        self.classes = list(CLASSES)
        self.feature_num = BASE_V3.feature_num  # 6
        self.seq_len = BASE_V3.seq_len          # 20
        self._norm = Preprocess4Normalization(self.feature_num)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        inner = ClassifierGRU(GRU_V3, input=BASE_V3.hidden, output=len(self.classes))
        model = BERTClassifier(BASE_V3, classifier=inner, frozen_bert=False)
        state = torch.load(str(model_path), map_location=self.device)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> tuple[str, float, list[float]]:
        """window: (seq_len, 6) raw [ax,ay,az (m/s^2), gx,gy,gz (rad/s)].

        Returns (class_name, confidence, per_class_probs). Normalization matches training.
        """
        arr = np.asarray(window, dtype=np.float32)
        norm = self._norm(arr).astype(np.float32)
        x = torch.from_numpy(norm).unsqueeze(0).to(self.device)  # (1, seq_len, 6)
        logits = self.model(x, False)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx]), [float(p) for p in probs]
