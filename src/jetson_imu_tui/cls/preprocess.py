"""Per-window normalization — verbatim numpy copy of ``utils.Preprocess4Normalization``.

For the 6-dim (accel + gyro) case this divides the 3 accel columns by 9.8 (so accel is in
g, still gravity-inclusive) and leaves the 3 gyro columns unchanged. Must match training
exactly, so the live window is normalized the same way before inference.
"""

from __future__ import annotations

import numpy as np


class Preprocess4Normalization:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new
