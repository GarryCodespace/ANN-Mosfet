from __future__ import annotations

import numpy as np
import pandas as pd

from mosfet_ann.model import FEATURE_COLUMNS, MosfetAnnModel


BIAS_COLUMNS = ["vgs", "vds", "vbs"]


def evaluate_bias_noise(
    model: MosfetAnnModel,
    frame: pd.DataFrame,
    noise_sigmas_mv: list[float],
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    baseline = model.predict_log_current(frame)
    for sigma_mv in noise_sigmas_mv:
        noisy = frame.copy()
        sigma_v = sigma_mv / 1000.0
        for column in BIAS_COLUMNS:
            noisy[column] = noisy[column] + rng.normal(0.0, sigma_v, len(noisy))
        noisy["vgs"] = noisy["vgs"].clip(0.0, 1.2)
        noisy["vds"] = noisy["vds"].clip(0.0, 1.2)
        noisy["vbs"] = noisy["vbs"].clip(-0.4, 0.0)
        predicted = model.predict_log_current(noisy[FEATURE_COLUMNS])
        error = np.abs(predicted - baseline)
        rows.append(
            {
                "bias_noise_sigma_mv": sigma_mv,
                "mean_abs_delta_log10_a": float(np.mean(error)),
                "p95_abs_delta_log10_a": float(np.percentile(error, 95)),
            }
        )
    return pd.DataFrame(rows)
