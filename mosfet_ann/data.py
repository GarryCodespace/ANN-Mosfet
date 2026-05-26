from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CURRENT_EPS = 1e-15


@dataclass(frozen=True)
class MosfetParameters:
    """Compact-model parameters for an nMOS synthetic data point."""

    vth0: float = 0.42
    kp: float = 220e-6
    lambda_: float = 0.06
    gamma: float = 0.45
    phi: float = 0.65


def drain_current(
    vgs: np.ndarray,
    vds: np.ndarray,
    vbs: np.ndarray,
    width_um: np.ndarray,
    length_um: np.ndarray,
    temperature_c: np.ndarray,
    params: MosfetParameters | None = None,
) -> np.ndarray:
    """Return nMOS drain current using a smooth Level-1 style equation."""

    p = params or MosfetParameters()
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)
    vbs = np.asarray(vbs, dtype=float)
    width_um = np.asarray(width_um, dtype=float)
    length_um = np.asarray(length_um, dtype=float)
    temperature_c = np.asarray(temperature_c, dtype=float)

    phi_term = np.sqrt(np.maximum(p.phi - vbs, 1e-9)) - np.sqrt(p.phi)
    temp_shift = -1.2e-3 * (temperature_c - 27.0)
    vth = p.vth0 + p.gamma * phi_term + temp_shift
    mobility_scale = np.power((temperature_c + 273.15) / 300.15, -1.5)
    beta = p.kp * mobility_scale * (width_um / np.maximum(length_um, 1e-9))

    vov = vgs - vth
    subthreshold = 1e-12 * (width_um / length_um) * np.exp(np.clip(vov / 0.08, -60, 20))
    linear = beta * (vov * vds - 0.5 * vds**2) * (1.0 + p.lambda_ * vds)
    saturation = 0.5 * beta * vov**2 * (1.0 + p.lambda_ * vds)

    ids = np.where(vov <= 0.0, subthreshold * (1.0 - np.exp(-np.maximum(vds, 0) / 0.026)), 0.0)
    ids = np.where((vov > 0.0) & (vds < vov), linear, ids)
    ids = np.where((vov > 0.0) & (vds >= vov), saturation, ids)
    return np.maximum(ids, 0.0)


def generate_dataset(samples: int, seed: int = 7) -> pd.DataFrame:
    """Generate random I-V samples with device and process variation."""

    rng = np.random.default_rng(seed)
    width_um = rng.uniform(0.12, 10.0, samples)
    length_um = rng.uniform(0.045, 1.0, samples)
    temperature_c = rng.uniform(-40.0, 125.0, samples)
    vgs = rng.uniform(0.0, 1.2, samples)
    vds = rng.uniform(0.0, 1.2, samples)
    vbs = rng.uniform(-0.4, 0.0, samples)

    vth0 = rng.normal(0.42, 0.035, samples)
    kp = rng.lognormal(np.log(220e-6), 0.12, samples)
    lambda_ = rng.lognormal(np.log(0.06), 0.25, samples)
    gamma = rng.normal(0.45, 0.05, samples)
    phi = rng.normal(0.65, 0.04, samples)

    ids = np.empty(samples)
    for index in range(samples):
        params = MosfetParameters(
            vth0=float(vth0[index]),
            kp=float(kp[index]),
            lambda_=float(lambda_[index]),
            gamma=float(gamma[index]),
            phi=float(phi[index]),
        )
        ids[index] = drain_current(
            vgs[index],
            vds[index],
            vbs[index],
            width_um[index],
            length_um[index],
            temperature_c[index],
            params,
        )

    frame = pd.DataFrame(
        {
            "vgs": vgs,
            "vds": vds,
            "vbs": vbs,
            "width_um": width_um,
            "length_um": length_um,
            "temperature_c": temperature_c,
            "vth0": vth0,
            "kp": kp,
            "lambda_": lambda_,
            "gamma": gamma,
            "phi": phi,
            "id_a": ids,
        }
    )
    frame["log_id"] = np.log10(np.abs(frame["id_a"]) + CURRENT_EPS)
    return frame
