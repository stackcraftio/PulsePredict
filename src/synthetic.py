"""Synthetic stand-in dataset that mirrors the schema of the Kaggle cardiovascular dataset.

Why this exists: so the project runs end-to-end out of the box with no download and contains
no real patient data. It deliberately includes the kinds of dirt found in the real file
(duplicates, impossible blood pressures, implausible heights/weights) so the cleaning step is
meaningful. To use the real data, replace data/raw/cardio_train.csv and re-run `python -m src.train`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


def _solve_intercept(z: np.ndarray, target_rate: float = 0.5) -> float:
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_synthetic_cardio(n: int = 70_000, seed: int = 42, dirty: bool = True,
                              signal: float = 1.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice([1, 2], size=n, p=[0.65, 0.35])
    age = np.clip(rng.normal(53.3, 6.8, n), 30, 65)
    height = np.where(gender == 1, rng.normal(161.5, 6.3, n), rng.normal(170.5, 7.0, n))
    bmi = np.clip(16 + rng.gamma(shape=6.0, scale=1.8, size=n), 16, 55)
    weight = bmi * (height / 100) ** 2

    hyper = rng.random(n) < 0.14
    sbp = 117 + 0.5 * (age - 53) + 0.8 * (bmi - 27) + rng.normal(0, 10, n) + hyper * rng.normal(30, 10, n)
    sbp = np.clip(sbp, 85, 230)
    snap = rng.random(n) < 0.45
    sbp = np.where(snap, np.round(sbp / 10) * 10, np.round(sbp))
    dbp = 0.5 * sbp + 18 + 0.15 * (bmi - 27) + rng.normal(0, 5, n)
    dbp = np.clip(np.round(dbp), 55, 130)
    dbp = np.minimum(dbp, sbp - 20)

    lat_c = 0.035 * (age - 53) + 0.05 * (bmi - 27) + 0.012 * (sbp - 127) + rng.normal(0, 1, n)
    chol = 1 + (lat_c > np.quantile(lat_c, 0.75)) + (lat_c > np.quantile(lat_c, 0.88))
    lat_g = 0.5 * lat_c + 0.03 * (bmi - 27) + rng.normal(0, 1, n)
    gluc = 1 + (lat_g > np.quantile(lat_g, 0.85)) + (lat_g > np.quantile(lat_g, 0.925))
    smoke = (rng.random(n) < 0.088).astype(int)
    alco = (rng.random(n) < 0.054).astype(int)
    active = (rng.random(n) < 0.80).astype(int)

    z_age, z_sbp, z_bmi = (age - 53.3) / 6.8, (sbp - 127) / 17, (bmi - 26.8) / 4.4
    z = (0.50 * z_age + 0.85 * z_sbp + 0.30 * z_bmi + 0.25 * z_age * z_sbp
         + 0.60 * (sbp >= 140) + 0.25 * (sbp >= 160) + 0.20 * z_bmi ** 2
         + 0.30 * ((age > 58) & (chol >= 2))
         + 0.45 * (chol == 2) + 1.00 * (chol == 3)
         + 0.12 * (gluc == 2) + 0.28 * (gluc == 3)
         - 0.20 * active + 0.10 * smoke) * signal
    z = z + _solve_intercept(z)
    cardio = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "id": np.sort(rng.choice(np.arange(1, 100_000), size=n, replace=False)),
        "age": np.round(age * DAYS_PER_YEAR).astype(int),
        "gender": gender,
        "height": np.round(height).astype(int),
        "weight": np.round(weight).astype(float),
        "ap_hi": sbp.astype(int),
        "ap_lo": dbp.astype(int),
        "cholesterol": chol.astype(int),
        "gluc": gluc.astype(int),
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "cardio": cardio,
    })

    if dirty and n >= 1000:
        order = rng.permutation(n)
        cur = 0

        def take(k: int) -> np.ndarray:
            nonlocal cur
            rows = order[cur:cur + k]
            cur += k
            return rows

        rows = take(int(0.012 * n))               # blood-pressure entry errors
        modes = rng.integers(0, 5, size=len(rows))
        for r, m in zip(rows, modes):
            hi, lo = df.at[r, "ap_hi"], df.at[r, "ap_lo"]
            if m == 0:
                df.at[r, "ap_hi"] = hi * 10
            elif m == 1:
                df.at[r, "ap_hi"] = hi * 100
            elif m == 2:
                df.at[r, "ap_hi"], df.at[r, "ap_lo"] = lo, hi
            elif m == 3:
                df.at[r, "ap_lo"] = lo * 10
            else:
                df.at[r, "ap_hi"], df.at[r, "ap_lo"] = -hi, -lo
        rows = take(int(0.0015 * n))              # height errors
        df.loc[rows, "height"] = rng.choice([50, 55, 65, 250], size=len(rows))
        rows = take(int(0.0015 * n))              # weight errors
        df.loc[rows, "weight"] = rng.choice([10.0, 11.0, 21.0, 300.0], size=len(rows))
        src, dst = take(24), take(24)             # duplicated observations (new ids)
        cols = [c for c in df.columns if c != "id"]
        df.loc[dst, cols] = df.loc[src, cols].to_numpy()

    return df


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH

    out = generate_synthetic_cardio()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW_DATA_PATH, sep=";", index=False)
    print(f"Wrote {len(out):,} synthetic rows to {RAW_DATA_PATH}")
