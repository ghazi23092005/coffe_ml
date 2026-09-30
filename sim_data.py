import numpy as np
import pandas as pd


def simulate_batches(n_batches: int = 60, seed: int = 0, interval_min: float = 30.0):
    rng = np.random.default_rng(seed)
    rows, scores = [], {}
    dt = interval_min / 60.0
    for s in range(1, n_batches + 1):
        dur = rng.uniform(24, 72)
        t = np.arange(int(dur / dt)) * dt
        k = rng.uniform(0.12, 0.45)
        t_mid = rng.uniform(6, min(28, dur * 0.7))
        ph0, ph_floor = rng.normal(5.9, 0.15), rng.uniform(3.4, 4.8)
        prog = 1 / (1 + np.exp(-k * (t - t_mid)))
        act = 4 * prog * (1 - prog)

        ph = ph0 - (ph0 - ph_floor) * prog + rng.normal(0, 0.02, len(t))
        temp = (rng.uniform(22, 32) + 1.5 * act * rng.uniform(0.5, 1.5)
                + 1.0 * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6.28))
                + rng.normal(0, 0.08, len(t)))
        co2 = 430 + rng.uniform(600, 3000) * prog + 150 * act + rng.normal(0, 15, len(t))
        o2 = 20.9 - rng.uniform(0.3, 2.8) * (0.5 * prog + 0.5 * act) + rng.normal(0, 0.02, len(t))
        rh = rng.uniform(60, 80) + 5 * prog + rng.normal(0, 0.3, len(t))

        h_below_4 = dt * np.sum(ph < 4.0)
        ph_end = ph[-3:].mean()
        temp_at_peak = temp[act > 0.7].mean() if np.any(act > 0.7) else temp.mean()
        score = (84 - 7 * abs(ph_end - 4.3) - 0.25 * h_below_4
                 - 0.6 * max(0, temp.mean() - 28) - 1.2 * temp.std()
                 - 1.5 * max(0, temp_at_peak - 28)
                 + rng.normal(0, 1.2))
        scores[s] = round(float(np.clip(score, 60, 92)), 1)

        for i in range(len(t)):
            rows.append({"session": s, "sample": i + 1, "elapsed_h": round(float(t[i]), 3),
                         "o2_percent": round(float(o2[i]), 2), "ph_value": round(float(ph[i]), 2),
                         "scd41_co2_ppm": int(co2[i]), "scd41_temp_c": round(float(temp[i]), 2),
                         "scd41_rh_percent": round(float(rh[i]), 2)})
    return pd.DataFrame(rows), pd.Series(scores, name="score")
