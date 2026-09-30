"""Pendekatan 2 — DEEP LEARNING SEQUENCE: tiap batch = deret waktu multivariat
(pH, O2, CO2, suhu, RH + laju perubahan pH & CO2) di-resample ke panjang tetap
pada sumbu waktu ternormalisasi 0..1; durasi asli masuk sebagai fitur statis.
Arsitektur kecil (GRU / 1D-CNN) karena batch masih sedikit."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import SENSOR_COLS

torch.set_num_threads(1)
T_STEPS = 32
N_CHANNELS = len(SENSOR_COLS) + 2      # + dpH/dt, dCO2/dt


def batches_to_tensor(batches: dict):
    """-> X (N, T, C) float32, S (N, 1) float32 [durasi jam], sessions (list)."""
    grid = np.linspace(0, 1, T_STEPS)
    X, S, ids = [], [], []
    for s, g in batches.items():
        t = g["elapsed_h"].values.astype(float)
        dur = max(float(t[-1] - t[0]), 1e-6)
        u = (t - t[0]) / dur
        ch = [np.interp(grid, u, g[c].values.astype(float)) for c in SENSOR_COLS]
        hours = grid * dur
        d_ph = np.gradient(ch[1], hours)          # per jam
        d_co2 = np.gradient(ch[2], hours)
        X.append(np.stack(ch + [d_ph, d_co2], axis=1))
        S.append([dur]); ids.append(s)
    return np.asarray(X, np.float32), np.asarray(S, np.float32), ids


class GRUNet(nn.Module):
    def __init__(self, c, n_static, h=32, p=0.2):
        super().__init__()
        self.gru = nn.GRU(c, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h + n_static, 32), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(32, 1))
    def hidden(self, x):
        return self.gru(x)[1][-1]
    def forward(self, x, st):
        return self.head(torch.cat([self.hidden(x), st], 1)).squeeze(1)


class CNNNet(nn.Module):
    def __init__(self, c, n_static, p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2, stride=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 + n_static, 32), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(32, 1))
    def hidden(self, x):
        z = self.conv(x.transpose(1, 2))
        return torch.cat([z.mean(2), z.amax(2)], 1)
    def forward(self, x, st):
        return self.head(torch.cat([self.hidden(x), st], 1)).squeeze(1)


def _build(arch, n_static=1):
    return GRUNet(N_CHANNELS, n_static) if arch == "gru" else CNNNet(N_CHANNELS, n_static)


def fit_bundle(arch, X, S, y, epochs=120, seeds=(0, 1), lr=3e-3, wd=1e-2,
               bs=16, noise=0.05, offset=0.1):
    """Latih ensemble kecil (beberapa seed). Return 'bundle' berisi semua yang
    dibutuhkan untuk prediksi (bobot + statistik normalisasi)."""
    xm, xs = X.mean((0, 1)), np.maximum(X.std((0, 1)), 1e-3)
    sm, ss = S.mean(0), np.maximum(S.std(0), 1e-3)
    ym, ys = float(y.mean()), float(max(y.std(), 1e-3))
    Xn = torch.tensor((X - xm) / xs, dtype=torch.float32)
    Sn = torch.tensor((S - sm) / ss, dtype=torch.float32)
    yn = torch.tensor((y - ym) / ys, dtype=torch.float32)
    n, states, history = len(y), [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = _build(arch)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n)
            ep_loss, ep_n = 0.0, 0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:              # BatchNorm-free, tapi hindari batch 1 saat dropout
                    continue
                xb = Xn[idx] + noise * torch.randn_like(Xn[idx]) \
                     + offset * torch.randn(len(idx), 1, N_CHANNELS)   # augmentasi: noise + drift kalibrasi
                loss = nn.functional.smooth_l1_loss(model(xb, Sn[idx]), yn[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss.detach()) * len(idx); ep_n += len(idx)
            sched.step()
            if seed == seeds[0]:
                history.append(ep_loss / max(ep_n, 1))
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    return {"arch": arch, "xm": xm, "xs": xs, "sm": sm, "ss": ss, "ym": ym, "ys": ys,
            "states": states, "history": history}


def _load(bundle, k):
    m = _build(bundle["arch"]); m.load_state_dict(bundle["states"][k]); m.eval(); return m


def _norm(bundle, X, S):
    return (torch.tensor((X - bundle["xm"]) / bundle["xs"], dtype=torch.float32),
            torch.tensor((S - bundle["sm"]) / bundle["ss"], dtype=torch.float32))


def predict_bundle(bundle, X, S):
    Xn, Sn = _norm(bundle, X, S)
    with torch.no_grad():
        p = np.mean([_load(bundle, k)(Xn, Sn).numpy() for k in range(len(bundle["states"]))], axis=0)
    return p * bundle["ys"] + bundle["ym"]


def embed_bundle(bundle, X, S):
    """Representasi tersembunyi tiap batch (dipakai untuk clustering)."""
    Xn, _ = _norm(bundle, X, S)
    with torch.no_grad():
        return _load(bundle, 0).hidden(Xn).numpy()


def make_pred_fn(arch, X, S, y, epochs, seeds):
    def fn(tr, te):
        b = fit_bundle(arch, X[tr], S[tr], y[tr], epochs=epochs, seeds=seeds)
        return predict_bundle(b, X[te], S[te])
    return fn
