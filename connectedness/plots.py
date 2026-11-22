"""Grafiklar — R paketidagi Plot* funksiyalarining matplotlib portlari.

Barcha yorliqlar o'zbek tilida. Har funksiya PNG fayl yo'lini qaytaradi.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .measures import ConnectednessTable

__all__ = ["plot_tci", "plot_net", "plot_to_from", "plot_network", "plot_npt"]

# O'zbekcha oyma ranglar (R paketidagi paletraga o'xshash)
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
           "#1b9e77", "#d95f02"]


def _dates(dca):
    cfg = dca.get("config", {})
    dates = cfg.get("dates")
    if dates is None:
        t = dca["TCI"].shape[0]
        dates = [str(i + 1) for i in range(t)]
    try:
        import pandas as pd
        d = pd.to_datetime(dates, errors="coerce")
        if not d.isna().any():
            return d
    except Exception:
        pass
    return dates


def _xticks(ax, dates, n=8):
    n_t = len(dates)
    if n_t <= 1:
        return
    idx = np.linspace(0, n_t - 1, min(n, n_t)).astype(int)
    labels = [str(dates[i])[:10] for i in idx]
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


def plot_tci(dca, path, title=None):
    """Umumiy bog'liqlik indeksi (TCI) dinamikasi."""
    dates = _dates(dca)
    tci = np.asarray(dca["TCI"])
    TCI = tci[:, None] if tci.ndim == 1 else tci    # (t, bandlar)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    if TCI.shape[1] == 1:
        ax.plot(np.arange(len(dates)), TCI[:, 0], color="#1f77b4", lw=1.4)
        ax.fill_between(np.arange(len(dates)), TCI[:, 0], alpha=0.15,
                        color="#1f77b4")
    else:
        names = dca["config"].get("period_names",
                                  [f"Band {i+1}" for i in range(TCI.shape[1])])
        for j in range(TCI.shape[1]):
            ax.plot(np.arange(len(dates)), TCI[:, j], lw=1.4,
                    label=str(names[j]),
                    color=PALETTE[j % len(PALETTE)], alpha=0.55 if j else 1.0)
        ax.legend(fontsize=8, ncol=min(4, TCI.shape[1]))
    ax.set_title(title or "Umumiy bog'liqlik indeksi (TCI)", fontsize=11)
    ax.set_ylabel("TCI (%)", fontsize=9)
    ax.grid(alpha=0.25)
    _xticks(ax, dates)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_net(dca, path, names=None, max_series=12):
    """Sof (NET) yo'nalishli spilloverlar."""
    dates = _dates(dca)
    NET = dca["NET"]
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(NET.shape[1])]
    if NET.ndim == 3:                       # chastota yondashuvi — umumiy band
        NET = NET[:, :, 0]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    k = NET.shape[1]
    order = np.argsort(-NET.mean(axis=0))
    for rank, j in enumerate(order[:min(max_series, k)]):
        ax.plot(np.arange(len(dates)), NET[:, j], lw=1.2,
                label=str(names[j]), color=PALETTE[rank % len(PALETTE)])
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.set_title("Sof yo'nalishli spilloverlar (NET = TO − FROM)", fontsize=11)
    ax.set_ylabel("NET (%)", fontsize=9)
    ax.legend(fontsize=7, ncol=min(6, k), loc="best")
    ax.grid(alpha=0.25)
    _xticks(ax, dates)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_to_from(dca, path, names=None, max_series=8):
    """TO (uzatilgan) va FROM (qabul qilingan) spilloverlar."""
    dates = _dates(dca)
    TO = dca["TO"]
    FROM = dca["FROM"]
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(TO.shape[1])]
    if TO.ndim == 3:
        TO, FROM = TO[:, :, 0], FROM[:, :, 0]
    k = TO.shape[1]
    order = np.argsort(-TO.mean(axis=0))
    nser = min(max_series, k)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=110,
                             sharex=True, sharey=True)
    for rank, j in enumerate(order[:nser]):
        axes[0].plot(np.arange(len(dates)), TO[:, j], lw=1.1,
                     label=str(names[j]), color=PALETTE[rank % len(PALETTE)])
        axes[1].plot(np.arange(len(dates)), FROM[:, j], lw=1.1,
                     label=str(names[j]), color=PALETTE[rank % len(PALETTE)])
    axes[0].set_title("Uzatilgan spilloverlar (TO)", fontsize=10)
    axes[1].set_title("Qabul qilingan spilloverlar (FROM)", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_ylabel("%", fontsize=9)
        _xticks(ax, dates)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_npt(dca, path, names=None, max_series=12):
    """Juftlik bo'yicha sof yo'nalishlar soni (NPT)."""
    if dca.get("NPT") is None:
        return None
    dates = _dates(dca)
    NPT = dca["NPT"]
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(NPT.shape[1])]
    if NPT.ndim == 3:
        NPT = NPT[:, :, 0]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    k = NPT.shape[1]
    order = np.argsort(-NPT.mean(axis=0))
    for rank, j in enumerate(order[:min(max_series, k)]):
        ax.plot(np.arange(len(dates)), NPT[:, j], lw=1.2,
                label=str(names[j]), color=PALETTE[rank % len(PALETTE)])
    ax.set_title("Juftlik bo'yicha sof yo'nalishlar soni (NPT)", fontsize=11)
    ax.set_ylabel("NPT", fontsize=9)
    ax.legend(fontsize=7, ncol=min(6, k), loc="best")
    ax.grid(alpha=0.25)
    _xticks(ax, dates)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_network(dca, path, threshold=1.0, names=None):
    """O'rtacha bog'liqlik to'ri — doira shaklida.

    Tugun o'lchami — TO, rangi — NET (qizil: uzatuvchi, yashil: qabulqiluvchi).
    Qirralar qalinligi — o'rtacha spillover kuchi.
    """
    CT = dca["CT"]
    if CT.ndim == 4:
        CT = CT[:, :, :, 0]
    if CT.ndim == 3:
        CTm = CT.mean(axis=2)
    else:
        CTm = CT * 100
    k = CTm.shape[0]
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(k)]
    tbl = ConnectednessTable(CTm / 100)
    TO = tbl["TO"]
    NET = tbl["NET"]

    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) - np.pi / 2
    xs, ys = np.cos(ang), np.sin(ang)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    # qirralar
    w = (CTm + CTm.T) / 2
    np.fill_diagonal(w, 0)
    if w.max() > 0:
        w = w / w.max()
    for i in range(k):
        for j in range(i + 1, k):
            strength = w[i, j] * 100
            if strength < threshold:
                continue
            lw = 0.6 + 4.5 * strength / max(1e-9, 100 * w.max())
            # bir tomonlama ustun bo'lsa — yo'nalish strelkasi
            di, dj = CTm[i, j], CTm[j, i]
            if abs(di - dj) < 1e-9:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#8899aa",
                        lw=lw, alpha=0.55, zorder=1)
            else:
                s, t = (j, i) if CTm[i, j] > CTm[j, i] else (i, j)
                ax.annotate("", xy=(xs[t], ys[t]), xytext=(xs[s], ys[s]),
                            zorder=1, alpha=0.6,
                            arrowprops=dict(arrowstyle="-|>", color="#c44e52",
                                           lw=lw, shrinkA=18, shrinkB=18,
                                           alpha=0.55))
    # tugunlar
    size = 300 + 2200 * (TO - TO.min()) / max(1e-9, TO.max() - TO.min())
    for i in range(k):
        color = "#c44e52" if NET[i] > 0 else "#55a868"
        ax.scatter(xs[i], ys[i], s=size[i], color=color, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ha = "center"
        dx, dy = 0.14 * xs[i], 0.14 * ys[i]
        if abs(xs[i]) > 0.7:
            ha = "left" if xs[i] > 0 else "right"
        ax.text(xs[i] + dx, ys[i] + dy, str(names[i]), fontsize=8.5,
                ha=ha, va="center", zorder=4, fontweight="bold")
    ax.set_title("O'rtacha bog'liqlik to'ri\n(qizil — sof uzatuvchi, "
                 "yashil — sof qabulqiluvchi)", fontsize=11)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
