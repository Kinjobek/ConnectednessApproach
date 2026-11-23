"""Grafiklar — R paketidagi Plot* funksiyalarining matplotlib portlari.

Barcha yorliqlar o'zbek tilida. Har funksiya PNG fayl yo'lini qaytaradi.
Dizayn: toza oqlar, legenda ma'lumot ustiga tushmaydi, statik tahlil
uchun ustunli (bar) grafiklar, dinamik uchun chiziqli grafiklar.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from .measures import ConnectednessTable

__all__ = ["plot_tci", "plot_net", "plot_to_from", "plot_network",
           "plot_npt", "plot_bars", "plot_band_tci"]

PALETTE = ["#2456e6", "#d64545", "#0e9f6e", "#8045d6", "#e08214",
           "#5aa8d6", "#d65998", "#14b8c4", "#a3a324", "#8a8f9e",
           "#1b9e77", "#d95f02"]

C_TO = "#2456e6"       # uzatilgan — ko'k
C_FROM = "#e08214"     # qabul qilingan — to'q sariq
C_NET_POS = "#c4432b"  # sof uzatuvchi
C_NET_NEG = "#2e9e6b"  # sof qabulqiluvchi


def _style(ax):
    """Umumiy toza ko'rinish."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9cede")
    ax.grid(axis="y", color="#e6e9f2", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4b5670", labelsize=9)


def _dates(dca):
    cfg = dca.get("config", {})
    dates = cfg.get("dates")
    if dates is None:
        dates = [str(i + 1) for i in range(dca["TCI"].shape[0])]
    d = pd.to_datetime(dates, errors="coerce")
    if not d.isna().any():
        return d
    return dates


def _xticks(ax, dates, n=7):
    n_t = len(dates)
    if n_t <= 1:
        return
    idx = np.linspace(0, n_t - 1, min(n, n_t)).astype(int)
    if isinstance(dates, pd.DatetimeIndex):
        labels = [d.strftime("%Y-%m") for d in dates[idx]]
        if len(dates) > 5 * 365:
            labels = [d.strftime("%Y") for d in dates[idx]]
    else:
        labels = [str(dates[i])[:10] for i in idx]
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)


def _freq(dca):
    """Bandlar uchun yordamchi — TCI (t, band)."""
    tci = np.asarray(dca["TCI"])
    return tci[:, None] if tci.ndim == 1 else tci


# ================================================================ TCI
def plot_tci(dca, path, title=None):
    """Umumiy bog'liqlik indeksi (TCI) dinamikasi — faqat t > 1 uchun."""
    dates = _dates(dca)
    TCI = _freq(dca)
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=140)
    x = np.arange(len(dates))
    if TCI.shape[1] == 1:
        ax.plot(x, TCI[:, 0], color=C_TO, lw=1.7)
        ax.fill_between(x, TCI[:, 0], alpha=0.10, color=C_TO)
        ax.set_title(title or "Umumiy bog'liqlik indeksi (TCI)",
                     fontsize=12, pad=10, color="#1c2333")
    else:
        names = dca["config"].get("period_names",
                                  [f"Band {i+1}" for i in range(TCI.shape[1])])
        for j in range(TCI.shape[1]):
            ax.plot(x, TCI[:, j], lw=1.6, label=str(names[j]),
                    color=PALETTE[j % len(PALETTE)],
                    alpha=1.0 if j == 0 else 0.75)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4,
                  frameon=False, fontsize=8.5, handlelength=1.6)
        ax.set_title(title or "Chastota bo'yicha TCI", fontsize=12,
                     pad=28, color="#1c2333")
    ax.set_ylabel("TCI (%)", fontsize=9.5, color="#4b5670")
    _style(ax)
    _xticks(ax, dates)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================ NET
def plot_net(dca, path, names=None, max_series=8):
    """Sof yo'nalishli spilloverlar (dinamik, t > 1)."""
    dates = _dates(dca)
    NET = np.asarray(dca["NET"])
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(NET.shape[1])]
    if NET.ndim == 3:
        NET = NET[:, :, 0]
    k = NET.shape[1]
    order = np.argsort(-np.abs(NET).mean(axis=0))
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=140)
    x = np.arange(len(dates))
    for rank, j in enumerate(order[:min(max_series, k)]):
        ax.plot(x, NET[:, j], lw=1.5, label=str(names[j]),
                color=PALETTE[rank % len(PALETTE)])
    ax.axhline(0, color="#3b4560", lw=0.9, alpha=0.6)
    ax.set_title("Sof yo'nalishli spilloverlar (NET = TO − FROM)",
                 fontsize=12, pad=26, color="#1c2333")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=8,
              frameon=False, fontsize=8, handlelength=1.5)
    ax.set_ylabel("NET (%)", fontsize=9.5, color="#4b5670")
    _style(ax)
    _xticks(ax, dates)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================ TO/FROM
def plot_to_from(dca, path, names=None, max_series=6):
    """Uzatilgan va qabul qilingan spilloverlar (dinamik, t > 1)."""
    dates = _dates(dca)
    TO = np.asarray(dca["TO"])
    FROM = np.asarray(dca["FROM"])
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(TO.shape[1])]
    if TO.ndim == 3:
        TO, FROM = TO[:, :, 0], FROM[:, :, 0]
    k = TO.shape[1]
    order = np.argsort(-TO.mean(axis=0))[:min(max_series, k)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), dpi=140,
                             sharex=True, sharey=True)
    x = np.arange(len(dates))
    for rank, j in enumerate(order):
        axes[0].plot(x, TO[:, j], lw=1.4, color=PALETTE[rank % len(PALETTE)],
                     alpha=0.9)
        axes[1].plot(x, FROM[:, j], lw=1.4, color=PALETTE[rank % len(PALETTE)],
                     alpha=0.9)
    axes[0].set_title("Uzatilgan (TO)", fontsize=11, color="#1c2333")
    axes[1].set_title("Qabul qilingan (FROM)", fontsize=11, color="#1c2333")
    handles = [Line2D([0], [0], color=PALETTE[r % len(PALETTE)], lw=1.6,
                      label=str(names[j]))
               for r, j in enumerate(order)]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(order)),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    for ax in axes:
        _style(ax)
        _xticks(ax, dates)
        ax.set_ylabel("%", fontsize=9, color="#4b5670")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================ NPT
def plot_npt(dca, path, names=None, max_series=8):
    """Juftlik bo'yicha sof yo'nalishlar soni (dinamik, t > 1)."""
    if dca.get("NPT") is None:
        return None
    dates = _dates(dca)
    NPT = np.asarray(dca["NPT"])
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(NPT.shape[1])]
    if NPT.ndim == 3:
        NPT = NPT[:, :, 0]
    k = NPT.shape[1]
    order = np.argsort(-NPT.mean(axis=0))
    fig, ax = plt.subplots(figsize=(9.5, 4), dpi=140)
    x = np.arange(len(dates))
    for rank, j in enumerate(order[:min(max_series, k)]):
        ax.plot(x, NPT[:, j], lw=1.5, label=str(names[j]),
                color=PALETTE[rank % len(PALETTE)])
    ax.set_title("Juftlik bo'yicha sof yo'nalishlar soni (NPT)",
                 fontsize=12, pad=26, color="#1c2333")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=8,
              frameon=False, fontsize=8, handlelength=1.5)
    ax.set_ylabel("NPT (soni)", fontsize=9.5, color="#4b5670")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax)
    _xticks(ax, dates)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================ statik ustunlar
def plot_bars(dca, path, names=None):
    """Statik natija uchun: har o'zgaruvchi bo'yicha TO va FROM ustunlari."""
    TO = np.asarray(dca["TO"])
    FROM = np.asarray(dca["FROM"])
    NET = np.asarray(dca["NET"])
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(TO.shape[1])]
    if TO.ndim == 2:
        TO, FROM, NET = TO[-1], FROM[-1], NET[-1]
    k = len(names)
    order = np.argsort(-TO)
    fig, ax = plt.subplots(figsize=(9.5, max(2.6, 0.55 * k + 1.2)), dpi=140)
    y = np.arange(k)
    h = 0.38
    to = TO[order]
    fr = FROM[order]
    net = NET[order]
    lbl = [str(names[i]) for i in order]
    ax.barh(y + h / 2, to, height=h, color=C_TO, label="TO — uzatilgan")
    ax.barh(y - h / 2, fr, height=h, color=C_FROM, label="FROM — qabul qilingan")
    for i in range(k):
        ax.text(to[i] + to.max() * 0.015, y[i] + h / 2, f"{to[i]:.1f}",
                va="center", fontsize=8, color=C_TO)
        ax.text(fr[i] + to.max() * 0.015, y[i] - h / 2, f"{fr[i]:.1f}",
                va="center", fontsize=8, color="#b96a10")
        # NET qiymati o'ng tomda
        ax.text(to.max() * 1.14, y[i],
                f"NET {net[i]:+.1f}", va="center", ha="left", fontsize=8,
                color=C_NET_POS if net[i] > 0 else C_NET_NEG)
    ax.set_yticks(y)
    ax.set_yticklabels(lbl, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Spillover (%)", fontsize=9.5, color="#4b5670")
    ax.set_xlim(0, max(to.max(), fr.max()) * 1.24)
    ax.set_title("O'zgaruvchilar bo'yicha spilloverlar (bitta nuqta, %)",
                 fontsize=12, pad=10, color="#1c2333")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, fontsize=8.5)
    _style(ax)
    ax.grid(axis="x", color="#e6e9f2", lw=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_band_tci(dca, path):
    """Statik chastota tahlili uchun: davrlar bo'yicha TCI ustunlari."""
    TCI = _freq(dca)[0]                      # (bandlar,)
    names = dca["config"].get("period_names",
                              [f"Band {i+1}" for i in range(len(TCI))])
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=140)
    x = np.arange(len(TCI))
    colors = ["#2456e6"] + [PALETTE[(i + 1) % len(PALETTE)]
                            for i in range(len(TCI) - 1)]
    bars = ax.bar(x, TCI, width=0.62, color=colors)
    for b, v in zip(bars, TCI):
        ax.text(b.get_x() + b.get_width() / 2, v + TCI.max() * 0.02,
                f"{v:.1f}%", ha="center", fontsize=9, color="#1c2333")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in names], fontsize=9.5)
    ax.set_ylabel("TCI (%)", fontsize=9.5, color="#4b5670")
    ax.set_ylim(0, TCI.max() * 1.14)
    ax.set_title("Chastota davrlari bo'yicha bog'liqlik", fontsize=12,
                 pad=10, color="#1c2333")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ================================================================ to' grafigi
def plot_network(dca, path, names=None, max_edges=None):
    """O'rtacha bog'liqlik to'ri — egri strelkali, faqat eng kuchli qirralar.

    Tugun o'lchami — TO, rangi — NET (qizil: sof uzatuvchi,
    yashil: sof qabulqiluvchi). Qirralar — juftlik spillover kuchi.
    """
    CT = dca["CT"]
    if CT.ndim == 4:
        CT = CT[:, :, :, 0]
    if CT.ndim == 3:
        CTm = CT.mean(axis=2)
    else:
        CTm = np.asarray(CT)
    k = CTm.shape[0]
    names = names or dca["config"].get("names") or \
        [str(i + 1) for i in range(k)]
    tbl = ConnectednessTable(CTm)
    TO, NET = tbl["TO"], tbl["NET"]

    # qirralarni saralash (i<j juftlik kuchi = ikki yo'nalish yig'indisi)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append((CTm[i, j] + CTm[j, i], i, j))
    pairs.sort(reverse=True)
    if max_edges is None:
        max_edges = max(6, 2 * k)
    pairs = [p for p in pairs if p[0] >= 2.0][:max_edges]
    smax = pairs[0][0] if pairs else 1.0

    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) - np.pi / 2
    xs, ys = np.cos(ang), np.sin(ang)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=140)
    for s, i, j in pairs:
        # yo'nalish: kattaroq CTm qiymati manba
        if CTm[i, j] >= CTm[j, i]:
            src, dst = i, j
        else:
            src, dst = j, i
        lw = 0.7 + 3.8 * s / smax
        alpha = min(0.9, 0.25 + 0.65 * s / smax)
        rad = 0.16 if (i + j) % 2 == 0 else -0.16
        arrow = FancyArrowPatch(
            (xs[src], ys[src]), (xs[dst], ys[dst]),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_width=3.2,head_length=6.5",
            mutation_scale=8, lw=lw, color="#c4432b", alpha=alpha,
            shrinkA=16, shrinkB=16, zorder=1)
        ax.add_patch(arrow)

    size = 240 + 1900 * (TO - TO.min()) / max(1e-9, TO.max() - TO.min())
    for i in range(k):
        color = C_NET_POS if NET[i] > 0 else C_NET_NEG
        ax.scatter(xs[i], ys[i], s=size[i], color=color, zorder=3,
                   edgecolors="white", linewidths=1.6)
    # yorliqlar
    for i in range(k):
        lx, ly = 1.22 * xs[i], 1.22 * ys[i]
        ha = "center"
        if abs(xs[i]) > 0.6:
            ha = "left" if xs[i] > 0 else "right"
        va = "bottom" if ys[i] > 0.6 else ("top" if ys[i] < -0.6 else "center")
        ax.text(lx, ly, str(names[i]), fontsize=9.5, ha=ha, va=va,
                zorder=4, fontweight="bold", color="#1c2333")

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NET_POS,
              markersize=11, label="sof uzatuvchi (NET > 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NET_NEG,
              markersize=11, label="sof qabulqiluvchi (NET < 0)"),
        Line2D([0], [0], color="#c4432b", lw=2.4,
              label="qirralar — spillover kuchi"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("O'rtacha bog'liqlik to'ri", fontsize=13, pad=14,
                 color="#1c2333")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.42, 1.38)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
