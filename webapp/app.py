"""ConnectednessApproach — o'zbek tilidagi veb-ilova (Flask).

Ishga tushirish:  python3 webapp/app.py
Manzil:           http://127.0.0.1:5000
"""
import os
import sys
import time
import uuid
import shutil

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, abort)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from connectedness import (ConnectednessApproach, load_dataset,
                           list_datasets, DATASETS,
                           plot_tci, plot_net, plot_to_from, plot_network,
                           plot_npt, plot_bars, plot_band_tci)

BASE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE, "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "connectedness-uz-key"

MODELS = {
    "VAR": "VAR (klassik OLS)",
    "QVAR": "QVAR (kvantil VAR)",
    "LASSO": "LASSO VAR",
    "Ridge": "Ridge VAR",
    "Elastic": "Elastic Net VAR",
    "TVP-VAR": "TVP-VAR (vaqt o'zgaruvchan)",
}
APPROACHES = {
    "Time": "Vaqt sohasi (Diebold–Yilmaz)",
    "Frequency": "Chastota sohasi (Barunik–Krehlik)",
    "Joint": "Qo'shma spillover (Lastrapes–Wiesen)",
    "Extended Joint": "Kengaytirilgan qo'shma (Balcilar et al.)",
}
PRIORS = {
    "BayesPrior": "Bayes prior (trening namunasi)",
    "MinnesotaPrior": "Minnesota prior",
    "UninformativePrior": "Ma'lumotsiz prior",
}


def _clean_old_plots(max_age=3600):
    """Eski grafik fayllarni tozalash (1 soatdan eski)."""
    now = time.time()
    try:
        for fn in os.listdir(PLOT_DIR):
            p = os.path.join(PLOT_DIR, fn)
            if os.path.isfile(p) and now - os.path.getmtime(p) > max_age:
                os.remove(p)
    except OSError:
        pass


def _read_upload(file_storage):
    """Yuklangan CSV ni DataFrame ga o'qish."""
    df = pd.read_csv(file_storage)
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0].lower()
    date_col = None
    if first in ("date", "sana", "vaqt", "time", "day", "datum"):
        date_col = df.columns[0]
    else:
        try:
            d = pd.to_datetime(df[df.columns[0]], errors="raise")
            date_col = df.columns[0]
        except Exception:
            d = None
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df.columns = [c if c else f"V{i+1}" for i, c in enumerate(df.columns)]
    return df


def _table_html(table_df):
    """Bog'liqlik jadvalini chiroyli HTML ga aylantirish."""
    html = table_df.to_html(classes="ctable", border=0, na_rep="")
    return html


def _table_to_dict(tbl):
    rows = []
    for idx, row in tbl.iterrows():
        rows.append({"name": str(idx),
                     "values": [str(v) for v in row.tolist()],
                     "is_diag": False})
    return rows


@app.route("/")
def index():
    datasets = []
    for name in list_datasets():
        meta = DATASETS.get(name, {})
        try:
            ds = load_dataset(name)
            shape = f"{ds['data'].shape[0]}×{ds['data'].shape[1]}"
            period = (f"{ds['data'].index[0]:%Y-%m} — "
                      f"{ds['data'].index[-1]:%Y-%m}"
                      if hasattr(ds["data"].index[0], "year") else "—")
        except Exception:
            shape, period = "—", "—"
        datasets.append({"name": name,
                        "title": meta.get("title", name),
                        "desc": meta.get("desc", ""),
                        "paper": meta.get("paper", ""),
                        "shape": shape, "period": period})
    return render_template("index.html", datasets=datasets)


@app.route("/tahlil", methods=["GET"])
def analysis():
    datasets = [{"name": n, "title": DATASETS.get(n, {}).get("title", n)}
                for n in list_datasets()]
    return render_template("analysis.html", datasets=datasets,
                           models=MODELS, approaches=APPROACHES,
                           priors=PRIORS)


@app.route("/natija", methods=["POST"])
def results():
    _clean_old_plots()
    t_start = time.time()

    # ---------- 1. Ma'lumot
    source = request.form.get("source", "builtin")
    try:
        if source == "upload":
            f = request.files.get("csvfile")
            if f is None or f.filename == "":
                flash("CSV fayl tanlanmadi.")
                return redirect(url_for("analysis"))
            data = _read_upload(f)
            ds_title = os.path.splitext(f.filename)[0]
            ds_paper = ""
            ds_name = "upload"
        else:
            name = request.form.get("dataset")
            ds = load_dataset(name)
            data = ds["data"]
            ds_title, ds_paper, ds_name = ds["title"], ds["paper"], name
    except Exception as ex:
        flash(f"Ma'lumotni o'qishda xatolik: {ex}")
        return redirect(url_for("analysis"))

    if data.shape[0] < 20 or data.shape[1] < 2:
        flash("Ma'lumot juda kichik: kamida 20 qator va 2 o'zgaruvchi kerak.")
        return redirect(url_for("analysis"))

    # ---------- 2. Parametrlar
    def _int(key, default, lo, hi):
        try:
            v = int(request.form.get(key, default))
            return max(lo, min(hi, v))
        except (TypeError, ValueError):
            return default

    model = request.form.get("model", "VAR")
    approach = request.form.get("connectedness", "Time")
    if model not in MODELS:
        model = "VAR"
    if approach not in APPROACHES:
        approach = "Time"
    nlag = _int("nlag", 1, 1, 12)
    nfore = _int("nfore", 10, 1, 200)
    window = request.form.get("window_size", "").strip()
    window_size = _int("window_size", 0, 0, data.shape[0]) or None
    if window_size and window_size - nlag < 25:
        flash("Oyna hajmi juda kichik — kamida 25 koeffitsient qolishi kerak.")
        window_size = None
    corrected = request.form.get("corrected") == "on"
    generalized = request.form.get("generalized") == "on"

    VAR_config = {}
    Connectedness_config = {}
    if model == "QVAR":
        tau = request.form.get("tau", "0.5")
        try:
            tau = float(tau)
            tau = min(0.99, max(0.01, tau))
        except ValueError:
            tau = 0.5
        VAR_config["QVAR"] = {"tau": tau}
    elif model in ("LASSO", "Ridge", "Elastic"):
        VAR_config["ElasticNet"] = {"nfolds": 10, "loss": "mae",
                                    "delta_alpha": 0.1,
                                    "alpha": None if model == "Elastic"
                                    else (1 if model == "LASSO" else 0)}
    elif model == "TVP-VAR":
        prior = request.form.get("tvp_prior", "BayesPrior")
        if prior not in PRIORS:
            prior = "BayesPrior"
        tvp = {"prior": prior, "kappa1": 0.99, "kappa2": 0.99, "gamma": 0.1}
        for kk in ("kappa1", "kappa2", "gamma"):
            try:
                v = float(request.form.get(kk, tvp[kk]))
                tvp[kk] = min(0.999, max(0.01, v)) if kk != "gamma" \
                    else max(0.0001, v)
            except ValueError:
                pass
        VAR_config["TVPVAR"] = tvp
    if approach == "Time":
        Connectedness_config["TimeConnectedness"] = {
            "generalized": generalized}
    elif approach == "Frequency":
        band = request.form.get("band", "pi2")
        partition = {"pi2": [np.pi, np.pi / 2, 0],
                     "pi4": [np.pi, np.pi / 4, 0],
                     "pi8": [np.pi, np.pi / 8, 0],
                     "pi24": [np.pi, np.pi / 2, np.pi / 4, 0]}[band]
        Connectedness_config["FrequencyConnectedness"] = {
            "partition": partition,
            "generalized": True,
            "scenario": request.form.get("scenario", "ABS"),
        }

    # ---------- 3. Hisoblash
    try:
        dca = ConnectednessApproach(
            data, nlag=nlag, nfore=nfore, window_size=window_size,
            corrected=corrected, model=model, connectedness=approach,
            VAR_config=VAR_config,
            Connectedness_config=Connectedness_config)
    except Exception as ex:
        flash(f"Hisoblashda xatolik: {ex}")
        return redirect(url_for("analysis"))
    elapsed = time.time() - t_start

    # ---------- 4. Grafiklar
    uid = uuid.uuid4().hex[:10]
    plots = {}
    def _png(key, fn, func, *a, **kw):
        p = os.path.join(PLOT_DIR, f"{uid}_{fn}.png")
        try:
            func(*a, path=p, **kw)
            plots[key] = f"plots/{uid}_{fn}.png"
        except Exception as ex:
            app.logger.warning("Grafik xatosi (%s): %s", key, ex)

    freq = approach == "Frequency"
    tci_arr = np.asarray(dca["TCI"])
    t_n = int(tci_arr.shape[0])
    dynamic = t_n > 1

    if approach in ("Time", "Frequency"):
        if dynamic:
            _png("tci", "tci", plot_tci, dca,
                 title="Chastota bo'yicha TCI" if freq else
                 "Umumiy bog'liqlik indeksi (TCI)")
            _png("net", "net", plot_net, dca)
            _png("tofrom", "tofrom", plot_to_from, dca)
            _png("npt", "npt", plot_npt, dca)
        else:
            _png("bars", "bars", plot_bars, dca)
            if freq:
                _png("bandtci", "bandtci", plot_band_tci, dca)
    else:
        if dynamic:
            _png("tci", "tci", plot_tci, dca,
                 title="Qo'shma spillover indeksi (jTCI)")
            _png("net", "net", plot_net, dca)
        else:
            _png("bars", "bars", plot_bars, dca)
    _png("network", "netgraph", plot_network, dca)

    # ---------- 5. Natija ma'lumotlari
    tables = []          # [(nom, TABLE_df)]
    if freq:
        for pname, tbl in dca.get("TABLES", {}).items():
            tables.append((f"{pname} davri", tbl))
    else:
        tables.append(("Umumiy jadval", dca["TABLE"]))

    tci = np.asarray(dca["TCI"]).ravel()
    summary = {
        "tci_mean": float(np.mean(tci)),
        "tci_min": float(np.min(tci)),
        "tci_max": float(np.max(tci)),
        "tci_first": float(tci[0]),
        "tci_last": float(tci[-1]),
        "n_t": int(tci.shape[0]),
    }
    cfg = dca["config"]
    names = cfg.get("names", [])
    net_last = None
    if dca.get("NET") is not None:
        NET = np.asarray(dca["NET"])
        if NET.ndim == 3:                 # chastota yondashuvi — umumiy band
            NET = NET[:, :, 0]
        if NET.ndim == 2 and NET.shape[0] > 0:
            net_last = list(zip(names, NET[-1].tolist()))

    params_desc = {
        "Ma'lumotlar": ds_title,
        "Model": MODELS[model],
        "Yondashuv": APPROACHES[approach],
        "Lag soni (nlag)": nlag,
        "Bashorat ufqi (nfore)": nfore,
        "Oyna hajmi": window_size or "to'liq namuna",
        "TCI tuzatilgan": "ha" if corrected else "yo'q",
    }
    if freq:
        params_desc["Chastota bo'linishi"] = request.form.get("band", "pi2")
        params_desc["Ssenariy"] = request.form.get("scenario", "ABS")
    if model == "QVAR":
        params_desc["Kvantil (tau)"] = VAR_config.get("QVAR", {}).get("tau")
    if model == "TVP-VAR":
        tvp = VAR_config.get("TVPVAR", {})
        params_desc["Prior"] = PRIORS[tvp.get("prior", "BayesPrior")]
        params_desc["kappa1 / kappa2"] = f"{tvp.get('kappa1')} / {tvp.get('kappa2')}"
    if approach == "Time":
        params_desc["Umumlashtirilgan FEVD"] = \
            "ha" if Connectedness_config.get("TimeConnectedness", {}).get("generalized") else "yo'q"

    return render_template(
        "results.html",
        tables=[(n, _table_html(t)) for n, t in tables],
        plots=plots, summary=summary, net_last=net_last,
        params=params_desc, elapsed=round(elapsed, 2),
        ds_title=ds_title, ds_paper=ds_paper, approach=approach, freq=freq)


@app.route("/uslub")
def method():
    return render_template("method.html")


@app.route("/haqida")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  ConnectednessApproach veb-ilovasi")
    print(f"  Manzil: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
