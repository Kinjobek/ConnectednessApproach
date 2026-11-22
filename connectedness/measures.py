"""Bog'liqlik o'lchovlari: Diebold–Yilmaz va boshqa yondashuvlar.

R ConnectednessApproach paketidagi ConnectednessTable, TimeConnectedness,
FrequencyConnectedness, JointConnectedness, ExtendedJointConnectedness va
umumiy ConnectednessApproach wrapper portlari.
"""
import numpy as np
import pandas as pd

from .core import FEVD as _FEVD, Wold, get_partition
from .models import (VAR, QVAR, ElasticNetVAR, TVPVAR,
                     BayesPrior, MinnesotaPrior, UninformativePrior)

__all__ = ["ConnectednessTable", "TimeConnectedness", "FrequencyConnectedness",
           "JointConnectedness", "ExtendedJointConnectedness", "ConnectednessApproach"]


def _fmt(x, digit=2):
    """R `format(round(x, digit), nsmall=digit)` ko'rinishi."""
    return f"{round(float(x), digit):.{digit}f}"


def ConnectednessTable(FEVD, digit=2, names=None):
    """Standart bog'liqlik jadvali. R `ConnectednessTable()` porti.

    FEVD : (k, k) yoki (k, k, t) — [qator = qabulqachi, ustun = manba].
    """
    fevd = np.asarray(FEVD, dtype=float)
    if fevd.ndim < 2:
        raise ValueError("FEVD kamida 2 o'lchamli matritsa bo'lishi kerak")
    if fevd.ndim == 3:
        CT = fevd.mean(axis=2) * 100
    else:
        CT = fevd * 100
    k = CT.shape[0]
    if names is None:
        names = [str(i + 1) for i in range(k)]
    names = [str(n) for n in names]

    own = np.diag(np.diag(CT))
    TO = (CT - own).sum(axis=0)          # ustun bo'yicha — uzatilgan
    FROM = (CT - own).sum(axis=1)        # qator bo'yicha — qabul qilingan
    NET = TO - FROM
    TCI = TO.mean()
    cTCI = TCI * k / (k - 1) if k > 1 else TCI
    NPDC = CT - CT.T
    NPT = (NPDC < 0).sum(axis=1)
    INFLUENCE = 100 * np.abs(NPDC / (CT + CT.T))
    PCI = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            PCI[i, j] = (200 * (CT[i, j] + CT[j, i]) /
                         (CT[i, i] + CT[i, j] + CT[j, i] + CT[j, j]))

    # TABLE — R formatidagi matnli jadval
    table = []
    for i in range(k):
        table.append([_fmt(v, digit) for v in CT[i]] + [_fmt(FROM[i], digit)])
    table.append([_fmt(v, digit) for v in TO] + [_fmt(TO.sum(), digit)])
    table.append([_fmt(v, digit) for v in CT.sum(axis=0)] + ["cTCI/TCI"])
    table.append([_fmt(v, digit) for v in NET] +
                 [f"{_fmt(cTCI, digit)}/{_fmt(TCI, digit)}"])
    table.append([_fmt(v, digit) for v in NPT] + [""])
    TABLE = pd.DataFrame(table,
                         index=list(names) + ["TO", "Inc.Own", "NET", "NPT"],
                         columns=names + ["FROM"])
    return {"FEVD": CT, "TCI": TCI, "cTCI": cTCI, "PCI": PCI,
            "TO": TO, "FROM": FROM, "NET": NET, "NPDC": NPDC,
            "TABLE": TABLE, "NPT": NPT, "INFLUENCE": INFLUENCE}


def _as_3d(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 2:
        return a[:, :, None]
    return a


def TimeConnectedness(Phi=None, Sigma=None, nfore=10, generalized=True,
                      corrected=False, FEVD=None, names=None, dates=None):
    """Diebold–Yilmaz (2009, 2012) vaqt sohasi bog'liqlik yondashuvi.

    R `TimeConnectedness()` porti. Natija dict:
    TABLE, CT (k,k,t), TCI (t,), TO/FROM/NET/NPT (t,k), NPDC/PCI/INFLUENCE (k,k,t).
    """
    if FEVD is None:
        if Phi is None or Sigma is None:
            raise ValueError("Phi va Sigma yoki FEVD berilishi kerak")
        Phi3 = _as_3d(Phi)
        Sig3 = _as_3d(Sigma)
        t = Sig3.shape[2]
        if names is None:
            names = [str(i + 1) for i in range(Sig3.shape[0])]
        date_labels = ([str(d) for d in dates[-t:]] if dates is not None
                       else [str(i + 1) for i in range(t)])
        CT = np.zeros((Sig3.shape[0], Sig3.shape[0], t))
        for i in range(t):
            CT[:, :, i] = _FEVD(Phi=Phi3[:, :, i], Sigma=Sig3[:, :, i],
                               nfore=nfore, type="time",
                               generalized=generalized)["FEVD"]
    else:
        CT = _as_3d(FEVD)
        t = CT.shape[2]
        k = CT.shape[0]
        names = names or [str(i + 1) for i in range(k)]
        date_labels = ([str(d) for d in dates[-t:]] if dates is not None
                       else [str(i + 1) for i in range(t)])

    k = CT.shape[0]
    TCI = np.zeros(t)
    TO = np.zeros((t, k))
    FROM = np.zeros((t, k))
    NET = np.zeros((t, k))
    NPT = np.zeros((t, k))
    NPDC = np.zeros((k, k, t))
    PCI = np.zeros((k, k, t))
    INFLUENCE = np.zeros((k, k, t))
    for i in range(t):
        ct = ConnectednessTable(CT[:, :, i], names=names)
        TO[i] = ct["TO"]
        FROM[i] = ct["FROM"]
        NET[i] = ct["NET"]
        NPT[i] = ct["NPT"]
        NPDC[:, :, i] = ct["NPDC"]
        INFLUENCE[:, :, i] = ct["INFLUENCE"]
        PCI[:, :, i] = ct["PCI"]
        TCI[i] = ct["cTCI"] if corrected else ct["TCI"]
    TABLE = ConnectednessTable(CT, names=names)["TABLE"]
    return {"TABLE": TABLE, "CT": CT, "TCI": TCI, "TO": TO, "FROM": FROM,
            "NET": NET, "NPT": NPT, "NPDC": NPDC, "PCI": PCI,
            "INFLUENCE": INFLUENCE,
            "config": {"nfore": nfore, "approach": "Time",
                       "generalized": generalized, "corrected": corrected,
                       "names": [str(n) for n in names],
                       "dates": date_labels}}


def FrequencyConnectedness(Phi, Sigma, nfore=100, partition=None,
                           generalized=True, orth=False, scenario="ABS",
                           corrected=False, names=None, dates=None):
    """Barunik–Krehlik (2018) chastota bog'liqlik yondashuvi.

    R `FrequencyConnectedness()` porti.
    """
    if partition is None:
        partition = [np.pi, np.pi / 2, 0]
    partition = list(partition)
    if nfore <= 0:
        raise ValueError("nfore musbat butun son bo'lishi kerak")
    Phi3 = _as_3d(Phi)
    Sig3 = _as_3d(Sigma)
    k, _, t = Sig3.shape
    if names is None:
        names = [str(i + 1) for i in range(k)]
    names = [str(n) for n in names]

    with np.errstate(divide="ignore"):
        periods = np.round(np.pi / np.array(partition, dtype=float))

    def _p(p):
        s = f"{p:g}"
        return "∞" if s in ("inf", "Inf") else s

    period_names = ["Total"]
    for i in range(len(periods) - 1):
        period_names.append(f"{_p(periods[i])}-{_p(periods[i + 1])}")

    date_labels = ([str(d) for d in dates[-t:]] if dates is not None
                   else [str(i + 1) for i in range(t)])
    interval = len(period_names)
    nb = interval - 1

    new_p = get_partition(partition, nfore)
    rng = np.array(sorted(set(np.concatenate(new_p).tolist())))

    TCI = np.zeros((t, interval))
    CT = np.zeros((k, k, t, interval))
    NPDC = np.zeros((k, k, t, interval))
    TO = np.zeros((t, k, interval))
    FROM = np.zeros((t, k, interval))
    NET = np.zeros((t, k, interval))
    NPT = np.zeros((t, k, nb))
    PCI = np.zeros((k, k, t, nb))
    INFLUENCE = np.zeros((k, k, t, nb))

    for i in range(t):
        decomp = _FEVD(Phi=Phi3[:, :, i], Sigma=Sig3[:, :, i], nfore=nfore,
                      type="frequency", generalized=generalized, orth=orth,
                      range_=rng)["FEVD"]                      # (nfore+1, k, k)
        tables = [decomp[band].sum(axis=0) for band in new_p]
        for b in range(nb):
            if scenario == "ABS":
                tb = tables[b]
            elif scenario == "WTH":
                tb = tables[b] / tables[b].sum() * k
            else:
                raise ValueError("scenario 'ABS' yoki 'WTH' bo'lishi kerak")
            dca = ConnectednessTable(tb, names=names)
            CT[:, :, i, b + 1] = dca["FEVD"]
            TO[i, :, b + 1] = dca["TO"]
            FROM[i, :, b + 1] = dca["FROM"]
            NET[i, :, b + 1] = dca["NET"]
            NPDC[:, :, i, b + 1] = dca["NPDC"]
            PCI[:, :, i, b] = dca["PCI"]
            INFLUENCE[:, :, i, b] = dca["INFLUENCE"]
            NPT[i, :, b] = dca["NPT"]
            TCI[i, b + 1] = dca["cTCI"] if corrected else dca["TCI"]

    CT[:, :, :, 0] = CT.sum(axis=3)
    TCI[:, 0] = TCI.sum(axis=1)
    TO[:, :, 0] = TO.sum(axis=2)
    FROM[:, :, 0] = FROM.sum(axis=2)
    NET[:, :, 0] = NET.sum(axis=2)
    NPDC[:, :, :, 0] = NPDC.sum(axis=3)

    TABLES = {}
    for j in range(interval):
        tbl = ConnectednessTable(CT[:, :, :, j] / 100, names=names)["TABLE"]
        TABLES[period_names[j]] = tbl

    return {"TABLE": TABLES[period_names[0]], "TABLES": TABLES,
            "CT": CT / 100, "TCI": TCI, "TO": TO, "FROM": FROM, "NET": NET,
            "NPT": NPT, "NPDC": NPDC, "PCI": PCI, "INFLUENCE": INFLUENCE,
            "config": {"partition": partition, "nfore": nfore,
                       "generalized": generalized, "orth": orth,
                       "scenario": scenario, "corrected": corrected,
                       "approach": "Frequency", "period_names": period_names,
                       "names": names, "dates": date_labels}}


def JointConnectedness(Phi, Sigma, nfore, names=None, dates=None):
    """Lastrapes–Wiesen (2021) qo'shma spillover indeksi.

    R `JointConnectedness()` porti.
    """
    if nfore <= 0:
        raise ValueError("nfore musbat butun son bo'lishi kerak")
    Phi3 = _as_3d(Phi)
    Sig3 = _as_3d(Sigma)
    k, _, t = Sig3.shape
    if names is None:
        names = [str(i + 1) for i in range(k)]
    names = [str(n) for n in names]
    date_labels = ([str(d) for d in dates[-t:]] if dates is not None
                   else [str(i + 1) for i in range(t)])

    TCI = np.zeros(t)
    NET = np.zeros((t, k))
    FROM = np.zeros((t, k))
    TO = np.zeros((t, k))
    CT = np.zeros((k, k, t))
    CT_ = np.zeros((t, k + 3, k + 1))

    I_K = np.eye(k)
    for ij in range(t):
        Phi_i = Phi3[:, :, ij]
        Sig_i = Sig3[:, :, ij]
        gSOT = 100 * _FEVD(Phi=Phi_i, Sigma=Sig_i, nfore=nfore, type="time",
                          generalized=True)["FEVD"]
        off = gSOT - np.diag(np.diag(gSOT))
        gSOI = off.sum(axis=1).mean()

        A = Wold(Phi_i, nfore)                       # (nfore+1, k, k)
        Xi = np.zeros((k, k))
        for h in range(1, nfore + 1):
            Xi += A[h - 1] @ Sig_i @ A[h - 1].T   # R: A[,,h] = Psi_{h-1}
        M = np.array([I_K[:, [i for i in range(k) if i != m]]
                      for m in range(k)]).transpose(1, 2, 0)   # (k, k-1, k)

        S_jnt = np.zeros(k)
        for i in range(k):
            Mi = M[:, :, i]
            num = 0.0
            for h in range(1, nfore + 1):
                Ah = A[h - 1]                     # R: A[,,h] = Psi_{h-1}
                v = (I_K[i] @ Ah @ Sig_i @ Mi @
                     np.linalg.pinv(Mi.T @ Sig_i @ Mi) @ Mi.T @ Sig_i @
                     Ah.T @ I_K[i])
                num += v
            S_jnt[i] = 100 * num / Xi[i, i]
        jSOI = S_jnt.mean()
        lam = gSOI / jSOI
        TCI[ij] = jSOI
        FROM[ij] = S_jnt
        TO[ij] = off.sum(axis=0) / lam
        NET[ij] = TO[ij] - FROM[ij]
        CT[:, :, ij] = gSOT / 100
        block = np.zeros((k + 3, k + 1))
        block[:k, :k] = gSOT
        block[:k, k] = FROM[ij]
        block[k, :k] = TO[ij]
        block[k, k] = TO[ij].sum()
        block[k + 1, :k] = gSOT.sum(axis=0)
        block[k + 2, :k] = NET[ij]
        block[k + 2, k] = TCI[ij]
        CT_[ij] = block

    avg = CT_.mean(axis=0)
    table = [[_fmt(v) for v in row] for row in avg]
    table[k + 1][k] = "TCI"
    TABLE = pd.DataFrame(table,
                         index=list(names) + ["TO", "Inc.Own", "NET"],
                         columns=names + ["FROM"])
    return {"TABLE": TABLE, "CT": CT, "TCI": TCI, "TO": TO, "FROM": FROM,
            "NET": NET, "NPDC": None, "NPT": None, "PCI": None,
            "INFLUENCE": None,
            "config": {"nfore": nfore, "approach": "Joint",
                       "generalized": True, "corrected": False,
                       "names": names, "dates": date_labels}}


def ExtendedJointConnectedness(Phi, Sigma, nfore, names=None, dates=None):
    """Balcilar et al. (2021) kengaytirilgan qo'shma yondashuv.

    R `ExtendedJointConnectedness()` porti.
    """
    if nfore <= 0:
        raise ValueError("nfore musbat butun son bo'lishi kerak")
    Phi3 = _as_3d(Phi)
    Sig3 = _as_3d(Sigma)
    k, _, t = Sig3.shape
    if names is None:
        names = [str(i + 1) for i in range(k)]
    names = [str(n) for n in names]
    date_labels = ([str(d) for d in dates[-t:]] if dates is not None
                   else [str(i + 1) for i in range(t)])

    TCI = np.zeros(t)
    NPT = np.zeros((t, k))
    NET = np.zeros((t, k))
    FROM = np.zeros((t, k))
    TO = np.zeros((t, k))
    CT = np.zeros((k, k, t))
    PCI = np.zeros((k, k, t))
    NPDC = np.zeros((k, k, t))
    INFLUENCE = np.zeros((k, k, t))

    I_K = np.eye(k)
    for ij in range(t):
        Phi_i = Phi3[:, :, ij]
        Sig_i = Sig3[:, :, ij]
        gSOT = 100 * _FEVD(Phi=Phi_i, Sigma=Sig_i, nfore=nfore, type="time",
                          generalized=True)["FEVD"]
        A = Wold(Phi_i, nfore)
        Xi = np.zeros((k, k))
        for h in range(1, nfore + 1):
            Xi += A[h - 1] @ Sig_i @ A[h - 1].T   # R: A[,,h] = Psi_{h-1}
        M = np.array([I_K[:, [i for i in range(k) if i != m]]
                      for m in range(k)]).transpose(1, 2, 0)

        S_jnt = np.zeros(k)
        for i in range(k):
            Mi = M[:, :, i]
            num = 0.0
            for h in range(1, nfore + 1):
                Ah = A[h - 1]                     # R: A[,,h] = Psi_{h-1}
                num += (I_K[i] @ Ah @ Sig_i @ Mi @
                        np.linalg.pinv(Mi.T @ Sig_i @ Mi) @ Mi.T @ Sig_i @
                        Ah.T @ I_K[i])
            S_jnt[i] = 100 * num / Xi[i, i]

        gSOT_diag = gSOT - np.diag(np.diag(gSOT))
        lam = S_jnt / gSOT_diag.sum(axis=1)
        jSOT = gSOT * lam[:, None]
        jSOT_diag = jSOT - np.diag(np.diag(jSOT))
        from_jnt = jSOT_diag.sum(axis=1)
        to_jnt = jSOT_diag.sum(axis=0)
        jSOT_filled = jSOT.copy()
        np.fill_diagonal(jSOT_filled, 100 - from_jnt)

        dca = ConnectednessTable(jSOT_filled / 100, names=names)
        CT[:, :, ij] = dca["FEVD"]
        TO[ij] = dca["TO"]
        FROM[ij] = dca["FROM"]
        NET[ij] = dca["NET"]
        NPDC[:, :, ij] = dca["NPDC"]
        TCI[ij] = dca["TCI"]
        PCI[:, :, ij] = dca["PCI"]
        NPT[ij] = dca["NPT"]
        INFLUENCE[:, :, ij] = dca["INFLUENCE"]

    TABLE = ConnectednessTable(CT / 100, names=names)["TABLE"]
    return {"TABLE": TABLE, "CT": CT / 100, "TCI": TCI, "TO": TO,
            "FROM": FROM, "NET": NET, "NPT": NPT, "NPDC": NPDC, "PCI": PCI,
            "INFLUENCE": INFLUENCE,
            "config": {"nfore": nfore, "approach": "Extended Joint",
                       "generalized": True, "corrected": False,
                       "names": names, "dates": date_labels}}


def ConnectednessApproach(x, nlag=1, nfore=10, window_size=None,
                          corrected=False, model="VAR", connectedness="Time",
                          VAR_config=None, Connectedness_config=None,
                          names=None, dates=None, progress=None):
    """Modulli asosiy wrapper — model + yondashuv kombinatsiyasi.

    R `ConnectednessApproach()` porti (DCC-GARCH bundan mustasno).

    Parametrlar
    -----------
    x : (T, k) massiv yoki DataFrame — zoo matritsa o'rniga
    model : "VAR" | "QVAR" | "LASSO" | "Ridge" | "Elastic" | "TVP-VAR"
    connectedness : "Time" | "Frequency" | "Joint" | "Extended Joint"
    window_size : siljish oynasi (rolling); None — butun namuna
    """
    if isinstance(x, pd.DataFrame):
        if names is None:
            names = list(x.columns)
        if dates is None:
            dates = [str(d.date()) if hasattr(d, "date") else str(d)
                     for d in x.index]
        x = x.values
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x — 2 o'lchamli matritsa bo'lishi kerak")
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    if nfore <= 0:
        raise ValueError("nfore musbat butun son bo'lishi kerak")
    if model not in ("VAR", "QVAR", "LASSO", "Ridge", "Elastic", "TVP-VAR"):
        raise ValueError(f"Noma'lum model: {model}")
    if connectedness not in ("Time", "Frequency", "Joint", "Extended Joint"):
        raise ValueError(f"Noma'lum yondashuv: {connectedness}")

    VAR_config = VAR_config or {}
    C_config = Connectedness_config or {}
    T, k = x.shape
    if names is None:
        names = [str(i + 1) for i in range(k)]
    names = [str(n) for n in names]
    if dates is None:
        dates = [str(i + 1) for i in range(T)]

    if window_size is None:
        window_size = T
        t0 = 1
    else:
        window_size = window_size - nlag
        t0 = T - window_size

    # ---- model sozlamasi
    if model == "VAR":
        def var_model(data, cfg):
            return VAR(data, {"nlag": cfg["nlag"]})
        configuration = {"nlag": nlag}
    elif model == "QVAR":
        def var_model(data, cfg):
            return QVAR(data, {"nlag": cfg["nlag"], "tau": cfg["tau"]})
        configuration = {"nlag": nlag,
                         "tau": VAR_config.get("QVAR", {}).get("tau", 0.5)}
    elif model in ("LASSO", "Ridge", "Elastic"):
        en_cfg = VAR_config.get("ElasticNet", {})
        alpha = 1 if model == "LASSO" else (0 if model == "Ridge" else
                                            en_cfg.get("alpha", None))
        def var_model(data, cfg):
            return ElasticNetVAR(data, {"nlag": cfg["nlag"],
                                         "alpha": cfg["alpha"],
                                         "nfolds": cfg["nfolds"],
                                         "loss": cfg["loss"],
                                         "delta_alpha": cfg["delta_alpha"]})
        configuration = {"nlag": nlag, "alpha": alpha,
                         "nfolds": en_cfg.get("nfolds", 10),
                         "loss": en_cfg.get("loss", "mae"),
                         "delta_alpha": en_cfg.get("delta_alpha", 0.1)}
    elif model == "TVP-VAR":
        tvp_cfg = VAR_config.get("TVPVAR", {})
        prior_name = tvp_cfg.get("prior", "BayesPrior")
        if prior_name == "MinnesotaPrior":
            prior = MinnesotaPrior(tvp_cfg.get("gamma", 0.1), k=k, nlag=nlag)
        elif prior_name == "UninformativePrior":
            prior = UninformativePrior(k=k, nlag=nlag)
        elif prior_name == "BayesPrior":
            prior = BayesPrior(x, size=window_size, nlag=nlag)
        else:
            raise ValueError("Noma'lum prior tanlovi")
        configuration = {"l": (tvp_cfg.get("kappa1", 0.99),
                              tvp_cfg.get("kappa2", 0.99)),
                         "nlag": nlag, "prior": prior}
        def var_model(data, cfg):
            return TVPVAR(data, cfg)
    else:
        raise ValueError("Model mavjud emas")

    # ---- modelni baholash
    if progress:
        progress(0.0, "Model baholanmoqda ...")
    if model == "TVP-VAR":
        fit = var_model(x, configuration)
        B_t, Q_t = fit["B_t"], fit["Q_t"]
    else:
        Q_t = np.zeros((k, k, t0))
        B_t = np.zeros((k, k * nlag, t0))
        for i in range(t0):
            fit = var_model(x[i:i + window_size], configuration)
            B_t[:, :, i] = fit["B"]
            Q_t[:, :, i] = fit["Q"][:, :, 0]
            if progress:
                progress((i + 1) / t0, f"Model baholanmoqda ... {i + 1}/{t0}")

    t_eff = Q_t.shape[2]
    date_slice = [str(d) for d in dates[T - t_eff:T]] if t_eff <= T else \
        [str(d) for d in dates]

    # ---- bog'liqlik o'lchovlari
    if progress:
        progress(0.0, "Bog'liqlik o'lchovlari hisoblanmoqda ...")
    if connectedness == "Time":
        gen = C_config.get("TimeConnectedness", {}).get("generalized", True)
        dca = TimeConnectedness(Phi=B_t, Sigma=Q_t, nfore=nfore,
                                generalized=gen, corrected=corrected,
                                names=names, dates=date_slice)
    elif connectedness == "Frequency":
        fcfg = C_config.get("FrequencyConnectedness", {})
        dca = FrequencyConnectedness(
            Phi=B_t, Sigma=Q_t, nfore=nfore,
            partition=fcfg.get("partition", [np.pi, np.pi / 2, 0]),
            generalized=fcfg.get("generalized", True),
            scenario=fcfg.get("scenario", "ABS"),
            corrected=corrected, names=names, dates=date_slice)
    elif connectedness == "Joint":
        dca = JointConnectedness(B_t, Q_t, nfore, names=names,
                                 dates=date_slice)
    else:
        dca = ExtendedJointConnectedness(B_t, Q_t, nfore, names=names,
                                        dates=date_slice)
    dca["config"]["model"] = model
    dca["config"]["nlag"] = nlag
    dca["config"]["window_size"] = None if window_size == T else window_size
    return dca
