"""VAR oilasi modellari va priors.

R ConnectednessApproach paketidagi VAR, QVAR, ElasticNetVAR (LASSO/Ridge/Elastic),
TVP-VAR modellari hamda Bayes/Minnesota/Uninformative priors portlari.
"""
import numpy as np
from scipy.optimize import linprog

from .core import embed

__all__ = ["VAR", "QVAR", "ElasticNetVAR", "TVPVAR",
           "BayesPrior", "MinnesotaPrior", "UninformativePrior"]


def _check_data(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Ma'lumot 2 o'lchamli matritsa bo'lishi kerak")
    return x


# ---------------------------------------------------------------- VAR (OLS)
def VAR(x, configuration=None):
    """Vektorli avtoregressiya (VAR) — tenglamama-tenglama OLS.

    R `VAR()` porti. Har bir tenglama intercept bilan baholanadi,
    `B` faqat lag koeffitsientlarini o'z ichiga oladi.

    Natija: dict(B=(k, k*nlag), Q=(k, k, 1), se=(k, k*nlag))
    """
    configuration = configuration or {}
    x = _check_data(x)
    nlag = int(configuration.get("nlag", 1))
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    T, k = x.shape
    z = embed(x, nlag)
    y = z[:, :k]
    X = z[:, k:]
    Xm = X - X.mean(axis=0)
    ym = y - y.mean(axis=0)
    coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)   # (k*nlag, k)
    resid = ym - Xm @ coef
    n = resid.shape[0]
    B = coef.T                                        # (k, k*nlag)
    Q = (resid.T @ resid / n)[:, :, None]
    # standart xatolar (R summary(lm) bilan mos)
    XtXi = np.linalg.pinv(Xm.T @ Xm)
    dof = n - Xm.shape[1] - 1
    se = np.empty_like(B)
    for i in range(k):
        sigma2 = resid[:, i] @ resid[:, i] / dof
        se[i] = np.sqrt(np.maximum(sigma2 * np.diag(XtXi), 0))
    return {"B": B, "Q": Q, "se": se}


# ---------------------------------------------------------------- QVAR (quantile)
def _quantile_reg(Xd, y, tau):
    """Kvantil regressiya — R `quantreg::rq()` (br usuli) ekvivalenti.

    LP shakllantirish: min tau*sum(u) + (1-tau)*sum(v)
    shart: y - Xd @ beta = u - v, u, v >= 0.
    """
    n, p = Xd.shape
    c = np.concatenate([np.zeros(p), np.full(n, tau), np.full(n, 1.0 - tau)])
    A_eq = np.concatenate([Xd, np.eye(n), -np.eye(n)], axis=1)
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Kvantil regressiya yechilmadi: {res.message}")
    return res.x[:p]


def QVAR(x, configuration=None):
    """Kvantil VAR (QVAR). R `QVAR()` porti.

    configuration: dict(nlag=1, tau=0.5)
    """
    configuration = configuration or {}
    x = _check_data(x)
    nlag = int(configuration.get("nlag", 1))
    tau = float(configuration.get("tau", 0.5))
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    if not 0 < tau < 1:
        raise ValueError("tau 0 va 1 orasida bo'lishi kerak")
    T, k = x.shape
    z = embed(x, nlag)
    n = z.shape[0]
    B = np.empty((k, k * nlag))
    Res = np.empty((n, k))
    for i in range(k):
        y = z[:, i]
        Xd = np.column_stack([np.ones(n), z[:, k:]])
        beta = _quantile_reg(Xd, y, tau)
        B[i] = beta[1:]
        Res[:, i] = y - Xd @ beta
    Q = (Res.T @ Res / n)[:, :, None]
    return {"B": B, "Q": Q}


# ---------------------------------------------------------------- ElasticNet VAR
def ElasticNetVAR(x, configuration=None):
    """LASSO / Ridge / Elastic Net VAR. R `ElasticNetVAR()` porti.

    configuration: dict(nlag=1, alpha=1|0|None|0..1, nfolds=10, loss="mae"|"mse",
                        delta_alpha=0.1)

    alpha=1 — LASSO, alpha=0 — Ridge, aks holda Elastic Net.
    Coefficientlar sklearn dvigatelkasi orqali CV bilan tanlanadi
    (glmnet bilan bir xil g'oya, biroq aniq sonlar biroz farq qilishi mumkin).
    """
    configuration = configuration or {}
    x = _check_data(x)
    nlag = int(configuration.get("nlag", 1))
    alpha = configuration.get("alpha", 1)
    nfolds = int(configuration.get("nfolds", 10))
    loss = configuration.get("loss", "mae")
    delta_alpha = float(configuration.get("delta_alpha", 0.1))
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    if nfolds <= 0:
        raise ValueError("nfolds musbat butun son bo'lishi kerak")

    from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet
    from sklearn.model_selection import cross_val_score

    T, k = x.shape
    z = embed(x, nlag)
    n = z.shape[0]
    y_all = z[:, :k]
    X = z[:, k:]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    cv = min(nfolds, max(2, n // 10))

    def _mae_cv(model):
        scores = cross_val_score(model, Xs, yc, cv=cv,
                                 scoring="neg_mean_absolute_error")
        return scores.mean()

    B = np.empty((k, k * nlag))
    Res = np.empty((n, k))
    alpha_out = np.empty(k)
    for i in range(k):
        yc = y_all[:, i] - y_all[:, i].mean()

        def _fit_penalty(l1_ratio):
            """glmnet uslubida: lambda yo'li bo'yicha K-fold MAE-CV.

            (mdl, best_alpha, best_cv_score) qaytaradi.
            """
            lam_max = np.max(np.abs(Xs.T @ yc)) / n
            if lam_max == 0:
                lam_max = 1.0
            path = np.geomspace(lam_max, lam_max * 1e-4, 50)
            best_alpha, best_score = None, -np.inf
            for a in path:
                if l1_ratio == 1:
                    mdl = Lasso(alpha=a, max_iter=100000)
                elif l1_ratio == 0:
                    mdl = Ridge(alpha=a)
                else:
                    mdl = ElasticNet(alpha=a, l1_ratio=l1_ratio,
                                    max_iter=100000)
                s = _mae_cv(mdl)
                if s > best_score:
                    best_score, best_alpha = s, a
            if l1_ratio == 1:
                model = Lasso(alpha=best_alpha, max_iter=100000)
            elif l1_ratio == 0:
                model = Ridge(alpha=best_alpha)
            else:
                model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio,
                                  max_iter=100000)
            model.fit(Xs, yc)
            return model, best_alpha, best_score

        if alpha == 0:
            model = RidgeCV(alphas=np.logspace(-3, 4, 60), cv=cv,
                            scoring="neg_mean_absolute_error")
            model.fit(Xs, yc)
            alpha_out[i] = model.alpha_
        else:
            if alpha is None:
                best = None
                best_score = -np.inf
                for lr in np.arange(delta_alpha, 1.0, delta_alpha):
                    mdl, a, s = _fit_penalty(lr)
                    if s > best_score:
                        best_score, best = s, mdl
                model = best
            else:
                model, a, _ = _fit_penalty(float(alpha))
            alpha_out[i] = getattr(model, "alpha_", np.nan)
        beta_std = model.coef_
        B[i] = beta_std / sd
        intercept = y_all[:, i].mean() - (beta_std / sd) @ mu
        Res[:, i] = y_all[:, i] - (intercept + X @ B[i])
    Q = (Res.T @ Res / n)[:, :, None]
    return {"B": B, "Q": Q, "alpha": alpha_out}


# ---------------------------------------------------------------- Priors
def BayesPrior(x, size=None, nlag=1):
    """Bayes prior (Primiceri, 2005). R `BayesPrior()` porti.

    Natija: dict(bprior=(m,), Vprior=(m, m), Q=(k, k)) — m = nlag * k^2.
    """
    x = _check_data(x)
    nlag = int(nlag)
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    T, p = x.shape
    if size is None:
        size = T
    size = size - nlag
    if size <= 0:
        raise ValueError("size nlag dan katta bo'lishi kerak")
    m = p + nlag * p * p

    # Stacked regressor matritsasi Zt: (size*p, m)
    Zt = np.zeros((size * p, m))
    yflat = np.empty(size * p)
    for idx in range(size):
        i = nlag + idx                       # 0-asosli vaqt indeksi
        ztemp = np.zeros((p, m))
        ztemp[:, :p] = np.eye(p)
        for j in range(1, nlag + 1):
            xlag = x[i - j, :]
            xtemp = np.zeros((p, p * p))
            for jj in range(p):
                xtemp[jj, jj * p:(jj + 1) * p] = xlag
            ztemp[:, p + (j - 1) * p * p: p + j * p * p] = xtemp
        Zt[idx * p:(idx + 1) * p, :] = ztemp
        yflat[idx * p:(idx + 1) * p] = x[i, :]

    vbar = Zt.T @ Zt
    xhy = Zt.T @ yflat
    aols = np.linalg.pinv(vbar) @ xhy
    rflat = yflat - Zt @ aols
    R = rflat.reshape(size, p)
    sse2 = R.T @ R
    hbar = sse2 / size
    vbar_inv = np.linalg.pinv(vbar)
    return {"bprior": aols[p:], "Vprior": vbar_inv[p:, p:], "Q": hbar}


def MinnesotaPrior(gamma=0.1, k=2, nlag=1):
    """Minnesota prior. R `MinnesotaPrior()` porti (natija R bilan identik).

    bprior — nol vektor; Vprior — diagonal, lag L uchun gamma/L^2.
    """
    if k <= 1:
        raise ValueError("k — o'zgaruvchilar soni, 1 dan katta butun son")
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    m = nlag * k * k
    bprior = np.zeros(m)
    diag = np.empty(m)
    for c in range(m):
        L = c // (k * k)                     # lag (0-asosli)
        diag[c] = gamma / (L + 1) ** 2
    return {"bprior": bprior, "Vprior": np.diag(diag)}


def UninformativePrior(k=2, nlag=1):
    """Ma'lumotsiz prior. R `UninformativePrior()` porti."""
    if k <= 1:
        raise ValueError("k — o'zgaruvchilar soni, 1 dan katta butun son")
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    m = nlag * k * k
    bprior = np.zeros(m)
    return {"bprior": bprior, "Vprior": 4.0 * np.eye(m)}


# ---------------------------------------------------------------- TVP-VAR
def _create_rhs_ni(templag, r, nlag, t):
    """R `create_RHS_NI()` porti — TVP-VAR holat regressorlari.

    Natija: ((t-nlag)*r, nlag*r^2) o'lchamli Flagtemp.
    """
    K = nlag * r * r
    nobs = t - nlag
    x_t = np.zeros((nobs * r, K))
    for i in range(nobs):
        blocks = []
        for j in range(nlag):
            xlag = templag[i, j * r:(j + 1) * r]
            blk = np.zeros((r, r * r))
            for eq in range(r):
                blk[eq, eq * r:(eq + 1) * r] = xlag
            blocks.append(blk)
        x_t[i * r:(i + 1) * r, :] = np.hstack(blocks)
    return x_t


def TVPVAR(x, configuration=None):
    """Vaqt bo'yicha o'zgaruvchan parametrli VAR (TVP-VAR).

    R `TVPVAR()` porti — Koop & Korobilis (2014) eshitish omillari bilan
    Kalman filtri. configuration: dict(l=(kappa1, kappa2), nlag=1, prior=...).
    """
    configuration = configuration or {}
    x = _check_data(x)
    nlag = int(configuration.get("nlag", 1))
    l = configuration.get("l", (0.99, 0.99))
    prior = configuration.get("prior")
    if nlag <= 0:
        raise ValueError("nlag musbat butun son bo'lishi kerak")
    if not 0 < l[0] < 1 or not 0 < l[1] < 1:
        raise ValueError("kappa1 va kappa2 0 va 1 orasida bo'lishi kerak")

    T0, r = x.shape
    if prior is None:
        prior = UninformativePrior(k=r, nlag=nlag)
    beta_0_mean = np.asarray(prior["bprior"], dtype=float).ravel()
    beta_0_var = np.asarray(prior["Vprior"], dtype=float)
    Q0 = prior.get("Q")
    if Q0 is None:
        Q0 = np.cov(x, rowvar=False)

    m = nlag * r * r
    kk = nlag * r                       # hamroh matritsa o'lchami
    l_4, l_2 = float(l[0]), float(l[1])

    x_c = x - x.mean(axis=0)            # R: scale(x, TRUE, FALSE)
    t = T0
    yy = x_c[nlag:, :]                                  # (t-nlag, r)
    xx = embed(x_c, nlag)[:, r:]                        # (t-nlag, r*nlag)
    Flagtemp = _create_rhs_ni(xx, r, nlag, t)
    Flag = np.vstack([np.zeros((kk, m)), Flagtemp])     # (t*r, m)

    beta_pred = np.zeros((m, t))
    beta_update = np.zeros((m, t))
    Rb_t = np.zeros((t, m, m))
    Sb_t = np.zeros((t, m, m))
    beta_t = np.zeros((t, kk, kk))
    Q_t = np.zeros((t, r, r))

    B = None
    for i in range(t):
        # ---- Q_t yangilash
        if i == 0:
            Q_t[i] = Q0
        else:
            if i <= nlag:
                Gf = 0.1 * np.outer(x_c[i], x_c[i])
            else:
                A = B[:r, :] if B is not None else np.zeros((r, kk))
                res = yy[i - nlag] - xx[i - nlag] @ A.T
                Gf = np.outer(res, res)
            Q_t[i] = l_2 * Q_t[i - 1] + (1 - l_2) * Gf
        # ---- beta holati
        if i <= nlag:
            beta_pred[:, i] = beta_0_mean
            beta_update[:, i] = beta_0_mean
            Rb_t[i] = beta_0_var
        else:
            beta_pred[:, i] = beta_update[:, i - 1]
            Rb_t[i] = (1.0 / l_4) * Sb_t[i - 1]
        # ---- Kalman yangilanishi
        if i >= nlag:
            Fb = Flag[i * r:(i + 1) * r, :]             # (r, m)
            Rx = Rb_t[i] @ Fb.T
            KV_b = Q_t[i] + Fb @ Rx
            KG = Rx @ np.linalg.pinv(KV_b)
            beta_update[:, i] = beta_pred[:, i] + KG @ (x_c[i] - Fb @ beta_pred[:, i])
            Sb_t[i] = Rb_t[i] - KG @ (Fb @ Rb_t[i])
        # ---- koeffitsientlarni yig'ish
        bb = beta_update[:, i]
        biga = np.zeros((r, r * nlag))
        splace = 0
        for ii in range(nlag):
            for iii in range(r):
                biga[iii, ii * r:(ii + 1) * r] = bb[splace:splace + r]
                splace += r
        if nlag > 1:
            companion = np.zeros((r * (nlag - 1), r))
            companion[:r * (nlag - 1), :r] = np.eye(r * (nlag - 1))
            B = np.vstack([biga, np.hstack([np.eye(r * (nlag - 1)),
                                            np.zeros((r * (nlag - 1), r))])]) \
                if nlag > 1 else biga
        else:
            B = biga
        # ---- barqarorlik tekshiruvi
        ev = np.max(np.abs(np.linalg.eigvals(B)))
        if ev <= 1 or i == 0:
            beta_t[i] = B
        else:
            beta_t[i] = beta_t[i - 1]
            beta_update[:, i] = 0.99 * beta_update[:, i - 1]

    B_t = beta_t[:, :r, :]              # (t, r, r*nlag) -> transpose to (r, r*nlag, t)
    return {"B_t": np.transpose(B_t, (1, 2, 0)), "Q_t": np.transpose(Q_t, (1, 2, 0))}
