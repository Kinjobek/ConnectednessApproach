"""Yadro funksiyalar: IRF, FEVD, Wold, chastota bo'linishi.

ConnectednessApproach R paketining (David Gabauer, 2022) sodda va sifatli Python porti.
Barcha formulalar R kodidan birma-bir olingan.
"""
import warnings
import numpy as np

__all__ = ["embed", "IRF", "FEVD", "Wold", "get_indices", "get_partition"]


def embed(x, nlag):
    """R `stats::embed(x, nlag + 1)` ekvivalenti.

    Qator i: [x_t, x_{t-1}, ..., x_{t-nlag}] — ya'ni joriy qiymat keyin laglar.
    Natija: (T - nlag, k * (nlag + 1)) o'lchamli massiv.
    """
    x = np.asarray(x, dtype=float)
    T, k = x.shape
    if T <= nlag:
        raise ValueError("Ma'lumot uzunligi lag sonidan katta bo'lishi kerak")
    rows = T - nlag
    out = np.empty((rows, k * (nlag + 1)))
    for i in range(nlag + 1):
        out[:, i * k:(i + 1) * k] = x[nlag - i: nlag - i + rows]
    return out


def _companion_irf(Phi, nfore):
    """PSI rekursiyasi: PSI_0 = I, PSI_h = sum_j A_j PSI_{h-j}.

    Phi: (k, k*p) — VAR koeffitsientlari (R dagi B).
    Natija: (nfore+1, k, k) — psi[h] = Psi_h.
    """
    k = Phi.shape[0]
    p = Phi.shape[1] // k
    psi = np.zeros((nfore + 1, k, k))
    psi[0] = np.eye(k)
    for h in range(1, nfore + 1):
        acc = np.zeros((k, k))
        iend = min(h, p)
        for j in range(1, iend + 1):
            A_j = Phi[:, (j - 1) * k: j * k]
            acc += A_j @ psi[h - j]
        psi[h] = acc
    return psi


def IRF(Phi, Sigma, nfore=10, orth=True):
    """Impuls-javob funksiyalari (Impulse Response Functions).

    R `IRF()` funksiyasining porti. `irf[i]` — i-shokka barcha o'zgaruvchilarning
    javobi: (nfore + 1, k) matritsa (qatorlar — gorizontal bosqichlar).

    Parametrlar
    -----------
    Phi : (k, k*nlag) VAR koeffitsient matritsasi
    Sigma : (k, k) qoldiqlar kovaryans matritsasi
    nfore : bashorat ufqi
    orth : True — ortogonalizatsiyalangan (Cholesky) shoklar
    """
    if nfore <= 0:
        raise ValueError("nfore musbat butun son bo'lishi kerak")
    Phi = np.asarray(Phi, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim > 2:
        Sigma = Sigma[:, :, 0]
    k = Phi.shape[0]
    if orth:
        # R: eigen(Sigma) -> P diag(sqrt(lambda)) P' (simmetrik, tartib ahamiyatsiz)
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.maximum(vals, 0.0)
        Sh = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    else:
        Sh = np.eye(k)
    psi = _companion_irf(Phi, nfore)
    wgt = psi @ Sh                                  # wgt[h] = Psi_h %*% Sh
    # R: irf[[i]] = t(WGT[qatorlar bloki i, ]) — blok i = Psi_h Sh ning i-USTUNI
    irf = [wgt[:, :, i].copy() for i in range(k)]   # irf[i][h, var] = (Psi_h Sh)[var, i]
    return {"irf": irf}


def _fevd_time(irf, Sigma, nfore, generalized):
    """Vaqt sohasi FEVD. R FEVD(type="time") porti.

    Natija FEVD[dest, source] — har bir qator (dest) yig'indisi 1 ga teng.
    """
    k = Sigma.shape[0]
    # M_h[resp, shk] = irf[shk][h, resp] = Psi_h[resp, shk]
    M = np.zeros((nfore, k, k))
    for h in range(nfore):
        for shk in range(k):
            M[h][:, shk] = irf[shk][h, :]
    denom = np.einsum("hij,jk,hik->i", M, Sigma, M)   # diag(sum_h M_h Sigma M_h')
    if generalized:
        enum = np.zeros((k, k))
        for h in range(nfore):
            enum += (M[h] @ Sigma) ** 2
        theta = enum / (denom[:, None] * np.diag(Sigma)[None, :])
        fevd = theta / theta.sum(axis=1, keepdims=True)
    else:
        # R: enum = sum_h (chol(Sigma) %*% t(M_h))^2; FEVD = t(enum[, i]/denom[i])
        L = np.linalg.cholesky(Sigma).T              # R chol() — yuqori uchburchak
        enum = np.zeros((k, k))
        for h in range(nfore):
            enum += (L @ M[h].T) ** 2                 # [shk, resp]
        fevd = (enum / denom[None, :]).T              # [dest, shk]
    return fevd


def _fevd_frequency(irf, Sigma, nfore, generalized, rng):
    """Chastota sohasi FEVD. R FEVD(type="frequency") porti.

    `rng` — ishlatiladigan chastota indekslari (0-asosli).
    Natija: (nfore+1, k, k) massiv — har bir chastota bini uchun FEVD.
    """
    k = Sigma.shape[0]
    N = nfore + 1
    # FFT har bir (shk, var) juftligi uchun gorizontal bo'yicha
    F = np.zeros((N, k, k), dtype=complex)  # F[w, resp, shk] = FFT_{w}(irf[shk][·, resp])
    for shk in range(k):
        F[:, :, shk] = np.fft.fft(irf[shk], axis=0, n=N) if irf[shk].shape[0] == N else \
            np.fft.fft(irf[shk][:N], axis=0)
    denom = np.zeros(k)
    for w in rng:
        Mw = F[w]                                   # [resp, shk]
        denom += np.real(np.diag(Mw @ Sigma @ np.conj(Mw).T)) / nfore
    fevd = np.zeros((N, k, k))
    tot = np.zeros(k)
    tabs = {}
    if generalized:
        for w in rng:
            enum = (np.abs(F[w] @ Sigma)) ** 2 / (nfore + 1)
            tab = enum / (denom[:, None] * np.diag(Sigma)[None, :])
            tabs[w] = tab
            tot += tab.sum(axis=1)     # R: apply(tab, 2, sum) — dest bo'yicha
        for w in rng:
            # R: FEVD_w = t(tab_w) / tot -> [dest, shk] = tab[dest, shk]/tot[dest]
            fevd[w] = tabs[w] / tot[:, None]
    else:
        L = np.linalg.cholesky(Sigma).T
        for w in rng:
            enum = (np.abs(F[w] @ L.T)) ** 2 / (nfore + 1)
            # R: FEVD_w[dest, shk] = enum_w[dest, shk] / denom[dest]
            fevd[w] = enum / denom[:, None]
    return fevd


def FEVD(Phi, Sigma, nfore=100, type="time", generalized=True, orth=False, range_=None):
    """Prognoz xatosi dispersiyasini dekompozitsiya (FEVD).

    R `FEVD()` funksiyasining to'liq porti — vaqt va chastota sohalari,
    umumlashtirilgan (Pesaran–Shin) va ortogonalizatsiyalangan variantlar.

    Parametrlar
    -----------
    Phi : (k, k*nlag) VAR koeffitsientlari
    Sigma : (k, k) qoldiq kovaryans matritsasi
    nfore : bashorat ufqi
    type : "time" yoki "frequency"
    generalized : umumlashtirilgan FEVD
    orth : IRF uchun ortogonalizatsiya
    range_ : chastota sohasi uchun ishlatiladigan bin indekslari (0-asosli)
    """
    if nfore <= 0:
        raise ValueError("nfore musbat butun son bo'lishi kerak")
    Phi = np.asarray(Phi, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim > 2:
        Sigma = Sigma[:, :, 0]
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma kamida 2 o'lchamli kvadrat matritsa bo'lishi kerak")
    if Phi.ndim != 2:
        raise ValueError("Phi kamida 2 o'lchamli matritsa bo'lishi kerak")
    irf = IRF(Phi=Phi, Sigma=Sigma, nfore=nfore, orth=orth)["irf"]
    if type == "time":
        fevd = _fevd_time(irf, Sigma, nfore, generalized)
    elif type == "frequency":
        N = nfore + 1
        rng = range_ if range_ is not None else np.arange(N)
        fevd = _fevd_frequency(irf, Sigma, nfore, generalized, rng)
    else:
        raise ValueError("type 'time' yoki 'frequency' bo'lishi kerak")
    return {"IRF": irf, "FEVD": fevd}


def Wold(x, nfore=10):
    """VAR koeffitsientlaridan VMA (Wold) ko'rsatkichlarini hisoblash.

    R `Wold()` porti. Natija: (nfore+1, k, k) — Phi[0] = I, Phi[h] = VMA h-bosqich.
    """
    if nfore <= 0:
        raise ValueError("nfore musbat butun son bo'lishi kerak")
    x = np.asarray(x, dtype=float)
    K = x.shape[0]
    nstep = int(abs(nfore))
    return _companion_irf(x, nstep)


def get_indices(n_ahead, up, down):
    """R `frequencyConnectedness::getIndeces()` porti.

    (down, up) oralig'idagi chastotalarga mos DFT bin indekslarini qaytaradi
    (0-asosli, to'liq spektr — manfiy chastotalar oynasi bilan birga).
    """
    half = n_ahead // 2
    if half == 0:
        half = 1
    space = np.arange(half + 1) / half * np.pi
    sel = (space >= down) & (space < up)
    full = np.zeros(n_ahead, dtype=bool)
    full[:half + 1] = sel
    if n_ahead % 2 == 0:
        full[half + 1:] = sel[half - 1:0:-1]
    else:
        full[half + 1:] = sel[half:0:-1]
    return np.flatnonzero(full)


def get_partition(partition, n_ahead):
    """R `frequencyConnectedness::getPartition()` porti.

    partition — kamayuvchi tartibdagi chegara nuqtalari (masalan [pi, pi/4, 0]).
    Qaytaradi: har bir chastota oralig'i uchun bin indekslari ro'yxati.
    """
    partition = np.asarray(partition, dtype=float)
    if not np.all(np.diff(partition) <= 0):
        raise ValueError("Chegaralar kamayuvchi tartibda bo'lishi kerak")
    N = n_ahead + 1
    part = [get_indices(N, partition[i], partition[i + 1])
            for i in range(len(partition) - 1)]
    if len(np.unique(np.concatenate(part))) != N:
        warnings.warn("Tanlangan bo'linish butun diapazonni qamrab olmaydi.")
    if any(len(p) == 0 for p in part):
        raise ValueError("Bo'linishni o'zgartiring — ayrim oraliq bo'sh.")
    return part
