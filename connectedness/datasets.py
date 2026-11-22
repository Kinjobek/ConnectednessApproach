"""Paket bilan birga kelgan ma'lumot to'plamlari (CSV formatda).

R paketidagi .rda fayllardan aylantirilgan 17 ta klassik ma'lumot to'plami.
Har biri: dict(name, description, paper, data — date ustuni bilan DataFrame).
"""
import os
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Ma'lumot to'plamlari tavsifi (R paketi hujjatlaridan)
DATASETS = {
    "dy2009": {
        "title": "Diebold–Yilmaz (2009) — 19 ta bozor volatililigi",
        "desc": ("Global aksiya bozorlari volatililigi spilloverlari. "
                 "AQSH, Buyuk Britaniya, Fransiya, Germaniya, Yaponiya va "
                 "rivojlanayotgan bozorlar (1992–2007)."),
        "paper": "Diebold, F. X., & Yilmaz, K. (2009). Measuring financial asset return and volatility spillovers. The Economic Journal.",
    },
    "dy2012": {
        "title": "Diebold–Yilmaz (2012) — 4 aktiv sinfi",
        "desc": ("S&P 500, 10 yillik davlat obligatsiyasi, Bloomberg "
                 "tovar indeksi va dollar indeksi volatililigi "
                 "(1999–2010) — umumlashtirilgan yondashuv namunasi."),
        "paper": "Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive. International Journal of Forecasting.",
    },
    "acg2020": {
        "title": "Antonakakis et al. (2020) — TVP-VAR valyutalari",
        "desc": ("EUR, GBP, JPY, CHF valyuta juftliklari — TVP-VAR "
                 "yondashuvi uchun asosiy ma'lumot (1975–2018, oylik)."),
        "paper": "Antonakakis, N., Chatziantoniou, I., & Gabauer, D. (2020). Refined measures of dynamic connectedness. JRFM.",
    },
    "g2020": {
        "title": "Gabauer (2020) — DCC-GARCH valyutalari",
        "desc": "EUR, GBP, CHF, JPY — volatililik uzatish mexanizmlari (2002–2018).",
        "paper": "Gabauer, D. (2020). Volatility impulse response analysis for DCC-GARCH models. Journal of Forecasting.",
    },
    "gg2018": {
        "title": "Gabauer–Gupta (2018) — pul siyosati noaniqligi",
        "desc": "AQSH va Yaponiya pul, fiskal, savdo va valyuta siyosati noaniqligi (1987–2017).",
        "paper": "Gabauer, D., & Gupta, R. (2018). On the transmission channels of monetary and fiscal policy.",
    },
    "cg2021": {
        "title": "Chatziantoniou–Gabauer (2021) — EMU davlat obligatsiyalari",
        "desc": "Yevrozonaning 12 a'zo davlati 10 yillik davlat obligatsiyalari daromadliligi (2003–2018).",
        "paper": "Chatziantoniou, I., & Gabauer, D. (2021). EMU risk-synchronisation and financial fragility. QREF.",
    },
    "cgs2021": {
        "title": "Chatziantoniou et al. (2021) — valyuta svoplarini almashtirish",
        "desc": "USD, EUR, GBP, JPY 1 yillik valyuta bazis svoplari (2005–2020).",
        "paper": "Chatziantoniou, I., Gabauer, D., & Stenfors, A. (2021). Interest rate swaps and the transmission mechanism of monetary policy. Economics Letters.",
    },
    "cgs2022": {
        "title": "Chatziantoniou et al. (2022) — ISDA marja bo'yicha svoplar",
        "desc": "Shvetsariya, Daniya, Yevro hududi, Buyuk Britaniya, Norvegiya 2/5/10 yillik ISDAFixed svop stavkalari (1999–2021).",
        "paper": "Chatziantoniou, I., Gabauer, D., & Stenfors, A. (2022).",
    },
    "cegg2022": {
        "title": "Chatziantoniou et al. (2022) — neft bozoridagi zarbalar",
        "desc": "Talab, taklif va risk zarbalari hamda 9 mamlakat valyutalari (2006–2021).",
        "paper": "Chatziantoniou, I., Elsayed, A. H., Gabauer, D., & Gozgor, G. (2022). Integration and risk transmission in the market for crude oil.",
    },
    "cgg2022": {
        "title": "Chatziantoniou et al. (2022) — neft eksportchilari",
        "desc": "Yevropa, AQSH, Dubay, Malayziya, Afrika va OPEC neft ishlab chiqarish samaradorligi (1996–2020).",
        "paper": "Chatziantoniou, I., Gabauer, D., & Gupta, R. (2022).",
    },
    "gcat2022": {
        "title": "Chatziantoniou et al. (2022) — yashil aktivlar",
        "desc": "Yashil obligatsiya, atrof-muhit, barqarorlik va toza energiya indekslari (2008–2020).",
        "paper": "Chatziantoniou, I., Aikins Abakah, E. J., Gabauer, D., & Tiwari, A. K. (2022). Quantile time-frequency price connectedness.",
    },
    "gghm2022": {
        "title": "Gabauer et al. (2022) — AQSH uy narxlari",
        "desc": "50 shtat va Kolumbiya okrugi uy narxlari indeksi (1976–2019) — Elastic Net modellari uchun.",
        "paper": "Gabauer, D., Gupta, R., Marfatia, H., & Miller, S. M. (2022). US Housing Price Network Connectedness.",
    },
    "aaacgo2022": {
        "title": "Adekoya et al. (2022) — neft va texnologiyalar",
        "desc": "Brent nefti, asosiy valyutalar va 9 sektor indekslari (2013–2021).",
        "paper": "Adekoya, O. B., Akinresanre, O., & Gabauer, D. (2022).",
    },
    "bgu2021": {
        "title": "Balcilar et al. (2021) — tovar bozorlari",
        "desc": "Xom neft, mol go'shti, kakao, kofe, bug'doy va boshqa tovarlar (2005–2020) — Extended Joint yondashuvi.",
        "paper": "Balcilar, M., Gabauer, D., & Umar, Z. (2021). Crude Oil futures contracts and commodity markets. Resources Policy.",
    },
    "lw2021": {
        "title": "Lastrapes–Wiesen (2021) — 9 sektor",
        "desc": "Energiya, xomashyo, sanoat, moliya, sog'liqni saqlash va boshqa sektorlar (1994–2018) — Joint Spillover indeksi.",
        "paper": "Lastrapes, W. D., & Wiesen, T. F. (2021). The joint spillover index. Economic Modelling.",
    },
    "jcggh2022": {
        "title": "Cunado et al. (2022) — qimmatbaho metallar",
        "desc": "Xom neft, ogyovgoyin moy, oltin va kumush realizatsiyalangan volatililigi (2006–2021).",
        "paper": "Cunado, J., Chatziantoniou, I., Gabauer, D., Hardik, M., & de Garcia, F. P. (2022).",
    },
}


def list_datasets():
    """Mavjud CSV ma'lumot to'plamlari ro'yxati."""
    out = []
    if not os.path.isdir(_DATA_DIR):
        return out
    for fn in sorted(os.listdir(_DATA_DIR)):
        if fn.endswith(".csv"):
            out.append(fn[:-4])
    return out


def load_dataset(name):
    """Nomli ma'lumot to'plamini yuklash (date indeksli DataFrame)."""
    path = os.path.join(_DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise KeyError(f"Ma'lumot to'plami topilmadi: {name}")
    df = pd.read_csv(path)
    date_col = df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        pass
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    meta = DATASETS.get(name, {})
    return {"name": name, "title": meta.get("title", name),
            "desc": meta.get("desc", ""), "paper": meta.get("paper", ""),
            "data": df}
