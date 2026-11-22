# ConnectednessApproach — R + Python + JavaScript + veb-sayt

Bu repozitoriy **Diebold–Yilmaz bog'liqlik yondashuvining** (Connectedness
Approach, David Gabauer 2022) uch bosqichli implementatsiyasini o'z ichiga oladi:

| Qism | Til | Joylashuvi | Holati |
|---|---|---|---|
| **Asl R paketi** | R | repo ildizi (`R/`, `man/`, `data/`) | o'zgarmagan |
| **Python port + Flask veb-ilova** | Python | `connectedness/`, `webapp/`, `validation/` | 31/31 test ✅ |
| **GitHub Pages sayt (o'zbek tilida)** | JavaScript | `docs/` | 26/26 test ✅ |

🌐 **Trafik sayt:** `https://kinjobek.github.io/ConnectednessApproach/` —
barcha hisob-kitoblar brauzerda bajariladi, server kerak emas.

---

## 🌐 Veb-sayt (docs/)

Sayt o'zbek tilida: 17 ta klassik ma'lumot to'plami (yoki o'z CSV faylingiz),
VAR va TVP-VAR modellari, to'rt xil bog'liqlik yondashuvi (vaqt, chastota,
qo'shma, kengaytirilgan qo'shma), to'liq bog'liqlik jadvallari va dinamik
grafiklar (TCI, NET, TO/FROM, NPT, to' grafigi).

Hisoblash dvigateli — `docs/js/engine.js` — asl R paketining to'g'ridan-to'g'ri
JavaScript tarjimasi. Aniqligini tekshirish:

```bash
node docs/js/validate.js      # R referens natijalari bilan 26/26 test
```

## 🐍 Python porti (connectedness/)

```python
from connectedness import ConnectednessApproach, load_dataset

ds = load_dataset("dy2012")
dca = ConnectednessApproach(
    ds["data"], nlag=1, nfore=10,
    model="TVP-VAR", connectedness="Time",
    VAR_config={"TVPVAR": dict(kappa1=0.99, kappa2=0.96,
                               prior="MinnesotaPrior", gamma=0.1)))
print(dca["TABLE"])
```

Modellar: VAR, QVAR (kvantil), LASSO/Ridge/Elastic Net VAR, TVP-VAR.
Yondashuvlar: Time, Frequency, Joint, Extended Joint.

Flask veb-ilovasi (serverli to'liq versiya, QVAR/penalized modellar bilan):

```bash
pip install -r requirements.txt
python3 webapp/app.py          # → http://127.0.0.1:5000
```

## ✅ Validatsiya (validation/)

Python va JavaScript portlari asl R paketi bilan solishtirilgan:

```bash
Rscript validation/reference_run.R    # R referens natijalari
python3 validation/compare.py         # Python: 31/31
node docs/js/validate.js              # JavaScript: 26/26
```

Maksimal nisbiy xato ~10⁻¹². LASSO/Ridge/Elastic Net uchun glmnet o'rniga
scikit-learn ishlatiladi (usul bir xil, tanlangan jarima biroz farq qilishi mumkin).

## 📦 Asl R paketi (repo ildizi)

```r
library(devtools)
install_github("Kinjobek/ConnectednessApproach")
```

## 📄 Litsenziya

GPL-3. Foydalanishda asl paket va metodika maqolalariga iqtibos keltiring:

> Gabauer, D. (2022). ConnectednessApproach: R package version 1.0.0.
> Diebold, F. X., & Yilmaz, K. (2009, 2012). Barunik, J., & Krehlik, T. (2018).
> Antonakakis, N., Chatziantoniou, I., & Gabauer, D. (2020). Koop, G., & Korobilis, D. (2014).
> Lastrapes, W. D., & Wiesen, T. F. (2021). Balcilar, M., Gabauer, D., & Umar, Z. (2021).
