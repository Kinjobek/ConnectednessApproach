"""Python portni R referens natijalari bilan solishtirish."""
import re
import sys
import numpy as np

sys.path.insert(0, ".")
from connectedness import (VAR, QVAR, ElasticNetVAR, TVPVAR, BayesPrior,
                           MinnesotaPrior, FEVD, ConnectednessTable,
                           TimeConnectedness, FrequencyConnectedness,
                           JointConnectedness, ExtendedJointConnectedness,
                           load_dataset)

REF = "validation/reference_results.txt"


def load_ref():
    out = {}
    name = None
    with open(REF) as f:
        for line in f:
            if line.startswith("### "):
                name = line[4:].strip()
                out[name] = []
            elif name is not None and line.strip():
                vals = []
                for tok in line.replace('"', "").replace("'", "").split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                if vals:
                    out[name].extend(vals)
    return {k: np.array(v) for k, v in out.items()}


def check(label, py, ref, tol=1e-6, verbose=True):
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float).ravel()
    if py.size != ref.size:
        print(f"FAIL {label}: o'lcham {py.size} != {ref.size}")
        return False
    ref = ref.reshape(py.shape, order="F")   # R write.table — ustunma-ustun
    py = py.ravel()
    ref = ref.ravel()
    denom = np.maximum(1.0, np.abs(ref))
    err = np.max(np.abs(py - ref) / denom)
    ok = err <= tol
    tag = "OK  " if ok else "FAIL"
    if verbose or not ok:
        print(f"{tag} {label}: maks.xato={err:.3e}")
    if not ok:
        print("   py :", np.array2string(py[:12], precision=8))
        print("   ref:", np.array2string(ref[:12], precision=8))
    return ok


ref = load_ref()
ds = load_dataset("dy2012")
x = ds["data"].values
T, k = x.shape
print(f"dy2012: {T} x {k}")
results = []

# ---- 1. VAR
fit = VAR(x, {"nlag": 1})
results.append(check("VAR_B", fit["B"], ref["VAR_B"]))
results.append(check("VAR_Q", fit["Q"][:, :, 0], ref["VAR_Q"]))
results.append(check("VAR_se", fit["se"], ref["VAR_se"]))

# ---- 2. FEVD
fevd_g = FEVD(Phi=fit["B"], Sigma=fit["Q"], nfore=10, type="time",
              generalized=True)["FEVD"]
results.append(check("FEVD_time_gen", fevd_g, ref["FEVD_time_gen"]))
fevd_o = FEVD(Phi=fit["B"], Sigma=fit["Q"], nfore=10, type="time",
              generalized=False)["FEVD"]
results.append(check("FEVD_time_orth", fevd_o, ref["FEVD_time_orth"]))

# ---- 3. ConnectednessTable
ct = ConnectednessTable(fevd_g)
results.append(check("CT_TCI", ct["TCI"], ref["CT_TCI"]))
results.append(check("CT_cTCI", ct["cTCI"], ref["CT_cTCI"]))
results.append(check("CT_TO", ct["TO"], ref["CT_TO"]))
results.append(check("CT_FROM", ct["FROM"], ref["CT_FROM"]))
results.append(check("CT_NET", ct["NET"], ref["CT_NET"]))
results.append(check("CT_NPT", ct["NPT"], ref["CT_NPT"]))
results.append(check("CT_PCI", ct["PCI"], ref["CT_PCI"]))
results.append(check("CT_INFLUENCE", ct["INFLUENCE"], ref["CT_INFLUENCE"]))

# ---- 4. TimeConnectedness
dca = TimeConnectedness(Phi=fit["B"], Sigma=fit["Q"], nfore=10,
                        generalized=True)
results.append(check("TIME_TCI", dca["TCI"], ref["TIME_TCI"]))

# ---- 5. QVAR
qfit = QVAR(x, {"nlag": 1, "tau": 0.5})
results.append(check("QVAR_B", qfit["B"], ref["QVAR_B"], tol=1e-4))
results.append(check("QVAR_Q", qfit["Q"][:, :, 0], ref["QVAR_Q"], tol=1e-4))

# ---- 6. BayesPrior + TVPVAR
prior = BayesPrior(x, nlag=1)
results.append(check("BAYESPRIOR_bprior", prior["bprior"],
                     ref["BAYESPRIOR_bprior"]))
tfit = TVPVAR(x, {"l": (0.99, 0.99), "nlag": 1, "prior": prior})
results.append(check("TVPVAR_Bt_last", tfit["B_t"][:, :, -1],
                     ref["TVPVAR_Bt_last"]))
results.append(check("TVPVAR_Qt_last", tfit["Q_t"][:, :, -1],
                     ref["TVPVAR_Qt_last"]))
results.append(check("TVPVAR_Bt_first", tfit["B_t"][:, :, 0],
                     ref["TVPVAR_Bt_first"]))
tdca = TimeConnectedness(Phi=tfit["B_t"], Sigma=tfit["Q_t"], nfore=10,
                        generalized=True)
results.append(check("TVPTIME_TCI_head", tdca["TCI"][:5],
                     ref["TVPTIME_TCI_head"]))
results.append(check("TVPTIME_TCI_tail", tdca["TCI"][-5:],
                     ref["TVPTIME_TCI_tail"]))

mprior = MinnesotaPrior(0.1, k=4, nlag=1)
mfit = TVPVAR(x, {"l": (0.99, 0.99), "nlag": 1, "prior": mprior})
results.append(check("TVPVAR_MINN_Bt_last", mfit["B_t"][:, :, -1],
                     ref["TVPVAR_MINN_Bt_last"]))

# ---- 7. Frequency
partition = [np.pi + 0.00001, np.pi / 4, 0]
fdca = FrequencyConnectedness(Phi=fit["B"], Sigma=fit["Q"], nfore=100,
                              partition=partition)
results.append(check("FREQ_TCI", fdca["TCI"], ref["FREQ_TCI"]))
results.append(check("FREQ_CT_band2", fdca["CT"][:, :, 0, 1],
                     ref["FREQ_CT_band2"]))

# ---- 8. Joint
jfit = VAR(x, {"nlag": 2})
jdca = JointConnectedness(jfit["B"], jfit["Q"], 30)
results.append(check("JOINT_TCI", jdca["TCI"], ref["JOINT_TCI"]))
results.append(check("JOINT_TO", jdca["TO"], ref["JOINT_TO"]))
results.append(check("JOINT_FROM", jdca["FROM"], ref["JOINT_FROM"]))
results.append(check("JOINT_NET", jdca["NET"], ref["JOINT_NET"]))

# ---- 9. Extended joint
bfit = TVPVAR(x, {"l": (0.99, 0.99), "nlag": 1, "prior": mprior})
ejdca = ExtendedJointConnectedness(bfit["B_t"], bfit["Q_t"], 20)
results.append(check("EXTJOINT_TCI_head", ejdca["TCI"][:200],
                     ref["EXTJOINT_TCI"][:200]))
results.append(check("EXTJOINT_NET_last", ejdca["NET"][-1],
                     ref["EXTJOINT_NET_last"]))

# ---- 10. LASSO (glmnet != sklearn — faqat yaqinligini ko'ramiz)
try:
    efit = ElasticNetVAR(x, {"nlag": 1, "alpha": 1, "nfolds": 10, "loss": "mae"})
    rb = np.array(ref["LASSO_B"]).reshape(efit["B"].shape, order="F")
    rel = np.abs(efit["B"] - rb) / np.maximum(np.abs(rb), 1e-8)
    print(f"INFO LASSO_B (glmnet vs sklearn): o'rt.rel.farq={rel.mean():.3f}")
except Exception as ex:
    print("INFO LASSO o'tkazib yuborildi:", ex)

n_ok = sum(results)
print(f"\n{n_ok}/{len(results)} test o'tdi")
sys.exit(0 if n_ok == len(results) else 1)
