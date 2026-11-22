/* JS dvigatelini R referens natijalari bilan solishtirish (Node).
 * Ishga tushirish:  node docs/js/validate.js  (repo ildizidan)
 */
const fs = require("fs");
const path = require("path");
const CC = require("./engine.js");

// ---------- R referensini o'qish
function loadRef(p) {
  const out = {};
  let name = null;
  for (const line of fs.readFileSync(p, "utf8").split("\n")) {
    if (line.startsWith("### ")) { name = line.slice(4).trim(); out[name] = []; }
    else if (name && line.trim()) {
      for (const tok of line.replace(/"/g, "").split(/\s+/)) {
        const v = parseFloat(tok);
        if (Number.isFinite(v)) out[name].push(v);
      }
    }
  }
  return out;
}

// ---------- CSV o'qish
function readCSV(p) {
  const lines = fs.readFileSync(p, "utf8").trim().split("\n");
  const header = lines[0].split(",").map(s => s.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(",").map(s => parseFloat(s));
    if (vals.length === header.length && vals.every(Number.isFinite))
      rows.push(vals.slice(1));
  }
  return rows;                                    // faqat raqamli ustunlar
}

function firstExisting(paths) {
  for (const p of paths) if (fs.existsSync(p)) return p;
  throw new Error("Fayl topilmadi: " + paths.join(" | "));
}
const here = __dirname;
const refFile = firstExisting([
  path.join(here, "..", "..", "..", "validation", "reference_results.txt"),
  path.join(here, "..", "..", "validation", "reference_results.txt"),
  path.join(process.cwd(), "validation", "reference_results.txt"),
]);
const ref = loadRef(refFile);
const dataFile = firstExisting([
  path.join(here, "..", "data", "dy2012.csv"),
  path.join(process.cwd(), "target_repo", "docs", "data", "dy2012.csv"),
  path.join(process.cwd(), "docs", "data", "dy2012.csv"),
]);
const x = readCSV(dataFile);

let passed = 0, failed = 0;
function flatF(v, out) {          // matritsalar — ustunma-ustun (R write.table kabi)
  out = out || [];
  if (!Array.isArray(v)) { out.push(v); return out; }
  if (v.length && Array.isArray(v[0])) {
    for (let c = 0; c < v[0].length; c++)
      for (let r = 0; r < v.length; r++) flatF(v[r][c], out);
  } else {
    for (const e of v) out.push(e);
  }
  return out;
}
function check(label, py, refv, tol) {
  tol = tol || 1e-8;
  const flat = flatF(py);
  const r = flatF(refv);
  if (flat.length !== r.length) {
    console.log(`FAIL ${label}: o'lcham ${flat.length} != ${r.length}`);
    failed++; return;
  }
  let maxErr = 0;
  for (let i = 0; i < flat.length; i++) {
    const d = Math.abs(flat[i] - r[i]) / Math.max(1, Math.abs(r[i]));
    if (d > maxErr) maxErr = d;
  }
  if (maxErr <= tol) { console.log(`OK   ${label}: maks.xato=${maxErr.toExponential(2)}`); passed++; }
  else {
    console.log(`FAIL ${label}: maks.xato=${maxErr.toExponential(2)}`);
    console.log("   js :", JSON.stringify(flat.slice(0, 8)));
    console.log("   ref:", JSON.stringify(r.slice(0, 8)));
    failed++;
  }
}

console.log(`dy2012: ${x.length} x ${x[0].length}`);

// ---------- 1. VAR
const fit = CC.VAR(x, 1);
check("VAR_B", fit.B, ref.VAR_B);
check("VAR_Q", fit.Q, ref.VAR_Q);

// ---------- 2. FEVD time
const fevd = CC.FEVDtime(fit.B, fit.Q, 10);
check("FEVD_time_gen", fevd, ref.FEVD_time_gen);

// ---------- 3. ConnectednessTable
const ct = CC.connectednessTable(fevd);
check("CT_TCI", ct.TCI, ref.CT_TCI, 1e-6);
check("CT_TO", ct.TO, ref.CT_TO, 1e-6);
check("CT_FROM", ct.FROM, ref.CT_FROM, 1e-6);
check("CT_NET", ct.NET, ref.CT_NET, 1e-6);
check("CT_NPT", ct.NPT, ref.CT_NPT);
check("CT_PCI", ct.PCI, ref.CT_PCI, 1e-6);
check("CT_INFLUENCE", ct.INFLUENCE, ref.CT_INFLUENCE, 1e-6);

// ---------- 4. TimeConnectedness (statik)
const tca = CC.timeConnectedness([fit.B], [fit.Q], 10);
check("TIME_TCI", tca.TCI, ref.TIME_TCI, 1e-6);

// ---------- 5. BayesPrior + TVPVAR
const prior = CC.bayesPrior(x, 1);
check("BAYESPRIOR_bprior", prior.bprior, ref.BAYESPRIOR_bprior, 1e-6);
const tf = CC.TVPVAR(x, { nlag: 1, kappa1: 0.99, kappa2: 0.99, prior });
check("TVPVAR_Bt_first", tf.B_t[0], ref.TVPVAR_Bt_first, 1e-6);
check("TVPVAR_Bt_last", tf.B_t[tf.B_t.length - 1], ref.TVPVAR_Bt_last, 1e-6);
check("TVPVAR_Qt_last", tf.Q_t[tf.Q_t.length - 1], ref.TVPVAR_Qt_last, 1e-6);
const ttdca = CC.timeConnectedness(tf.B_t, tf.Q_t, 10);
check("TVPTIME_TCI_head", ttdca.TCI.slice(0, 5), ref.TVPTIME_TCI_head, 1e-6);
check("TVPTIME_TCI_tail", ttdca.TCI.slice(-5), ref.TVPTIME_TCI_tail, 1e-6);

// Minnesota prior
const mp = CC.minnesotaPrior(0.1, 4, 1);
const mf = CC.TVPVAR(x, { nlag: 1, kappa1: 0.99, kappa2: 0.99, prior: mp });
check("TVPVAR_MINN_Bt_last", mf.B_t[mf.B_t.length - 1], ref.TVPVAR_MINN_Bt_last, 1e-6);

// ---------- 6. Chastota
const part = [Math.PI + 0.00001, Math.PI / 4, 0];
const fdca = CC.frequencyConnectedness([fit.B], [fit.Q], 100, part);
check("FREQ_TCI", fdca.TCI, ref.FREQ_TCI, 1e-6);
check("FREQ_CT_band2", fdca.CT[0][1].map(r => r.map(v => v / 100)), ref.FREQ_CT_band2, 1e-6);

// ---------- 7. Joint (nlag=2 VAR)
const fit2 = CC.VAR(x, 2);
const jdca = CC.jointConnectedness([fit2.B], [fit2.Q], 30);
check("JOINT_TCI", jdca.TCI, ref.JOINT_TCI, 1e-6);
check("JOINT_TO", jdca.TO, ref.JOINT_TO, 1e-6);
check("JOINT_FROM", jdca.FROM, ref.JOINT_FROM, 1e-6);
check("JOINT_NET", jdca.NET, ref.JOINT_NET, 1e-6);

// ---------- 8. Extended Joint (TVPVAR + Minnesota)
const mf2 = CC.TVPVAR(x, { nlag: 1, kappa1: 0.99, kappa2: 0.99, prior: CC.minnesotaPrior(0.1, 4, 1) });
const ejdca = CC.extendedJointConnectedness(mf2.B_t, mf2.Q_t, 20);
check("EXTJOINT_TCI_head", ejdca.TCI.slice(0, 200), ref.EXTJOINT_TCI.slice(0, 200), 1e-6);
check("EXTJOINT_NET_last", ejdca.NET[ejdca.NET.length - 1], ref.EXTJOINT_NET_last, 1e-6);

console.log(`\n${passed}/${passed + failed} test o'tdi`);
process.exit(failed ? 1 : 0);
