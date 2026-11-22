/* ConnectednessApproach — JavaScript dvigateli.
 *
 * R paketi (David Gabauer, 2022) va uning Python portining to'g'ridan-to'g'ri
 * tarjimasi. Node.js da R referens natijalari bilan validatsiya qilingan.
 * Matritsalar: qator-massivlar (number[][]). Natijalar: oddiy obyektlar.
 */
(function (global) {
  "use strict";

  // ================================================ matritsa yordamchilari
  const M = {
    zeros(r, c) { return Array.from({ length: r }, () => new Array(c).fill(0)); },
    eye(n) { const A = M.zeros(n, n); for (let i = 0; i < n; i++) A[i][i] = 1; return A; },
    T(A) {
      const r = A.length, c = A[0].length, B = M.zeros(c, r);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) B[j][i] = A[i][j];
      return B;
    },
    mul(A, B) {
      const n = A.length, m = B[0].length, p = B.length, C = M.zeros(n, m);
      for (let i = 0; i < n; i++) {
        const Ai = A[i], Ci = C[i];
        for (let k = 0; k < p; k++) {
          const a = Ai[k];
          if (a === 0) continue;
          const Bk = B[k];
          for (let j = 0; j < m; j++) Ci[j] += a * Bk[j];
        }
      }
      return C;
    },
    sub(A, B) { return A.map((r, i) => r.map((v, j) => v - B[i][j])); },
    add(A, B) { return A.map((r, i) => r.map((v, j) => v + B[i][j])); },
    scale(A, s) { return A.map(r => r.map(v => v * s)); },
    diag(v) { const n = v.length, A = M.zeros(n, n); for (let i = 0; i < n; i++) A[i][i] = v[i]; return A; },
    diagOf(A) { return A.map((r, i) => r[i]); },
    rowSums(A) { return A.map(r => r.reduce((a, b) => a + b, 0)); },
    colSums(A) {
      const c = A[0].length, s = new Array(c).fill(0);
      for (const r of A) for (let j = 0; j < c; j++) s[j] += r[j];
      return s;
    },
    meanCols(A) {
      const c = A[0].length, s = new Array(c).fill(0);
      for (const r of A) for (let j = 0; j < c; j++) s[j] += r[j];
      return s.map(v => v / A.length);
    },
    // Gauss–Jordan inversiya (qisman pivotlash)
    inv(A) {
      const n = A.length;
      const Aug = A.map((r, i) => r.concat(M.eye(n)[i]));
      for (let col = 0; col < n; col++) {
        let piv = col;
        for (let r = col + 1; r < n; r++)
          if (Math.abs(Aug[r][col]) > Math.abs(Aug[piv][col])) piv = r;
        if (Math.abs(Aug[piv][col]) < 1e-300) throw new Error("Singulyar matritsa");
        if (piv !== col) { const t = Aug[col]; Aug[col] = Aug[piv]; Aug[piv] = t; }
        const d = Aug[col][col];
        for (let j = 0; j < 2 * n; j++) Aug[col][j] /= d;
        for (let r = 0; r < n; r++) {
          if (r === col) continue;
          const f = Aug[r][col];
          if (f === 0) continue;
          const Rr = Aug[r], Rc = Aug[col];
          for (let j = 0; j < 2 * n; j++) Rr[j] -= f * Rc[j];
        }
      }
      return Aug.map(r => r.slice(n));
    },
    pinv(A) {                    // kvadrat uchun — R MASS::ginv o'rnini bosadi
      try { return M.inv(A); }
      catch (e) { return M.inv(M.add(A, M.scale(M.eye(A.length), 1e-12))); }
    },
    spectralRadius(A) {          // power iteratsiya — max |lambda|
      const n = A.length;
      let v = new Array(n).fill(1 / Math.sqrt(n));
      let rho = 0;
      for (let it = 0; it < 200; it++) {
        const w = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
          const Ai = A[i];
          let s = 0;
          for (let j = 0; j < n; j++) s += Ai[j] * v[j];
          w[i] = s;
        }
        let mx = 0;
        for (const x of w) mx = Math.max(mx, Math.abs(x));
        if (mx < 1e-300) return 0;
        for (let i = 0; i < n; i++) v[i] = w[i] / mx;
        rho = mx;
      }
      return rho;
    }
  };

  // ================================================ embed (R stats::embed)
  function embed(x, nlag) {
    const T = x.length, k = x[0].length, rows = T - nlag;
    if (rows <= 0) throw new Error("Ma'lumot lag sonidan katta bo'lishi kerak");
    const out = Array.from({ length: rows }, () => new Array(k * (nlag + 1)).fill(0));
    for (let i = 0; i <= nlag; i++)
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < k; c++)
          out[r][i * k + c] = x[nlag - i + r][c];
    return out;
  }

  // ================================================ VAR (OLS)
  function VAR(x, nlag) {
    const k = x[0].length;
    const z = embed(x, nlag), n = z.length, p = nlag * k;
    const X = z.map(r => r.slice(k));
    const mu = M.meanCols(X);
    const ymu = new Array(k).fill(0);
    for (let t = 0; t < n; t++) for (let c = 0; c < k; c++) ymu[c] += z[t][c];
    for (let c = 0; c < k; c++) ymu[c] /= n;
    const XtX = M.zeros(p, p), XtY = M.zeros(p, k);
    for (let t = 0; t < n; t++) {
      const xt = X[t];
      for (let i = 0; i < p; i++) {
        const xi = xt[i] - mu[i];
        for (let j = i; j < p; j++) XtX[i][j] += xi * (xt[j] - mu[j]);
        for (let c = 0; c < k; c++) XtY[i][c] += xi * (z[t][c] - ymu[c]);
      }
    }
    for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) XtX[i][j] = XtX[j][i];
    const coef = M.mul(M.inv(XtX), XtY);            // (p, k)
    const resid = Array.from({ length: n }, () => new Array(k).fill(0));
    for (let t = 0; t < n; t++)
      for (let c = 0; c < k; c++) {
        let s = z[t][c] - ymu[c];
        for (let i = 0; i < p; i++) s -= (X[t][i] - mu[i]) * coef[i][c];
        resid[t][c] = s;
      }
    const Q = M.zeros(k, k);
    for (let t = 0; t < n; t++)
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++) Q[i][j] += resid[t][i] * resid[t][j];
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) Q[i][j] /= n;
    return { B: M.T(coef), Q, resid };
  }

  // ================================================ IRF
  function companionIRF(Phi, nfore) {
    const k = Phi.length, p = Phi[0].length / k;
    const psi = [M.eye(k)];
    for (let h = 1; h <= nfore; h++) {
      const acc = M.zeros(k, k);
      const iend = Math.min(h, p);
      for (let j = 1; j <= iend; j++) {
        const A = M.zeros(k, k);
        for (let r = 0; r < k; r++)
          for (let c = 0; c < k; c++) A[r][c] = Phi[r][(j - 1) * k + c];
        const P = M.mul(A, psi[h - j]);
        for (let r = 0; r < k; r++) for (let c = 0; c < k; c++) acc[r][c] += P[r][c];
      }
      psi.push(acc);
    }
    return psi;
  }

  function IRF(Phi, nfore) {
    const psi = companionIRF(Phi, nfore);
    const k = Phi.length;
    const irf = Array.from({ length: k }, (_, i) =>
      Array.from({ length: nfore + 1 }, (_, h) =>
        psi[h].map(row => row[i])));                // irf[i][h][var] = Psi_h[var][i]
    return irf;
  }

  // ================================================ FEVD — vaqt sohasi
  function FEVDtime(Phi, Sigma, nfore) {
    const irf = IRF(Phi, nfore);
    const k = Sigma.length, sdiag = M.diagOf(Sigma);
    const denom = new Array(k).fill(0);
    const enumM = M.zeros(k, k);
    for (let h = 0; h < nfore; h++) {
      // M_h[dest][shk] = Psi_h[dest][shk];  P = M_h Sigma
      const P = M.zeros(k, k);
      for (let d = 0; d < k; d++)
        for (let s = 0; s < k; s++) {
          let acc = 0;
          for (let c = 0; c < k; c++) acc += irf[c][h][d] * Sigma[c][s];
          P[d][s] = acc;
        }
      for (let d = 0; d < k; d++) {
        let mm = 0;
        for (let s = 0; s < k; s++)
          for (let c = 0; c < k; c++) mm += irf[s][h][d] * Sigma[s][c] * irf[c][h][d];
        denom[d] += mm;
        for (let s = 0; s < k; s++) enumM[d][s] += P[d][s] * P[d][s];
      }
    }
    // theta[dest][shk] va qator-normalizatsiya
    const fevd = M.zeros(k, k);
    for (let d = 0; d < k; d++) {
      let rowsum = 0;
      for (let s = 0; s < k; s++) {
        fevd[d][s] = enumM[d][s] / (denom[d] * sdiag[s]);
        rowsum += fevd[d][s];
      }
      for (let s = 0; s < k; s++) fevd[d][s] /= rowsum;
    }
    return fevd;
  }

  // ================================================ getPartition (chastota)
  function getIndices(nAhead, up, down) {
    const half = Math.max(1, Math.floor(nAhead / 2));
    const sel = [];
    for (let j = 0; j <= half; j++) {
      const w = j / half * Math.PI;
      sel.push(w >= down && w < up);
    }
    const full = new Array(nAhead).fill(false);
    for (let j = 0; j <= half; j++) full[j] = sel[j];
    if (nAhead % 2 === 0) {
      for (let j = 0; j < half - 1; j++) full[half + 1 + j] = sel[half - 1 - j];
    } else {
      for (let j = 0; j < half; j++) full[half + 1 + j] = sel[half - j];
    }
    const idx = [];
    for (let i = 0; i < nAhead; i++) if (full[i]) idx.push(i);
    return idx;
  }

  function getPartition(partition, nfore) {
    const N = nfore + 1;
    const part = [];
    let total = 0;
    for (let i = 0; i < partition.length - 1; i++) {
      const idx = getIndices(N, partition[i], partition[i + 1]);
      if (idx.length === 0) throw new Error("Bo'linishni o'zgartiring — oraliq bo'sh");
      total += idx.length;
      part.push(idx);
    }
    if (total !== N)
      console.warn("Tanlangan bo'linish butun diapazonni qamrab olmaydi (R dagi kabi).");
    return part;
  }

  // ================================================ FEVD — chastota sohasi
  function FEVDFreq(Phi, Sigma, nfore, rng) {
    const irf = IRF(Phi, nfore);
    const k = Sigma.length, sdiag = M.diagOf(Sigma), N = nfore + 1;
    // F[w][dest][shk] — kompleks (re, im juftligi sifatida ikki massiv)
    const Fre = Array.from({ length: N }, () => M.zeros(k, k));
    const Fim = Array.from({ length: N }, () => M.zeros(k, k));
    for (let w = 0; w < N; w++) {
      for (let s = 0; s < k; s++)
        for (let d = 0; d < k; d++) {
          let re = 0, im = 0;
          const series = irf[s];
          for (let h = 0; h < N; h++) {
            const ang = -2 * Math.PI * w * h / N;
            const v = series[h][d];
            re += v * Math.cos(ang);
            im += v * Math.sin(ang);
          }
          Fre[w][d][s] = re;
          Fim[w][d][s] = im;
        }
    }
    const denom = new Array(k).fill(0);
    for (const w of rng) {
      for (let d = 0; d < k; d++) {
        let re = 0;
        for (let s = 0; s < k; s++)
          for (let c = 0; c < k; c++) {
            const a = Fre[w][d][s], ai = Fim[w][d][s];
            const b = Fre[w][d][c], bi = -Fim[w][d][c];   // conj
            const sr = Sigma[s][c];
            re += sr * (a * b - ai * bi);
          }
        denom[d] += re / nfore;
      }
    }
    const tot = new Array(k).fill(0);
    const tabs = {};
    for (const w of rng) {
      const tab = M.zeros(k, k);
      for (let d = 0; d < k; d++) {
        for (let s = 0; s < k; s++) {
          let are = 0, aim = 0;                        // (F_w Sigma)[d][s]
          for (let c = 0; c < k; c++) {
            are += Fre[w][d][c] * Sigma[c][s];
            aim += Fim[w][d][c] * Sigma[c][s];
          }
          const mag2 = (are * are + aim * aim) / (nfore + 1);
          tab[d][s] = mag2 / (denom[d] * sdiag[s]);
        }
      }
      tabs[w] = tab;
      for (let d = 0; d < k; d++) {
        let s = 0;
        for (let j = 0; j < k; j++) s += tab[d][j];
        tot[d] += s;
      }
    }
    // FEVD_w[dest][shk] = tab[dest][shk] / tot[dest]
    const fevd = {};
    for (const w of rng) {
      const f = M.zeros(k, k);
      for (let d = 0; d < k; d++)
        for (let s = 0; s < k; s++) f[d][s] = tabs[w][d][s] / tot[d];
      fevd[w] = f;
    }
    return fevd;
  }

  // ================================================ Bog'liqlik jadvali
  function fmt(v, digit) {
    digit = digit || 2;
    return (Math.round(v * Math.pow(10, digit)) / Math.pow(10, digit))
      .toFixed(digit);
  }

  function connectednessTable(FEVDm, names) {
    const k = FEVDm.length;
    if (!names) names = Array.from({ length: k }, (_, i) => String(i + 1));
    const CT = FEVDm.map(r => r.map(v => v * 100));
    const TO = new Array(k).fill(0), FROM = new Array(k).fill(0);
    const NET = new Array(k).fill(0), NPT = new Array(k).fill(0);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        if (i !== j) { TO[j] += CT[i][j]; FROM[i] += CT[i][j]; }
      }
    }
    for (let i = 0; i < k; i++) NET[i] = TO[i] - FROM[i];
    for (let i = 0; i < k; i++) {
      let n = 0;
      for (let j = 0; j < k; j++) if (CT[i][j] - CT[j][i] < 0) n++;
      NPT[i] = n;
    }
    const TCI = TO.reduce((a, b) => a + b, 0) / k;
    const cTCI = TCI * k / (k - 1);
    const NPDC = M.zeros(k, k), INFLUENCE = M.zeros(k, k), PCI = M.zeros(k, k);
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++) {
        NPDC[i][j] = CT[i][j] - CT[j][i];
        INFLUENCE[i][j] = 100 * Math.abs(NPDC[i][j]) / (CT[i][j] + CT[j][i]);
        PCI[i][j] = 200 * (CT[i][j] + CT[j][i]) /
          (CT[i][i] + CT[i][j] + CT[j][i] + CT[j][j]);
      }
    // TABLE — matnli massiv: (k+4) x (k+1)
    const table = [];
    for (let i = 0; i < k; i++)
      table.push(CT[i].map(v => fmt(v)).concat([fmt(FROM[i])]));
    table.push(TO.map(v => fmt(v)).concat([fmt(TO.reduce((a, b) => a + b, 0))]));
    table.push(M.colSums(CT).map(v => fmt(v)).concat(["cTCI/TCI"]));
    table.push(NET.map(v => fmt(v))
      .concat([fmt(cTCI) + "/" + fmt(TCI)]));
    table.push(NPT.map(v => fmt(v)).concat([""]));
    return { FEVD: CT, TCI, cTCI, TO, FROM, NET, NPDC, NPT, PCI, INFLUENCE,
             TABLE: { rows: table, rowNames: names.concat(["TO", "Inc.Own", "NET", "NPT"]),
                      colNames: names.concat(["FROM"]) } };
  }

  // ================================================ TimeConnectedness
  function timeConnectedness(PhiT, SigT, nfore, names, dates, corrected) {
    const t = SigT.length, k = SigT[0].length;
    const TCI = new Array(t), TO = [], FROM = [], NET = [], NPT = [];
    const NPDC = [], PCI = [], INFL = [];
    const CTs = [];                                  // har bir nuqtadagi FEVD
    const avgFEVD = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < t; i++) {
      const fevd = FEVDtime(PhiT[i], SigT[i], nfore);
      CTs.push(fevd);
      for (let a = 0; a < k; a++)
        for (let b = 0; b < k; b++) avgFEVD[a][b] += fevd[a][b] / t;
      const ct = connectednessTable(fevd, names);
      TCI[i] = corrected ? ct.cTCI : ct.TCI;
      TO.push(ct.TO); FROM.push(ct.FROM); NET.push(ct.NET); NPT.push(ct.NPT);
      NPDC.push(ct.NPDC); PCI.push(ct.PCI); INFL.push(ct.INFLUENCE);
    }
    const table = connectednessTable(avgFEVD, names);
    return { TCI, TO, FROM, NET, NPT, NPDC, PCI, INFLUENCE: INFL,
             CT: CTs, avgFEVD, TABLE: table.TABLE,
             config: { approach: "Time", nfore, names, dates } };
  }

  // ================================================ FrequencyConnectedness
  function frequencyConnectedness(PhiT, SigT, nfore, partition, names, dates, scenario, corrected) {
    const t = SigT.length, k = SigT[0].length;
    const periodNames = ["Total"];
    for (let i = 0; i < partition.length - 1; i++) {
      const p1 = Math.round(Math.PI / partition[i]);
      const p2 = partition[i + 1] === 0 ? "∞" : Math.round(Math.PI / partition[i + 1]);
      periodNames.push(p1 + "-" + p2);
    }
    const interval = periodNames.length, nb = interval - 1;
    const newP = getPartition(partition, nfore);
    const rngSet = new Set();
    for (const band of newP) for (const w of band) rngSet.add(w);
    const rng = [...rngSet].sort((a, b) => a - b);

    const TCI = Array.from({ length: t }, () => new Array(interval).fill(0));
    const mk = () => Array.from({ length: interval }, () => new Array(k).fill(0));
    const TO = Array.from({ length: t }, mk);
    const FROM = Array.from({ length: t }, mk);
    const NET = Array.from({ length: t }, mk);
    const NPT = Array.from({ length: t }, () =>
      Array.from({ length: nb }, () => new Array(k).fill(0)));
    const CTs = [];
    for (let i = 0; i < t; i++) {
      const decomp = FEVDFreq(PhiT[i], SigT[i], nfore, rng);
      const CTt = Array.from({ length: interval }, () => M.zeros(k, k));
      for (let b = 0; b < nb; b++) {
        let tb = M.zeros(k, k);
        for (const w of newP[b])
          for (let d = 0; d < k; d++)
            for (let s = 0; s < k; s++) tb[d][s] += decomp[w][d][s];
        let useTb = tb;
        if (scenario === "WTH") {
          let sum = 0;
          for (const r of tb) for (const v of r) sum += v;
          useTb = M.scale(tb, k / sum);
        }
        const ct = connectednessTable(useTb, names);
        CTt[b + 1] = ct.FEVD;
        TO[i][b + 1] = ct.TO; FROM[i][b + 1] = ct.FROM; NET[i][b + 1] = ct.NET;
        NPT[i][b] = ct.NPT;
        TCI[i][b + 1] = corrected ? ct.cTCI : ct.TCI;
      }
      for (let j = 1; j < interval; j++)
        for (let d = 0; d < k; d++)
          for (let s = 0; s < k; s++) CTt[0][d][s] += CTt[j][d][s];
      CTs.push(CTt);
    }
    for (let i = 0; i < t; i++) {
      TCI[i][0] = TCI[i].slice(1).reduce((a, b) => a + b, 0);
      for (let j = 1; j < interval; j++)
        for (let d = 0; d < k; d++) {
          TO[i][0][d] += TO[i][j][d];
          FROM[i][0][d] += FROM[i][j][d];
          NET[i][0][d] += NET[i][j][d];
        }
    }
    return { TCI, TO, FROM, NET, NPT, CT: CTs,
             config: { approach: "Frequency", nfore, names, dates, periodNames, partition } };
  }

  // ================================================ Priors
  function minnesotaPrior(gamma, k, nlag) {
    const m = nlag * k * k;
    const bprior = new Array(m).fill(0);
    const Vdiag = new Array(m);
    for (let c = 0; c < m; c++) {
      const L = Math.floor(c / (k * k));
      Vdiag[c] = gamma / Math.pow(L + 1, 2);
    }
    return { bprior, Vprior: M.diag(Vdiag), Q: null };
  }

  function uninformativePrior(k, nlag) {
    const m = nlag * k * k;
    return { bprior: new Array(m).fill(0),
             Vprior: M.scale(M.eye(m), 4), Q: null };
  }

  function bayesPrior(x, nlag) {
    const T = x.length, p = x[0].length;
    const size = T - nlag;
    if (size <= 0) throw new Error("size nlag dan katta bo'lishi kerak");
    const m = p + nlag * p * p;
    const Zt = M.zeros(size * p, m);
    const yflat = new Array(size * p);
    for (let idx = 0; idx < size; idx++) {
      const i = nlag + idx;
      for (let eq = 0; eq < p; eq++) {
        Zt[idx * p + eq][eq] = 1;                  // identifikatsiya bloki
        yflat[idx * p + eq] = x[i][eq];
        for (let j = 1; j <= nlag; j++) {           // lag bloklari
          const base = p + (j - 1) * p * p;
          for (let jj = 0; jj < p; jj++)
            for (let v = 0; v < p; v++)
              Zt[idx * p + eq][base + jj * p + v] =
                (jj === eq) ? x[i - j][v] : 0;
        }
      }
    }
    const vbar = M.zeros(m, m), xhy = new Array(m).fill(0);
    for (let r = 0; r < size * p; r++) {
      for (let c = 0; c < m; c++) {
        const zr = Zt[r][c];
        if (zr === 0) continue;
        xhy[c] += zr * yflat[r];
        for (let c2 = c; c2 < m; c2++) vbar[c][c2] += zr * Zt[r][c2];
      }
    }
    for (let c = 0; c < m; c++) for (let c2 = 0; c2 < c; c2++) vbar[c][c2] = vbar[c2][c];
    const vbarInv = M.pinv(vbar);
    const aols = M.mul(vbarInv, xhy.map(v => [v])).map(r => r[0]);
    const rflat = yflat.map((y, r) => {
      let s = y;
      for (let c = 0; c < m; c++) s -= Zt[r][c] * aols[c];
      return s;
    });
    const sse = M.zeros(p, p);
    for (let idx = 0; idx < size; idx++)
      for (let i = 0; i < p; i++)
        for (let j = 0; j < p; j++)
          sse[i][j] += rflat[idx * p + i] * rflat[idx * p + j];
    const hbar = M.scale(sse, 1 / size);
    const Vp = vbarInv.slice(p).map(r => r.slice(p));
    return { bprior: aols.slice(p), Vprior: Vp, Q: hbar };
  }

  // ================================================ TVP-VAR
  function createRHS(templag, r, nlag, t) {
    const K = nlag * r * r, nobs = t - nlag;
    const x_t = M.zeros(nobs * r, K);
    for (let i = 0; i < nobs; i++) {
      const block = M.zeros(r, K);
      for (let j = 0; j < nlag; j++) {
        for (let eq = 0; eq < r; eq++)
          for (let v = 0; v < r; v++)
            block[eq][j * r * r + eq * r + v] = templag[i][j * r + v];
      }
      for (let rr = 0; rr < r; rr++)
        x_t[i * r + rr] = block[rr].slice();
    }
    return x_t;
  }

  function TVPVAR(x, cfg) {
    const nlag = cfg.nlag, l1 = cfg.kappa1, l2 = cfg.kappa2;
    const T0 = x.length, r = x[0].length;
    let prior = cfg.prior;
    if (!prior) prior = uninformativePrior(r, nlag);
    const beta0 = prior.bprior, V0 = prior.Vprior;
    let Q0 = prior.Q;
    if (!Q0) {
      Q0 = M.zeros(r, r);
      const mu = M.meanCols(x);
      for (let t = 0; t < T0; t++)
        for (let i = 0; i < r; i++)
          for (let j = 0; j < r; j++)
            Q0[i][j] += (x[t][i] - mu[i]) * (x[t][j] - mu[j]);
      for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) Q0[i][j] /= (T0 - 1);
    }
    const m = nlag * r * r, kk = nlag * r, t = T0;
    const xc = x.map(row => row.map((v, i) => v - 0)); // markazlash pastda
    const mu = M.meanCols(x);
    for (let t2 = 0; t2 < T0; t2++) for (let i = 0; i < r; i++) xc[t2][i] = x[t2][i] - mu[i];
    const yy = xc.slice(nlag);
    const xx = embed(xc, nlag).map(row => row.slice(r));
    const Flagtemp = createRHS(xx, r, nlag, t);
    const Flag = M.zeros(t * r, m);
    for (let i = nlag * r; i < t * r; i++) Flag[i] = Flagtemp[i - nlag * r].slice();

    const beta_pred = M.zeros(m, t), beta_upd = M.zeros(m, t);
    const Rb = [], Sb = [];
    const beta_t = [], Q_t = [];
    let B = null;

    for (let i = 0; i < t; i++) {
      // Q_t
      const Qi = M.zeros(r, r);
      if (i === 0) {
        for (let a = 0; a < r; a++) for (let b = 0; b < r; b++) Qi[a][b] = Q0[a][b];
      } else {
        if (i <= nlag) {
          for (let a = 0; a < r; a++)
            for (let b = 0; b < r; b++) Qi[a][b] = 0.1 * xc[i][a] * xc[i][b];
        } else {
          const res = new Array(r).fill(0);
          const A = B ? B.slice(0, r) : M.zeros(r, kk);
          for (let a = 0; a < r; a++) {
            let s = yy[i - nlag][a];
            for (let c = 0; c < kk; c++) s -= xx[i - nlag][c] * A[a][c];
            res[a] = s;
          }
          for (let a = 0; a < r; a++)
            for (let b = 0; b < r; b++) Qi[a][b] = res[a] * res[b];
        }
        const Qprev = Q_t[i - 1];
        for (let a = 0; a < r; a++)
          for (let b = 0; b < r; b++)
            Qi[a][b] = l2 * Qprev[a][b] + (1 - l2) * Qi[a][b];
      }
      Q_t.push(Qi);
      // beta holati
      const Rbi = M.zeros(m, m);
      if (i <= nlag) {
        for (let c = 0; c < m; c++) {
          beta_pred[c][i] = beta0[c];
          beta_upd[c][i] = beta0[c];
          for (let c2 = 0; c2 < m; c2++) Rbi[c][c2] = V0[c][c2];
        }
      } else {
        for (let c = 0; c < m; c++) beta_pred[c][i] = beta_upd[c][i - 1];
        const Sprev = Sb[i - 1];
        for (let c = 0; c < m; c++)
          for (let c2 = 0; c2 < m; c2++) Rbi[c][c2] = Sprev[c][c2] / l1;
      }
      Rb.push(Rbi);
      // Kalman yangilanishi
      if (i >= nlag) {
        const Fb = M.zeros(r, m);
        for (let rr = 0; rr < r; rr++)
          for (let c = 0; c < m; c++) Fb[rr][c] = Flag[i * r + rr][c];
        const FbT = M.T(Fb);
        const Rx = M.mul(Rbi, FbT);
        const KV = M.add(Qi, M.mul(Fb, Rx));
        const KG = M.mul(Rx, M.pinv(KV));
        const innov = new Array(r);
        for (let rr = 0; rr < r; rr++) {
          let s = xc[i][rr];
          for (let c = 0; c < m; c++) s -= Fb[rr][c] * beta_pred[c][i];
          innov[rr] = s;
        }
        const Sbi = M.zeros(m, m);
        for (let c = 0; c < m; c++) {
          let s = beta_pred[c][i];
          for (let rr = 0; rr < r; rr++) s += KG[c][rr] * innov[rr];
          beta_upd[c][i] = s;
        }
        const FR = M.mul(Fb, Rbi);
        for (let c = 0; c < m; c++)
          for (let c2 = 0; c2 < m; c2++) {
            let s = Rbi[c][c2];
            for (let rr = 0; rr < r; rr++) s -= KG[c][rr] * FR[rr][c2];
            Sbi[c][c2] = s;
          }
        Sb.push(Sbi);
      } else {
        Sb.push(M.zeros(m, m));
      }
      // biga yig'ish
      const biga = M.zeros(r, kk);
      let sp = 0;
      for (let ii = 0; ii < nlag; ii++)
        for (let iii = 0; iii < r; iii++) {
          for (let v = 0; v < r; v++) biga[iii][ii * r + v] = beta_upd[sp + v][i];
          sp += r;
        }
      // hamroh matritsa
      B = M.zeros(kk, kk);
      for (let rr = 0; rr < r; rr++)
        for (let c = 0; c < kk; c++) B[rr][c] = biga[rr][c];
      for (let rr = r; rr < kk; rr++) B[rr][rr - r] = 1;
      const rho = M.spectralRadius(B);
      if (rho <= 1 || i === 0) {
        beta_t.push(B.map(row => row.slice()));
      } else {
        beta_t.push(beta_t[i - 1].map(row => row.slice()));
        for (let c = 0; c < m; c++) beta_upd[c][i] = 0.99 * beta_upd[c][i - 1];
      }
    }
    const B_t = beta_t.map(Bm => Bm.slice(0, r).map(row => row.slice(0, kk)));
    return { B_t, Q_t };
  }

  // ================================================ Wold + Joint
  function jointConnectedness(PhiT, SigT, nfore, names, dates) {
    const t = SigT.length, k = SigT[0].length;
    const TCI = new Array(t), TO = [], FROM = [], NET = [];
    for (let ij = 0; ij < t; ij++) {
      const Phi = PhiT[ij], Sig = SigT[ij];
      const gSOT = M.scale(FEVDtime(Phi, Sig, nfore), 100);
      let gSOI = 0;
      for (let d = 0; d < k; d++) {
        let s = 0;
        for (let s2 = 0; s2 < k; s2++) if (s2 !== d) s += gSOT[d][s2];
        gSOI += s;
      }
      gSOI /= k;
      const psi = companionIRF(Phi, nfore);
      const Xi = M.zeros(k, k);
      for (let h = 1; h <= nfore; h++) {
        const Ph = psi[h - 1];                       // R: A[,,h] = Psi_{h-1}
        const P1 = M.mul(Ph, Sig), P2 = M.mul(P1, M.T(Ph));
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) Xi[a][b] += P2[a][b];
      }
      const Sjnt = new Array(k).fill(0);
      for (let i = 0; i < k; i++) {
        // M_i: k x (k-1) — I dan i-ustunni olib tashlash
        const Mi = M.zeros(k, k - 1);
        let col = 0;
        for (let c = 0; c < k; c++) {
          if (c === i) continue;
          for (let rr = 0; rr < k; rr++) Mi[rr][col] = rr === c ? 1 : 0;
          col++;
        }
        const MiT = M.T(Mi);
        const MiTSMi = M.pinv(M.mul(MiT, M.mul(Sig, Mi)));
        let num = 0;
        for (let h = 1; h <= nfore; h++) {
          const Ph = psi[h - 1];
          // v = I_K[i] A_h Sigma Mi (Mi' Sigma Mi)^-1 Mi' Sigma A_h' I_K[i]
          const P1 = M.mul(Ph, Sig);                 // A Sigma
          const P2 = M.mul(P1, Mi);                   // (k x k-1)
          const P3 = M.mul(P2, MiTSMi);
          const P4 = M.mul(P3, MiT);                  // (k x k)
          const P5 = M.mul(P4, Sig);
          const P6 = M.mul(P5, M.T(Ph));             // A Sigma Mi ginv Mi' Sigma A'
          num += P6[i][i];
        }
        Sjnt[i] = 100 * num / Xi[i][i];
      }
      const jSOI = Sjnt.reduce((a, b) => a + b, 0) / k;
      const lam = gSOI / jSOI;
      TCI[ij] = jSOI;
      const to = new Array(k).fill(0), from = new Array(k).fill(0);
      for (let d = 0; d < k; d++) {
        let s = 0;
        for (let s2 = 0; s2 < k; s2++) if (s2 !== d) s += gSOT[s2][d] / lam;
        to[d] = s;
        let s2 = 0;
        for (let j = 0; j < k; j++) if (j !== d) s2 += gSOT[d][j];
        from[d] = Sjnt[d];
      }
      TO.push(to); FROM.push(from);
      NET.push(to.map((v, i2) => v - from[i2]));
    }
    return { TCI, TO, FROM, NET,
             config: { approach: "Joint", nfore, names, dates } };
  }

  function extendedJointConnectedness(PhiT, SigT, nfore, names, dates, corrected) {
    const t = SigT.length, k = SigT[0].length;
    const TCI = new Array(t), TO = [], FROM = [], NET = [], NPT = [];
    const CTall = [];
    for (let ij = 0; ij < t; ij++) {
      const Phi = PhiT[ij], Sig = SigT[ij];
      const gSOT = M.scale(FEVDtime(Phi, Sig, nfore), 100);
      const psi = companionIRF(Phi, nfore);
      const Xi = M.zeros(k, k);
      for (let h = 1; h <= nfore; h++) {
        const P1 = M.mul(psi[h - 1], Sig), P2 = M.mul(P1, M.T(psi[h - 1]));
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) Xi[a][b] += P2[a][b];
      }
      const Sjnt = new Array(k).fill(0);
      for (let i = 0; i < k; i++) {
        const Mi = M.zeros(k, k - 1);
        let col = 0;
        for (let c = 0; c < k; c++) {
          if (c === i) continue;
          for (let rr = 0; rr < k; rr++) Mi[rr][col] = rr === c ? 1 : 0;
          col++;
        }
        const MiT = M.T(Mi);
        const MiTSMi = M.pinv(M.mul(MiT, M.mul(Sig, Mi)));
        let num = 0;
        for (let h = 1; h <= nfore; h++) {
          const P1 = M.mul(psi[h - 1], Sig);
          const P2 = M.mul(P1, Mi);
          const P3 = M.mul(P2, MiTSMi);
          const P4 = M.mul(P3, MiT);
          const P5 = M.mul(P4, Sig);
          const P6 = M.mul(P5, M.T(psi[h - 1]));
          num += P6[i][i];
        }
        Sjnt[i] = 100 * num / Xi[i][i];
      }
      const lam = new Array(k);
      for (let i = 0; i < k; i++) {
        let s = 0;
        for (let j = 0; j < k; j++) if (j !== i) s += gSOT[i][j];
        lam[i] = Sjnt[i] / s;
      }
      const jSOT = M.zeros(k, k);
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++) jSOT[i][j] = gSOT[i][j] * lam[i];
      const fromJ = new Array(k).fill(0), toJ = new Array(k).fill(0);
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          if (i !== j) { fromJ[i] += jSOT[i][j]; toJ[j] += jSOT[i][j]; }
        }
      }
      const jSOTf = jSOT.map((r, i) => r.map((v, j) => i === j ? 100 - fromJ[i] : v));
      const ct = connectednessTable(M.scale(jSOTf, 0.01), names);
      TCI[ij] = ct.TCI;
      TO.push(ct.TO); FROM.push(ct.FROM); NET.push(ct.NET); NPT.push(ct.NPT);
      CTall.push(ct.FEVD);
    }
    return { TCI, TO, FROM, NET, NPT, CT: CTall,
             config: { approach: "Extended Joint", nfore, names, dates } };
  }

  // ================================================ Umumiy wrapper
  function connectednessApproach(x, opts) {
    opts = opts || {};
    const nlag = opts.nlag || 1, nfore = opts.nfore || 10;
    const model = opts.model || "VAR";
    const approach = opts.connectedness || "Time";
    const names = opts.names, dates = opts.dates;
    const T = x.length, k = x[0].length;
    let windowSize = opts.window_size || T;
    let t0 = 1;
    if (opts.window_size) { windowSize = opts.window_size - nlag; t0 = T - windowSize; }
    let B_t = [], Q_t = [];
    if (model === "TVP-VAR") {
      let prior = null;
      const pname = (opts.VAR_config && opts.VAR_config.TVPVAR &&
                     opts.VAR_config.TVPVAR.prior) || "BayesPrior";
      if (pname === "MinnesotaPrior")
        prior = minnesotaPrior(opts.VAR_config.TVPVAR.gamma || 0.1, k, nlag);
      else if (pname === "UninformativePrior")
        prior = uninformativePrior(k, nlag);
      else
        prior = bayesPrior(x, nlag);
      const fit = TVPVAR(x, { nlag, kappa1: opts.VAR_config.TVPVAR.kappa1 || 0.99,
                              kappa2: opts.VAR_config.TVPVAR.kappa2 || 0.99,
                              prior });
      B_t = fit.B_t; Q_t = fit.Q_t;
    } else {
      for (let i = 0; i < t0; i++) {
        const win = x.slice(i, i + windowSize);
        const fit = VAR(win, nlag);
        B_t.push(fit.B); Q_t.push(fit.Q);
      }
    }
    let dca;
    if (approach === "Time") {
      dca = timeConnectedness(B_t, Q_t, nfore, names, dates, !!opts.corrected);
    } else if (approach === "Frequency") {
      const fc = (opts.Connectedness_config && opts.Connectedness_config.FrequencyConnectedness) || {};
      dca = frequencyConnectedness(B_t, Q_t, nfore, fc.partition || [Math.PI, Math.PI / 2, 0],
                                    names, dates, fc.scenario || "ABS", !!opts.corrected);
    } else if (approach === "Joint") {
      dca = jointConnectedness(B_t, Q_t, nfore, names, dates);
    } else {
      dca = extendedJointConnectedness(B_t, Q_t, nfore, names, dates);
    }
    dca.config.model = model;
    dca.config.nlag = nlag;
    return dca;
  }

  const CC = {
    M, embed, VAR, IRF, FEVDtime, getIndices, getPartition, FEVDFreq,
    connectednessTable, timeConnectedness, frequencyConnectedness,
    minnesotaPrior, uninformativePrior, bayesPrior, TVPVAR,
    jointConnectedness, extendedJointConnectedness, connectednessApproach, fmt
  };
  if (typeof module !== "undefined") module.exports = CC;
  global.CC = CC;
})(typeof window !== "undefined" ? window : globalThis);
