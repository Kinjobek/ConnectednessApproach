/* Canvas grafiklar — tashqi kutubxonasiz (TCI, NET, TO/FROM, NPT chiziqlari
 * va doiraviy bog'liqlik to'ri). */
(function (global) {
  "use strict";

  const PAL = ["#2456e6", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
               "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
               "#1b9e77", "#d95f02"];

  function setupCanvas(cv, h) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.parentElement.clientWidth - 2;
    cv.width = w * dpr;
    cv.height = h * dpr;
    cv.style.width = w + "px";
    cv.style.height = h + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function niceTicks(min, max, n) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min, step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const lo = Math.floor(min / step) * step;
    const hi = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = lo; v <= hi + step / 2; v += step) ticks.push(v);
    return { ticks, lo, hi };
  }

  /** Ko'p qatorli chiziqli grafik.
   * series: [{name, values[], color?}], labels: x o'qi yorliqlari (sana). */
  function lineChart(cv, opts) {
    const { ctx, w, h } = setupCanvas(cv, opts.height || 280);
    const pad = { l: 46, r: 12, t: 10, b: 34 };
    const series = opts.series, n = series[0].values.length;
    let lo = Infinity, hi = -Infinity;
    for (const s of series)
      for (const v of s.values) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (opts.zeroLine && lo > 0) lo = 0;
    if (opts.zeroLine && hi < 0) hi = 0;
    const t = niceTicks(lo, hi, 5);
    lo = t.lo; hi = t.hi;
    const xs = i => pad.l + (n <= 1 ? 0 : (w - pad.l - pad.r) * i / (n - 1));
    const ys = v => h - pad.b - (h - pad.t - pad.b) * (v - lo) / (hi - lo || 1);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "10px sans-serif";
    ctx.strokeStyle = "#e8ebf4";
    ctx.fillStyle = "#66708a";
    ctx.textAlign = "right";
    for (const tv of t.ticks) {
      const y = ys(tv);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(String(Math.round(tv * 100) / 100), pad.l - 6, y + 3);
    }
    // x yorliqlari
    const nTick = Math.min(8, n);
    ctx.textAlign = "center";
    for (let j = 0; j < nTick; j++) {
      const idx = Math.round((n - 1) * j / (nTick - 1 || 1));
      const lbl = (opts.labels && opts.labels[idx] ? String(opts.labels[idx]) : idx)
        .slice(0, 10);
      ctx.fillText(lbl, xs(idx), h - pad.b + 14);
    }
    // seriyalar
    for (let s = 0; s < series.length; s++) {
      const ser = series[s];
      const color = ser.color || PAL[s % PAL.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = ser.width || 1.5;
      ctx.globalAlpha = opts.fillFirst && s === 0 ? 1 : 0.9;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const X = xs(i), Y = ys(ser.values[i]);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.stroke();
      if (opts.fillFirst && s === 0 && n > 1) {
        ctx.globalAlpha = 0.12;
        ctx.fillStyle = color;
        ctx.lineTo(xs(n - 1), ys(lo)); ctx.lineTo(xs(0), ys(lo));
        ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    // nol chizig'i
    if (opts.zeroLine && lo < 0 && hi > 0) {
      ctx.strokeStyle = "#3b4560"; ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, ys(0)); ctx.lineTo(w - pad.r, ys(0)); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // legenda
    if (series.length > 1 && opts.legend !== false) {
      ctx.font = "10.5px sans-serif";
      const per = Math.min(6, series.length);
      const colW = (w - pad.l - pad.r) / per;
      let x = pad.l, y = 14;
      for (let s = 0; s < series.length; s++) {
        const color = series[s].color || PAL[s % PAL.length];
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 7, 9, 3);
        ctx.fillStyle = "#3c465f";
        ctx.textAlign = "left";
        const nm = series[s].name.length > 14 ? series[s].name.slice(0, 13) + "…" : series[s].name;
        ctx.fillText(nm, x + 13, y - 1);
        x += colW;
        if ((s + 1) % per === 0) { x = pad.l; y += 16; }
      }
    }
    return cv;
  }

  /** Doiraviy bog'liqlik to'ri. CT — o'rtacha FEVD (0..1), names, TO, NET. */
  function networkChart(cv, CT, names, TO, NET, threshold) {
    const size = setupCanvas(cv, 460);
    const { ctx, w, h } = size;
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 60;
    const k = CT.length;
    const ang = i => -Math.PI / 2 + 2 * Math.PI * i / k;
    const xs = i => cx + R * Math.cos(ang(i));
    const ys = i => cy + R * Math.sin(ang(i));
    ctx.clearRect(0, 0, w, h);
    // qirralar
    let wmax = 0;
    const W = [];
    for (let i = 0; i < k; i++) {
      W.push([]);
      for (let j = 0; j < k; j++) {
        const v = i === j ? 0 : (CT[i][j] + CT[j][i]) / 2 * 100;
        W[i].push(v);
        if (v > wmax) wmax = v;
      }
    }
    for (let i = 0; i < k; i++)
      for (let j = i + 1; j < k; j++) {
        const strength = W[i][j];
        if (strength < (threshold || 1)) continue;
        const lw = 0.5 + 4 * strength / (wmax || 1);
        const di = CT[i][j] * 100, dj = CT[j][i] * 100;
        ctx.globalAlpha = 0.55;
        if (Math.abs(di - dj) < 1e-9) {
          ctx.strokeStyle = "#8899aa"; ctx.lineWidth = lw;
          ctx.beginPath(); ctx.moveTo(xs(i), ys(i)); ctx.lineTo(xs(j), ys(j)); ctx.stroke();
        } else {
          // yo'nalishli qirra: kuchli manbadan maqsadga, egri strelka bilan
          const s = CT[i][j] > CT[j][i] ? j : i;
          const t = CT[i][j] > CT[j][i] ? i : j;
          const x1 = xs(s), y1 = ys(s), x2 = xs(t), y2 = ys(t);
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
          const bend = 0.10;
          ctx.strokeStyle = "#c44e52"; ctx.lineWidth = lw;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.quadraticCurveTo(mx + (y2 - y1) * bend, my - (x2 - x1) * bend, x2, y2);
          ctx.stroke();
          const ang2 = Math.atan2(y2 - my, x2 - mx);
          const head = 5 + lw * 1.2;
          ctx.fillStyle = "#c44e52";
          ctx.beginPath();
          ctx.moveTo(x2, y2);
          ctx.lineTo(x2 - head * Math.cos(ang2 - 0.5), y2 - head * Math.sin(ang2 - 0.5));
          ctx.lineTo(x2 - head * Math.cos(ang2 + 0.5), y2 - head * Math.sin(ang2 + 0.5));
          ctx.closePath(); ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
    // tugunlar
    const toMin = Math.min(...TO), toMax = Math.max(...TO);
    for (let i = 0; i < k; i++) {
      const r = 7 + 16 * (TO[i] - toMin) / (toMax - toMin || 1);
      const color = NET[i] > 0 ? "#c44e52" : "#55a868";
      ctx.fillStyle = color;
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(xs(i), ys(i), r, 0, 2 * Math.PI);
      ctx.fill(); ctx.stroke();
      // yorliq
      const lx = cx + (R + 22) * Math.cos(ang(i));
      const ly = cy + (R + 22) * Math.sin(ang(i));
      ctx.fillStyle = "#1c2333";
      ctx.font = "bold 10.5px sans-serif";
      ctx.textAlign = Math.abs(Math.cos(ang(i))) < 0.4 ? "center" :
        (Math.cos(ang(i)) > 0 ? "left" : "right");
      ctx.fillText(String(names[i]).slice(0, 12), lx, ly + 3);
    }
    return cv;
  }

  global.Charts = { lineChart, networkChart, PAL };
})(window);
