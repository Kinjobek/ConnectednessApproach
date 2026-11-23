/* Canvas grafiklar — tashqi kutubxonasiz.
 * Sifat: legenda alohida joyda (ma'lumot ustiga tushmaydi), sana yorliqlari
 * qiya, statik natija uchun ustunli (bar) grafiklar, to' grafigi faqat eng
 * kuchli qirralar bilan. */
(function (global) {
  "use strict";

  const PAL = ["#2456e6", "#d64545", "#0e9f6e", "#8045d6", "#e08214",
               "#5aa8d6", "#d65998", "#14b8c4", "#a3a324", "#8a8f9e",
               "#1b9e77", "#d95f02"];
  const C_TO = "#2456e6", C_FROM = "#e08214";

  function setupCanvas(cv, h) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(320, cv.parentElement.clientWidth - 2);
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

  /** Ko'p qatorli chiziqli grafik — legenda tepada alohida qatorda. */
  function lineChart(cv, opts) {
    const H = opts.height || 300;
    const series = opts.series, n = series[0].values.length;
    // legenda uchun joy ajratish (ma'lumot ustiga tushmaydi)
    const perRow = Math.min(6, series.length);
    const legendRows = series.length > 1 ? Math.ceil(series.length / perRow) : 0;
    const hasLegend = legendRows > 0;
    const { ctx, w, h } = setupCanvas(cv, H);
    const pad = { l: 52, r: 14, t: 14 + legendRows * 17, b: 52 };

    let lo = Infinity, hi = -Infinity;
    for (const s of series)
      for (const v of s.values) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo === hi) { lo -= 1; hi += 1; }
    if (opts.zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    const t = niceTicks(lo, hi, 5);
    lo = t.lo; hi = t.hi;
    const xs = i => pad.l + (n <= 1 ? (w - pad.l - pad.r) / 2 :
      (w - pad.l - pad.r) * i / (n - 1));
    const ys = v => h - pad.b - (h - pad.t - pad.b) * (v - lo) / (hi - lo || 1);

    ctx.clearRect(0, 0, w, h);
    ctx.font = "10px sans-serif";
    // gorizontal to'r + y qiymatlari
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of t.ticks) {
      const y = ys(tv);
      ctx.strokeStyle = "#e6e9f2"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#66708a";
      ctx.fillText(String(Math.round(tv * 100) / 100), pad.l - 7, y);
    }
    // nol chizig'i
    if (opts.zeroLine && lo < 0 && hi > 0) {
      ctx.strokeStyle = "#3b4560"; ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, ys(0)); ctx.lineTo(w - pad.r, ys(0)); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // sanalar — qiya, ustma-ust tushmaydi
    ctx.textAlign = "right"; ctx.textBaseline = "top";
    const nTick = Math.min(7, n);
    for (let j = 0; j < nTick; j++) {
      const idx = Math.round((n - 1) * j / (nTick - 1 || 1));
      const lbl = (opts.labels && opts.labels[idx] ? String(opts.labels[idx]) : idx)
        .slice(0, 10);
      const x = xs(idx), y = h - pad.b + 8;
      ctx.save(); ctx.translate(x, y); ctx.rotate(-0.6);
      ctx.fillStyle = "#66708a";
      ctx.fillText(lbl, 0, 0);
      ctx.restore();
      ctx.strokeStyle = "#d8dce9"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, h - pad.b); ctx.lineTo(x, h - pad.b + 4); ctx.stroke();
    }
    // seriyalar
    for (let s = 0; s < series.length; s++) {
      const ser = series[s];
      const color = ser.color || PAL[s % PAL.length];
      if (n === 1) {                                  // bitta nuqta — doira
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(xs(0), ys(ser.values[0]), 4, 0, 6.29); ctx.fill();
        continue;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = ser.width || 1.6;
      ctx.lineJoin = "round";
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const X = xs(i), Y = ys(ser.values[i]);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.stroke();
      if (opts.fillFirst && s === 0) {
        ctx.globalAlpha = 0.10; ctx.fillStyle = color;
        ctx.lineTo(xs(n - 1), ys(lo)); ctx.lineTo(xs(0), ys(lo));
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
    }
    // legenda — tepada, ma'lumotdan yuqorida
    if (hasLegend) {
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      const colW = (w - pad.l - pad.r) / perRow;
      for (let s = 0; s < series.length; s++) {
        const row = Math.floor(s / perRow);
        const x = pad.l + (s % perRow) * colW;
        const y = 10 + row * 17;
        ctx.fillStyle = series[s].color || PAL[s % PAL.length];
        ctx.fillRect(x, y - 4, 10, 4);
        ctx.fillStyle = "#3c465f"; ctx.font = "10.5px sans-serif";
        const nm = series[s].name;
        ctx.fillText(nm.length > 16 ? nm.slice(0, 15) + "…" : nm, x + 14, y - 2);
      }
    }
    return cv;
  }

  function roundBar(ctx, x, y, bw, bh, r) {
    if (bw < 0.5) return;
    r = Math.min(r, bh / 2, bw / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + bw, y, x + bw, y + bh, r);
    ctx.arcTo(x + bw, y + bh, x, y + bh, r);
    ctx.arcTo(x, y + bh, x, y, r);
    ctx.arcTo(x, y, x + bw, y, r);
    ctx.fill();
  }

  /** Statik natija: guruhlangan gorizontal ustunlar (TO / FROM). */
  function barChart(cv, opts) {
    const names = opts.names, TO = opts.TO, FROM = opts.FROM, NET = opts.NET;
    const k = names.length;
    const order = [...names.keys()].sort((a, b) => TO[b] - TO[a]);
    const H = Math.max(230, 46 * k + 70);
    const { ctx, w, h } = setupCanvas(cv, H);
    const padL = 108, padR = 96, padT = 30, padB = 18;
    const rowH = (h - padT - padB) / k;
    const bh = Math.min(14, rowH * 0.36);
    const vmax = Math.max(...TO, ...FROM) * 1.02 || 1;
    ctx.clearRect(0, 0, w, h);
    // vertikal to'r
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle";
    const t = niceTicks(0, vmax, 4);
    for (const tv of t.ticks) {
      const x = padL + (w - padL - padR) * tv / vmax;
      ctx.strokeStyle = "#e6e9f2"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, padT - 8); ctx.lineTo(x, h - padB); ctx.stroke();
      ctx.fillStyle = "#66708a"; ctx.textAlign = "center";
      ctx.fillText(String(Math.round(tv * 10) / 10), x, h - padB + 10);
    }
    for (let r = 0; r < k; r++) {
      const i = order[r];
      const cy = padT + rowH * (r + 0.5);
      // nom
      ctx.fillStyle = "#1c2333"; ctx.textAlign = "right";
      ctx.font = "11px sans-serif";
      const nm = String(names[i]);
      ctx.fillText(nm.length > 15 ? nm.slice(0, 14) + "…" : nm, padL - 12, cy);
      // TO
      const toW = (w - padL - padR) * TO[i] / vmax;
      ctx.fillStyle = C_TO;
      roundBar(ctx, padL, cy - bh - 1, toW, bh, 2);
      ctx.fillStyle = "#2456e6"; ctx.textAlign = "left";
      ctx.font = "9.5px sans-serif";
      ctx.fillText(TO[i].toFixed(1), padL + toW + 5, cy - bh / 2 - 1);
      // FROM
      const frW = (w - padL - padR) * FROM[i] / vmax;
      ctx.fillStyle = C_FROM;
      roundBar(ctx, padL, cy + 1, frW, bh, 2);
      ctx.fillStyle = "#b96a10";
      ctx.fillText(FROM[i].toFixed(1), padL + frW + 5, cy + bh / 2 + 1);
      // NET
      const net = NET ? NET[i] : (TO[i] - FROM[i]);
      ctx.fillStyle = net > 0 ? "#c4432b" : "#2e9e6b";
      ctx.textAlign = "left"; ctx.font = "bold 10px sans-serif";
      ctx.fillText((net > 0 ? "+" : "") + net.toFixed(1), w - padR + 14, cy);
    }
    // sarlavha + legenda
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillStyle = "#1c2333"; ctx.font = "bold 12px sans-serif";
    ctx.fillText("Spilloverlar (%)", padL, 12);
    ctx.font = "9.5px sans-serif";
    ctx.fillStyle = C_TO; ctx.fillRect(padL + 128, 9, 10, 6);
    ctx.fillStyle = "#4b5670"; ctx.fillText("TO — uzatilgan", padL + 142, 12);
    ctx.fillStyle = C_FROM; ctx.fillRect(padL + 225, 9, 10, 6);
    ctx.fillStyle = "#4b5670"; ctx.fillText("FROM — qabul qilingan", padL + 239, 12);
    ctx.fillStyle = "#c4432b"; ctx.fillText("NET", w - padR + 14, 12);
    return cv;
  }

  /** Statik chastota: davrlar bo'yicha TCI vertikal ustunlari. */
  function bandBarChart(cv, opts) {
    const names = opts.names, values = opts.values;
    const { ctx, w, h } = setupCanvas(cv, 280);
    const pad = { l: 52, r: 14, t: 34, b: 40 };
    ctx.clearRect(0, 0, w, h);
    const vmax = Math.max(...values) * 1.15 || 1;
    const step = (w - pad.l - pad.r) / names.length;
    const bw = Math.min(90, step * 0.6);
    // to'r
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle";
    const t = niceTicks(0, vmax, 4);
    for (const tv of t.ticks) {
      const y = h - pad.b - (h - pad.t - pad.b) * tv / vmax;
      ctx.strokeStyle = "#e6e9f2"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#66708a"; ctx.textAlign = "right";
      ctx.fillText(String(Math.round(tv * 10) / 10), pad.l - 6, y);
    }
    names.forEach((nm, i) => {
      const x = pad.l + step * i + (step - bw) / 2;
      const bh2 = (h - pad.t - pad.b) * values[i] / vmax;
      ctx.fillStyle = i === 0 ? "#2456e6" : PAL[i % PAL.length];
      roundBar(ctx, x, h - pad.b - bh2, bw, bh2, 3);
      ctx.fillStyle = "#1c2333"; ctx.textAlign = "center"; ctx.font = "10px sans-serif";
      ctx.fillText(values[i].toFixed(1) + "%", x + bw / 2, h - pad.b - bh2 - 9);
      ctx.fillStyle = "#4b5670";
      ctx.fillText(String(nm), x + bw / 2, h - pad.b + 13);
    });
    ctx.fillStyle = "#1c2333"; ctx.textAlign = "left";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText("Chastota davrlari bo'yicha TCI (%)", pad.l, 14);
    return cv;
  }

  /** Doiraviy bog'liqlik to'ri — faqat eng kuchli qirralar, egri strelkalar. */
  function networkChart(cv, CT, names, TO, NET) {
    const size = setupCanvas(cv, 470);
    const { ctx, w, h } = size;
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 68;
    const k = CT.length;
    const ang = i => -Math.PI / 2 + 2 * Math.PI * i / k;
    const xs = i => cx + R * Math.cos(ang(i));
    const ys = i => cy + R * Math.sin(ang(i));
    ctx.clearRect(0, 0, w, h);

    // juftlik kuchi (ikkala yo'nalish yig'indisi, %) — faqat eng kuchlilari
    const pairs = [];
    for (let i = 0; i < k; i++)
      for (let j = i + 1; j < k; j++) {
        const s = (CT[i][j] + CT[j][i]) * 100;
        if (s >= 2.0) pairs.push([s, i, j]);
      }
    pairs.sort((a, b) => b[0] - a[0]);
    const edges = pairs.slice(0, Math.max(6, 2 * k));
    const smax = edges.length ? edges[0][0] : 1;

    for (const [s, i, j] of edges) {
      const lw = 0.7 + 3.6 * s / smax;
      const alpha = Math.min(0.9, 0.3 + 0.6 * s / smax);
      const src = CT[i][j] >= CT[j][i] ? j : i;   // kuchsiz uzatuvchi → kuchli
      const dst = CT[i][j] >= CT[j][i] ? i : j;
      const x1 = xs(src), y1 = ys(src), x2 = xs(dst), y2 = ys(dst);
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      const bend = ((i + j) % 2 === 0 ? 1 : -1) * 0.16;
      const qx = mx + (y2 - y1) * bend, qy = my - (x2 - x1) * bend;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = "#c4432b"; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo(qx, qy, x2, y2);
      ctx.stroke();
      const a2 = Math.atan2(y2 - qy, x2 - qx);
      const head = 5 + lw * 1.1;
      ctx.fillStyle = "#c4432b";
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - head * Math.cos(a2 - 0.45), y2 - head * Math.sin(a2 - 0.45));
      ctx.lineTo(x2 - head * Math.cos(a2 + 0.45), y2 - head * Math.sin(a2 + 0.45));
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }
    // tugunlar
    const toMin = Math.min(...TO), toMax = Math.max(...TO);
    for (let i = 0; i < k; i++) {
      const r = 8 + 15 * (TO[i] - toMin) / (toMax - toMin || 1);
      ctx.fillStyle = NET[i] > 0 ? "#c4432b" : "#2e9e6b";
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(xs(i), ys(i), r, 0, 6.29);
      ctx.fill(); ctx.stroke();
      const ca = Math.cos(ang(i)), sa = Math.sin(ang(i));
      const lx = cx + (R + 26) * ca, ly = cy + (R + 26) * sa;
      ctx.fillStyle = "#1c2333"; ctx.font = "bold 10.5px sans-serif";
      ctx.textAlign = Math.abs(ca) < 0.4 ? "center" : (ca > 0 ? "left" : "right");
      ctx.textBaseline = sa > 0.85 ? "top" : (sa < -0.85 ? "bottom" : "middle");
      ctx.fillText(String(names[i]).slice(0, 12), lx, ly);
    }
    // legenda
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.font = "10px sans-serif";
    const ly2 = h - 16, lx = cx - 150;
    ctx.fillStyle = "#c4432b"; ctx.beginPath(); ctx.arc(lx, ly2, 6, 0, 6.29); ctx.fill();
    ctx.fillStyle = "#4b5670"; ctx.fillText("sof uzatuvchi", lx + 12, ly2);
    ctx.fillStyle = "#2e9e6b"; ctx.beginPath(); ctx.arc(lx + 100, ly2, 6, 0, 6.29); ctx.fill();
    ctx.fillStyle = "#4b5670"; ctx.fillText("sof qabulqiluvchi", lx + 112, ly2);
    ctx.strokeStyle = "#c4432b"; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(lx + 215, ly2); ctx.lineTo(lx + 240, ly2); ctx.stroke();
    ctx.fillText("qirralar — spillover kuchi", lx + 248, ly2);
    return cv;
  }

  global.Charts = { lineChart, barChart, bandBarChart, networkChart, PAL };
})(window);
