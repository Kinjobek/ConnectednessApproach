/* Veb-ilova mantiqi — barcha hisob-kitoblar brauzerda (engine.js). */
(function () {
  "use strict";

  // -------------------------------------------------- ma'lumot to'plamlari
  const DATASETS = {
    dy2009: ["Diebold–Yilmaz (2009) — 19 ta bozor volatililigi",
      "Global aksiya bozorlari volatililigi spilloverlari: AQSH, Buyuk Britaniya, Fransiya, Germaniya, Yaponiya va rivojlanayotgan bozorlar (1992–2007)."],
    dy2012: ["Diebold–Yilmaz (2012) — 4 aktiv sinfi",
      "S&P 500, 10 yillik davlat obligatsiyasi, Bloomberg tovar indeksi va dollar indeksi volatililigi (1999–2010) — umumlashtirilgan yondashuv namunasi."],
    acg2020: ["Antonakakis et al. (2020) — TVP-VAR valyutalari",
      "EUR, GBP, JPY, CHF valyuta juftliklari — TVP-VAR yondashuvi uchun asosiy ma'lumot (1975–2018, oylik)."],
    g2020: ["Gabauer (2020) — DCC-GARCH valyutalari",
      "EUR, GBP, CHF, JPY — volatilik uzatish mexanizmlari (2002–2018)."],
    gg2018: ["Gabauer–Gupta (2018) — pul siyosati noaniqligi",
      "AQSH va Yaponiya pul, fiskal, savdo va valyuta siyosati noaniqligi (1987–2017)."],
    cg2021: ["Chatziantoniou–Gabauer (2021) — EMU davlat obligatsiyalari",
      "Yevrozonaning 12 a'zo davlati 10 yillik davlat obligatsiyalari daromadliligi (2003–2018)."],
    cgs2021: ["Chatziantoniou et al. (2021) — valyuta svoplarini almashtirish",
      "USD, EUR, GBP, JPY 1 yillik valyuta bazis svoplari (2005–2020)."],
    cgs2022: ["Chatziantoniou et al. (2022) — ISDA marja bo'yicha svoplar",
      "Shvetsariya, Daniya, Yevro hududi, Buyuk Britaniya, Norvegiya 2/5/10 yillik ISDAFixed svop stavkalari (1999–2021)."],
    cegg2022: ["Chatziantoniou et al. (2022) — neft bozoridagi zarbalar",
      "Talab, taklif va risk zarbalari hamda 9 mamlakat valyutalari (2006–2021)."],
    cgg2022: ["Chatziantoniou et al. (2022) — neft eksportchilari",
      "Yevropa, AQSH, Dubay, Malayziya, Afrika va OPEC neft ishlab chiqarish samaradorligi (1996–2020)."],
    gcat2022: ["Chatziantoniou et al. (2022) — yashil aktivlar",
      "Yashil obligatsiya, atrof-muhit, barqarorlik va toza energiya indekslari (2008–2020)."],
    gghm2022: ["Gabauer et al. (2022) — AQSH uy narxlari",
      "50 shtat va Kolumbiya okrugi uy narxlari indeksi (1976–2019) — katta tizim (52 o'zgaruvchi)."],
    aaacgo2022: ["Adekoya et al. (2022) — neft va texnologiyalar",
      "Brent nefti, asosiy valyutalar va 9 sektor indekslari (2013–2021)."],
    bgu2021: ["Balcilar et al. (2021) — tovar bozorlari",
      "Xom neft, mol go'shti, kakao, kofe, bug'doy va boshqa tovarlar (2005–2020) — Extended Joint yondashuvi."],
    lw2021: ["Lastrapes–Wiesen (2021) — 9 sektor",
      "Energiya, xomashyo, sanoat, moliya va boshqa sektorlar (1994–2018) — Joint Spillover indeksi."],
    jcggh2022: ["Cunado et al. (2022) — qimmatbaho metallar",
      "Xom neft, ogyovgoyin moy, oltin va kumush realizatsiyalangan volatililigi (2006–2021)."]
  };
  const MODELS = {
    "VAR": "VAR (klassik OLS)",
    "TVP-VAR": "TVP-VAR (vaqt o'zgaruvchan)"
  };
  const APPROACHES = {
    "Time": "Vaqt sohasi (Diebold–Yilmaz)",
    "Frequency": "Chastota sohasi (Barunik–Krehlik)",
    "Joint": "Qo'shma spillover (Lastrapes–Wiesen)",
    "Extended Joint": "Kengaytirilgan qo'shma (Balcilar et al.)"
  };
  const PRIORS = {
    "BayesPrior": "Bayes prior (trening namunasi)",
    "MinnesotaPrior": "Minnesota prior",
    "UninformativePrior": "Ma'lumotsiz prior"
  };
  const TVP_K_LIMIT = 8;   // brauzer uchun xotira/hisob chegarasi

  // -------------------------------------------------- yordamchilar
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const fmt2 = v => (Math.round(v * 100) / 100).toFixed(2);

  function show(id) {
    $$(".page").forEach(p => p.classList.remove("active"));
    $("#" + id).classList.add("active");
    window.scrollTo(0, 0);
  }
  function route() {
    const h = location.hash || "#/";
    const map = { "#/": "page-home", "#/tahlil": "page-tahlil",
                  "#/natija": "page-natija",
                  "#/uslub": "page-uslub", "#/haqida": "page-haqida" };
    show(map[h] || "page-home");
  }
  window.addEventListener("hashchange", route);

  // dataset kartochkalari
  function renderHome() {
    const grid = $("#dsGrid");
    grid.innerHTML = "";
    const names = Object.keys(DATASETS);
    names.forEach((n, i) => {
      const [title, desc] = DATASETS[n];
      const card = document.createElement("div");
      card.className = "ds-card";
      card.innerHTML = `<h3>${title}</h3>
        <div class="meta"><span>📦 ${n}.csv</span></div>
        <p>${desc}</p>
        <a class="go" href="#/tahlil" data-ds="${n}">Shu to'plamni tahlil qilish →</a>`;
      grid.appendChild(card);
    });
    grid.addEventListener("click", e => {
      const a = e.target.closest("a[data-ds]");
      if (!a) return;
      setTimeout(() => {
        const sel = $("#datasetSel");
        if (sel) {
          sel.value = a.dataset.ds;
          sel.dispatchEvent(new Event("change"));
        }
      }, 60);
    });
  }

  // -------------------------------------------------- CSV o'qish
  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 3) throw new Error("CSV juda qisqa");
    const header = lines[0].split(",").map(s => s.trim());
    const names = header.slice(1).map((h, i) => h || ("V" + (i + 1)));
    const dates = [], rows = [];
    for (let i = 1; i < lines.length; i++) {
      const toks = lines[i].split(",");
      if (toks.length !== header.length) continue;
      const vals = toks.slice(1).map(parseFloat);
      if (!vals.every(Number.isFinite)) continue;
      dates.push(toks[0].trim());
      rows.push(vals);
    }
    if (rows.length < 20 || names.length < 2)
      throw new Error("Kamida 20 qator va 2 raqamli ustun kerak");
    return { names, dates, rows };
  }

  async function loadBuiltin(name) {
    const res = await fetch("data/" + name + ".csv");
    if (!res.ok) throw new Error("Ma'lumot to'plami yuklanmadi: " + name);
    return parseCSV(await res.text());
  }

  // -------------------------------------------------- forma hodisalari
  function initForm() {
    const modelSel = $("#modelSel"), apprSel = $("#approachSel");
    const tvpCard = $("#tvpCard"), freqCard = $("#freqCard");
    const genCheck = $("#genCheck");
    function refresh() {
      tvpCard.style.display = modelSel.value === "TVP-VAR" ? "" : "none";
      freqCard.style.display = apprSel.value === "Frequency" ? "" : "none";
      genCheck.closest(".check").style.opacity =
        apprSel.value === "Time" ? "1" : "0.45";
    }
    modelSel.onchange = refresh; apprSel.onchange = refresh; refresh();

    // manba tablitsasi
    $("#tabBuiltin").onclick = () => {
      $("#srcField").value = "builtin";
      $("#srcBuiltin").style.display = ""; $("#srcUpload").style.display = "none";
      $("#tabBuiltin").classList.add("active"); $("#tabUpload").classList.remove("active");
    };
    $("#tabUpload").onclick = () => {
      $("#srcField").value = "upload";
      $("#srcBuiltin").style.display = "none"; $("#srcUpload").style.display = "";
      $("#tabUpload").classList.add("active"); $("#tabBuiltin").classList.remove("active");
    };

    $("#analysisForm").onsubmit = async e => {
      e.preventDefault();
      await runAnalysis();
    };
  }

  // -------------------------------------------------- natijalarni chizish
  function htmlTable(T) {
    let html = '<table class="ctable"><thead><tr><th></th>';
    for (const c of T.colNames) html += "<th>" + c + "</th>";
    html += "</tr></thead><tbody>";
    for (let i = 0; i < T.rows.length; i++) {
      html += "<tr><td>" + T.rowNames[i] + "</td>";
      for (const v of T.rows[i]) html += "<td>" + v + "</td>";
      html += "</tr>";
    }
    return html + "</tbody></table>";
  }

  function lastNames(dca) {
    // NET oxirgi nuqtasi (chastota bo'lsa Total band)
    let net = dca.NET;
    if (!net || !net.length) return null;
    let last = net[net.length - 1];
    const names = dca.config.names;
    return names.map((n, i) => [n, Array.isArray(last[i]) ? last[i][0] : last[i]]);
  }

  function drawResult(dca, meta) {
    // parametrlar
    $("#resParams").innerHTML = Object.entries(meta.params)
      .map(([k, v]) => `<span class="chip"><b>${k}:</b>&nbsp;${v}</span>`).join("");
    // statistika
    const tci = dca.TCI.map(v => Array.isArray(v) ? v[0] : v);
    const mn = tci.reduce((a, b) => a + b, 0) / tci.length;
    $("#resStats").innerHTML = `
      <div class="stat"><div class="v">${fmt2(mn)}%</div><div class="l">O'rtacha TCI</div></div>
      <div class="stat"><div class="v">${fmt2(Math.max(...tci))}%</div><div class="l">Maksimal TCI</div></div>
      <div class="stat"><div class="v">${fmt2(Math.min(...tci))}%</div><div class="l">Minimal TCI</div></div>
      <div class="stat"><div class="v">${fmt2(tci[tci.length - 1])}%</div><div class="l">Oxirgi TCI</div></div>`;
    // NET xulosasi
    const ln = lastNames(dca);
    $("#resNet").style.display = ln ? "" : "none";
    if (ln) {
      $("#resNet").querySelector(".chips").innerHTML = ln.map(([n, v]) =>
        `<span class="chip" style="background:${v > 0 ? "#fdeceb;color:#9d2b1a;border-color:#f6cfcb" : "#e8f7f0;color:#14684a;border-color:#c9ecdc"}">
         <b>${n}:</b>&nbsp;${fmt2(v)}%</span>`).join("");
    }
    // jadvallar
    const wrap = $("#resTables");
    wrap.innerHTML = "";
    if (dca.config.approach === "Frequency") {
      const k = dca.CT[0][0].length, names = dca.config.names;
      const pn = dca.config.periodNames;
      for (let b = 0; b < pn.length; b++) {
        const avg = Array.from({ length: k }, () => new Array(k).fill(0));
        for (const CTt of dca.CT)
          for (let i = 0; i < k; i++)
            for (let j = 0; j < k; j++) avg[i][j] += CTt[b][i][j];
        for (let i = 0; i < k; i++)
          for (let j = 0; j < k; j++) avg[i][j] /= dca.CT.length;
        const ct = CC.connectednessTable(avg, names);
        wrap.innerHTML += `<div class="card plot-card"><h3>🧾 Bog'liqlik jadvali — ${pn[b]} davri</h3>
          <div class="tablewrap">${htmlTable(ct.TABLE)}</div></div>`;
      }
    } else if (dca.TABLE) {
      wrap.innerHTML = `<div class="card plot-card"><h3>🧾 Bog'liqlik jadvali (o'rtacha)</h3>
        <p class="hint">Diogonal — o'z ulushi. <b>TO</b> — uzatilgan, <b>FROM</b> — qabul qilingan,
        <b>NET</b> — sof yo'nalish, <b>NPT</b> — juftlik bo'yicha yo'nalishlar soni.</p>
        <div class="tablewrap">${htmlTable(dca.TABLE)}</div></div>`;
    }
    // grafiklar
    const labels = dca.config.dates || [];
    const tciSeries = dca.TCI.map(v => Array.isArray(v) ? v[0] : v);
    const dynamic = tciSeries.length > 1;
    const freq = dca.config.approach === "Frequency";
    const names = dca.config.names;

    const get1 = (arr, i) => arr.map(row => Array.isArray(row[i]) ? row[i][0] : row[i]);
    const avgOf = (arr, i) => { let s = 0, n = 0;
      for (const row of arr) { s += Array.isArray(row[i]) ? row[i][0] : row[i]; n++; }
      return s / n; };

    if (dynamic) {
      // ---- dinamik chiziqli grafiklar
      if (freq) {
        Charts.lineChart($("#chartTCI"), {
          series: dca.config.periodNames.map((pn, j) => ({
            name: pn, values: dca.TCI.map(row => row[j]) })),
          labels, height: 310 });
        $("#cardTCI").querySelector("h3").textContent = "📈 Chastota bo'yicha TCI dinamikasi";
      } else {
        Charts.lineChart($("#chartTCI"), {
          series: [{ name: "TCI", values: tciSeries }],
          labels, height: 310, fillFirst: true });
        $("#cardTCI").querySelector("h3").textContent =
          dca.config.approach === "Joint" ? "📈 Qo'shma spillover indeksi (jTCI)"
          : "📈 Umumiy bog'liqlik indeksi (TCI)";
      }
      // NET
      const hasNet = dca.NET && dca.NET.length && dca.NET[0].length;
      $("#cardNET").style.display = hasNet ? "" : "none";
      if (hasNet) {
        const order = names.map((_, i2) => i2)
          .sort((a, b) => Math.abs(avgOf(dca.NET, b)) - Math.abs(avgOf(dca.NET, a)))
          .slice(0, 8);
        Charts.lineChart($("#chartNET"), {
          series: order.map(i2 => ({ name: names[i2], values: get1(dca.NET, i2) })),
          labels, height: 310, zeroLine: true });
      }
      // TO / FROM
      const hasTO = dca.TO && dca.TO.length && dca.TO[0].length;
      $("#cardTOFROM").style.display = hasTO ? "" : "none";
      if (hasTO) {
        const order = names.map((_, i2) => i2)
          .sort((a, b) => avgOf(dca.TO, b) - avgOf(dca.TO, a)).slice(0, 6);
        Charts.lineChart($("#chartTO"), { series: order.map(i2 => ({ name: names[i2], values: get1(dca.TO, i2) })), labels, height: 270 });
        Charts.lineChart($("#chartFROM"), { series: order.map(i2 => ({ name: names[i2], values: get1(dca.FROM, i2) })), labels, height: 270 });
      }
      // NPT
      const hasNPT = dca.NPT && dca.NPT.length && !Array.isArray(dca.NPT[0][0]);
      $("#cardNPT").style.display = hasNPT ? "" : "none";
      if (hasNPT) {
        const order = names.map((_, i2) => i2)
          .sort((a, b) => avgOf(dca.NPT, b) - avgOf(dca.NPT, a)).slice(0, 8);
        Charts.lineChart($("#chartNPT"), {
          series: order.map(i2 => ({ name: names[i2], values: get1(dca.NPT, i2) })),
          labels, height: 290 });
      }
    } else {
      // ---- statik natija: ustunli grafiklar (chiziqlar bo'sh chiqmasligi uchun)
      $("#cardNET").style.display = "none";
      $("#cardTOFROM").style.display = "none";
      $("#cardNPT").style.display = "none";
      const pick = (arr, i2) => {
        const v = arr[arr.length - 1][i2];
        return Array.isArray(v) ? v[0] : v;
      };
      if (freq) {
        Charts.bandBarChart($("#chartTCI"), {
          names: dca.config.periodNames, values: dca.TCI[0] });
        $("#cardTCI").querySelector("h3").textContent = "📊 Chastota davrlari bo'yicha TCI";
      } else if (dca.TO && dca.TO.length) {
        Charts.barChart($("#chartTCI"), {
          names,
          TO: names.map((_, i2) => pick(dca.TO, i2)),
          FROM: names.map((_, i2) => pick(dca.FROM, i2)),
          NET: names.map((_, i2) => pick(dca.NET, i2)) });
        $("#cardTCI").querySelector("h3").textContent = "📊 Spilloverlar: TO va FROM";
      } else {
        $("#cardTCI").style.display = "none";
      }
    }
    // To' grafigi — o'rtacha FEVD dan
    let CTavg = null;
    if (dca.avgFEVD) CTavg = dca.avgFEVD;
    else if (dca.CT && dca.CT.length && dca.config.approach === "Frequency") {
      const k = dca.CT[0][0].length;
      const avg = Array.from({ length: k }, () => new Array(k).fill(0));
      for (const CTt of dca.CT)
        for (let i = 0; i < k; i++)
          for (let j = 0; j < k; j++) avg[i][j] += CTt[0][i][j];
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++) avg[i][j] /= dca.CT.length;
      CTavg = avg;
    }
    $("#cardNetwork").style.display = CTavg && CTavg.length <= 30 ? "" : "none";
    if (CTavg) {
      const ct = CC.connectednessTable(CTavg, dca.config.names);
      Charts.networkChart($("#chartNetwork"), CTavg, dca.config.names,
                          ct.TO, ct.NET);
    }
    location.hash = "#/natija";
    show("page-natija");
  }

  function avgOf(arr2d, i) {
    let s = 0, n = 0;
    for (const row of arr2d) { const v = row[i]; s += Array.isArray(v) ? v[0] : v; n++; }
    return s / n;
  }

  // -------------------------------------------------- hisoblash
  async function runAnalysis() {
    const btn = $("#submitBtn");
    const note = $("#calcNote");
    btn.disabled = true; btn.textContent = "⏳ Hisoblanmoqda...";
    note.textContent = "Brauzerda hisoblanmoqda — kichik to'plamlar bir necha soniya oladi.";
    try {
      const source = $("#srcField").value;
      let data, dsTitle;
      if (source === "upload") {
        const f = $("#csvfile").files[0];
        if (!f) throw new Error("CSV fayl tanlanmadi");
        data = parseCSV(await f.text());
        dsTitle = f.name;
      } else {
        const name = $("#datasetSel").value;
        data = await loadBuiltin(name);
        dsTitle = (DATASETS[name] || [name])[0];
      }
      const k = data.names.length, T = data.rows.length;
      const model = $("#modelSel").value;
      const approach = $("#approachSel").value;
      const nlag = Math.max(1, Math.min(12, parseInt($("#nlag").value) || 1));
      let nfore = Math.max(1, Math.min(200, parseInt($("#nfore").value) || 10));
      if (approach === "Frequency" && nfore < 100) nfore = 100;
      const winRaw = $("#windowSize").value.trim();
      const windowSize = winRaw ? Math.max(0, parseInt(winRaw) || 0) : null;
      const corrected = $("#corrCheck").checked;

      if (model === "TVP-VAR" && k > TVP_K_LIMIT)
        throw new Error(`TVP-VAR brauzer versiyasida kamida ${TVP_K_LIMIT} ta o'zgaruvchi uchun mo'ljallangan (sizda ${k}). VAR yoki kichikroq to'plam tanlang.`);
      if (k * k * nlag > 4000 && windowSize)
        throw new Error("Siljuvchi oyna + katta tizim brauzer uchun juda og'ir — nlag ni kamaytiring.");

      const VAR_config = {};
      if (model === "TVP-VAR") {
        VAR_config.TVPVAR = {
          prior: $("#tvpPrior").value,
          kappa1: parseFloat($("#kappa1").value) || 0.99,
          kappa2: parseFloat($("#kappa2").value) || 0.99,
          gamma: parseFloat($("#gamma").value) || 0.1
        };
      }
      const Connectedness_config = {};
      if (approach === "Frequency") {
        const band = $("#bandSel").value;
        const part = { pi2: [Math.PI, Math.PI / 2, 0],
                       pi4: [Math.PI, Math.PI / 4, 0],
                       pi8: [Math.PI, Math.PI / 8, 0],
                       pi24: [Math.PI, Math.PI / 2, Math.PI / 4, 0] }[band];
        Connectedness_config.FrequencyConnectedness = {
          partition: part, scenario: $("#scenarioSel").value };
      }

      // UI nafas olishi uchun
      await new Promise(r => setTimeout(r, 30));
      const t0 = performance.now();
      const dca = CC.connectednessApproach(data.rows, {
        nlag, nfore, model, connectedness: approach,
        window_size: windowSize, corrected,
        names: data.names, dates: data.dates,
        VAR_config, Connectedness_config
      });
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

      const params = {
        "Ma'lumotlar": dsTitle,
        "O'zgaruvchilar": k,
        "Kuzatuvlar": T,
        "Model": { VAR: "VAR (OLS)", "TVP-VAR": "TVP-VAR" }[model],
        "Yondashuv": APPROACHES[approach],
        "nlag": nlag, "nfore": nfore,
        "Oyna": windowSize || "to'liq namuna",
        "cTCI": corrected ? "ha" : "yo'q"
      };
      if (model === "TVP-VAR") {
        const tv = VAR_config.TVPVAR;
        params["Prior"] = PRIORS[tv.prior];
        params["kappa1/kappa2"] = tv.kappa1 + " / " + tv.kappa2;
      }
      if (approach === "Frequency") {
        params["Ssenariy"] = $("#scenarioSel").value;
      }
      params["Vaqt"] = elapsed + " s";
      drawResult(dca, { params });
    } catch (ex) {
      alert("Xatolik: " + ex.message);
    } finally {
      btn.disabled = false; btn.textContent = "⚡ Hisoblash (brauzerda)";
      note.textContent = "Barcha hisob-kitoblar brauzeringizda bajariladi — server kerak emas.";
    }
  }

  // -------------------------------------------------- boshlanish
  document.addEventListener("DOMContentLoaded", () => {
    renderHome();
    initForm();
    route();
  });
})();
