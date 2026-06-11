/* ==========================================================
   DISCIPLINED TRADER — terminal frontend
   ========================================================== */
"use strict";

const TF_SECONDS = { "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400 };
const TF_FACTOR = { "1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 390 };
const TFS = ["1m", "5m", "15m", "1h", "1d"];
const BASE_TS = 1736065800; // must match server
const SPEEDS = [1, 2, 5, 10, 25];

const BULL_RE = /(bull|hammer|morning|white_soldiers|double_bottom|triple_bottom|cup|reverse_head|falling_wedge|dragonfly)/;
const BEAR_RE = /(bear|shooting|hanging|evening|black_crows|double_top|triple_top|head_and_shoulders$|rising_wedge|gravestone)/;

const state = {
  candles: Object.fromEntries(TFS.map(tf => [tf, []])),
  markers: [],          // {bar, price, type, shares?, profit?}
  equity: [],
  trades: [],
  account: null,
  obs: null,
  rewardTotal: 0,
  activeTask: "easy",
  activeSeed: 42,
  activeTf: "1m",
  running: false,
  inFlight: false,
  done: false,
  timer: null,
};

const $ = id => document.getElementById(id);
const fmtUsd = v => (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtNum = (v, d = 2) => Number(v).toFixed(d);

/* ---------------------------------------------------------- charts */
const chartOpts = {
  layout: {
    background: { type: "solid", color: "transparent" },
    textColor: "#7e8ca8",
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 11,
  },
  grid: {
    vertLines: { color: "rgba(27,39,64,0.45)" },
    horzLines: { color: "rgba(27,39,64,0.45)" },
  },
  rightPriceScale: { borderColor: "#1b2740" },
  timeScale: { borderColor: "#1b2740", timeVisible: true, secondsVisible: false, barSpacing: 6, rightOffset: 4 },
  crosshair: {
    mode: LightweightCharts.CrosshairMode.Normal,
    vertLine: { color: "#38bdf8", width: 1, style: 3, labelBackgroundColor: "#1e293b" },
    horzLine: { color: "#38bdf8", width: 1, style: 3, labelBackgroundColor: "#1e293b" },
  },
};

const mainChart = LightweightCharts.createChart($("main-chart"), chartOpts);
const candleSeries = mainChart.addCandlestickSeries({
  upColor: "#00e08a", downColor: "#ff4d6d",
  wickUpColor: "#00e08a", wickDownColor: "#ff4d6d",
  borderVisible: false,
});
const volumeSeries = mainChart.addHistogramSeries({
  priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false,
});
mainChart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
const smaSeries = mainChart.addLineSeries({ color: "#fbbf24", lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
const bbUpSeries = mainChart.addLineSeries({ color: "rgba(56,189,248,0.45)", lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
const bbLoSeries = mainChart.addLineSeries({ color: "rgba(56,189,248,0.45)", lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });

const equityChart = LightweightCharts.createChart($("equity-chart"), { ...chartOpts, timeScale: { ...chartOpts.timeScale, visible: false } });
const equitySeries = equityChart.addBaselineSeries({
  baseValue: { type: "price", price: 10000 },
  topLineColor: "#00e08a", topFillColor1: "rgba(0,224,138,0.25)", topFillColor2: "rgba(0,224,138,0.02)",
  bottomLineColor: "#ff4d6d", bottomFillColor1: "rgba(255,77,109,0.02)", bottomFillColor2: "rgba(255,77,109,0.25)",
  lineWidth: 2, priceLineVisible: false,
});

const chartResizer = new ResizeObserver(() => {
  const m = $("main-chart"), e = $("equity-chart");
  mainChart.applyOptions({ width: m.clientWidth, height: m.clientHeight });
  equityChart.applyOptions({ width: e.clientWidth, height: e.clientHeight });
});
chartResizer.observe(document.body);
chartResizer.observe($("main-chart"));
chartResizer.observe($("equity-chart"));

let entryLine = null, stopLine = null;

/* ---------------------------------------------------------- overlays */
function computeOverlays(candles) {
  const sma = [], up = [], lo = [], period = 20;
  let sum = 0;
  const closes = candles.map(c => c.close);
  for (let i = 0; i < closes.length; i++) {
    sum += closes[i];
    if (i >= period) sum -= closes[i - period];
    if (i >= period - 1) {
      const mean = sum / period;
      let varSum = 0;
      for (let j = i - period + 1; j <= i; j++) varSum += (closes[j] - mean) ** 2;
      const sd = Math.sqrt(varSum / period);
      sma.push({ time: candles[i].time, value: mean });
      up.push({ time: candles[i].time, value: mean + 2 * sd });
      lo.push({ time: candles[i].time, value: mean - 2 * sd });
    }
  }
  return { sma, up, lo };
}

function markerToChart(m, tf) {
  const t = BASE_TS + Math.floor(m.bar / TF_FACTOR[tf]) * TF_SECONDS[tf];
  if (m.type === "long") return { time: t, position: "belowBar", shape: "arrowUp", color: "#00e08a", text: `LONG ${m.shares}` };
  if (m.type === "short") return { time: t, position: "aboveBar", shape: "arrowDown", color: "#ff4d6d", text: `SHORT ${m.shares}` };
  if (m.type === "stop") {
    const win = m.profit >= 0;
    return { time: t, position: "aboveBar", shape: "square", color: win ? "#00e08a" : "#fbbf24", text: `${win ? "TRAIL +" : "STOP "}${fmtNum(m.profit)}` };
  }
  return { time: t, position: "aboveBar", shape: "circle", color: m.profit >= 0 ? "#00e08a" : "#ff4d6d", text: `${m.profit >= 0 ? "+" : ""}${fmtNum(m.profit)}` };
}

function redrawChart(full = false) {
  const tf = state.activeTf;
  const candles = state.candles[tf];
  if (full) {
    candleSeries.setData(candles);
    volumeSeries.setData(candles.map(c => ({
      time: c.time, value: c.volume,
      color: c.close >= c.open ? "rgba(0,224,138,0.35)" : "rgba(255,77,109,0.35)",
    })));
  }
  const ov = computeOverlays(candles);
  smaSeries.setData(ov.sma);
  bbUpSeries.setData(ov.up);
  bbLoSeries.setData(ov.lo);

  // dedupe markers landing on the same resampled candle
  const seen = new Map();
  for (const m of state.markers) {
    const cm = markerToChart(m, tf);
    seen.set(`${cm.time}|${cm.text}`, cm);
  }
  candleSeries.setMarkers([...seen.values()].sort((a, b) => a.time - b.time));

  // entry / stop price lines
  if (entryLine) { candleSeries.removePriceLine(entryLine); entryLine = null; }
  if (stopLine) { candleSeries.removePriceLine(stopLine); stopLine = null; }
  const a = state.account;
  if (a && a.position_shares !== 0) {
    entryLine = candleSeries.createPriceLine({
      price: a.entry_price, color: a.position_shares > 0 ? "#00e08a" : "#ff4d6d",
      lineWidth: 1, lineStyle: 2, title: "ENTRY",
    });
    if (a.stop_loss) {
      stopLine = candleSeries.createPriceLine({
        price: a.stop_loss, color: "#fbbf24", lineWidth: 1, lineStyle: 3, title: "STOP",
      });
    }
  }
}

function appendCandles(newCandles) {
  for (const tf of TFS) {
    const incoming = newCandles[tf] || [];
    state.candles[tf].push(...incoming);
    if (tf === state.activeTf) {
      for (const c of incoming) {
        candleSeries.update(c);
        volumeSeries.update({
          time: c.time, value: c.volume,
          color: c.close >= c.open ? "rgba(0,224,138,0.35)" : "rgba(255,77,109,0.35)",
        });
      }
    }
  }
}

/* ---------------------------------------------------------- HUD */
function updateHud() {
  const a = state.account;
  if (!a) return;
  const delta = a.equity - 10000;
  const pct = (delta / 10000) * 100;

  $("hud-equity").textContent = fmtUsd(a.equity);
  $("hud-equity").style.color = delta >= 0 ? "#00e08a" : "#ff4d6d";
  const dEl = $("hud-delta");
  dEl.textContent = `${delta >= 0 ? "+" : ""}${fmtUsd(delta)} (${fmtNum(pct)}%)`;
  dEl.className = `hud-delta mono ${delta >= 0 ? "up" : "down"}`;

  $("hud-cash").textContent = fmtUsd(a.cash);
  $("hud-price").textContent = fmtNum(a.price, 2);
  $("hud-unreal").textContent = fmtUsd(a.unrealized);
  $("hud-unreal").style.color = a.unrealized > 0 ? "#00e08a" : a.unrealized < 0 ? "#ff4d6d" : "";
  $("hud-real").textContent = fmtUsd(a.realized);
  $("hud-real").style.color = a.realized > 0 ? "#00e08a" : a.realized < 0 ? "#ff4d6d" : "";
  $("hud-dd").textContent = fmtNum(a.max_drawdown * 100, 1) + "%";
  $("hud-wr").textContent = a.total_trades ? fmtNum(a.win_rate * 100, 0) + "% / " + a.total_trades : "—";

  const chip = $("position-chip");
  if (a.position_shares > 0) {
    chip.className = "position-chip long";
    chip.textContent = `▲ LONG ${a.position_shares} @ ${fmtNum(a.entry_price)}`;
  } else if (a.position_shares < 0) {
    chip.className = "position-chip short";
    chip.textContent = `▼ SHORT ${-a.position_shares} @ ${fmtNum(a.entry_price)}`;
  } else {
    chip.className = "position-chip flat";
    chip.textContent = "FLAT — NO POSITION";
  }

  const risk = Math.min(100, Math.abs(state.obs ? state.obs.risk_usage : 0) * 100);
  $("risk-fill").style.width = risk + "%";
  $("risk-pct").textContent = fmtNum(risk, 0) + "%";

  $("bar-label").textContent = `${a.bar} / ${a.max_bars}`;
  $("progress-fill").style.width = (a.bar / a.max_bars) * 100 + "%";
}

function updateEpisodeLabel(task) {
  if (task) {
    state.activeTask = task;
    $("episode-label").textContent = task.toUpperCase();
  }
}

function updateBadges() {
  const o = state.obs;
  if (!o) return;
  const rb = $("regime-badge");
  rb.textContent = o.regime.toUpperCase();
  rb.className = `regime-badge ${o.regime}`;

  const t = o.tf[state.activeTf];
  const setBadge = (el, label, val) => {
    el.textContent = val === "none" ? `— ${label} —` : val.replace(/_/g, " ");
    el.className = "badge" + (BULL_RE.test(val) ? " bull" : BEAR_RE.test(val) ? " bear" : "");
  };
  setBadge($("candle-badge"), "candle", t.candle);
  setBadge($("chart-badge"), "chart", t.chart);
  const st = $("st-badge");
  st.textContent = t.supertrend === 1 ? "ST ▲" : t.supertrend === -1 ? "ST ▼" : "ST —";
  st.className = "badge" + (t.supertrend === 1 ? " bull" : t.supertrend === -1 ? " bear" : "");
}

function updateRadar() {
  const o = state.obs;
  if (!o) return;
  const radar = $("radar");
  radar.innerHTML = "";
  for (const tf of TFS) {
    const t = o.tf[tf];
    const pattern = t.chart !== "none" ? t.chart : t.candle;
    const cls = BULL_RE.test(pattern) ? "bull" : BEAR_RE.test(pattern) ? "bear" : "";
    const dotColor = t.rsi >= 70 ? "#ff4d6d" : t.rsi <= 30 ? "#00e08a" : "#38bdf8";
    radar.insertAdjacentHTML("beforeend", `
      <div class="radar-row">
        <span class="radar-tf">${tf}</span>
        <div class="rsi-track" title="RSI ${fmtNum(t.rsi, 0)}">
          <div class="rsi-dot" style="left:${t.rsi}%; background:${dotColor}; box-shadow:0 0 8px ${dotColor}"></div>
        </div>
        <span class="radar-pattern ${cls}" title="${pattern}">${pattern === "none" ? "·" : pattern.replace(/_/g, " ")}</span>
        <span class="radar-st ${t.supertrend === 1 ? "up" : t.supertrend === -1 ? "down" : ""}">${t.supertrend === 1 ? "▲" : t.supertrend === -1 ? "▼" : "—"}</span>
      </div>`);
  }
}

function updateConfluence() {
  const c = state.obs && state.obs.confluence;
  if (!c) return;
  // Scale bars so the entry threshold sits at ~55% of the track width.
  const scale = v => Math.min(100, (v / c.entry_score) * 55);
  $("conf-bull").style.width = scale(c.bull) + "%";
  $("conf-bear").style.width = scale(c.bear) + "%";
  $("conf-bull-val").textContent = c.bull.toFixed(1);
  $("conf-bear-val").textContent = c.bear.toFixed(1);
  $("conf-threshold").textContent = c.entry_score.toFixed(1);

  const v = $("conf-verdict");
  if (c.bull >= c.entry_score && c.bull - c.bear >= c.entry_margin) {
    v.textContent = "BULLISH EDGE"; v.className = "conf-verdict mono bull";
  } else if (c.bear >= c.entry_score && c.bear - c.bull >= c.entry_margin) {
    v.textContent = "BEARISH EDGE"; v.className = "conf-verdict mono bear";
  } else {
    v.textContent = "NO EDGE — WAIT"; v.className = "conf-verdict mono";
  }
}

let cumPL = 0;
function appendTrades(trades) {
  if (!trades.length) return;
  $("blotter-empty").style.display = "none";
  const body = $("blotter-body");
  for (const t of trades) {
    state.trades.push(t);
    cumPL += t.profit;
    const row = document.createElement("tr");
    row.className = t.profit >= 0 ? "win" : "loss";
    row.innerHTML = `<td>${state.trades.length}</td><td>${t.entry_bar}</td><td>${t.exit_bar}</td>` +
      `<td class="pl">${t.profit >= 0 ? "+" : ""}${fmtNum(t.profit)}</td>` +
      `<td>${cumPL >= 0 ? "+" : ""}${fmtNum(cumPL)}</td>`;
    body.prepend(row);
  }
  $("blotter-count").textContent = `${state.trades.length} closed`;
}

function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 3400);
}

/* ---------------------------------------------------------- payload */
function applyPayload(p) {
  const bigJump = p.candles && (p.candles["1m"] || []).length > 40;
  if (p.candles) appendCandles(p.candles);
  if (p.equity) for (const pt of p.equity) equitySeries.update(pt);
  if (p.account) state.account = p.account;
  if (p.obs) state.obs = p.obs;
  if (p.markers && p.markers.length) {
    state.markers.push(...p.markers);
    for (const m of p.markers) {
      if (m.type === "long") toast(`▲ LONG ${m.shares} @ ${fmtNum(m.price)}`, "good");
      else if (m.type === "short") toast(`▼ SHORT ${m.shares} @ ${fmtNum(m.price)}`, "bad");
      else if (m.type === "stop") toast(m.profit >= 0 ? `◼ TRAIL LOCKED +${fmtNum(m.profit)}` : `◼ STOP HIT ${fmtNum(m.profit)}`, m.profit >= 0 ? "good" : "bad");
      else toast(`✕ CLOSED ${m.profit >= 0 ? "+" : ""}${fmtNum(m.profit)}`, m.profit >= 0 ? "good" : "bad");
    }
  }
  if (p.trades) appendTrades(p.trades);
  if (typeof p.reward_step === "number") {
    const el = $("reward-step");
    el.textContent = fmtNum(p.reward_step, 3);
    el.style.color = p.reward_step > 0 ? "#00e08a" : p.reward_step < 0 ? "#ff4d6d" : "";
  }
  if (typeof p.reward_total === "number") {
    state.rewardTotal = p.reward_total;
    const el = $("reward-total");
    el.textContent = fmtNum(p.reward_total, 3);
    el.style.color = p.reward_total > 0 ? "#00e08a" : p.reward_total < 0 ? "#ff4d6d" : "";
  }
  updateHud();
  updateBadges();
  updateRadar();
  updateConfluence();
  redrawChart(false);
  if (bigJump) mainChart.timeScale().fitContent();

  if (p.done) {
    state.done = true;
    stopAutopilot();
    showModal(p);
  }
}

/* ---------------------------------------------------------- API */
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function doReset() {
  stopAutopilot();
  checkLlmAvailability(); // picks up freshly trained adapters (hot-reload)
  state.done = false;
  state.markers = [];
  state.trades = [];
  cumPL = 0;
  for (const tf of TFS) state.candles[tf] = [];
  candleSeries.setData([]); volumeSeries.setData([]);
  smaSeries.setData([]); bbUpSeries.setData([]); bbLoSeries.setData([]);
  equitySeries.setData([]);
  $("blotter-body").innerHTML = "";
  $("blotter-empty").style.display = "";
  $("blotter-count").textContent = "";
  $("modal").classList.add("hidden");

  try {
    const p = await api("/api/reset", {
      task: $("task-select").value,
      seed: parseInt($("seed-input").value) || 42,
    });
    applyPayload({ ...p, reward_step: 0, reward_total: 0 });
    state.activeSeed = p.seed;
    updateEpisodeLabel(p.task);
    redrawChart(true);
    toast(`NEW EPISODE · ${p.task.toUpperCase()} · seed ${p.seed}`, "info");
  } catch (e) {
    toast("Reset failed: " + e.message, "bad");
  }
}

async function doStep(body) {
  if (state.inFlight || state.done) return;
  state.inFlight = true;
  try {
    if (body.mode === "llm") body.steps = 1;
    const p = await api("/api/step", body);
    applyPayload(p);
  } catch (e) {
    toast(e.message, "bad");
    stopAutopilot();
  } finally {
    state.inFlight = false;
  }
}

function autopilotSteps() {
  if ($("policy-select").value === "llm") return 1;
  return SPEEDS[parseInt($("speed-slider").value)];
}

function updateSpeedUi() {
  const isLlm = $("policy-select").value === "llm";
  $("speed-slider").disabled = isLlm;
  $("llm-speed-note").classList.toggle("hidden", !isLlm);
  if (isLlm) {
    $("speed-label").textContent = "1 bar";
  } else {
    $("speed-label").textContent = SPEEDS[parseInt($("speed-slider").value)] + "×";
  }
}

/* ---------------------------------------------------------- autopilot */
function startAutopilot() {
  if (state.done) return;
  state.running = true;
  const btn = $("play-btn");
  btn.textContent = "■ STOP";
  btn.classList.add("running");
  state.timer = setInterval(() => {
    if (state.inFlight) return;
    doStep({ mode: $("policy-select").value, steps: autopilotSteps() });
  }, 220);
}

function stopAutopilot() {
  state.running = false;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  const btn = $("play-btn");
  btn.textContent = "▶ RUN";
  btn.classList.remove("running");
}

/* ---------------------------------------------------------- modal */
function gradeMeta(score) {
  if (score >= 0.9) return { letter: "S", color: "#fbbf24" };
  if (score >= 0.8) return { letter: "A", color: "#00e08a" };
  if (score >= 0.65) return { letter: "B", color: "#38bdf8" };
  if (score >= 0.5) return { letter: "C", color: "#8b5cf6" };
  if (score >= 0.35) return { letter: "D", color: "#fb923c" };
  return { letter: "F", color: "#ff4d6d" };
}

function showModal(p) {
  const score = p.grade ?? 0;
  const meta = gradeMeta(score);
  const a = state.account;
  $("grade-letter").textContent = meta.letter;
  $("grade-letter").style.color = meta.color;
  $("grade-score").textContent = `grade ${fmtNum(score, 3)}`;
  const ring = $("ring-fg");
  ring.style.stroke = meta.color;

  const stats = [
    ["FINAL EQUITY", fmtUsd(a.equity)],
    ["TOTAL REWARD", fmtNum(state.rewardTotal, 3)],
    ["NET P/L", fmtUsd(a.equity - 10000)],
    ["MAX DRAWDOWN", fmtNum(a.max_drawdown * 100, 1) + "%"],
    ["TRADES", String(a.total_trades)],
    ["WIN RATE", a.total_trades ? fmtNum(a.win_rate * 100, 0) + "%" : "—"],
  ];
  $("modal-stats").innerHTML = stats.map(([k, v]) =>
    `<div class="hud-cell"><span class="muted">${k}</span><span class="mono">${v}</span></div>`).join("");

  $("modal").classList.remove("hidden");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ring.style.strokeDashoffset = String(326.7 * (1 - score));
    });
  });
}

/* ---------------------------------------------------------- wiring */
$("reset-btn").addEventListener("click", doReset);
$("modal-restart").addEventListener("click", doReset);

$("play-btn").addEventListener("click", () => state.running ? stopAutopilot() : startAutopilot());

$("policy-select").addEventListener("change", updateSpeedUi);

$("task-select").addEventListener("change", () => {
  toast("Task changed — starting new episode", "info");
  doReset();
});

$("seed-input").addEventListener("change", () => {
  toast("Seed changed — starting new episode", "info");
  doReset();
});

$("speed-slider").addEventListener("input", updateSpeedUi);

document.querySelectorAll(".step-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const mode = $("policy-select").value;
    let steps = parseInt(btn.dataset.steps);
    if (mode === "llm") steps = 1;
    doStep({ mode, steps });
  });
});

const manualShares = () => parseInt($("shares-input").value) || 999999;
$("long-btn").addEventListener("click", () => doStep({ mode: "manual", action_type: "open_long", amount_shares: manualShares(), steps: 1 }));
$("short-btn").addEventListener("click", () => doStep({ mode: "manual", action_type: "open_short", amount_shares: manualShares(), steps: 1 }));
$("close-btn").addEventListener("click", () => doStep({ mode: "manual", action_type: "close_position", steps: 1 }));
$("wait-btn").addEventListener("click", () => doStep({ mode: "manual", action_type: "do_nothing", steps: 1 }));
$("stop-btn").addEventListener("click", () => {
  const pct = parseFloat($("stop-input").value);
  if (!pct || pct <= 0) { toast("Enter a stop %, e.g. 2", "bad"); return; }
  doStep({ mode: "manual", action_type: "set_stop_loss", stop_loss_percent: pct / 100, steps: 1 });
});

document.querySelectorAll(".tf-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tf-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    state.activeTf = tab.dataset.tf;
    redrawChart(true);
    updateBadges();
    mainChart.timeScale().fitContent();
  });
});

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); state.running ? stopAutopilot() : startAutopilot(); }
  else if (e.code === "ArrowRight") doStep({ mode: $("policy-select").value, steps: autopilotSteps() });
  else if (e.key === "l" || e.key === "L") $("long-btn").click();
  else if (e.key === "s" || e.key === "S") $("short-btn").click();
  else if (e.key === "c" || e.key === "C") $("close-btn").click();
});

/* boot — supports ?task=hard&seed=7&autorun=200 for shareable demo links */
async function checkLlmAvailability() {
  try {
    const res = await fetch("/api/status");
    const { llm_available, llm_loaded, load_error } = await res.json();
    const opt = document.querySelector('#policy-select option[value="llm"]');
    if (llm_available) {
      opt.disabled = false;
      opt.textContent = llm_loaded
        ? "🧠 Trained LLM (GRPO)"
        : "🧠 Trained LLM (GRPO) — loads on first step";
      if (load_error) toast(load_error, "bad");
    } else {
      opt.disabled = true;
      opt.textContent = "🧠 Trained LLM — train first (inference.py)";
      if ($("policy-select").value === "llm") $("policy-select").value = "bot";
    }
  } catch { /* status is cosmetic; ignore */ }
}

(async () => {
  const params = new URLSearchParams(location.search);
  if (params.get("task")) $("task-select").value = params.get("task");
  if (params.get("seed")) $("seed-input").value = params.get("seed");
  await checkLlmAvailability();
  updateSpeedUi();
  await doReset();
  const autorun = parseInt(params.get("autorun"));
  if (autorun > 0) await doStep({ mode: $("policy-select").value, steps: Math.min(autorun, 500) });
})();
