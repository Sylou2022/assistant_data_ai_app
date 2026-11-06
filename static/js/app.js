/* =========================================================
   Assistant Data AI — interface web (HTML / CSS / JS natif)
   Communique avec l'API Python (app/web/server.py).
   ========================================================= */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const OFFLINE_MESSAGE = "⚠️ **Le serveur de l'assistant n'est pas démarré.**\n\n"
  + "Dans un terminal, à la racine du projet, lancez :\n\n`python -m app.server`\n\n"
  + "puis ouvrez **http://localhost:7860** dans votre navigateur.";
const ZOOM_PREFIX = "Détail du chiffre d'affaires de la région";
const VIEWS = ["chat", "mission", "research", "guide", "about"];
const LOGO = "static/img/robot.svg";

/* Adresse de l'API : même origine quand la page est servie par « python -m app.server ».
   Si index.html est ouvert directement (double-clic) ou via un autre serveur
   (ex. Live Server de VS Code), on se rabat sur le serveur Python local. */
const DEFAULT_API = "http://127.0.0.1:7860";
let API = window.ASSISTANT_API_URL || "";
const api = (path) => `${API}${path}`;

async function detectApi() {
  const candidates = [...new Set([API, ...(location.protocol === "file:" ? [] : [""]), DEFAULT_API, "http://localhost:7860"])]
    .filter((c) => c !== undefined && !(c === "" && location.protocol === "file:"));
  for (const base of candidates) {
    try {
      const res = await fetch(`${base}/api/config`, { cache: "no-store" });
      if (res.ok && (res.headers.get("content-type") || "").includes("json")) {
        API = base;
        return res.json();
      }
    } catch { /* candidat suivant */ }
  }
  return null;
}

const state = {
  config: null,
  history: [],      // [{role, content}] envoyé au serveur pour le contexte
  memory: [],       // requêtes mémorisées (question + SQL), 5 dernières
  questions: [],    // historique affiché dans le menu
  regions: [],      // options du filtre Région
  busy: false,
  missionBusy: false,
  researchBusy: false,
  guideLoaded: false,
};

/* ---------------- Utilitaires ---------------- */

const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

function md(text) {
  const raw = window.marked ? marked.parse(String(text || ""), { breaks: true }) : `<p>${escapeHtml(text)}</p>`;
  return safe(raw);
}

function safe(html) {
  return window.DOMPurify ? DOMPurify.sanitize(html || "", { ADD_ATTR: ["target"] }) : (html || "");
}

const plainKey = (s) => String(s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

function toast(message, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  $("#toasts").appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.remove(), 300); }, 3800);
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || data.detail || `HTTP ${res.status}`);
  return data;
}

/** Lit un flux NDJSON (une ligne JSON par événement). */
async function streamJSON(url, body, onEvent) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while ((i = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, i).trim();
      buffer = buffer.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
}

/** Entrée = envoyer, Maj + Entrée = nouvelle ligne. */
function bindComposer(textarea, form, button) {
  const refresh = () => { autoResize(textarea); if (button) button.disabled = !textarea.value.trim(); };
  textarea.addEventListener("input", refresh);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  return refresh;
}

function renderChips(container, items, onClick) {
  container.innerHTML = "";
  items.filter(Boolean).forEach((text) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = text;
    b.addEventListener("click", () => onClick(text));
    container.appendChild(b);
  });
}

function renderDownloads(container, downloads) {
  const icons = { csv: "⬇", excel: "📗", pptx: "📊", md: "📝" };
  container.innerHTML = (downloads || []).map((d) => (
    `<a class="btn btn-download" href="${escapeHtml(api(d.url))}" download>${icons[d.kind] || "⬇"} ${escapeHtml(d.label)}</a>`
  )).join("");
}

function plot(el, figure, onClick) {
  if (!figure || !window.Plotly) { el.hidden = true; return; }
  el.hidden = false;
  const layout = Object.assign({}, figure.layout, {
    autosize: true,
    separators: ", ",          // format français : 1 254 300,5
    font: Object.assign({ family: "Inter, system-ui, sans-serif" }, (figure.layout || {}).font),
  });
  delete layout.width;
  Plotly.react(el, figure.data, layout, {
    responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: { format: "png", filename: "assistant-data-ai", scale: 2 },
  });
  if (onClick && !el.__clickBound) {
    el.__clickBound = true;
    el.on("plotly_click", (ev) => {
      const p = ev.points && ev.points[0];
      if (!p) return;
      onClick(p.location || (typeof p.x === "string" ? p.x : p.y));
    });
  }
}

/* ---------------- Navigation ---------------- */

function showView(view) {
  if (!VIEWS.includes(view)) view = "chat";
  $$(".view").forEach((s) => { s.hidden = s.dataset.view !== view; });
  $$("[data-view-link]").forEach((a) => a.classList.toggle("active", a.dataset.viewLink === view));
  $("#nav").classList.remove("open");
  $("#nav-toggle").setAttribute("aria-expanded", "false");
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (view === "guide" && !state.guideLoaded) loadGuide("");
  if (view === "chat") window.dispatchEvent(new Event("resize"));   // Plotly recalcule sa taille
}

window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

$("#nav-toggle").addEventListener("click", () => {
  const open = $("#nav").classList.toggle("open");
  $("#nav-toggle").setAttribute("aria-expanded", String(open));
});

/* ---------------- Conversation ---------------- */

const chatEl = $("#chat");

function addMessage(role, html, extra = {}) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role}`;
  if (role === "bot") wrap.innerHTML = `<img class="avatar" src="${LOGO}" alt="">`;
  const bubble = document.createElement("div");
  bubble.className = "bubble md";
  bubble.innerHTML = html;
  wrap.appendChild(bubble);
  if (extra.feedback) wrap.appendChild(feedbackBar(extra.feedback));
  chatEl.appendChild(wrap);
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
  return wrap;
}

function feedbackBar({ question, answer, sql }) {
  const bar = document.createElement("div");
  bar.className = "msg-actions";
  bar.innerHTML = `
    <button type="button" data-like="1" title="Réponse utile">👍</button>
    <button type="button" data-like="0" title="Réponse à améliorer">👎</button>
    <button type="button" data-copy-answer title="Copier la réponse">⧉</button>`;
  bar.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    if (btn.hasAttribute("data-copy-answer")) {
      await navigator.clipboard?.writeText(answer);
      toast("Réponse copiée.");
      return;
    }
    const liked = btn.dataset.like === "1";
    try {
      const res = await postJSON(api("/api/feedback"), { question, answer, liked, sql });
      $$("[data-like]", bar).forEach((b) => b.classList.toggle("on", b === btn));
      toast(res.message, "success");
    } catch {
      toast("Votre avis n'a pas pu être enregistré.", "error");
    }
  });
  return bar;
}

function loadingHtml(steps) {
  if (!steps.length) {
    return `<div class="chat-loading"><span class="spinner"></span><span>Analyse des données en cours…</span></div>`;
  }
  const items = steps.slice(-5);
  const current = items.pop();
  return `<div class="chat-loading chat-steps"><ul>${items.map((s) => `<li class="step-done">✅ ${escapeHtml(s)}</li>`).join("")}
    <li class="step-current"><span class="spinner"></span> ${escapeHtml(current)}…</li></ul></div>`;
}

function welcome() {
  chatEl.innerHTML = "";
  addMessage("bot", md(state.config?.welcome || "Bonjour 👋 Posez votre question."));
}

async function ask(question, { focusRegion = null } = {}) {
  question = (question || "").trim();
  if (!question || state.busy) return;
  state.busy = true;
  setAskBusy(true);

  addMessage("user", escapeHtml(question).replace(/\n/g, "<br>"));
  const loader = addMessage("bot", loadingHtml([]));
  const bubble = $(".bubble", loader);
  showResultsLoading();

  const steps = [];
  const payload = {
    question,
    history: state.history,
    memory: state.memory,
    year: $("#year-filter").value,
    region: $("#region-filter").value,
    focus_region: focusRegion,
  };

  try {
    await streamJSON(api("/api/ask"), payload, (ev) => {
      if (ev.type === "step" && !steps.includes(ev.label)) {
        steps.push(ev.label);
        bubble.innerHTML = loadingHtml(steps);
      } else if (ev.type === "result") {
        loader.remove();
        const sql = ev.panel?.sql || "";
        addMessage("bot", md(ev.message), { feedback: { question, answer: ev.message, sql } });
        state.history.push({ role: "user", content: question }, { role: "assistant", content: ev.message });
        state.history = state.history.slice(-12);
        if (ev.memo) state.memory = [...state.memory, ev.memo].slice(-5);
        rememberQuestion(question);
        renderPanel(ev.panel);
      }
    });
  } catch (err) {
    loader.remove();
    // Serveur Python arrêté (fetch impossible) ou page servie sans l'API (404/405/501)
    const unreachable = state.offline || err instanceof TypeError || /HTTP (404|405|501)/.test(err.message);
    addMessage("bot", md(unreachable
      ? OFFLINE_MESSAGE
      : "Je rencontre actuellement un problème pour récupérer ces informations. Vous pouvez réessayer dans un instant."));
    resetPanel();
    toast(`Erreur : ${err.message}`, "error");
  } finally {
    state.busy = false;
    setAskBusy(false);
  }
}

function setAskBusy(busy) {
  const btn = $("#ask-btn");
  btn.classList.toggle("loading", busy);
  btn.disabled = busy || !$("#ask-input").value.trim();
  $(".btn-label", btn).textContent = busy ? "Analyse…" : "Analyser";
}

function rememberQuestion(q) {
  if (!state.questions.includes(q)) state.questions.push(q);
  const box = $("#history");
  box.innerHTML = "";
  state.questions.slice(-8).reverse().forEach((text) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "side-item";
    b.textContent = text;
    b.addEventListener("click", () => {
      showView("chat"); location.hash = "chat";
      const input = $("#ask-input");
      input.value = text; input.dispatchEvent(new Event("input")); input.focus();
    });
    box.appendChild(b);
  });
}

/* ---------------- Panneau de résultats ---------------- */

function showResultsLoading() {
  $("#results-empty").hidden = true;
  $("#results-loading").hidden = false;
  ["#highlights", "#downloads"].forEach((s) => { $(s).innerHTML = ""; });
  $("#chart").hidden = true;
  $("#table-wrap").hidden = true;
  $("#badges").innerHTML = "";
}

function resetPanel() {
  $("#results-loading").hidden = true;
  $("#results-empty").hidden = false;
  ["#highlights", "#downloads", "#badges"].forEach((s) => { $(s).innerHTML = ""; });
  $("#chart").hidden = true;
  $("#table-wrap").hidden = true;
  $("#sql-fold").hidden = true;
  $("#plain").hidden = true;
  $("#tab-trace").innerHTML = '<p class="muted">La méthode de calcul s\'affichera ici.</p>';
  $("#tab-warnings").innerHTML = '<p class="muted">Aucun avertissement.</p>';
  $("#glossary").innerHTML = safe(state.config?.empty_glossary || "");
  renderChips($("#chips"), (state.config?.examples || []).slice(0, 3), (q) => ask(q));
}

function renderPanel(p) {
  $("#results-loading").hidden = true;
  $("#badges").innerHTML = safe(p.badges);
  $("#highlights").innerHTML = safe(p.highlights);
  plot($("#chart"), p.chart, onRegionClick);
  renderTable(p.table);
  renderDownloads($("#downloads"), p.downloads);
  $("#results-empty").hidden = Boolean(p.chart || p.table || p.highlights);

  $("#sql-fold").hidden = !p.sql;
  $("#sql-code").textContent = p.sql || "";
  $("#plain").hidden = !p.plain;
  $("#plain").textContent = p.plain || "";

  $("#tab-trace").innerHTML = p.trace ? md(p.trace) : '<p class="muted">Aucune information de traçabilité.</p>';
  $("#tab-warnings").innerHTML = p.warnings ? md(p.warnings) : '<p class="muted">Aucun avertissement.</p>';
  $("#glossary").innerHTML = safe(p.glossary);
  renderChips($("#chips"), p.suggestions || [], (q) => ask(q));
  if (p.alerts !== null && p.alerts !== undefined) renderAlerts(p.alerts);
}

function renderTable(t) {
  const wrap = $("#table-wrap");
  if (!t || !t.rows || !t.rows.length) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const withBars = Array.isArray(t.bars);
  const head = t.columns.map((c, i) => `<th class="${t.numeric[i] ? "num" : ""}">${escapeHtml(c)}</th>`).join("")
    + (withBars ? `<th class="bar-col">Poids relatif</th>` : "");
  const body = t.rows.map((row, r) => {
    const cells = row.map((v, i) => `<td class="${t.numeric[i] ? "num" : ""}">${escapeHtml(v)}</td>`).join("");
    const bar = withBars
      ? `<td class="bar-col"><span class="bar"><span style="width:${Math.round((t.bars[r] || 0) * 100)}%"></span></span></td>`
      : "";
    return `<tr data-row="${r}">${cells}${bar}</tr>`;
  }).join("");
  $("#table").innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
  $$("#table tbody tr").forEach((tr) => {
    const row = t.rows[Number(tr.dataset.row)];
    const region = row.map(matchRegionLocal).find(Boolean);
    if (region) {
      tr.classList.add("clickable");
      tr.title = `Zoomer sur ${region}`;
      tr.addEventListener("click", () => zoomRegion(region));
    }
  });
}

function renderAlerts(html) {
  const box = $("#alerts");
  box.innerHTML = safe(html);
  box.hidden = !html;
}

/* ---------------- Régions : carte, tableau, filtres ---------------- */

function matchRegionLocal(name) {
  const key = plainKey(name);
  return key ? state.regions.find((r) => plainKey(r) === key) || null : null;
}

async function onRegionClick(name) {
  let region = matchRegionLocal(name);
  if (!region && name) {
    try { region = (await getJSON(api(`/api/match-region?name=${encodeURIComponent(name)}`))).region; } catch { /* ignoré */ }
  }
  if (region) zoomRegion(region);
}

function zoomRegion(region) {
  $("#region-filter").value = region;
  ask(`${ZOOM_PREFIX} ${region}`, { focusRegion: region });
}

/** Changement d'Année / Région : on relance la dernière question avec les filtres. */
function refilter() {
  const users = state.history.filter((m) => m.role === "user").map((m) => m.content);
  const last = users[users.length - 1];
  const lastReal = [...users].reverse().find((q) => !q.startsWith(ZOOM_PREFIX));
  const region = $("#region-filter").value;
  if (last && last.startsWith(ZOOM_PREFIX) && region !== "Toutes") zoomRegion(region);
  else if (lastReal) ask(lastReal);
  else if (region !== "Toutes") zoomRegion(region);
}

async function loadFilters(attempt = 1) {
  try {
    const { years, regions } = await getJSON(api("/api/filters"));
    if (!years.length && !regions.length && attempt < 5) {
      // Base probablement en cours de réveil : on réessaie un peu plus tard.
      setTimeout(() => loadFilters(attempt + 1), 15000);
      return;
    }
    state.regions = regions;
    const fill = (sel, values) => {
      sel.innerHTML = ['<option value="Toutes">Toutes</option>', ...values.map((v) => `<option>${escapeHtml(v)}</option>`)].join("");
    };
    fill($("#year-filter"), years);
    fill($("#region-filter"), regions);
  } catch {
    toast("Filtres indisponibles : la base de données ne répond pas.", "error");
  }
}

/* ---------------- Agent analyste ---------------- */

async function runMission(objective) {
  objective = (objective || "").trim();
  if (!objective || state.missionBusy) return;
  state.missionBusy = true;
  const btn = $("#mission-btn");
  btn.disabled = true; btn.textContent = "⏳ Mission en cours…";
  $("#mission-report").hidden = true;
  $("#mission-steps").innerHTML = "";
  $("#mission-downloads").innerHTML = "";

  try {
    await streamJSON(api("/api/mission"), {
      objective, year: $("#year-filter").value, region: $("#region-filter").value,
    }, (ev) => {
      if (ev.type === "progress") {
        $("#mission-progress").innerHTML = safe(ev.html);
      } else if (ev.type === "result") {
        $("#mission-progress").innerHTML = safe(ev.progress);
        const report = $("#mission-report");
        report.innerHTML = md(ev.report);
        report.hidden = !ev.report;
        renderDownloads($("#mission-downloads"), ev.downloads);
        renderMissionSteps(ev.steps || []);
      }
    });
  } catch (err) {
    toast(`Mission impossible : ${err.message}`, "error");
  } finally {
    state.missionBusy = false;
    btn.disabled = false; btn.textContent = "🚀 Lancer la mission";
  }
}

function renderMissionSteps(steps) {
  const grid = $("#mission-steps");
  grid.innerHTML = "";
  steps.forEach((step) => {
    const card = document.createElement("div");
    card.className = "mission-step-card";
    card.innerHTML = `<div class="md">${md(step.markdown)}</div>`;
    if (step.chart) {
      const chart = document.createElement("div");
      chart.className = "chart chart-sm";
      card.appendChild(chart);
      grid.appendChild(card);
      plot(chart, step.chart);
    } else {
      grid.appendChild(card);
    }
  });
}

/* ---------------- Recherche documentée ---------------- */

async function runResearch(question) {
  question = (question || "").trim();
  if (!question || state.researchBusy) return;
  state.researchBusy = true;
  const btn = $("#research-btn");
  btn.disabled = true; btn.textContent = "⏳ Recherche…";
  $("#research-results").hidden = true;
  $("#research-status").innerHTML = `<div class="research-loading"><span class="spinner spinner-lg"></span>
    <div><b>Recherche en cours…</b><br><span>${escapeHtml(question)}</span></div></div>`;

  try {
    const res = await postJSON(api("/api/research"), { question });
    const n = res.sources.length;
    $("#research-status").innerHTML = `<div class="research-status">✅ Synthèse prête en ${String(res.elapsed).replace(".", ",")} s · ${n} source${n > 1 ? "s" : ""}</div>`;
    $("#research-answer").innerHTML = md(res.answer);
    $("#research-sources").innerHTML = safe(res.sources_html);
    $("#research-results").hidden = false;
  } catch (err) {
    $("#research-status").innerHTML = `<div class="alert-banner"><div class="alert-title">⚠️ Recherche impossible</div>${escapeHtml(err.message)}</div>`;
  } finally {
    state.researchBusy = false;
    btn.disabled = false; btn.textContent = "🔎 Rechercher";
  }
}

/* ---------------- Guide ---------------- */

let guideTimer = null;

async function loadGuide(q) {
  try {
    const res = await fetch(api(`/api/guide?q=${encodeURIComponent(q || "")}`));
    $("#guide-content").innerHTML = safe(await res.text());
    state.guideLoaded = true;
  } catch {
    $("#guide-content").innerHTML = '<div class="guide-empty">Guide indisponible pour le moment.</div>';
  }
}

/* ---------------- Initialisation ---------------- */

function bindEvents() {
  const askInput = $("#ask-input");
  const refreshAsk = bindComposer(askInput, $("#ask-form"), $("#ask-btn"));
  $("#ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = askInput.value;
    askInput.value = ""; refreshAsk();
    ask(q);
  });

  $("#reset-btn").addEventListener("click", () => {
    if (state.busy) return;
    state.history = []; state.memory = [];
    welcome(); resetPanel();
    toast("Nouvelle conversation.");
  });

  $("#year-filter").addEventListener("change", refilter);
  $("#region-filter").addEventListener("change", refilter);

  $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $("#tab-trace").hidden = tab.dataset.tab !== "trace";
    $("#tab-warnings").hidden = tab.dataset.tab !== "warnings";
  }));

  $$("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
    await navigator.clipboard?.writeText($(`#${b.dataset.copy}`).textContent);
    toast("Requête SQL copiée.");
  }));

  const missionInput = $("#mission-input");
  bindComposer(missionInput, $("#mission-form"));
  $("#mission-form").addEventListener("submit", (e) => { e.preventDefault(); runMission(missionInput.value); });

  const researchInput = $("#research-input");
  bindComposer(researchInput, $("#research-form"));
  $("#research-form").addEventListener("submit", (e) => { e.preventDefault(); runResearch(researchInput.value); });

  $("#guide-search").addEventListener("input", (e) => {
    clearTimeout(guideTimer);
    guideTimer = setTimeout(() => loadGuide(e.target.value), 250);
  });
}

async function init() {
  bindEvents();
  showView(location.hash.slice(1) || "chat");

  state.config = await detectApi();
  if (!state.config) {
    state.config = { examples: [], mission_examples: [], research_examples: [] };
    state.offline = true;
  }
  welcome();
  resetPanel();
  if (state.offline) {
    addMessage("bot", md(OFFLINE_MESSAGE));
    $("#ask-btn").disabled = true;
    return;
  }

  const sideExamples = $("#examples");
  state.config.examples.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "side-item"; b.textContent = q;
    b.addEventListener("click", () => { location.hash = "chat"; ask(q); });
    sideExamples.appendChild(b);
  });
  const fillInput = (id) => (q) => { const el = $(id); el.value = q; el.dispatchEvent(new Event("input")); el.focus(); };
  renderChips($("#mission-examples"), state.config.mission_examples, fillInput("#mission-input"));
  renderChips($("#research-examples"), state.config.research_examples, fillInput("#research-input"));

  loadFilters();
  getJSON(api("/api/alerts")).then((r) => renderAlerts(r.html)).catch(() => {});
}

document.addEventListener("DOMContentLoaded", init);
