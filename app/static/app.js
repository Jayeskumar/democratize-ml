"use strict";
/* Democratize ML front end. No framework, no build step, no outside requests. */

/* Icons: 24×24 stroke paths, drawn for this app. */
const ICONS = {
  upload: "M12 15V4M7 9l5-5 5 5M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4",
  table: "M3 5h18v14H3zM3 10h18M3 15h18M9 5v14",
  users: "M16 20v-1.5a3.5 3.5 0 0 0-3.5-3.5h-5A3.5 3.5 0 0 0 4 18.5V20M10 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM20 20v-1.5a3.5 3.5 0 0 0-2.5-3.35M15.5 5.2a3 3 0 0 1 0 5.6",
  home: "M3 11l9-7 9 7M5 10v10h14V10M10 20v-6h4v6",
  leaf: "M12 21v-9M12 12C12 7 8.5 4 4 4c0 5 3 8 8 8zM12 15c0-5 3.5-8 8-8 0 5-3 8-8 8",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  eraser: "M18 20H8l-4.3-4.3a1 1 0 0 1 0-1.4l9.6-9.6a1 1 0 0 1 1.4 0l5.6 5.6a1 1 0 0 1 0 1.4L12 20M9 9.5l6 6",
  ruler: "M3.5 16.5l13-13 4 4-13 13zM7.5 12.5l2 2M10.5 9.5l2 2M13.5 6.5l2 2",
  scale: "M12 4v16M7 20h10M5 7h14M5 7l-2.5 6a2.5 2.5 0 0 0 5 0zM19 7l-2.5 6a2.5 2.5 0 0 0 5 0z",
  spark: "M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9zM18.5 15l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z",
  line: "M4 19L20 6M6.5 11.5h.01M10 15.5h.01M13.5 8h.01M17.5 13h.01",
  tree: "M12 4v4M6 12V8h12v4M6 12v4M3 16h6M18 12v4M15 16h6",
  forest: "M7 3l-4 8h8zM7 11v9M17 3l-4 8h8zM17 11v9",
  boost: "M3 17l6-6 4 4 8-8M15 7h6v6",
  near: "M12 12h.01M5 6h.01M19 6h.01M5 19h.01M19 18h.01M12 12L5 6M12 12l7 6M12 12l-7 7",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 12h.01",
  eyeoff: "M3 3l18 18M10.6 6.1A9 9 0 0 1 12 6c6 0 9.5 6 9.5 6a16 16 0 0 1-2.4 3.1M6.6 6.6C4 8.3 2.5 12 2.5 12S6 18 12 18c1.9 0 3.5-.5 4.9-1.3M9.9 9.9a3 3 0 0 0 4.2 4.2",
  check: "M5 12l5 5L20 7",
  download: "M12 4v11M7 10l5 5 5-5M4 20h16",
  code: "M8 7l-5 5 5 5M16 7l5 5-5 5",
  chart: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  wand: "M15 4V2M15 10V8M11 6h2M17 6h2M3 21l12-12M18 13l1 1M14 3l-1-1",
  plus: "M12 5v14M5 12h14",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5M12 8h.01",
};
const SAMPLE_ICONS = { "customer-churn": "users", "house-prices": "home", "iris-flowers": "leaf" };

const STEPS = {
  dedupe: ["Remove duplicate rows", "Drops rows that appear more than once.", "copy"],
  drop_missing: ["Remove rows with blanks", "Otherwise blanks are filled in with typical values.", "eraser"],
  scale: ["Put numbers on one scale", "Helps when columns have very different sizes.", "ruler"],
  balance: ["Balance rare answers", "Pays extra attention to rare answers, like fraud.", "scale"],
};
const MODELS = {
  auto: ["Best pick (automatic)", "Tries several learners and keeps the best. Recommended.", "spark"],
  linear: ["Simple formula", "Fast and easy to explain.", "line"],
  tree: ["Decision tree", "A flowchart of yes/no questions.", "tree"],
  forest: ["Random forest", "Many trees voting together. A reliable all-rounder.", "forest"],
  boosting: ["Gradient boosting", "Often the most accurate on spreadsheets.", "boost"],
  knn: ["Nearest neighbours", "Looks up the most similar past examples.", "near"],
};
const TRAIN_STEPS = [
  "Reading your data",
  "Filling blanks and preparing columns",
  "Trying different learners",
  "Testing on rows it hasn't seen",
  "Working out what matters most",
];

const state = { dataset: null, target: null, ignore: new Set(), steps: [], model: "auto", task: "auto", busy: false };
let samples = [];
let dragging = null; // {kind, value} of the block or chip being dragged
let justPlaced = null; // "kind:value" of the last thing added, so only it animates in
const mark = (kind, value) => (justPlaced = `${kind}:${value}`);
const fresh = (kind, value) => (justPlaced === `${kind}:${value}` ? " fresh-drop" : "");

const $ = (sel) => document.querySelector(sel);

/** Tiny DOM builder. Text always goes in as text nodes, so data can never inject HTML. */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) el.append(kid);
  return el;
}
const NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, text) {
  const el = document.createElementNS(NS, tag);
  for (const [a, v] of Object.entries(attrs)) el.setAttribute(a, v);
  if (text != null) el.textContent = text;
  return el;
}
function icon(name) {
  const svg = s("svg", { viewBox: "0 0 24 24", class: "i", "aria-hidden": "true" });
  svg.append(s("path", { d: ICONS[name] || ICONS.info }));
  return svg;
}

const fmt = (n, digits = 3) =>
  n == null ? "" : new Intl.NumberFormat(undefined, { maximumSignificantDigits: Math.abs(n) >= 1000 ? 4 : digits }).format(n);
const pct = (x) => Math.round(x * 100);
const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 7000);
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch("/api" + path, options);
  } catch {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" && detail ? detail : `Something went wrong (error ${res.status}).`);
  }
  return res;
}
const getJSON = async (path, options) => (await api(path, options)).json();
const postJSON = (path, body) =>
  getJSON(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/* ------------------------------------------------------------ drag and drop */

function draggable(el, kind, value) {
  el.draggable = true;
  el.addEventListener("dragstart", (e) => {
    dragging = { kind, value };
    e.dataTransfer.effectAllowed = "copyMove";
    e.dataTransfer.setData("text/plain", value);
    document.body.classList.add("dragging");
    document.querySelectorAll(".zone").forEach((z) => z.classList.toggle("can", accepts(z, kind)));
  });
  el.addEventListener("dragend", endDrag);
  return el;
}
function endDrag() {
  dragging = null;
  document.body.classList.remove("dragging");
  document.querySelectorAll(".zone").forEach((z) => z.classList.remove("can", "over"));
}
const accepts = (zone, kind) => (zone.dataset.accept || "").split(" ").includes(kind);
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");

function dropzone(el, onDrop) {
  const ok = (e) => (dragging && accepts(el, dragging.kind)) || (accepts(el, "file") && hasFiles(e));
  el.addEventListener("dragover", (e) => {
    if (!ok(e)) return;
    e.preventDefault();
    el.classList.add("over");
  });
  el.addEventListener("dragleave", (e) => {
    if (!el.contains(e.relatedTarget)) el.classList.remove("over");
  });
  el.addEventListener("drop", (e) => {
    if (!ok(e)) return;
    e.preventDefault();
    const item = dragging || { kind: "file", value: e.dataTransfer.files[0] };
    endDrag();
    onDrop(item);
  });
  return el;
}

/** One entry point for "this block goes into the recipe", used by both dropping and clicking. */
function place(kind, value, toggle = false) {
  if (state.busy) return;
  if (kind === "data") return loadSample(value);
  if (kind === "file") return uploadFile(value);
  if (kind === "step") {
    if (!state.steps.includes(value)) { state.steps.push(value); mark("step", value); }
    else if (toggle) state.steps = state.steps.filter((x) => x !== value);
  }
  if (kind === "model") { state.model = value; mark("model", value); }
  render();
}

// Dropping a file anywhere outside a drop area should not navigate the browser away.
window.addEventListener("dragover", (e) => hasFiles(e) && e.preventDefault());
window.addEventListener("drop", (e) => hasFiles(e) && e.preventDefault());

/* ------------------------------------------------------------------ builder */

function block(kind, value, title, text, iconName, { removable, onRemove, used } = {}) {
  const tail = removable
    ? h("button", { class: "x", type: "button", "aria-label": `Remove ${title}`, onclick: onRemove }, "×")
    : used ? h("span", { class: "tick" }, icon("check")) : h("span", { class: "grip", "aria-hidden": "true" });
  const el = h(removable ? "div" : "button", {
    class: `block k-${kind}${used ? " used" : ""}${removable ? fresh(kind, value) : ""}`, type: removable ? null : "button",
    "aria-pressed": removable ? null : String(!!used),
  }, h("span", { class: "ic" }, icon(iconName)), h("span", {}, h("b", {}, title), h("span", { class: "d" }, text)), tail);
  if (!removable) {
    draggable(el, kind, value);
    el.addEventListener("click", () => place(kind, value, true));
  }
  return el;
}

function renderPalette() {
  $("#palette-data").replaceChildren(...samples.map((x) =>
    block("data", x.id, x.name, x.description, SAMPLE_ICONS[x.id] || "table", { used: state.dataset?.sample === x.id })));
  $("#palette-step").replaceChildren(...Object.entries(STEPS).map(([id, [t, d, ic]]) =>
    block("step", id, t, d, ic, { used: state.steps.includes(id) })));
  $("#palette-model").replaceChildren(...Object.entries(MODELS).map(([id, [t, d, ic]]) =>
    block("model", id, t, d, ic, { used: state.model === id })));
}

function renderData() {
  const body = $("#data-body");
  const d = state.dataset;
  if (!d) {
    const zone = dropzone(
      h("div", { class: "zone drop-data", "data-accept": "data file" },
        h("span", { class: "big-ic" }, icon("upload")),
        h("strong", {}, "Drop a spreadsheet here"),
        h("p", {}, "CSV or Excel · one row per example · column names in the first row"),
        h("div", { class: "row", style: "justify-content:center;margin-top:.35rem" },
          h("button", { class: "primary", type: "button", onclick: () => $("#file-input").click() }, icon("table"), "Choose a file")),
        samples.length ? h("div", { class: "quick" },
          h("span", { class: "lab", style: "justify-content:center" }, "No data handy? Try an example"),
          samples.map((x) => h("button", { class: "pill", type: "button", onclick: () => place("data", x.id) },
            icon(SAMPLE_ICONS[x.id] || "table"), x.name))) : null),
      (item) => place(item.kind, item.value));
    body.replaceChildren(zone);
    return;
  }
  const numeric = d.columns.map((c) => c.type === "number");
  const cls = (c, i) => [numeric[i] && "n", c.name === state.target && "tgt", state.ignore.has(c.name) && "ign"].filter(Boolean).join(" ") || null;
  const blanks = d.columns.reduce((n, c) => n + c.missing, 0);
  body.replaceChildren(
    h("div", { class: "data-head" },
      h("span", { class: "ic" }, icon(SAMPLE_ICONS[d.sample] || "table")),
      h("div", {},
        h("b", {}, d.name),
        h("span", { class: "meta" }, `${plural(d.rows, "row")} · ${plural(d.columns.length, "column")}${blanks ? ` · ${plural(blanks, "blank")} (we'll handle them)` : ""}`)),
      h("button", { class: "ghost", type: "button", onclick: resetData }, "Use different data")),
    h("div", { class: "table-wrap", tabindex: "0", "aria-label": "First rows of your data" },
      h("table", {},
        h("thead", {}, h("tr", {}, d.columns.map((c, i) => h("th", { class: cls(c, i) }, c.name)))),
        h("tbody", {}, d.preview.map((row) => h("tr", {}, row.map((v, i) =>
          v == null ? h("td", { class: ["blank", cls(d.columns[i], i)].filter(Boolean).join(" ") }, "blank")
            : h("td", { class: cls(d.columns[i], i) }, typeof v === "number" ? fmt(v, 6) : String(v)))))))));
}

function chip(col, { placed, onRemove, dim } = {}) {
  const parts = [h("span", { class: "cn" }, col.name), h("i", {}, col.type === "number" ? "123" : "abc")];
  if (placed) {
    return h("span", { class: "chip placed" + fresh("column", col.name) }, parts,
      h("button", { class: "x", type: "button", "aria-label": `Put ${col.name} back`, onclick: onRemove }, "×"));
  }
  const tips = [];
  if (dim) tips.push("Skipped automatically: it is different on almost every row (like an ID), or the same on every row.");
  if (col.missing) tips.push(`${plural(col.missing, "blank")} — filled in for you.`);
  const el = h("button", { class: "chip" + (dim ? " dim" : ""), type: "button", "aria-haspopup": "menu", title: tips.join(" ") || "Drag, or tap for options" },
    col.missing && !dim ? h("span", { class: "dot", "aria-label": "has blanks" }) : null, parts);
  draggable(el, "column", col.name);
  el.addEventListener("click", (e) => { e.stopPropagation(); chipMenu(el, col.name); });
  return el;
}

function chipMenu(anchor, name) {
  closeMenu();
  const act = (fn) => () => { closeMenu(); fn(); render(); };
  const menu = h("div", { class: "menu", role: "menu" },
    h("button", { type: "button", role: "menuitem", onclick: act(() => { setTarget(name); mark("column", name); }) }, icon("target"), "Predict this"),
    h("button", { type: "button", role: "menuitem", onclick: act(() => { state.ignore.add(name); mark("column", name); }) }, icon("eyeoff"), "Ignore this column"));
  const r = anchor.getBoundingClientRect();
  menu.style.left = `${Math.max(scrollX + 8, Math.min(r.left + scrollX, scrollX + innerWidth - 216))}px`;
  menu.style.top = `${r.bottom + scrollY + 6}px`;
  document.body.append(menu);
  menu.firstChild.focus();
}
const closeMenu = () => document.querySelectorAll(".menu").forEach((m) => m.remove());
document.addEventListener("click", closeMenu);
document.addEventListener("keydown", (e) => e.key === "Escape" && closeMenu());

function setTarget(name) {
  state.target = name;
  state.task = "auto";
  state.ignore.delete(name);
}

const taskOf = (col) => (state.task === "auto" ? col.task : state.task);

function renderTarget() {
  const body = $("#target-body");
  const d = state.dataset;
  if (!d) {
    body.replaceChildren(h("p", { class: "meta" }, "Your columns will show up here once you add data."));
    return;
  }
  const byName = Object.fromEntries(d.columns.map((c) => [c.name, c]));
  const pool = d.columns.filter((c) => c.name !== state.target && !state.ignore.has(c.name));

  const targetZone = dropzone(h("div", { class: "zone target", "data-accept": "column" },
    state.target
      ? chip(byName[state.target], { placed: true, onRemove: () => { state.target = null; render(); } })
      : h("span", { class: "empty" }, icon("target"), "Drop the answer column here")),
    (item) => { setTarget(item.value); mark("column", item.value); render(); });

  const ignoreZone = dropzone(h("div", { class: "zone", "data-accept": "column" },
    state.ignore.size
      ? [...state.ignore].map((n) => chip(byName[n], { placed: true, onRemove: () => { state.ignore.delete(n); render(); } }))
      : h("span", { class: "empty" }, "Columns the computer shouldn't look at")),
    (item) => { if (state.target === item.value) state.target = null; state.ignore.add(item.value); mark("column", item.value); render(); });

  let kind = null;
  if (state.target) {
    const col = byName[state.target];
    const task = taskOf(col);
    kind = h("p", { class: "kind" }, icon(task === "classification" ? "table" : "chart"),
      h("span", {}, task === "classification"
        ? ["It will pick ", h("b", {}, "a category"), ` — one of the ${col.unique} different values in “${col.name}”.`]
        : ["It will estimate ", h("b", {}, "a number"), `, like the values in “${col.name}”.`]),
      col.type === "number" && h("button", { class: "link", type: "button", onclick: () => {
        state.task = task === "classification" ? "regression" : "classification";
        render();
      } }, task === "classification" ? "Treat as a number instead" : "Treat as a category instead"));
  }

  const usable = pool.filter((c) => !c.id_like && !c.empty).length;
  body.replaceChildren(h("div", { class: "cols" },
    h("div", {},
      h("span", { class: "lab" }, state.target
        ? `Clues the computer will learn from (${usable})`
        : "Your columns — drag the one you want predicted into the box below, or tap it"),
      h("div", { class: "pool" },
        pool.length ? pool.map((c) => chip(c, { dim: c.id_like || c.empty })) : h("span", { class: "meta" }, "No columns left to use as clues."))),
    h("div", { class: "pair" },
      h("div", {}, h("span", { class: "lab" }, icon("target"), "Predict this"), targetZone),
      h("div", {}, h("span", { class: "lab" }, icon("eyeoff"), "Ignore these (optional)"), ignoreZone)),
    kind));
}

function renderSteps() {
  $("#steps-zone").replaceChildren(...(state.steps.length
    ? state.steps.map((id) => block("step", id, ...STEPS[id], {
        removable: true, onRemove: () => { state.steps = state.steps.filter((x) => x !== id); render(); } }))
    : [h("span", { class: "empty" }, icon("wand"), "Nothing needed. Drop clean-up blocks here if you want extra control.")]));
}

function renderModel() {
  $("#model-zone").replaceChildren(block("model", state.model, ...MODELS[state.model], {
    removable: state.model !== "auto", onRemove: () => { state.model = "auto"; render(); } }));
}

function renderStages() {
  const d = !!state.dataset, t = d && !!state.target;
  const set = (id, { done, current, locked }) => {
    const el = $(id);
    el.classList.toggle("done", !!done);
    el.classList.toggle("current", !!current);
    el.classList.toggle("locked", !!locked);
  };
  set("#stage-data", { done: d, current: !d });
  set("#stage-target", { done: t, current: d && !t, locked: !d });
  set("#stage-steps", { done: t, locked: !t });
  set("#stage-model", { done: t, locked: !t });
}

function render() {
  renderPalette(); renderData(); renderTarget(); renderSteps(); renderModel(); renderStages();
  justPlaced = null;
  const ready = !!(state.dataset && state.target);
  const btn = $("#train");
  btn.disabled = state.busy || !ready;
  btn.replaceChildren(icon("spark"), "Train my model");
  $(".train-bar").classList.toggle("ready", ready);
  const msg = $("#train-msg");
  if (!state.dataset) msg.replaceChildren("Step 1: add some data to get started.");
  else if (!state.target) msg.replaceChildren("Step 2: choose the column you want to predict.");
  else {
    const col = state.dataset.columns.find((c) => c.name === state.target);
    const clues = state.dataset.columns.filter((c) => c.name !== state.target && !state.ignore.has(c.name) && !c.id_like && !c.empty).length;
    msg.replaceChildren("Ready: predict ", h("b", {}, `“${state.target}”`), ` as ${taskOf(col) === "classification" ? "a category" : "a number"} from ${plural(clues, "clue")}, using `,
      h("b", {}, MODELS[state.model][0].replace(" (automatic)", "")),
      state.steps.length ? ` with ${plural(state.steps.length, "clean-up step")}.` : ".");
  }
}

function resetData() {
  Object.assign(state, { dataset: null, target: null, ignore: new Set(), task: "auto" });
  render();
}

async function withData(promise, sample) {
  if (state.busy) return;
  const body = $("#data-body");
  body.replaceChildren(h("div", { class: "skeleton" }));
  try {
    const d = await promise;
    resetData();
    state.dataset = { ...d, sample };
    render();
    $("#stage-target").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (err) { toast(err.message); render(); }
}
const loadSample = (id) => withData(getJSON(`/datasets/sample/${id}`, { method: "POST" }), id);
function uploadFile(file) {
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  return withData(getJSON("/datasets", { method: "POST", body }));
}

function trainingOverlay() {
  const steps = TRAIN_STEPS.filter((x, i) => i !== 2 || state.model === "auto");
  const items = steps.map((x) => h("li", {}, x));
  const el = h("div", { class: "overlay", role: "dialog", "aria-modal": "true", "aria-label": "Training" },
    h("div", { class: "card" },
      h("h2", {}, "Training your model…"),
      h("p", { class: "sub" }, `Learning to predict “${state.target}” from ${plural(state.dataset.rows, "example")}.`),
      h("ul", { class: "ticks", "aria-live": "polite" }, items),
      h("div", { class: "track" }, h("div", {}))));
  let i = 0;
  const tick = () => items.forEach((li, j) => { li.className = j < i ? "ok" : j === i ? "now" : ""; });
  tick();
  const timer = setInterval(() => { if (i < items.length - 1) { i++; tick(); } }, 700);
  document.body.append(el);
  return {
    async finish(ok) {
      clearInterval(timer);
      if (ok) { i = items.length; tick(); await new Promise((r) => setTimeout(r, 350)); }
      el.remove();
    },
  };
}

async function trainModel() {
  state.busy = true;
  render();
  const overlay = trainingOverlay();
  try {
    const m = await postJSON("/models", {
      dataset_id: state.dataset.id, target: state.target, ignore: [...state.ignore],
      steps: state.steps, model: state.model, task: state.task,
    });
    await overlay.finish(true);
    location.hash = `model/${m.id}`;
    refreshCount();
  } catch (err) { await overlay.finish(false); toast(err.message); }
  state.busy = false;
  render();
}

/* ------------------------------------------------------------ model results */

function verdict(m) {
  const k = m.metrics;
  const gain = m.task === "classification" ? k.accuracy - k.baseline : (k.baseline_mae ? 1 - k.mae / k.baseline_mae : 0);
  const [weak, strong] = m.task === "classification" ? [0.02, 0.1] : [0.05, 0.3];
  if (gain <= weak) return ["bad", "No better than guessing", "Try adding more rows, or columns that relate more closely to the answer."];
  if (gain <= strong) return ["ok", "A little better than guessing", "It has found a weak pattern. More or better data would help."];
  const great = m.task === "classification" ? k.accuracy >= 0.9 : gain >= 0.6;
  return great ? ["good", "Very reliable", m.task === "classification" ? "It gets the answer right almost every time." : "Its estimates land close to the real values."]
    : ["good", "Useful", "Clearly better than guessing, though not perfect."];
}

function ring(frac, big, small, tone, size = "") {
  const r = 50, c = 2 * Math.PI * r;
  const fg = s("circle", { class: "fg", cx: 60, cy: 60, r, "stroke-dasharray": c, "stroke-dashoffset": c });
  const svg = s("svg", { viewBox: "0 0 120 120", "aria-hidden": "true" });
  svg.append(s("circle", { class: "bg", cx: 60, cy: 60, r }), fg);
  requestAnimationFrame(() => requestAnimationFrame(() => fg.setAttribute("stroke-dashoffset", c * (1 - Math.max(0, Math.min(1, frac))))));
  return h("div", { class: `ring tone-${tone} ${size}`, role: "img", "aria-label": `${big} ${small || ""}` },
    svg, h("div", { class: "c" }, h("b", {}, big), small ? h("span", {}, small) : null));
}

function scoreCard(m) {
  const k = m.metrics;
  const [tone, title, text] = verdict(m);
  let gauge, rows;
  if (m.task === "classification") {
    gauge = ring(k.accuracy, `${pct(k.accuracy)}%`, "correct", tone);
    rows = [["This model", k.accuracy, `${pct(k.accuracy)}%`, ""], ["Guessing the most common answer", k.baseline, `${pct(k.baseline)}%`, " base"]];
  } else {
    const top = Math.max(k.mae, k.baseline_mae) || 1;
    gauge = ring(Math.max(0, k.r2), `${pct(Math.max(0, k.r2))}%`, "explained", tone);
    rows = [["This model is typically off by", k.mae / top, `±${fmt(k.mae)}`, ""], ["Guessing the middle value is off by", k.baseline_mae / top, `±${fmt(k.baseline_mae)}`, " base"]];
  }
  return h("section", { class: `card hero tone-${tone}` }, gauge,
    h("div", {},
      h("span", { class: "verdict" }, icon(tone === "good" ? "check" : "info"), title),
      h("p", { class: "say" }, text),
      h("div", { class: "cmp" }, rows.map(([label, frac, value, cls]) => [
        h("span", {}, label),
        h("div", { class: "bar" + cls }, h("div", { style: `width:${Math.max(0, Math.min(1, frac)) * 100}%` })),
        h("b", {}, value)]))),
    h("p", { class: "meta foot" },
      `Tested on ${plural(m.rows_tested, "row")} the model never saw while learning from the other ${m.rows_trained.toLocaleString()}.`,
      m.task === "regression" ? ` “Explained” means how much of the up-and-down in “${m.target}” it accounts for.` : ""));
}

function importanceCard(m) {
  const top = m.importance.slice(0, 10);
  const max = Math.max(...top.map((r) => r.share), 0.0001);
  return h("section", { class: "card" },
    h("h2", {}, icon("chart"), "What matters most"),
    h("p", { class: "sub" }, "How much each column sways the prediction."),
    h("div", { class: "imp" }, top.map((r) => [
      h("span", { class: "nm", title: r.name }, r.name),
      h("div", { class: "bar" }, h("div", { style: `width:${(r.share / max) * 100}%` })),
      h("span", { class: "v" }, `${pct(r.share)}%`)])));
}

function detailCard(m) {
  const k = m.metrics;
  if (m.task === "classification") {
    if (k.labels.length > 8) return null;
    const max = Math.max(...k.confusion.flat(), 1);
    return h("section", { class: "card" },
      h("h2", {}, icon("table"), "Where it gets confused"),
      h("p", { class: "sub" }, "Rows are the real answers, columns are what the model said. The diagonal is where it was right."),
      h("div", { style: "overflow:auto" }, h("table", { class: "cm" },
        h("thead", {}, h("tr", {}, h("th", { class: "axis" }, "real ↓  said →"), k.labels.map((l) => h("th", {}, l)))),
        h("tbody", {}, k.confusion.map((row, i) => h("tr", {}, h("th", { class: "side" }, k.labels[i]),
          row.map((v, j) => h("td", {
            class: i === j ? "hit" : null,
            style: `background:color-mix(in srgb, var(${i === j ? "--good" : "--bad"}) ${Math.round((v / max) * (i === j ? 30 : 22))}%, transparent)`,
          }, String(v)))))))));
  }
  const all = k.points.flat();
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const S = 300, P = 28, pos = (v) => P + ((v - lo) / span) * (S - 2 * P);
  const svg = s("svg", { viewBox: `0 0 ${S} ${S}`, class: "scatter", role: "img", "aria-label": "Predicted values plotted against real values" });
  svg.append(s("rect", { x: P, y: P, width: S - 2 * P, height: S - 2 * P }),
    s("line", { x1: P, y1: S - P, x2: S - P, y2: P }),
    ...k.points.map(([a, p]) => s("circle", { cx: pos(a), cy: S - pos(p), r: 3 })),
    s("text", { x: P, y: S - 8 }, fmt(lo)), s("text", { x: S - P, y: S - 8, "text-anchor": "end" }, fmt(hi)),
    s("text", { x: S / 2, y: S - 8, "text-anchor": "middle" }, "real value →"),
    s("text", { x: P, y: 18 }, "↑ predicted"));
  return h("section", { class: "card" },
    h("h2", {}, icon("chart"), "Predicted vs. real"),
    h("p", { class: "sub" }, "Each dot is a test row. Dots close to the dashed line are good predictions."),
    svg);
}

function tryCard(m) {
  const answer = h("div", { class: "answer waiting", "aria-live": "polite" }, "Change any value, then press Predict.");
  const form = h("form", { class: "form" }, m.features.map((f) => {
    const input = f.type === "number"
      ? h("input", { type: "number", name: f.name, value: f.default ?? "", step: f.integer ? "1" : "any", placeholder: `${fmt(f.min)} – ${fmt(f.max)}` })
      : h("select", { name: f.name }, f.choices.map((c) => h("option", { value: c, selected: c === f.default }, c)));
    return h("label", { title: f.name }, h("span", {}, f.name), input);
  }));
  const btn = h("button", { class: "primary", type: "submit" }, icon("spark"), "Predict");
  const go = async (e) => {
    e?.preventDefault();
    const row = {};
    for (const f of m.features) {
      const v = form.elements[f.name].value;
      row[f.name] = f.type === "number" ? (v === "" ? null : Number(v)) : v;
    }
    try {
      const { predictions: [p] } = await postJSON(`/models/${m.id}/predict`, { rows: [row] });
      const chances = p.chances ? Object.entries(p.chances).sort((a, b) => b[1] - a[1]).slice(0, 5) : [];
      answer.className = "answer fresh";
      answer.replaceChildren(
        h("span", { class: "meta" }, `Predicted ${m.target}`),
        h("span", { class: "res" }, m.task === "regression" ? fmt(p.prediction, 4) : p.prediction),
        p.confidence != null && h("span", { class: "meta" }, `About ${pct(p.confidence)}% sure`),
        chances.length > 1 && h("div", { class: "chances" }, chances.map(([label, v]) => [
          h("span", { class: "nm", title: label }, label),
          h("div", { class: "bar" + (label === p.prediction ? "" : " base") }, h("div", { style: `width:${v * 100}%` })),
          h("span", { class: "v" }, `${pct(v)}%`)])));
    } catch (err) { toast(err.message); }
  };
  form.append(h("div", { class: "row", style: "grid-column:1/-1;margin-top:.2rem" }, btn));
  form.addEventListener("submit", go);
  return h("section", { class: "card" },
    h("h2", {}, icon("wand"), "Try it"),
    h("p", { class: "sub" }, `Fill in the details and see what the model predicts for “${m.target}”. Typical values are filled in for you.`),
    h("div", { class: "try" }, form, answer));
}

function useCard(m) {
  const example = Object.fromEntries(m.features.map((f) => [f.name, f.default]));
  const curl = `curl -X POST ${location.origin}/api/models/${m.id}/predict \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify({ rows: [example] }).replaceAll("'", "'\\''")}'`;
  const file = h("input", { type: "file", accept: ".csv,.tsv,.txt,.xlsx,.xls", hidden: true });
  const batchBtn = h("button", { class: "ghost", type: "button", onclick: () => file.click() }, "Upload spreadsheet");
  file.addEventListener("change", async () => {
    if (!file.files[0]) return;
    const body = new FormData();
    body.append("file", file.files[0]);
    batchBtn.classList.add("spin");
    try {
      const blob = await (await api(`/models/${m.id}/predict-file`, { method: "POST", body })).blob();
      const a = h("a", { href: URL.createObjectURL(blob), download: "predictions.csv" });
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err) { toast(err.message); }
    batchBtn.classList.remove("spin");
    file.value = "";
  });
  const copy = h("button", { class: "ghost", type: "button", onclick: async () => {
    try { await navigator.clipboard.writeText(curl); copy.textContent = "Copied!"; setTimeout(() => (copy.textContent = "Copy"), 1500); }
    catch { toast("Could not copy. Select the text and copy it by hand."); }
  } }, "Copy");
  return h("section", { class: "card" },
    h("h2", {}, icon("download"), "Use it"),
    h("p", { class: "sub" }, "Put the model to work on new data."),
    h("div", { class: "uses" },
      h("div", { class: "use" }, h("span", { class: "ic" }, icon("table")), h("b", {}, "Predict a whole spreadsheet"),
        h("p", {}, `Upload a file with the same columns. You get it back with a “predicted ${m.target}” column added.`), batchBtn, file),
      h("div", { class: "use" }, h("span", { class: "ic" }, icon("download")), h("b", {}, "Download the model"),
        h("p", {}, "A standard scikit-learn file your developers can run anywhere."),
        h("a", { class: "ghost", href: `/api/models/${m.id}/download` }, "Download .joblib")),
      h("div", { class: "use" }, h("span", { class: "ic" }, icon("code")), h("b", {}, "Call it from another app"),
        h("p", {}, "Send a row over HTTP, get a prediction back."),
        h("a", { class: "ghost", href: "/api/docs", target: "_blank", rel: "noopener" }, "API reference"))),
    h("details", { style: "margin-top:1rem" },
      h("summary", {}, "Show example request"),
      h("pre", {}, curl),
      h("div", { class: "row", style: "margin-top:.5rem" }, copy)));
}

async function showModel(id) {
  const view = $("#view-model");
  view.replaceChildren(h("div", { class: "skeleton" }), h("div", { class: "skeleton" }));
  let m;
  try { m = await getJSON(`/models/${id}`); }
  catch (err) { view.replaceChildren(h("div", { class: "card blank-state" }, h("span", { class: "big-ic" }, icon("info")), err.message, h("a", { class: "ghost", href: "#models" }, "Back to my models"))); return; }
  const remove = async () => {
    if (!confirm(`Delete “${m.name}”? This cannot be undone.`)) return;
    try { await api(`/models/${id}`, { method: "DELETE" }); refreshCount(); location.hash = "models"; } catch (err) { toast(err.message); }
  };
  const detail = detailCard(m);
  view.replaceChildren(...[
    h("div", { class: "crumbs" },
      h("h1", {}, m.name),
      h("a", { class: "ghost", href: "#build" }, icon("plus"), "Build another"),
      h("button", { class: "ghost danger", type: "button", onclick: remove }, "Delete")),
    h("div", { class: "tags" },
      h("span", { class: "tag" }, icon("target"), "Predicts ", h("b", {}, m.target)),
      h("span", { class: "tag" }, icon(MODELS[m.model]?.[2] || "spark"), h("b", {}, m.model_label), m.leaderboard.length ? " · won the try-out" : ""),
      h("span", { class: "tag" }, `Trained in ${m.seconds}s`),
      h("span", { class: "tag" }, `From “${m.dataset}”`)),
    scoreCard(m),
    detail ? h("div", { class: "grid2" }, importanceCard(m), detail) : importanceCard(m),
    tryCard(m),
    useCard(m),
    m.notes.length ? h("section", { class: "card" }, h("h2", {}, icon("info"), "Good to know"), h("ul", { class: "notes" }, m.notes.map((n) => h("li", {}, n)))) : null,
  ].filter(Boolean));
}

async function showModels() {
  const list = $("#models-list");
  list.className = "";
  list.replaceChildren(h("div", { class: "skeleton" }));
  let models;
  try { models = await getJSON("/models"); } catch (err) { toast(err.message); list.replaceChildren(); return; }
  setCount(models.length);
  if (!models.length) {
    list.replaceChildren(h("div", { class: "card blank-state" },
      h("span", { class: "big-ic" }, icon("spark")),
      h("b", { style: "color:var(--ink)" }, "No models yet"),
      h("p", {}, "Your trained models will be kept here."),
      h("a", { class: "primary", href: "#build", style: "text-decoration:none;padding:.55rem 1.1rem;border-radius:10px;background:var(--accent);color:var(--accent-ink);font-weight:700" }, "Build your first one")));
    return;
  }
  list.className = "mlist";
  list.replaceChildren(...models.map((m) => {
    const tone = verdictTone(m);
    const g = m.task === "classification"
      ? ring(m.metrics.accuracy, `${pct(m.metrics.accuracy)}%`, "", tone, "small")
      : ring(Math.max(0, m.metrics.r2), `${pct(Math.max(0, m.metrics.r2))}%`, "", tone, "small");
    return h("a", { class: "card mcard", href: `#model/${m.id}` }, g,
      h("div", {},
        h("b", { title: m.name }, m.name),
        h("div", { class: "meta" }, m.task === "classification" ? `${pct(m.metrics.accuracy)}% correct` : `±${fmt(m.metrics.mae)} typical miss · ${pct(Math.max(0, m.metrics.r2))}% explained`),
        h("div", { class: "meta" }, `${m.model_label} · ${new Date(m.created).toLocaleDateString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}`)));
  }));
}
// The list endpoint returns headline numbers only, so the tone is a simpler version of verdict().
const verdictTone = (m) => {
  const v = m.task === "classification" ? m.metrics.accuracy : m.metrics.r2;
  return v >= 0.75 ? "good" : v >= 0.5 ? "ok" : "bad";
};

function setCount(n) {
  const c = $("#model-count");
  c.textContent = n;
  c.hidden = !n;
}
const refreshCount = () => getJSON("/models").then((x) => setCount(x.length)).catch(() => {});

/* ------------------------------------------------------------------ routing */

function route() {
  closeMenu();
  const [name, arg] = location.hash.slice(1).split("/");
  const view = name === "models" ? "models" : name === "model" && arg ? "model" : "build";
  for (const v of ["build", "models", "model"]) $(`#view-${v}`).hidden = v !== view;
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("on", a.dataset.tab === (view === "model" ? "models" : view)));
  if (view === "models") showModels();
  if (view === "model") showModel(arg);
  window.scrollTo(0, 0);
}

dropzone($("#steps-zone"), (item) => place(item.kind, item.value));
dropzone($("#model-zone"), (item) => place(item.kind, item.value));
$("#train").addEventListener("click", trainModel);
$("#file-input").addEventListener("change", (e) => { uploadFile(e.target.files[0]); e.target.value = ""; });
window.addEventListener("hashchange", route);

(async function start() {
  render();
  route();
  refreshCount();
  try { samples = await getJSON("/samples"); } catch (err) { toast(err.message); }
  render();
})();
