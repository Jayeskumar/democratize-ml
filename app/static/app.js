"use strict";
/* Democratize ML front end. No framework, no build step, no outside requests. */

const STEPS = {
  dedupe: ["Remove duplicate rows", "Drops rows that appear more than once."],
  drop_missing: ["Remove rows with blanks", "Otherwise blanks are filled in with typical values."],
  scale: ["Put numbers on one scale", "Helps when columns have very different sizes."],
  balance: ["Balance rare answers", "Pays extra attention to answers that are rare, like fraud."],
};
const MODELS = {
  auto: ["Best pick (automatic)", "Tries several learners and keeps the best. Recommended."],
  linear: ["Simple formula", "Fast and easy to explain."],
  tree: ["Decision tree", "A flowchart of yes/no questions."],
  forest: ["Random forest", "Many trees voting together. A reliable all-rounder."],
  boosting: ["Gradient boosting", "Often the most accurate on spreadsheets."],
  knn: ["Nearest neighbours", "Looks up the most similar past examples."],
};

const state = { dataset: null, target: null, ignore: new Set(), steps: [], model: "auto", task: "auto", busy: false };
let samples = [];
let dragging = null; // {kind, value} of the block or chip being dragged

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

const fmt = (n, digits = 3) =>
  n == null ? "" : new Intl.NumberFormat(undefined, { maximumSignificantDigits: Math.abs(n) >= 1000 ? 4 : digits }).format(n);
const pct = (x) => Math.round(x * 100);

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
    document.querySelectorAll(".zone").forEach((z) => z.classList.toggle("can", accepts(z, kind)));
  });
  el.addEventListener("dragend", endDrag);
  return el;
}
function endDrag() {
  dragging = null;
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
function place(kind, value) {
  if (state.busy) return;
  if (kind === "data") return loadSample(value);
  if (kind === "file") return uploadFile(value);
  if (kind === "step" && !state.steps.includes(value)) state.steps.push(value);
  if (kind === "model") state.model = value;
  render();
}

// Dropping a file anywhere outside a drop area should not navigate the browser away.
window.addEventListener("dragover", (e) => hasFiles(e) && e.preventDefault());
window.addEventListener("drop", (e) => hasFiles(e) && e.preventDefault());

/* ------------------------------------------------------------------ builder */

function block(kind, value, title, text, { removable, onRemove } = {}) {
  const el = h(removable ? "div" : "button", { class: `block k-${kind}`, type: removable ? null : "button" },
    h("b", {}, title), h("span", {}, text));
  if (removable) {
    el.append(h("button", { class: "x", type: "button", "aria-label": `Remove ${title}`, onclick: onRemove }, "×"));
  } else {
    draggable(el, kind, value);
    el.addEventListener("click", () => place(kind, value));
  }
  return el;
}

function renderPalette() {
  $("#palette-data").replaceChildren(...samples.map((s) => block("data", s.id, s.name, s.description)));
  $("#palette-step").replaceChildren(...Object.entries(STEPS).map(([id, [t, d]]) => {
    const b = block("step", id, t, d);
    b.classList.toggle("used", state.steps.includes(id));
    return b;
  }));
  $("#palette-model").replaceChildren(...Object.entries(MODELS).map(([id, [t, d]]) => {
    const b = block("model", id, t, d);
    b.classList.toggle("used", state.model === id);
    return b;
  }));
}

function renderData() {
  const body = $("#data-body");
  const d = state.dataset;
  if (!d) {
    const zone = dropzone(
      h("div", { class: "zone drop-data", "data-accept": "data file" },
        h("strong", {}, "Drop a spreadsheet here"),
        h("p", {}, "CSV or Excel. One row per example, with column names in the first row."),
        h("button", { class: "primary", type: "button", onclick: () => $("#file-input").click() }, "Choose a file"),
        h("p", {}, "No data handy? Drag in one of the examples on the left.")),
      (item) => place(item.kind, item.value));
    body.replaceChildren(zone);
    return;
  }
  const numeric = d.columns.map((c) => c.type === "number");
  body.replaceChildren(
    h("div", { class: "data-head" },
      h("b", {}, d.name),
      h("span", {}, `${d.rows.toLocaleString()} rows · ${d.columns.length} columns`),
      h("button", { class: "ghost", type: "button", onclick: resetData }, "Use different data")),
    h("div", { class: "table-wrap", tabindex: "0", "aria-label": "First rows of your data" },
      h("table", {},
        h("thead", {}, h("tr", {}, d.columns.map((c, i) => h("th", { class: numeric[i] ? "n" : null }, c.name)))),
        h("tbody", {}, d.preview.map((row) => h("tr", {}, row.map((v, i) =>
          v == null ? h("td", { class: "blank" }, "blank")
            : h("td", { class: numeric[i] ? "n" : null }, typeof v === "number" ? fmt(v, 6) : String(v)))))))));
}

function chip(col, { placed, onRemove, dim } = {}) {
  const tag = h("i", {}, col.type === "number" ? "123" : "abc");
  if (placed) {
    return h("span", { class: "chip placed" }, col.name, tag,
      h("button", { class: "x", type: "button", "aria-label": `Put ${col.name} back`, onclick: onRemove }, "×"));
  }
  const el = h("button", { class: "chip" + (dim ? " dim" : ""), type: "button", "aria-haspopup": "menu",
    title: dim ? "Skipped automatically: it is different on almost every row (like an ID), or the same on every row." : null },
    col.name, tag);
  draggable(el, "column", col.name);
  el.addEventListener("click", (e) => { e.stopPropagation(); chipMenu(el, col.name); });
  return el;
}

function chipMenu(anchor, name) {
  closeMenu();
  const act = (fn) => () => { closeMenu(); fn(); render(); };
  const menu = h("div", { class: "menu", role: "menu" },
    h("button", { type: "button", role: "menuitem", onclick: act(() => setTarget(name)) }, "Predict this"),
    h("button", { type: "button", role: "menuitem", onclick: act(() => state.ignore.add(name)) }, "Ignore this column"));
  const r = anchor.getBoundingClientRect();
  menu.style.left = `${Math.min(r.left + scrollX, scrollX + innerWidth - 200)}px`;
  menu.style.top = `${r.bottom + scrollY + 4}px`;
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

function renderTarget() {
  const body = $("#target-body");
  const d = state.dataset;
  $("#stage-target").classList.toggle("locked", !d);
  if (!d) {
    body.replaceChildren(h("p", { class: "hint" }, "Your columns will show up here once you add data."));
    return;
  }
  const byName = Object.fromEntries(d.columns.map((c) => [c.name, c]));
  const pool = d.columns.filter((c) => c.name !== state.target && !state.ignore.has(c.name));

  const targetZone = dropzone(h("div", { class: "zone target", "data-accept": "column" },
    state.target
      ? chip(byName[state.target], { placed: true, onRemove: () => { state.target = null; render(); } })
      : h("span", { class: "empty" }, "Drop one column here")),
    (item) => { setTarget(item.value); render(); });

  const ignoreZone = dropzone(h("div", { class: "zone", "data-accept": "column" },
    state.ignore.size
      ? [...state.ignore].map((n) => chip(byName[n], { placed: true, onRemove: () => { state.ignore.delete(n); render(); } }))
      : h("span", { class: "empty" }, "Drop columns the computer should not look at")),
    (item) => { if (state.target === item.value) state.target = null; state.ignore.add(item.value); render(); });

  let kind = null;
  if (state.target) {
    const col = byName[state.target];
    const task = state.task === "auto" ? col.task : state.task;
    kind = h("p", { class: "kind" },
      task === "classification"
        ? `The answer will be a category — one of the ${col.unique} different values found in “${col.name}”. `
        : `The answer will be a number, like the values in “${col.name}”. `,
      col.type === "number" && h("button", { class: "link", type: "button", onclick: () => {
        state.task = task === "classification" ? "regression" : "classification";
        render();
      } }, task === "classification" ? "Treat it as a number instead" : "Treat it as a category instead"));
  }

  body.replaceChildren(h("div", { class: "cols" },
    h("div", {},
      h("span", { class: "lab" }, "Your columns — drag one into “Predict this”. The rest are used as clues."),
      h("div", { class: "zone", style: "border-style:solid" },
        pool.length ? pool.map((c) => chip(c, { dim: c.id_like || c.empty })) : h("span", { class: "empty" }, "No columns left to use as clues."))),
    h("div", { class: "pair" },
      h("div", {}, h("span", { class: "lab" }, "Predict this"), targetZone),
      h("div", {}, h("span", { class: "lab" }, "Ignore these (optional)"), ignoreZone)),
    kind));
}

function renderSteps() {
  const zone = $("#steps-zone");
  zone.replaceChildren(...(state.steps.length
    ? state.steps.map((id) => block("step", id, ...STEPS[id], {
        removable: true, onRemove: () => { state.steps = state.steps.filter((s) => s !== id); render(); } }))
    : [h("span", { class: "empty" }, "Nothing needed — blanks are handled for you. Drop clean-up blocks here if you want them.")]));
}

function renderModel() {
  $("#model-zone").replaceChildren(block("model", state.model, ...MODELS[state.model], {
    removable: state.model !== "auto", onRemove: () => { state.model = "auto"; render(); } }));
}

function render() {
  renderPalette(); renderData(); renderTarget(); renderSteps(); renderModel();
  const btn = $("#train");
  btn.disabled = state.busy || !state.dataset || !state.target;
  btn.classList.toggle("spin", state.busy);
  btn.textContent = state.busy ? "Learning from your data…" : "Train my model";
  $("#train-msg").textContent = state.busy ? "This usually takes a few seconds."
    : !state.dataset ? "Start by adding data."
    : !state.target ? "Next: choose the column to predict." : "";
}

function resetData() {
  Object.assign(state, { dataset: null, target: null, ignore: new Set(), task: "auto" });
  render();
}

async function withData(promise) {
  if (state.busy) return;
  try {
    const d = await promise;
    resetData();
    state.dataset = d;
    render();
    $("#stage-target").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) { toast(err.message); }
}
const loadSample = (id) => withData(getJSON(`/datasets/sample/${id}`, { method: "POST" }));
function uploadFile(file) {
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  return withData(getJSON("/datasets", { method: "POST", body }));
}

async function trainModel() {
  state.busy = true;
  render();
  try {
    const m = await postJSON("/models", {
      dataset_id: state.dataset.id, target: state.target, ignore: [...state.ignore],
      steps: state.steps, model: state.model, task: state.task,
    });
    location.hash = `model/${m.id}`;
  } catch (err) { toast(err.message); }
  state.busy = false;
  render();
}

/* ------------------------------------------------------------ model results */

function verdict(m) {
  const k = m.metrics;
  if (m.task === "classification") {
    const lift = k.accuracy - k.baseline;
    if (lift <= 0.02) return ["bad", "No better than guessing", "Try adding more rows, or columns that relate more closely to the answer."];
    if (lift <= 0.1) return ["ok", "A little better than guessing", "It has found a weak pattern. More or better data would help."];
    return k.accuracy >= 0.9 ? ["good", "Very reliable", "It gets the answer right most of the time."]
      : ["good", "Useful", "Clearly better than guessing, though not perfect."];
  }
  const gain = k.baseline_mae ? 1 - k.mae / k.baseline_mae : 0;
  if (gain <= 0.05) return ["bad", "No better than guessing", "Try adding more rows, or columns that relate more closely to the answer."];
  if (gain <= 0.3) return ["ok", "A little better than guessing", "It has found a weak pattern. More or better data would help."];
  return gain >= 0.6 ? ["good", "Very reliable", "Its estimates land close to the real values."]
    : ["good", "Useful", "Clearly better than guessing, though not perfect."];
}

function scoreCard(m) {
  const k = m.metrics;
  const [tone, title, text] = verdict(m);
  let big, rows;
  if (m.task === "classification") {
    big = h("div", { class: "big-n" }, `${pct(k.accuracy)}%`, h("span", {}, "correct answers"));
    rows = [["This model", k.accuracy, `${pct(k.accuracy)}%`, ""], ["Always guessing the most common answer", k.baseline, `${pct(k.baseline)}%`, " base"]];
  } else {
    const top = Math.max(k.mae, k.baseline_mae) || 1;
    big = h("div", { class: "big-n" }, `±${fmt(k.mae)}`, h("span", {}, "typical miss"));
    rows = [["This model is typically off by", k.mae / top, fmt(k.mae), ""], ["Always guessing the middle value is off by", k.baseline_mae / top, fmt(k.baseline_mae), " base"]];
  }
  return h("section", { class: "card" },
    h("div", { class: "score" }, big,
      h("div", {},
        h("p", { class: `verdict ${tone}` }, title),
        h("p", {}, text),
        h("div", { class: "cmp" }, rows.map(([label, frac, value, cls]) => [
          h("span", {}, label),
          h("div", { class: "bar" + cls }, h("div", { style: `width:${Math.max(0, Math.min(1, frac)) * 100}%` })),
          h("b", {}, value)])))),
    h("p", { class: "meta", style: "margin-top:.8rem" },
      `Checked on ${m.rows_tested.toLocaleString()} rows the model never saw while learning from the other ${m.rows_trained.toLocaleString()}.`,
      m.task === "regression" ? ` It explains about ${pct(Math.max(0, k.r2))}% of the differences in “${m.target}”.` : ""));
}

function importanceCard(m) {
  const top = m.importance.slice(0, 10);
  const max = Math.max(...top.map((r) => r.share), 0.0001);
  return h("section", { class: "card" },
    h("h2", {}, "What matters most"),
    h("p", { class: "sub" }, "How much each column influences the prediction."),
    h("div", { class: "imp" }, top.map((r) => [
      h("span", { class: "nm", title: r.name }, r.name),
      h("div", { class: "bar" }, h("div", { style: `width:${(r.share / max) * 100}%` })),
      h("span", { class: "v" }, `${pct(r.share)}%`)])));
}

function detailCard(m) {
  const k = m.metrics;
  if (m.task === "classification") {
    if (k.labels.length > 8) return null;
    return h("section", { class: "card" },
      h("h2", {}, "Where it gets confused"),
      h("p", { class: "sub" }, "Rows are the real answers, columns are what the model said. Highlighted cells are correct."),
      h("div", { style: "overflow:auto" }, h("table", { class: "cm" },
        h("thead", {}, h("tr", {}, h("th", { class: "axis" }, "real ↓ / predicted →"), k.labels.map((l) => h("th", {}, l)))),
        h("tbody", {}, k.confusion.map((row, i) => h("tr", {}, h("th", { class: "side" }, k.labels[i]),
          row.map((v, j) => h("td", { class: i === j ? "hit" : null }, String(v)))))))));
  }
  const all = k.points.flat();
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const S = 300, P = 28, pos = (v) => P + ((v - lo) / span) * (S - 2 * P);
  const ns = "http://www.w3.org/2000/svg";
  const s = (tag, attrs, text) => {
    const el = document.createElementNS(ns, tag);
    for (const [a, v] of Object.entries(attrs)) el.setAttribute(a, v);
    if (text) el.textContent = text;
    return el;
  };
  const svg = s("svg", { viewBox: `0 0 ${S} ${S}`, class: "scatter", role: "img", "aria-label": "Predicted values plotted against real values" });
  svg.append(s("rect", { x: P, y: P, width: S - 2 * P, height: S - 2 * P }),
    s("line", { x1: P, y1: S - P, x2: S - P, y2: P }),
    ...k.points.map(([a, p]) => s("circle", { cx: pos(a), cy: S - pos(p), r: 3 })),
    s("text", { x: P, y: S - 8 }, fmt(lo)), s("text", { x: S - P, y: S - 8, "text-anchor": "end" }, fmt(hi)),
    s("text", { x: S / 2, y: S - 8, "text-anchor": "middle" }, "real value →"),
    s("text", { x: P, y: 18 }, "↑ predicted"));
  return h("section", { class: "card" },
    h("h2", {}, "Predicted vs. real"),
    h("p", { class: "sub" }, "Each dot is a row it was checked on. Dots near the dashed line are good predictions."),
    svg);
}

function tryCard(m) {
  const answer = h("div", { class: "answer", hidden: true, "aria-live": "polite" });
  const form = h("form", { class: "form" }, m.features.map((f) => {
    const input = f.type === "number"
      ? h("input", { type: "number", name: f.name, value: f.default ?? "", step: f.integer ? "1" : "any" })
      : h("select", { name: f.name }, f.choices.map((c) => h("option", { value: c, selected: c === f.default }, c)));
    return h("label", {}, f.name, input);
  }));
  const go = async (e) => {
    e.preventDefault();
    const row = {};
    for (const f of m.features) {
      const v = form.elements[f.name].value;
      row[f.name] = f.type === "number" ? (v === "" ? null : Number(v)) : v;
    }
    try {
      const { predictions: [p] } = await postJSON(`/models/${m.id}/predict`, { rows: [row] });
      answer.replaceChildren(
        h("span", { class: "meta" }, `Predicted ${m.target}`),
        h("b", {}, m.task === "regression" ? fmt(p.prediction, 4) : p.prediction),
        p.confidence != null && h("span", {}, `The model is about ${pct(p.confidence)}% sure.`));
      answer.hidden = false;
    } catch (err) { toast(err.message); }
  };
  form.addEventListener("submit", go);
  form.append(h("button", { type: "submit", hidden: true }));
  return h("section", { class: "card" },
    h("h2", {}, "Try it"),
    h("p", { class: "sub" }, `Fill in the details and see what the model predicts for “${m.target}”.`),
    form,
    h("div", { class: "row" }, h("button", { class: "primary", type: "button", onclick: go }, "Predict")),
    answer);
}

function useCard(m) {
  const example = Object.fromEntries(m.features.map((f) => [f.name, f.default]));
  const curl = `curl -X POST ${location.origin}/api/models/${m.id}/predict \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify({ rows: [example] }).replaceAll("'", "'\\''")}'`;
  const file = h("input", { type: "file", accept: ".csv,.tsv,.txt,.xlsx,.xls", hidden: true });
  const batchBtn = h("button", { class: "ghost", type: "button", onclick: () => file.click() }, "Predict a whole spreadsheet");
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
  return h("section", { class: "card" },
    h("h2", {}, "Use it"),
    h("p", { class: "sub" }, "Upload a spreadsheet with the same columns to get it back with predictions filled in, or hand the model to your developers."),
    h("div", { class: "row", style: "margin-top:0" }, batchBtn, file,
      h("a", { class: "ghost", href: `/api/models/${m.id}/download` }, "Download model file")),
    h("details", { style: "margin-top:.8rem" },
      h("summary", {}, "For developers: call it from another program"),
      h("pre", {}, curl),
      h("p", { class: "meta", style: "margin-top:.5rem" }, "The downloaded file is a scikit-learn pipeline; load it with joblib.load() and call .predict() on a DataFrame with the same columns. Full API reference at ", h("a", { href: "/api/docs" }, "/api/docs"), ".")));
}

async function showModel(id) {
  const view = $("#view-model");
  view.replaceChildren(h("p", { class: "meta spin" }, "Loading…"));
  let m;
  try { m = await getJSON(`/models/${id}`); }
  catch (err) { view.replaceChildren(h("div", { class: "card blank-state" }, err.message)); return; }
  const remove = async () => {
    if (!confirm(`Delete “${m.name}”? This cannot be undone.`)) return;
    try { await api(`/models/${id}`, { method: "DELETE" }); location.hash = "models"; } catch (err) { toast(err.message); }
  };
  view.replaceChildren(...[
    h("div", { class: "crumbs" },
      h("h1", {}, m.name),
      h("a", { class: "ghost", href: "#build" }, "Build another"),
      h("button", { class: "ghost danger", type: "button", onclick: remove }, "Delete")),
    h("p", { class: "meta" }, `Predicts “${m.target}” · ${m.model_label}${m.leaderboard.length ? " (won the automatic try-out)" : ""} · trained in ${m.seconds}s`),
    scoreCard(m),
    (() => { const d = detailCard(m); return d ? h("div", { class: "grid2" }, importanceCard(m), d) : importanceCard(m); })(),
    tryCard(m),
    useCard(m),
    m.notes.length ? h("section", { class: "card" }, h("h2", {}, "Good to know"), h("ul", { class: "notes" }, m.notes.map((n) => h("li", {}, n)))) : null,
  ].filter(Boolean));
}

async function showModels() {
  const list = $("#models-list");
  let models;
  try { models = await getJSON("/models"); } catch (err) { toast(err.message); return; }
  if (!models.length) {
    list.replaceChildren(h("div", { class: "card blank-state" },
      h("p", {}, "You have not trained a model yet."),
      h("p", { style: "margin-top:.8rem" }, h("a", { class: "ghost", href: "#build" }, "Build your first one"))));
    return;
  }
  list.className = "mlist";
  list.replaceChildren(...models.map((m) => h("div", { class: "card mrow" },
    h("div", { class: "t" },
      h("a", { href: `#model/${m.id}` }, m.name),
      h("div", { class: "meta" }, `${m.model_label} · ${new Date(m.created).toLocaleString()}`)),
    h("span", { class: "s" }, m.task === "classification" ? `${pct(m.metrics.accuracy)}% correct` : `±${fmt(m.metrics.mae)} typical miss`),
    h("a", { class: "ghost", href: `#model/${m.id}` }, "Open"))));
}

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
  try { samples = await getJSON("/samples"); } catch (err) { toast(err.message); }
  render();
  route();
})();
