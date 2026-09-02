/* Quire worksheet UI. Plain JavaScript, no build step.
   The engine lives on the server; this file owns the document, rendering and plots. */
(() => {
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const uid = () => Math.random().toString(36).slice(2, 10);
const DRAFT_KEY = "quire.draft";
const PALETTE = ["#2f6fed", "#e0523d", "#1f9d6f", "#b7791f", "#8e44ad", "#0e9aa7", "#d63384", "#556b2f"];

const state = {
  title: "Untitled",
  fileName: null,
  cells: [],
  results: new Map(),
  els: new Map(),
  catalog: null,
  dirty: false,
  activeInput: null,
  evalTimer: null,
  evalSeq: 0,
};

// ---------- KaTeX helpers ----------
function katexHtml(latex, display = false) {
  if (window.katex) {
    try { return katex.renderToString(latex, { throwOnError: false, displayMode: display, strict: "ignore" }); }
    catch (e) { /* fall through */ }
  }
  return `<code>${esc(latex)}</code>`;
}

// ---------- Markdown (small subset) ----------
function inlineMd(s) {
  return s.split(/(\$[^$\n]+\$)/g).map(part => {
    if (part.length > 2 && part.startsWith("$") && part.endsWith("$")) return katexHtml(part.slice(1, -1));
    let h = esc(part);
    h = h.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*]+)\*/g, "$1<i>$2</i>").replace(/`(.+?)`/g, "<code>$1</code>");
    return h;
  }).join("");
}
function markdown(text) {
  const out = []; let para = []; let list = [];
  const flush = () => {
    if (para.length) { out.push(`<p>${inlineMd(para.join(" "))}</p>`); para = []; }
    if (list.length) { out.push(`<ul>${list.map(l => `<li>${inlineMd(l)}</li>`).join("")}</ul>`); list = []; }
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) { flush(); out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`); continue; }
    const li = /^[-*]\s+(.*)$/.exec(line);
    if (li) { if (para.length) flush(); list.push(li[1]); continue; }
    if (!line.trim()) { flush(); continue; }
    if (list.length) flush();
    para.push(line);
  }
  flush();
  return out.join("");
}

// ---------- Document model ----------
function newCell(type) {
  const base = { id: uid(), type };
  if (type === "plot") return { ...base, exprs: "", var: "", xmin: "", xmax: "", samples: "" };
  return { ...base, source: "" };
}
function serialize() {
  return { quire: 1, title: state.title, cells: state.cells.map(c => ({ ...c })) };
}
function loadDoc(doc, fileName = null) {
  state.title = doc.title || "Untitled";
  state.fileName = fileName;
  state.cells = (doc.cells || []).map(c => ({ ...newCell(c.type || "math"), ...c, id: c.id || uid() }));
  if (!state.cells.length) state.cells.push(newCell("math"));
  state.results = new Map();
  state.els = new Map();
  $("#cells").innerHTML = "";  // drop the previous document's elements
  $("#title").value = state.title;
  mountCells();
  setDirty(false);
  evaluateNow();
}
function setDirty(v) { state.dirty = v; $("#dirty").classList.toggle("on", v); saveDraft(); }
function saveDraft() {
  try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ doc: serialize(), fileName: state.fileName })); } catch (e) { /* ignore */ }
}
function touch() { setDirty(true); scheduleEval(); }

// ---------- Cell DOM ----------
function autogrow(ta) { ta.style.height = "auto"; ta.style.height = (ta.scrollHeight + 2) + "px"; }

function cellIndex(id) { return state.cells.findIndex(c => c.id === id); }

function createCellEl(cell) {
  const el = document.createElement("div");
  el.className = `cell ${cell.type}`;
  el.dataset.id = cell.id;
  const badge = { math: "∑", text: "¶", plot: "⌁" }[cell.type];
  el.innerHTML = `
    <div class="gutter" title="${cell.type} cell">${badge}</div>
    <div class="body"></div>
    <div class="tools">
      <button data-act="up" title="Move up">↑</button>
      <button data-act="down" title="Move down">↓</button>
      <button data-act="del" title="Delete cell">✕</button>
    </div>
    <div class="adder">
      <button data-add="math">+ math</button>
      <button data-add="text">+ text</button>
      <button data-add="plot">+ plot</button>
    </div>`;
  const body = $(".body", el);
  if (cell.type === "math") buildMath(cell, body);
  else if (cell.type === "text") buildText(cell, body);
  else buildPlot(cell, body);

  el.addEventListener("focusin", () => { el.classList.add("focus"); });
  el.addEventListener("focusout", () => { el.classList.remove("focus"); });
  el.addEventListener("click", ev => {
    const b = ev.target.closest("button"); if (!b) return;
    if (b.dataset.act) cellAction(cell.id, b.dataset.act);
    if (b.dataset.add) insertCell(b.dataset.add, cellIndex(cell.id) + 1, true);
  });
  return el;
}

function keyNav(cell, ta, ev) {
  const i = cellIndex(cell.id);
  if (ev.key === "Enter" && !ev.shiftKey && cell.type === "math") {
    ev.preventDefault();
    const next = state.cells[i + 1];
    if (next && next.type === "math" && !next.source.trim()) focusCell(next.id);
    else insertCell("math", i + 1, true);
    evaluateNow();
    return;
  }
  if (ev.key === "Backspace" && !ta.value && state.cells.length > 1) {
    ev.preventDefault(); cellAction(cell.id, "del"); return;
  }
  if (ev.key === "ArrowUp" && ta.selectionStart === 0 && !ta.value.slice(0, ta.selectionStart).includes("\n")) {
    if (i > 0) { ev.preventDefault(); focusCell(state.cells[i - 1].id, "end"); }
  }
  if (ev.key === "ArrowDown" && ta.selectionEnd === ta.value.length) {
    if (i < state.cells.length - 1) { ev.preventDefault(); focusCell(state.cells[i + 1].id, "start"); }
  }
  if (ev.key === "Escape") ta.blur();
}

function buildMath(cell, body) {
  body.innerHTML = `<textarea class="src" rows="1" spellcheck="false" placeholder="e.g.  F = m a        or        solve(x^2 == 4, x)"></textarea>
    <div class="out"></div><div class="warn"></div><div class="err"></div><div class="meta"></div>`;
  const ta = $("textarea", body);
  ta.value = cell.source || "";
  ta.addEventListener("input", () => { cell.source = ta.value; autogrow(ta); touch(); });
  ta.addEventListener("keydown", ev => keyNav(cell, ta, ev));
  ta.addEventListener("focus", () => { state.activeInput = ta; });
  requestAnimationFrame(() => autogrow(ta));
}

function buildText(cell, body) {
  body.innerHTML = `<textarea class="src" rows="1" spellcheck="true" placeholder="Write text. *italic*, **bold**, # heading, $x^2$ for math."></textarea><div class="md"></div>`;
  const ta = $("textarea", body), md = $(".md", body);
  ta.value = cell.source || "";
  const show = () => {
    if (document.activeElement === ta) return;
    md.innerHTML = markdown(cell.source || "");
    md.style.display = ""; ta.style.display = "none";
  };
  const edit = () => { md.style.display = "none"; ta.style.display = ""; autogrow(ta); ta.focus(); };
  ta.addEventListener("input", () => { cell.source = ta.value; autogrow(ta); setDirty(true); });
  ta.addEventListener("blur", show);
  ta.addEventListener("keydown", ev => {
    if (ev.key === "Escape") ta.blur();
    if (ev.key === "Backspace" && !ta.value && state.cells.length > 1) { ev.preventDefault(); cellAction(cell.id, "del"); }
    if (ev.key === "ArrowDown" && ev.metaKey) { const i = cellIndex(cell.id); if (i < state.cells.length - 1) focusCell(state.cells[i + 1].id); }
  });
  ta.addEventListener("focus", () => { state.activeInput = ta; });
  md.addEventListener("click", edit);
  body._edit = edit;
  show();
}

function buildPlot(cell, body) {
  body.innerHTML = `
    <div class="plot-form">
      <span>y =</span><input class="exprs" placeholder="sin(x), cos(x)   — or a function name, like y(t)" spellcheck="false">
      <span>from</span><input class="xmin small" placeholder="-10" spellcheck="false">
      <span>to</span><input class="xmax small" placeholder="10" spellcheck="false">
      <span>variable</span><input class="var tiny" placeholder="auto" spellcheck="false">
      <span>points</span><input class="samples tiny" placeholder="400" spellcheck="false">
    </div>
    <svg class="plot" viewBox="0 0 700 340" style="display:none"></svg>
    <div class="legend"></div><div class="err"></div>`;
  for (const key of ["exprs", "xmin", "xmax", "var", "samples"]) {
    const inp = $(`input.${key}`, body);
    inp.value = cell[key] || "";
    inp.addEventListener("input", () => { cell[key] = inp.value; touch(); });
    inp.addEventListener("focus", () => { state.activeInput = inp; });
    inp.addEventListener("keydown", ev => { if (ev.key === "Enter") evaluateNow(); if (ev.key === "Escape") inp.blur(); });
  }
}

function mountCells() {
  const root = $("#cells");
  const wanted = new Set(state.cells.map(c => c.id));
  for (const [id, el] of state.els) if (!wanted.has(id)) { el.remove(); state.els.delete(id); }
  let prev = null;
  for (const c of state.cells) {
    let el = state.els.get(c.id);
    if (!el) { el = createCellEl(c); state.els.set(c.id, el); }
    if (prev ? prev.nextSibling !== el : root.firstChild !== el) root.insertBefore(el, prev ? prev.nextSibling : root.firstChild);
    prev = el;
  }
}

function focusCell(id, where = "end") {
  const el = state.els.get(id); if (!el) return;
  const body = $(".body", el);
  if (body._edit) { body._edit(); return; }
  const inp = $("textarea, input", el); if (!inp) return;
  inp.focus();
  if (inp.setSelectionRange) { const p = where === "start" ? 0 : inp.value.length; inp.setSelectionRange(p, p); }
}

function insertCell(type, at, focus) {
  const c = newCell(type);
  state.cells.splice(at, 0, c);
  mountCells();
  if (focus) focusCell(c.id);
  touch();
  return c;
}

function cellAction(id, act) {
  const i = cellIndex(id); if (i < 0) return;
  if (act === "del") {
    if (state.cells.length === 1) { state.cells[0] = newCell("math"); state.els.clear(); $("#cells").innerHTML = ""; }
    else state.cells.splice(i, 1);
    mountCells();
    const j = Math.max(0, Math.min(i - 1, state.cells.length - 1));
    if (state.cells[j]) focusCell(state.cells[j].id);
  } else if (act === "up" && i > 0) {
    [state.cells[i - 1], state.cells[i]] = [state.cells[i], state.cells[i - 1]]; mountCells();
  } else if (act === "down" && i < state.cells.length - 1) {
    [state.cells[i + 1], state.cells[i]] = [state.cells[i], state.cells[i + 1]]; mountCells();
  }
  touch();
}

// ---------- Evaluation ----------
function scheduleEval() {
  clearTimeout(state.evalTimer);
  state.evalTimer = setTimeout(evaluateNow, 220);
}
async function evaluateNow() {
  clearTimeout(state.evalTimer);
  const seq = ++state.evalSeq;
  const cells = state.cells.filter(c => c.type !== "text");
  let data;
  try {
    const r = await fetch("/api/eval", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cells }) });
    data = await r.json();
  } catch (e) {
    status(`<span class="bad">Cannot reach the Quire server. Is it still running?</span>`); return;
  }
  if (seq !== state.evalSeq) return; // a newer evaluation is in flight
  let errors = 0;
  for (const res of data.results || []) {
    state.results.set(res.id, res);
    if (res.ok === false) errors++;
    renderResult(res);
  }
  const mods = state.catalog ? ` · ${state.catalog.modules.length} modules` : "";
  status(errors ? `<span class="bad">${errors} cell${errors > 1 ? "s" : ""} with errors</span> · ${data.ms} ms${mods}`
                : `✓ evaluated in ${data.ms} ms${mods}`);
}
function status(html) { $("#status").innerHTML = html; }

function renderResult(res) {
  const el = state.els.get(res.id); if (!el) return;
  const cell = state.cells[cellIndex(res.id)]; if (!cell) return;
  el.classList.toggle("err", res.ok === false);
  const err = $(".err", el); if (err) err.textContent = res.error || "";
  if (cell.type === "math") {
    const out = $(".out", el), warn = $(".warn", el), meta = $(".meta", el);
    out.innerHTML = (res.outputs || []).map(o => {
      let h = o.head ? katexHtml(`${o.head} = ${o.latex}`) : katexHtml(o.latex);
      if (o.approx) h += `<span class="approx">${katexHtml(`\\approx ${o.approx}`)}</span>`;
      const notes = (o.notes || []).map(n => `<div class="note">${esc(n)}</div>`).join("");
      return `<div class="out-line">${h}</div>${notes}`;
    }).join("");
    warn.textContent = res.warning || "";
    const parts = [];
    if (res.defines && res.defines.length) parts.push(`defines <b>${esc(res.defines.join(", "))}</b>`);
    if (res.uses && res.uses.length) parts.push(`uses <b>${esc(res.uses.join(", "))}</b>`);
    meta.innerHTML = parts.join(" · ");
  } else if (cell.type === "plot") {
    drawPlot(el, res);
  }
}

// ---------- Plotting (SVG) ----------
function niceTicks(lo, hi, count = 6) {
  const span = (hi - lo) || 1, rough = span / count;
  const p = Math.pow(10, Math.floor(Math.log10(rough))), r = rough / p;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * p;
  const ticks = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(12));
  return ticks;
}
function fmtTick(v) {
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
  return String(+v.toPrecision(5));
}
function drawPlot(el, res) {
  const svg = $("svg.plot", el), legend = $(".legend", el);
  if (!res.ok || !res.series || !res.series.length) { svg.style.display = "none"; legend.innerHTML = ""; return; }
  const W = 700, H = 340, ml = 64, mr = 16, mt = 14, mb = 44;
  const xs = res.x;
  const xlo = xs[0], xhi = xs[xs.length - 1];
  let ylo = Infinity, yhi = -Infinity;
  for (const s of res.series) for (const y of s.y) if (y !== null) { if (y < ylo) ylo = y; if (y > yhi) yhi = y; }
  if (!isFinite(ylo)) { svg.style.display = "none"; legend.innerHTML = `<span class="err">Nothing to draw: all values are undefined on this range.</span>`; return; }
  if (ylo === yhi) { ylo -= 1; yhi += 1; }
  const pad = (yhi - ylo) * 0.06; ylo -= pad; yhi += pad;
  const sx = x => ml + (x - xlo) / (xhi - xlo) * (W - ml - mr);
  const sy = y => H - mb - (y - ylo) / (yhi - ylo) * (H - mt - mb);
  let g = "";
  for (const t of niceTicks(xlo, xhi, 8)) {
    g += `<line x1="${sx(t)}" y1="${mt}" x2="${sx(t)}" y2="${H - mb}" stroke="#eee"/>`;
    g += `<text x="${sx(t)}" y="${H - mb + 16}" font-size="11" text-anchor="middle" fill="#666">${fmtTick(t)}</text>`;
  }
  for (const t of niceTicks(ylo, yhi, 6)) {
    g += `<line x1="${ml}" y1="${sy(t)}" x2="${W - mr}" y2="${sy(t)}" stroke="#eee"/>`;
    g += `<text x="${ml - 6}" y="${sy(t) + 4}" font-size="11" text-anchor="end" fill="#666">${fmtTick(t)}</text>`;
  }
  if (ylo < 0 && yhi > 0) g += `<line x1="${ml}" y1="${sy(0)}" x2="${W - mr}" y2="${sy(0)}" stroke="#999"/>`;
  if (xlo < 0 && xhi > 0) g += `<line x1="${sx(0)}" y1="${mt}" x2="${sx(0)}" y2="${H - mb}" stroke="#999"/>`;
  g += `<rect x="${ml}" y="${mt}" width="${W - ml - mr}" height="${H - mt - mb}" fill="none" stroke="#ccc"/>`;
  res.series.forEach((s, k) => {
    let d = "", pen = false;
    s.y.forEach((y, i) => {
      if (y === null) { pen = false; return; }
      const yy = Math.max(sy(yhi) - 2000, Math.min(sy(ylo) + 2000, sy(y)));
      d += `${pen ? "L" : "M"}${sx(xs[i]).toFixed(1)} ${yy.toFixed(1)} `; pen = true;
    });
    g += `<path d="${d}" fill="none" stroke="${PALETTE[k % PALETTE.length]}" stroke-width="2" stroke-linejoin="round" clip-path="url(#clip-${res.id})"/>`;
  });
  g += `<text x="${(ml + W - mr) / 2}" y="${H - 8}" font-size="12" text-anchor="middle" fill="#444">${esc(res.xlabel || "")}</text>`;
  if (res.ylabel) g += `<text x="14" y="${(mt + H - mb) / 2}" font-size="12" text-anchor="middle" fill="#444" transform="rotate(-90 14 ${(mt + H - mb) / 2})">${esc(res.ylabel)}</text>`;
  svg.innerHTML = `<defs><clipPath id="clip-${res.id}"><rect x="${ml}" y="${mt}" width="${W - ml - mr}" height="${H - mt - mb}"/></clipPath></defs>${g}`;
  svg.style.display = "";
  legend.innerHTML = res.series.map((s, k) => `<span><span class="swatch" style="background:${PALETTE[k % PALETTE.length]}"></span>${katexHtml(s.label)}</span>`).join("");
}

// ---------- Reference panel ----------
async function loadCatalog(reload = false) {
  const r = await fetch(reload ? "/api/reload" : "/api/catalog", { method: reload ? "POST" : "GET" });
  state.catalog = await r.json();
  renderCatalog();
  if (reload) evaluateNow();
}
function renderCatalog() {
  const q = $("#ref-search").value.trim().toLowerCase();
  const cat = state.catalog; if (!cat) return;
  $("#ref-modules").innerHTML = "Modules: " + cat.modules.map(m =>
    m.error ? `<span class="bad" title="${esc(m.error)}">${esc(m.name)} (failed)</span>` : `<span title="${esc(m.description)}">${esc(m.name)}</span>`
  ).join(", ") + (cat.conflicts && cat.conflicts.length ? `<div class="bad">${cat.conflicts.map(esc).join("<br>")}</div>` : "");
  const groups = new Map();
  for (const e of cat.entries) {
    if (q && !(e.name.toLowerCase().includes(q) || e.doc.toLowerCase().includes(q) || e.signature.toLowerCase().includes(q))) continue;
    if (!groups.has(e.category)) groups.set(e.category, []);
    groups.get(e.category).push(e);
  }
  let h = "";
  for (const [c, items] of groups) {
    h += `<div class="ref-cat">${esc(c || "Other")}</div>`;
    for (const e of items) {
      const insert = e.example || (e.kind === "function" ? e.signature : e.name);
      h += `<div class="ref-item" data-insert="${esc(insert)}" title="Click to insert">
        <span class="sig">${esc(e.kind === "function" ? e.signature : e.name)}${e.module !== "core" ? ` <span class="mod">${esc(e.module)}</span>` : ""}</span>
        <span class="doc">${esc(e.doc)}</span></div>`;
    }
  }
  $("#ref-list").innerHTML = h || `<div class="ref-cat">No matches</div>`;
}
function insertAtCursor(text) {
  let inp = state.activeInput;
  if (!inp || !document.body.contains(inp)) {
    const c = insertCell("math", state.cells.length, true);
    inp = $("textarea", state.els.get(c.id));
  }
  const a = inp.selectionStart ?? inp.value.length, b = inp.selectionEnd ?? a;
  inp.value = inp.value.slice(0, a) + text + inp.value.slice(b);
  inp.setSelectionRange(a + text.length, a + text.length);
  inp.dispatchEvent(new Event("input"));
  inp.focus();
}

// ---------- Files ----------
async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function modal(title, bodyHtml) {
  $("#modal-title").textContent = title; $("#modal-body").innerHTML = bodyHtml; $("#modal").classList.remove("hidden");
}
function closeModal() { $("#modal").classList.add("hidden"); }

async function openDialog(examples = false) {
  const r = await fetch("/api/files"); const info = await r.json();
  const names = examples ? info.examples : info.files;
  const list = names.length ? names.map(n => `<button class="file" data-name="${esc(n)}">${esc(n)}</button>`).join("")
                            : `<div class="hint">No worksheets yet.</div>`;
  modal(examples ? "Examples" : "Open worksheet", list + (examples ? "" : `<div class="hint">Folder: ${esc(info.dir)}</div>`));
  $("#modal-body").onclick = async ev => {
    const b = ev.target.closest("button.file"); if (!b) return;
    if (state.dirty && !confirm("Discard unsaved changes?")) return;
    try {
      const { doc } = await api("/api/open", { name: b.dataset.name, example: examples });
      closeModal(); loadDoc(doc, examples ? null : b.dataset.name);
    } catch (e) { alert(e.message); }
  };
}
async function save(as = false) {
  let name = state.fileName;
  if (as || !name) {
    name = prompt("Save worksheet as:", state.title === "Untitled" ? "" : state.title);
    if (!name) return;
  }
  try {
    const r = await api("/api/save", { name, doc: serialize() });
    state.fileName = r.saved; setDirty(false);
    status(`Saved to ${esc(r.path)}`);
  } catch (e) { alert(e.message); }
}

// ---------- Welcome document ----------
const WELCOME = {
  title: "Welcome",
  cells: [
    { type: "text", source: "# Welcome to Quire\nA worksheet where definitions stay live: change a number above and everything below updates. Units travel with the math, and anything you leave undefined stays symbolic." },
    { type: "math", source: "m = 2 kg" },
    { type: "math", source: "a = 3 m/s^2" },
    { type: "math", source: "F = m a -> N" },
    { type: "text", source: "Write units with a space after the number: `3 m/s^2`. Convert with `->`. Press **Enter** for a new cell. Undefined names stay symbolic:" },
    { type: "math", source: "E = 1/2 M v^2" },
    { type: "math", source: "diff(E, v)" },
    { type: "text", source: "Functions, equations and calculus:" },
    { type: "math", source: "f(x) = x^3 - 2 x + 1" },
    { type: "math", source: "solve(f(x) == 0, x)" },
    { type: "math", source: "integrate(f(x), x, 0, 2)" },
    { type: "plot", exprs: "f(x), diff(f(x), x)", xmin: "-2", xmax: "2", var: "", samples: "" },
    { type: "text", source: "Open **Reference** (top right) to browse every function, unit and constant, or **Examples** for complete worksheets." },
  ],
};

// ---------- Wiring ----------
function init() {
  $("#title").addEventListener("input", ev => { state.title = ev.target.value; setDirty(true); });
  $("#btn-new").onclick = () => { if (!state.dirty || confirm("Discard unsaved changes?")) loadDoc({ title: "Untitled", cells: [{ type: "math", source: "" }] }); };
  $("#btn-open").onclick = () => openDialog(false);
  $("#btn-examples").onclick = () => openDialog(true);
  $("#btn-save").onclick = () => save(false);
  $("#btn-saveas").onclick = () => save(true);
  $("#btn-ref").onclick = () => { $("#ref").classList.toggle("hidden"); $("#btn-ref").classList.toggle("on"); };
  $("#btn-reload").onclick = () => loadCatalog(true);
  $("#ref-search").addEventListener("input", renderCatalog);
  $("#ref-list").addEventListener("click", ev => { const it = ev.target.closest(".ref-item"); if (it) insertAtCursor(it.dataset.insert); });
  $("#modal-close").onclick = closeModal;
  $("#modal").addEventListener("click", ev => { if (ev.target.id === "modal") closeModal(); });
  document.querySelector(".add-end").addEventListener("click", ev => { const b = ev.target.closest("button"); if (b) insertCell(b.dataset.add, state.cells.length, true); });
  document.addEventListener("keydown", ev => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "s") { ev.preventDefault(); save(false); }
    if (ev.key === "Escape") closeModal();
  });
  window.addEventListener("beforeunload", ev => { if (state.dirty) { ev.preventDefault(); ev.returnValue = ""; } });

  loadCatalog();
  let draft = null;
  try { draft = JSON.parse(localStorage.getItem(DRAFT_KEY)); } catch (e) { /* ignore */ }
  if (draft && draft.doc && draft.doc.cells && draft.doc.cells.length) loadDoc(draft.doc, draft.fileName);
  else loadDoc(WELCOME);
  // KaTeX loads async; re-render once it is available.
  const tick = setInterval(() => { if (window.katex) { clearInterval(tick); for (const r of state.results.values()) renderResult(r); for (const el of state.els.values()) { const md = $(".md", el); if (md && md.style.display !== "none") { const c = state.cells[cellIndex(el.dataset.id)]; md.innerHTML = markdown(c.source || ""); } } } }, 100);
}
document.addEventListener("DOMContentLoaded", init);
})();
