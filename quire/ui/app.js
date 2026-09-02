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
  if (type === "plot") return { ...base, kind: "function", exprs: "", expr2: "", var: "", xmin: "", xmax: "", ymin: "", ymax: "", samples: "", logx: false, logy: false };
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

const PLOT_KINDS = {
  function:   { label: "y = f(x)",        f1: "y =",     f2: null,   var: "variable", range: "x", yrange: false, ph1: "sin(x), cos(x)   — or a function name like y(t)", ph2: "" },
  parametric: { label: "parametric",      f1: "x(t) =",  f2: "y(t) =", var: "parameter", range: "t", yrange: false, ph1: "cos(t)", ph2: "sin(2 t)" },
  polar:      { label: "polar r(θ)",      f1: "r =",     f2: null,   var: "angle", range: "θ", yrange: false, ph1: "1 + cos(theta)", ph2: "" },
  scatter:    { label: "scatter (data)",  f1: "x data",  f2: "y data", var: null, range: null, yrange: false, ph1: "[1, 2, 3]", ph2: "[2, 4, 6.5]" },
  slope:      { label: "slope field",     f1: "dy/dx =", f2: null,   var: "variables", range: "x", yrange: true, ph1: "x - y", ph2: "" },
  implicit:   { label: "implicit F(x,y)=0", f1: "equation", f2: null, var: "variables", range: "x", yrange: true, ph1: "x^2 + y^2 == 4", ph2: "" },
};

function buildPlot(cell, body) {
  if (!cell.kind) cell.kind = "function";
  const K = PLOT_KINDS[cell.kind] || PLOT_KINDS.function;
  body.innerHTML = `
    <div class="plot-form">
      <select class="kind" title="Plot kind">${Object.entries(PLOT_KINDS).map(([k, v]) => `<option value="${k}" ${k === cell.kind ? "selected" : ""}>${v.label}</option>`).join("")}</select>
      <span class="f1-label">${K.f1}</span><input class="exprs" placeholder="${esc(K.ph1)}" spellcheck="false">
      <span class="f2-wrap" ${K.f2 ? "" : "hidden"}><span class="f2-label">${K.f2 || ""}</span><input class="expr2" placeholder="${esc(K.ph2)}" spellcheck="false"></span>
      <span class="range-wrap" ${K.range ? "" : "hidden"}><span>${K.range} from</span><input class="xmin small" spellcheck="false"><span>to</span><input class="xmax small" spellcheck="false"></span>
      <span class="yrange-wrap" ${K.yrange ? "" : "hidden"}><span>y from</span><input class="ymin small" spellcheck="false"><span>to</span><input class="ymax small" spellcheck="false"></span>
      <span class="var-wrap" ${K.var ? "" : "hidden"}><span>${K.var || ""}</span><input class="var tiny" placeholder="auto" spellcheck="false"></span>
      <span>points</span><input class="samples tiny" placeholder="400" spellcheck="false">
      <label class="chk"><input type="checkbox" class="logx"> log x</label>
      <label class="chk"><input type="checkbox" class="logy"> log y</label>
      <button class="export" title="Download as SVG">SVG</button>
    </div>
    <div class="plot-wrap"><svg class="plot" viewBox="0 0 700 340" style="display:none"></svg><div class="tip" hidden></div></div>
    <div class="legend"></div><div class="err"></div>`;
  for (const key of ["exprs", "expr2", "xmin", "xmax", "ymin", "ymax", "var", "samples"]) {
    const inp = $(`input.${key}`, body);
    inp.value = cell[key] || "";
    inp.addEventListener("input", () => { cell[key] = inp.value; touch(); });
    inp.addEventListener("focus", () => { state.activeInput = inp; });
    inp.addEventListener("keydown", ev => { if (ev.key === "Enter") evaluateNow(); if (ev.key === "Escape") inp.blur(); });
  }
  for (const key of ["logx", "logy"]) {
    const box = $(`input.${key}`, body);
    box.checked = !!cell[key];
    box.addEventListener("change", () => { cell[key] = box.checked; setDirty(true); const r = state.results.get(cell.id); if (r) drawPlot(state.els.get(cell.id), r); });
  }
  $("select.kind", body).addEventListener("change", ev => {
    cell.kind = ev.target.value;
    const el = state.els.get(cell.id);
    buildPlot(cell, body);
    touch();
  });
  $("button.export", body).addEventListener("click", () => exportSvg(cell));
  body._view = { hidden: new Set() };
}

function exportSvg(cell) {
  const el = state.els.get(cell.id); const svg = $("svg.plot", el); if (!svg || svg.style.display === "none") return;
  const src = `<?xml version="1.0" encoding="UTF-8"?>\n` + svg.outerHTML.replace("<svg ", `<svg xmlns="http://www.w3.org/2000/svg" `);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([src], { type: "image/svg+xml" }));
  a.download = (state.title || "plot").replace(/[^\w-]+/g, "_") + "-plot.svg";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
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
    const active = document.activeElement;
    if (active && active.classList && active.classList.contains("slider") && out.contains(active)) {
      // dragging: refresh the value labels only, keep the control alive
      (res.outputs || []).forEach((o, idx) => { if (o.slider) { const row = out.querySelectorAll(".slider-row")[0]; } });
      return;
    }
    out.innerHTML = (res.outputs || []).map((o, idx) => {
      let h = o.head ? katexHtml(`${o.head} = ${o.latex}`) : katexHtml(o.latex);
      if (o.approx) h += `<span class="approx">${katexHtml(`\\approx ${o.approx}`)}</span>`;
      const notes = (o.notes || []).map(n => `<div class="note">${esc(n)}</div>`).join("");
      const sl = o.slider ? `<div class="slider-row"><input type="range" class="slider" data-line="${o.slider.line}" min="${o.slider.min}" max="${o.slider.max}" step="${o.slider.step || (o.slider.max - o.slider.min) / 200}" value="${o.slider.value}"><span class="slider-val">${fmtVal(o.slider.value)}</span></div>` : "";
      return `<div class="out-line">${h}</div>${sl}${notes}`;
    }).join("");
    for (const sl of out.querySelectorAll("input.slider")) sl.addEventListener("input", () => onSlider(cell, el, sl));
    warn.textContent = res.warning || "";
    const parts = [];
    if (res.defines && res.defines.length) parts.push(`defines <b>${esc(res.defines.join(", "))}</b>`);
    if (res.uses && res.uses.length) parts.push(`uses <b>${esc(res.uses.join(", "))}</b>`);
    meta.innerHTML = parts.join(" · ");
  } else if (cell.type === "plot") {
    drawPlot(el, res);
  }
}

function onSlider(cell, el, sl) {
  const line = +sl.dataset.line, v = +sl.value;
  const lines = cell.source.split("\n");
  lines[line] = lines[line].replace(/slider\(\s*[-+0-9.eE]+/, `slider(${+v.toPrecision(6)}`);
  cell.source = lines.join("\n");
  const ta = $("textarea.src", el); if (ta && document.activeElement !== ta) ta.value = cell.source;
  sl.nextElementSibling.textContent = fmtVal(v);
  setDirty(true);
  clearTimeout(state.evalTimer);
  state.evalTimer = setTimeout(evaluateNow, 40);
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
function logTicks(lo, hi) { // lo, hi in log10 space
  const ticks = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) ticks.push(e);
  if (ticks.length < 2) return niceTicks(lo, hi, 5);
  return ticks;
}
function fmtTick(v, log) {
  if (log) { const a = Math.abs(v) < 4 && Number.isInteger(v) ? String(+Math.pow(10, v).toPrecision(4)) : `1e${v}`; return a; }
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
  return String(+v.toPrecision(5));
}
function fmtVal(v) { return Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(4) : String(+v.toPrecision(6)); }

function plotGeometry(cell, res, view) {
  const logx = !!cell.logx, logy = !!cell.logy;
  const tx = v => logx ? (v > 0 ? Math.log10(v) : NaN) : v, ty = v => logy ? (v > 0 ? Math.log10(v) : NaN) : v;
  let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
  const consider = (x, y) => { x = tx(x); y = ty(y); if (Number.isFinite(x)) { xlo = Math.min(xlo, x); xhi = Math.max(xhi, x); } if (Number.isFinite(y)) { ylo = Math.min(ylo, y); yhi = Math.max(yhi, y); } };
  for (const s of res.series) {
    if (view.hidden.has(s.label_plain)) continue;
    if (s.type === "segments") for (const g of s.segments) { consider(g[0], g[1]); consider(g[2], g[3]); }
    else for (let i = 0; i < s.x.length; i++) if (s.x[i] !== null && s.y[i] !== null) consider(s.x[i], s.y[i]);
  }
  if (res.xrange) { xlo = tx(res.xrange[0]); xhi = tx(res.xrange[1]); }
  if (res.yrange) { ylo = ty(res.yrange[0]); yhi = ty(res.yrange[1]); }
  if (res.ysuggest && !logy && !res.yrange) { ylo = Math.max(ylo, res.ysuggest[0]); yhi = Math.min(yhi, res.ysuggest[1]); }
  const ymin = parseFloat(cell.ymin), ymax = parseFloat(cell.ymax);
  if (!res.yrange && Number.isFinite(ymin) && Number.isFinite(ymax) && ymin < ymax) { ylo = ty(ymin); yhi = ty(ymax); }
  if (!Number.isFinite(xlo) || !Number.isFinite(ylo)) return null;
  if (xlo === xhi) { xlo -= 1; xhi += 1; }
  if (ylo === yhi) { ylo -= 1; yhi += 1; }
  if (!res.yrange && !(Number.isFinite(ymin) && Number.isFinite(ymax))) { const pad = (yhi - ylo) * 0.06; ylo -= pad; yhi += pad; }
  const W = 700, H = 340, ml = 64, mr = 16, mt = 14, mb = 44;
  if (res.equal && !logx && !logy) { // same scale on both axes
    const sx = (W - ml - mr) / (xhi - xlo), sy = (H - mt - mb) / (yhi - ylo);
    if (sx > sy) { const cx = (xlo + xhi) / 2, half = (W - ml - mr) / sy / 2; xlo = cx - half; xhi = cx + half; }
    else { const cy = (ylo + yhi) / 2, half = (H - mt - mb) / sx / 2; ylo = cy - half; yhi = cy + half; }
  }
  const sx = x => ml + (tx(x) - xlo) / (xhi - xlo) * (W - ml - mr);
  const sy = y => H - mb - (ty(y) - ylo) / (yhi - ylo) * (H - mt - mb);
  const ix = px => { const v = xlo + (px - ml) / (W - ml - mr) * (xhi - xlo); return logx ? Math.pow(10, v) : v; };
  const iy = py => { const v = ylo + (H - mb - py) / (H - mt - mb) * (yhi - ylo); return logy ? Math.pow(10, v) : v; };
  return { W, H, ml, mr, mt, mb, xlo, xhi, ylo, yhi, sx, sy, ix, iy, logx, logy };
}

function drawPlot(el, res) {
  const svg = $("svg.plot", el), legend = $(".legend", el), body = $(".body", el);
  const cell = state.cells[cellIndex(el.dataset.id)];
  const view = body._view || (body._view = { hidden: new Set() });
  if (!res.ok || !res.series || !res.series.length) { svg.style.display = "none"; legend.innerHTML = ""; return; }
  const g0 = plotGeometry(cell, res, view);
  if (!g0) { svg.style.display = "none"; legend.innerHTML = `<span class="err">Nothing to draw: all values are undefined on this range.</span>`; return; }
  const { W, H, ml, mr, mt, mb, xlo, xhi, ylo, yhi, sx, sy, logx, logy } = g0;
  let g = "";
  for (const t of (logx ? logTicks(xlo, xhi) : niceTicks(xlo, xhi, 8))) {
    const px = ml + (t - xlo) / (xhi - xlo) * (W - ml - mr);
    g += `<line x1="${px}" y1="${mt}" x2="${px}" y2="${H - mb}" stroke="#eee"/><text x="${px}" y="${H - mb + 16}" font-size="11" text-anchor="middle" fill="#666">${fmtTick(t, logx)}</text>`;
  }
  for (const t of (logy ? logTicks(ylo, yhi) : niceTicks(ylo, yhi, 6))) {
    const py = H - mb - (t - ylo) / (yhi - ylo) * (H - mt - mb);
    g += `<line x1="${ml}" y1="${py}" x2="${W - mr}" y2="${py}" stroke="#eee"/><text x="${ml - 6}" y="${py + 4}" font-size="11" text-anchor="end" fill="#666">${fmtTick(t, logy)}</text>`;
  }
  if (!logy && ylo < 0 && yhi > 0) g += `<line x1="${ml}" y1="${sy(0)}" x2="${W - mr}" y2="${sy(0)}" stroke="#999"/>`;
  if (!logx && xlo < 0 && xhi > 0) g += `<line x1="${sx(0)}" y1="${mt}" x2="${sx(0)}" y2="${H - mb}" stroke="#999"/>`;
  g += `<rect x="${ml}" y="${mt}" width="${W - ml - mr}" height="${H - mt - mb}" fill="none" stroke="#ccc"/>`;
  const clip = `clip-path="url(#clip-${res.id})"`;
  res.series.forEach((s, k) => {
    if (view.hidden.has(s.label_plain)) return;
    const color = PALETTE[k % PALETTE.length];
    if (s.type === "points") {
      g += `<g ${clip}>` + s.x.map((x, i) => (x === null || s.y[i] === null) ? "" : `<circle cx="${sx(x).toFixed(1)}" cy="${sy(s.y[i]).toFixed(1)}" r="3.5" fill="${color}"/>`).join("") + "</g>";
    } else if (s.type === "segments") {
      g += `<g ${clip} stroke="${color}" stroke-width="1.4">` + s.segments.map(q => `<line x1="${sx(q[0]).toFixed(1)}" y1="${sy(q[1]).toFixed(1)}" x2="${sx(q[2]).toFixed(1)}" y2="${sy(q[3]).toFixed(1)}"/>`).join("") + "</g>";
    } else {
      let d = "", pen = false;
      s.x.forEach((x, i) => {
        const y = s.y[i];
        if (x === null || y === null || (logx && x <= 0) || (logy && y <= 0)) { pen = false; return; }
        const py = Math.max(mt - 2000, Math.min(H - mb + 2000, sy(y)));
        d += `${pen ? "L" : "M"}${sx(x).toFixed(1)} ${py.toFixed(1)} `; pen = true;
      });
      g += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" ${clip}/>`;
    }
  });
  g += `<text x="${(ml + W - mr) / 2}" y="${H - 8}" font-size="12" text-anchor="middle" fill="#444">${esc(res.xlabel || "")}</text>`;
  if (res.ylabel) g += `<text x="14" y="${(mt + H - mb) / 2}" font-size="12" text-anchor="middle" fill="#444" transform="rotate(-90 14 ${(mt + H - mb) / 2})">${esc(res.ylabel)}</text>`;
  g += `<g class="cursor" hidden><line class="vx" y1="${mt}" y2="${H - mb}" stroke="#888" stroke-dasharray="3 3"/><g class="dots"></g></g>`;
  svg.innerHTML = `<defs><clipPath id="clip-${res.id}"><rect x="${ml}" y="${mt}" width="${W - ml - mr}" height="${H - mt - mb}"/></clipPath></defs>${g}`;
  svg.style.display = "";
  legend.innerHTML = res.series.map((s, k) => `<span class="lg ${view.hidden.has(s.label_plain) ? "off" : ""}" data-label="${esc(s.label_plain)}" title="Click to hide or show"><span class="swatch" style="background:${PALETTE[k % PALETTE.length]}"></span>${katexHtml(s.label)}</span>`).join("");
  legend.onclick = ev => { const lg = ev.target.closest(".lg"); if (!lg) return; const key = lg.dataset.label; if (view.hidden.has(key)) view.hidden.delete(key); else view.hidden.add(key); drawPlot(el, res); };
  attachPlotInteraction(el, cell, res, g0);
}

function attachPlotInteraction(el, cell, res, geo) {
  const svg = $("svg.plot", el), tip = $(".tip", el), body = $(".body", el);
  const toSvg = ev => { const r = svg.getBoundingClientRect(); return { px: (ev.clientX - r.left) * geo.W / r.width, py: (ev.clientY - r.top) * geo.H / r.height }; };
  const inside = p => p.px >= geo.ml && p.px <= geo.W - geo.mr && p.py >= geo.mt && p.py <= geo.H - geo.mb;
  let drag = null;
  svg.onmousemove = ev => {
    const p = toSvg(ev);
    if (drag) {
      const dx = geo.ix(p.px) - geo.ix(drag.px);
      if (drag.moved || Math.abs(p.px - drag.px) > 3) { drag.moved = true; }
      return;
    }
    const cursor = $(".cursor", svg);
    if (!inside(p) || !res.series.some(s => s.type === "line")) { cursor.hidden = true; tip.hidden = true; return; }
    const x = geo.ix(p.px);
    const rows = [];
    let dots = "";
    res.series.forEach((s, k) => {
      if (s.type !== "line" || body._view.hidden.has(s.label_plain)) return;
      let best = -1, bd = Infinity;
      for (let i = 0; i < s.x.length; i++) { if (s.x[i] === null || s.y[i] === null) continue; const d = Math.abs(s.x[i] - x); if (d < bd) { bd = d; best = i; } }
      if (best < 0) return;
      const y = s.y[best];
      rows.push(`<span class="swatch" style="background:${PALETTE[k % PALETTE.length]}"></span>${esc(s.label_plain)} = <b>${fmtVal(y)}</b>`);
      dots += `<circle cx="${geo.sx(s.x[best])}" cy="${geo.sy(y)}" r="4" fill="${PALETTE[k % PALETTE.length]}" stroke="#fff"/>`;
    });
    if (!rows.length) { cursor.hidden = true; tip.hidden = true; return; }
    cursor.hidden = false;
    $(".vx", cursor).setAttribute("x1", p.px); $(".vx", cursor).setAttribute("x2", p.px);
    $(".dots", cursor).innerHTML = dots;
    tip.hidden = false;
    tip.innerHTML = `<div>${esc(res.var || "x")} = <b>${fmtVal(x)}</b></div>` + rows.map(r => `<div>${r}</div>`).join("");
    const wrap = svg.parentElement.getBoundingClientRect();
    const left = ev.clientX - wrap.left + 14, top = ev.clientY - wrap.top + 10;
    tip.style.left = Math.min(left, wrap.width - tip.offsetWidth - 8) + "px"; tip.style.top = top + "px";
  };
  svg.onmouseleave = () => { const c = $(".cursor", svg); if (c) c.hidden = true; tip.hidden = true; drag = null; };
  const setRange = (lo, hi) => {
    if (!PLOT_KINDS[cell.kind].range) return;
    const f = v => String(+v.toPrecision(6));
    cell.xmin = f(lo); cell.xmax = f(hi);
    $("input.xmin", el).value = cell.xmin; $("input.xmax", el).value = cell.xmax;
    touch();
  };
  svg.onwheel = ev => {
    const p = toSvg(ev); if (!inside(p) || geo.logx) return;
    ev.preventDefault();
    const factor = ev.deltaY > 0 ? 1.25 : 0.8, x = geo.ix(p.px);
    const lo = x - (x - geo.ix(geo.ml)) * factor, hi = x + (geo.ix(geo.W - geo.mr) - x) * factor;
    setRange(lo, hi);
  };
  svg.onmousedown = ev => { const p = toSvg(ev); if (inside(p) && !geo.logx) { drag = { px: p.px, lo: geo.ix(geo.ml), hi: geo.ix(geo.W - geo.mr), moved: false }; ev.preventDefault(); } };
  svg.onmouseup = ev => {
    if (!drag) return;
    const p = toSvg(ev);
    if (drag.moved) { const shift = geo.ix(drag.px) - geo.ix(p.px); setRange(drag.lo + shift, drag.hi + shift); }
    drag = null;
  };
  svg.ondblclick = () => { if (cell._orig) { cell.xmin = cell._orig.xmin; cell.xmax = cell._orig.xmax; $("input.xmin", el).value = cell.xmin; $("input.xmax", el).value = cell.xmax; touch(); } };
  if (!cell._orig) cell._orig = { xmin: cell.xmin, xmax: cell.xmax };
}

// ---------- Reference panel ----------
async function loadCatalog(reload = false) {
  const r = await fetch(reload ? "/api/reload" : "/api/catalog", { method: reload ? "POST" : "GET" });
  state.catalog = await r.json();
  renderCatalog();
  if (reload) evaluateNow();
}
const REF_KEY = "quire.ref.collapsed";
function loadCollapsed() { try { return new Set(JSON.parse(localStorage.getItem(REF_KEY) || "null") || []); } catch (e) { return new Set(); } }
function saveCollapsed(set) { try { localStorage.setItem(REF_KEY, JSON.stringify([...set])); } catch (e) { /* ignore */ } }

function renderCatalog() {
  const q = $("#ref-search").value.trim().toLowerCase();
  const cat = state.catalog; if (!cat) return;
  $("#ref-modules").innerHTML = "Modules: " + cat.modules.map(m =>
    m.error ? `<span class="bad" title="${esc(m.error)}">${esc(m.name)} (failed)</span>` : `<span title="${esc(m.description)}">${esc(m.name)}</span>`
  ).join(", ") + (cat.conflicts && cat.conflicts.length ? `<div class="bad">${cat.conflicts.map(esc).join("<br>")}</div>` : "")
    + `<div class="ref-tools"><button data-all="open">expand all</button><button data-all="close">collapse all</button></div>`;
  const groups = new Map();
  for (const e of cat.entries) {
    if (q && !(e.name.toLowerCase().includes(q) || e.doc.toLowerCase().includes(q) || e.signature.toLowerCase().includes(q))) continue;
    const key = e.category || "Other";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(e);
  }
  if (!state.refCollapsed) state.refCollapsed = loadCollapsed();
  if (!state.refInit) { state.refInit = true; if (!localStorage.getItem(REF_KEY)) { for (const c of groups.keys()) state.refCollapsed.add(c); } }
  let h = "";
  for (const [c, items] of groups) {
    const open = q ? true : !state.refCollapsed.has(c);
    h += `<div class="ref-cat ${open ? "open" : ""}" data-cat="${esc(c)}"><span class="caret">${open ? "▾" : "▸"}</span>${esc(c)} <span class="count">${items.length}</span></div>`;
    if (!open) continue;
    h += `<div class="ref-items">`;
    for (const e of items) {
      const insert = e.example || (e.kind === "function" ? e.signature : e.name);
      h += `<div class="ref-item" data-insert="${esc(insert)}" title="Click to insert">
        <span class="sig">${esc(e.kind === "function" ? e.signature : e.name)}${e.module !== "core" ? ` <span class="mod">${esc(e.module)}</span>` : ""}</span>
        <span class="doc">${esc(e.doc)}</span></div>`;
    }
    h += `</div>`;
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
  $("#ref-list").addEventListener("click", ev => {
    const head = ev.target.closest(".ref-cat[data-cat]");
    if (head) {
      const c = head.dataset.cat;
      if (state.refCollapsed.has(c)) state.refCollapsed.delete(c); else state.refCollapsed.add(c);
      saveCollapsed(state.refCollapsed); renderCatalog(); return;
    }
    const it = ev.target.closest(".ref-item"); if (it) insertAtCursor(it.dataset.insert);
  });
  $("#ref-modules").addEventListener("click", ev => {
    const b = ev.target.closest("button[data-all]"); if (!b || !state.catalog) return;
    const cats = new Set(state.catalog.entries.map(e => e.category || "Other"));
    state.refCollapsed = b.dataset.all === "close" ? cats : new Set();
    saveCollapsed(state.refCollapsed); renderCatalog();
  });
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
