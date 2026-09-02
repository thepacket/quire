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

// ---------- Markdown (headings, lists, tables, quotes, images, links, footnotes, $math$, $$display$$, {{values}}) ----------
function imgSrc(src) {
  if (/^(https?:)?\/\//i.test(src) || /^data:image\//i.test(src)) return src;
  return "/files/" + src.replace(/^\/files\//, "");
}
function renderValue(ctx) {
  const v = ctx.values ? ctx.values[ctx.k] : null;
  ctx.k++;
  if (!v) return `<span class="val pending">…</span>`;
  if (v.error) return `<span class="val err" title="${esc(v.error)}">${esc(v.error)}</span>`;
  return `<span class="val">${katexHtml(v.latex)}${v.approx ? `<span class="approx">${katexHtml("\\approx " + v.approx)}</span>` : ""}</span>`;
}
function inlineMd(s, ctx = {}) {
  return s.split(/(`[^`]+`|\$[^$\n]+\$)/g).map(part => {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) return `<code>${esc(part.slice(1, -1))}</code>`;
    if (part.length > 2 && part.startsWith("$") && part.endsWith("$")) return katexHtml(part.slice(1, -1));
    let h = esc(part);
    h = h.replace(/\{\{[^}]*\}\}/g, () => renderValue(ctx));
    h = h.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (m, alt, src) => `<img src="${imgSrc(src)}" alt="${alt}" title="${alt}">`);
    h = h.replace(/\[\^([^\]\s]+)\]/g, (m, id) => `<sup class="fn"><a href="#fn-${ctx.cellId || ""}-${id}">${id}</a></sup>`);
    h = h.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, t, url) => /^(https?:\/\/|#|\/)/i.test(url) ? `<a href="${url}" target="_blank" rel="noopener">${t}</a>` : m);
    h = h.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*]+)\*/g, "$1<i>$2</i>");
    return h;
  }).join("");
}
function markdown(text, ctx = {}) {
  ctx.k = 0;
  const lines = text.split("\n"), out = [], foot = [];
  let para = [], list = null, quote = [];
  const flush = () => {
    if (para.length) { out.push(`<p>${inlineMd(para.join(" "), ctx)}</p>`); para = []; }
    if (list) { out.push(`<${list.tag}>${list.items.map(l => `<li>${inlineMd(l, ctx)}</li>`).join("")}</${list.tag}>`); list = null; }
    if (quote.length) { out.push(`<blockquote>${inlineMd(quote.join(" "), ctx)}</blockquote>`); quote = []; }
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trimEnd(), t = line.trim();
    if (t.startsWith("$$")) {
      flush();
      let buf = t;
      if (!(t.length > 4 && t.endsWith("$$"))) while (++i < lines.length) { buf += "\n" + lines[i]; if (lines[i].trim().endsWith("$$")) break; }
      out.push(`<div class="dmath">${katexHtml(buf.replace(/^\$\$/, "").replace(/\$\$$/, ""), true)}</div>`);
      continue;
    }
    if (/^(-{3,}|\*{3,})$/.test(t)) { flush(); out.push("<hr>"); continue; }
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) { flush(); out.push(`<h${h[1].length}>${inlineMd(h[2], ctx)}</h${h[1].length}>`); continue; }
    const fn = /^\[\^([^\]\s]+)\]:\s*(.*)$/.exec(t);
    if (fn) { flush(); foot.push([fn[1], fn[2]]); continue; }
    if (t.startsWith("|") && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1].trim())) {
      flush();
      const cells = r => r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
      const head = cells(t), align = cells(lines[i + 1]).map(a => a.startsWith(":") && a.endsWith(":") ? "center" : a.endsWith(":") ? "right" : "left");
      const rows = [];
      i += 1;
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith("|")) rows.push(cells(lines[++i]));
      const td = (c, k, tag) => `<${tag} style="text-align:${align[k] || "left"}">${inlineMd(c, ctx)}</${tag}>`;
      out.push(`<table><thead><tr>${head.map((c, k) => td(c, k, "th")).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c, k) => td(c, k, "td")).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    const q = /^>\s?(.*)$/.exec(t);
    if (q) { if (para.length || list) flush(); quote.push(q[1]); continue; }
    const ol = /^\d+[.)]\s+(.*)$/.exec(t), ul = /^[-*+]\s+(.*)$/.exec(t);
    if (ol || ul) {
      const tag = ol ? "ol" : "ul";
      if (para.length || quote.length || (list && list.tag !== tag)) flush();
      if (!list) list = { tag, items: [] };
      list.items.push((ol || ul)[1]);
      continue;
    }
    if (!t) { flush(); continue; }
    if (list || quote.length) flush();
    para.push(line);
  }
  flush();
  if (foot.length) out.push(`<div class="footnotes"><ol>${foot.map(([id, txt]) => `<li id="fn-${ctx.cellId || ""}-${id}"><sup>${esc(id)}</sup> ${inlineMd(txt, ctx)}</li>`).join("")}</ol></div>`);
  return out.join("");
}

// ---------- Document model ----------
function newCell(type) {
  const base = { id: uid(), type };
  if (type === "plot") return { ...base, kind: "function", exprs: "", expr2: "", expr3: "", annot: "", var: "", xmin: "", xmax: "", ymin: "", ymax: "", samples: "", logx: false, logy: false };
  return { ...base, source: "" };
}
function serialize() {
  return { quire: 1, title: state.title, author: state.author || "", cells: state.cells.map(c => ({ ...c })) };
}
function renderTitleBlock() {
  $("#tb-title").textContent = state.title || "Untitled";
  $("#tb-meta").textContent = [state.author, state.savedAt ? "saved " + state.savedAt : ""].filter(Boolean).join(" · ");
}
function loadDoc(doc, fileName = null) {
  state.title = doc.title || "Untitled";
  state.author = doc.author || "";
  state.savedAt = doc.saved_at || null;
  state.fileName = fileName;
  renderTitleBlock();
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
  body.parentElement.dataset.src = cell.source || "";  // shown instead of the textarea when printing
  ta.addEventListener("input", () => { cell.source = ta.value; body.parentElement.dataset.src = ta.value; autogrow(ta); touch(); acShow(ta); });
  ta.addEventListener("keydown", ev => { if (acKey(ev)) return; keyNav(cell, ta, ev); });
  ta.addEventListener("focus", () => { state.activeInput = ta; });
  ta.addEventListener("blur", () => setTimeout(acClose, 120));
  requestAnimationFrame(() => autogrow(ta));
}

function buildText(cell, body) {
  body.innerHTML = `<textarea class="src" rows="1" spellcheck="true" placeholder="Write text. *italic*, **bold**, # heading, $x^2$ for math."></textarea><div class="md"></div>`;
  const ta = $("textarea", body), md = $(".md", body);
  ta.value = cell.source || "";
  const show = () => {
    if (document.activeElement === ta) return;
    const res = state.results.get(cell.id) || {};
    md.innerHTML = markdown(cell.source || "", { cellId: cell.id, values: res.values || null });
    md.style.display = ""; ta.style.display = "none";
  };
  body._show = show;
  const edit = () => { md.style.display = "none"; ta.style.display = ""; autogrow(ta); ta.focus(); };
  ta.addEventListener("input", () => { cell.source = ta.value; autogrow(ta); setDirty(true); if (/\{\{/.test(ta.value)) scheduleEval(); });
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

// Core plot kinds. The catalog replaces this table with the server's, which also lists module kinds.
let PLOT_KINDS = {
  function:   { label: "y = f(x)", f1: "y =", var: "variable", range: "x", ph1: "sin(x), x^2/10", annot: true, samples: "400" },
  parametric: { label: "parametric", f1: "x(t) =", f2: "y(t) =", var: "variable", range: "t", ph1: "cos(3 t)", ph2: "sin(2 t)", annot: true, samples: "600" },
  polar:      { label: "polar r(θ)", f1: "r(θ) =", var: "variable", range: "θ", ph1: "1 + cos(theta)", annot: true, samples: "600" },
  scatter:    { label: "scatter (data)", f1: "x data", f2: "y data", f3: "y errors", ph1: "[1, 2, 3]", ph2: "[2, 4, 6.5]", annot: true },
  slope:      { label: "slope field", f1: "dy/dx =", var: "variables", range: "x", yrange: true, ph1: "x - y", samples: "20" },
  implicit:   { label: "implicit F(x,y)=0", f1: "equation", var: "variables", range: "x", yrange: true, ph1: "x^2 + y^2 == 4", samples: "200", annot: true },
  shapes:     { label: "shapes (geometry)", f1: "draw", range: "x", yrange: true, ph1: "circle(point(0, 0), 2), triangle(point(-1, -1), point(2, 0), point(0, 1))", annot: true },
  contour:    { label: "contour F(x,y)", f1: "F(x, y) =", f2: "levels", var: "variables", range: "x", yrange: true, ph1: "sin(x) cos(y)", ph2: "12", renderer: "plotly", samples: "80" },
  heatmap:    { label: "heatmap F(x,y)", f1: "F(x, y) =", var: "variables", range: "x", yrange: true, ph1: "exp(-(x^2 + y^2)/4)", renderer: "plotly", samples: "80" },
  surface:    { label: "3D surface z = F(x,y)", f1: "z =", var: "variables", range: "x", yrange: true, ph1: "sin(x) cos(y)", renderer: "plotly", samples: "50" },
  curve3d:    { label: "3D curve", f1: "x(t), y(t), z(t) =", var: "variable", range: "t", ph1: "cos(t), sin(t), t/5", renderer: "plotly" },
};
function applyPlotKinds(list) {
  if (!list || !list.length) return;
  const kinds = {};
  for (const k of list) kinds[k.name] = k;
  PLOT_KINDS = kinds;
  for (const [id, el] of state.els) {
    const cell = state.cells[cellIndex(id)];
    const sel = cell && cell.type === "plot" ? $("select.kind", el) : null;
    if (sel) sel.innerHTML = kindOptions(cell.kind);
  }
}
function kindOptions(current) {
  return Object.entries(PLOT_KINDS).map(([k, v]) => `<option value="${k}" ${k === current ? "selected" : ""}>${esc(v.label)}</option>`).join("")
    + (PLOT_KINDS[current] ? "" : `<option value="${esc(current)}" selected>${esc(current)}</option>`);
}

function buildPlot(cell, body) {
  if (!cell.kind) cell.kind = "function";
  const K = PLOT_KINDS[cell.kind] || PLOT_KINDS.function;
  const plotly = K.renderer === "plotly";
  body.innerHTML = `
    <div class="plot-form">
      <select class="kind" title="Plot kind">${kindOptions(cell.kind)}</select>
      <span class="f1-label">${esc(K.f1 || "")}</span><input class="exprs" placeholder="${esc(K.ph1 || "")}" spellcheck="false">
      <span class="f2-wrap" ${K.f2 ? "" : "hidden"}><span class="f2-label">${esc(K.f2 || "")}</span><input class="expr2" placeholder="${esc(K.ph2 || "")}" spellcheck="false"></span>
      <span class="f3-wrap" ${K.f3 ? "" : "hidden"}><span class="f3-label">${esc(K.f3 || "")}</span><input class="expr3" placeholder="${esc(K.ph3 || "")}" spellcheck="false"></span>
      <span class="range-wrap" ${K.range ? "" : "hidden"}><span>${esc(K.range || "")} from</span><input class="xmin small" spellcheck="false"><span>to</span><input class="xmax small" spellcheck="false"></span>
      <span class="yrange-wrap" ${K.yrange ? "" : "hidden"}><span>y from</span><input class="ymin small" spellcheck="false"><span>to</span><input class="ymax small" spellcheck="false"></span>
      <span class="var-wrap" ${K.var ? "" : "hidden"}><span>${esc(K.var || "")}</span><input class="var tiny" placeholder="auto" spellcheck="false"></span>
      <span>points</span><input class="samples tiny" placeholder="${esc(K.samples || "400")}" spellcheck="false">
      <label class="chk" ${plotly ? "hidden" : ""}><input type="checkbox" class="logx"> log x</label>
      <label class="chk" ${plotly ? "hidden" : ""}><input type="checkbox" class="logy"> log y</label>
      <button class="export" title="Download the picture">${plotly ? "PNG" : "SVG"}</button>
    </div>
    <div class="annot-wrap" ${K.annot ? "" : "hidden"}><span>annotate</span><input class="annot" placeholder='mark(1, "peak"), shade(0, 2), hline(0.5), vline(1), band(2, 3), text(2, 1, "note")' spellcheck="false"></div>
    <div class="plot-wrap"><div class="panels"></div><div class="plotly-box" hidden></div><div class="tip" hidden></div></div>
    <div class="legend"></div><div class="err"></div>`;
  for (const key of ["exprs", "expr2", "expr3", "xmin", "xmax", "ymin", "ymax", "var", "samples", "annot"]) {
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
    buildPlot(cell, body);
    touch();
  });
  $("button.export", body).addEventListener("click", () => exportPlot(cell));
  body._view = { hidden: new Set() };
}

function exportPlot(cell) {
  const el = state.els.get(cell.id); if (!el) return;
  const name = (state.title || "plot").replace(/[^\w-]+/g, "_") + "-plot";
  const box = $(".plotly-box", el);
  if (box && !box.hidden && window.Plotly) { Plotly.downloadImage(box, { format: "png", width: 1000, height: 640, filename: name }); return; }
  const svg = $(".panels svg", el); if (!svg) return;
  const src = `<?xml version="1.0" encoding="UTF-8"?>\n` + svg.outerHTML.replace("<svg ", `<svg xmlns="http://www.w3.org/2000/svg" `);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([src], { type: "image/svg+xml" }));
  a.download = name + ".svg";
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
  const cells = state.cells;
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
      const reading = o.reading ? `<div class="reading"><span class="lbl">read as</span>${katexHtml(o.reading)}</div>` : "";
      const steps = o.steps ? `<div class="steps">${o.steps_title ? `<div class="steps-title">${esc(o.steps_title)}</div>` : ""}${o.steps.map((st, i) => `<div class="step"><span class="step-n">${i + 1}.</span><span class="step-text">${esc(st.text)}</span>${st.latex ? `<span class="step-math">${katexHtml(st.latex)}</span>` : ""}</div>`).join("")}</div>` : "";
      const sl = o.slider ? `<div class="slider-row"><input type="range" class="slider" data-line="${o.slider.line}" data-name="${esc(o.slider.name || "")}" min="${o.slider.min}" max="${o.slider.max}" step="${o.slider.step || (o.slider.max - o.slider.min) / 200}" value="${o.slider.value}"><span class="slider-val">${fmtVal(o.slider.value)}</span></div>` : "";
      return `${reading}${steps}<div class="out-line">${h}</div>${sl}${notes}`;
    }).join("");
    for (const sl of out.querySelectorAll("input.slider")) sl.addEventListener("input", () => onSlider(cell, el, sl));
    warn.textContent = res.warning || "";
    const parts = [];
    if (res.defines && res.defines.length) parts.push(`defines <b>${esc(res.defines.join(", "))}</b>`);
    if (res.uses && res.uses.length) parts.push(`uses <b>${esc(res.uses.join(", "))}</b>`);
    meta.innerHTML = parts.join(" · ");
  } else if (cell.type === "plot") {
    drawPlot(el, res);
  } else if (cell.type === "text") {
    const body = $(".body", el);
    if (body._show) body._show();
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
  fastRedraw();
  clearTimeout(state.evalTimer);
  state.evalTimer = setTimeout(evaluateNow, 120);
}
function sliderValues() {
  const m = {};
  for (const sl of document.querySelectorAll("input.slider[data-name]")) if (sl.dataset.name) m[sl.dataset.name] = +sl.value;
  return m;
}
// Series that depend on sliders arrive compiled to JavaScript; redraw them at once while dragging.
function fastRedraw() {
  const vals = sliderValues();
  for (const [id, res] of state.results) {
    const cell = state.cells[cellIndex(id)];
    if (!cell || cell.type !== "plot" || !res.ok || !res.series) continue;
    let changed = false;
    for (const s of res.series) {
      if (!s.js) continue;
      try {
        if (!s._fns) s._fns = s.js.map(code => new Function(res.var || "x", ...s.params, "return " + code));
        const args = s.params.map(p => vals[p]);
        if (args.some(v => v === undefined || !Number.isFinite(v))) continue;
        const fin = v => Number.isFinite(v) ? v : null;
        if (s.grid) { s.x = s.grid.map(t => fin(s._fns[0](t, ...args))); s.y = s.grid.map(t => fin(s._fns[1](t, ...args))); }
        else s.y = s.x.map(x => x === null ? null : fin(s._fns[0](x, ...args)));
        changed = true;
      } catch (e) { s.js = null; }
    }
    if (changed) drawPlot(state.els.get(id), res);
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
const SVG2D = new Set(["line", "points", "segments"]);

function interpSeries(s, x) {
  for (let i = 0; i < s.x.length - 1; i++) {
    const x1 = s.x[i], x2 = s.x[i + 1], y1 = s.y[i], y2 = s.y[i + 1];
    if (x1 === null || x2 === null || y1 === null || y2 === null) continue;
    if (x1 <= x && x <= x2) return y1 + (x - x1) / ((x2 - x1) || 1) * (y2 - y1);
  }
  return null;
}

function plotGeometry(cell, res, view) {
  const logx = !!cell.logx || !!res.logx, logy = !!cell.logy || !!res.logy;
  const tx = v => logx ? (v > 0 ? Math.log10(v) : NaN) : v, ty = v => logy ? (v > 0 ? Math.log10(v) : NaN) : v;
  let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
  const consider = (x, y) => { x = tx(x); y = ty(y); if (Number.isFinite(x)) { xlo = Math.min(xlo, x); xhi = Math.max(xhi, x); } if (Number.isFinite(y)) { ylo = Math.min(ylo, y); yhi = Math.max(yhi, y); } };
  for (const s of res.series) {
    if (!SVG2D.has(s.type) || view.hidden.has(s.label_plain)) continue;
    if (s.type === "segments") for (const g of s.segments) { consider(g[0], g[1]); consider(g[2], g[3]); }
    else for (let i = 0; i < s.x.length; i++) if (s.x[i] !== null && s.y[i] !== null) { consider(s.x[i], s.y[i]); if (s.yerr) { consider(s.x[i], s.y[i] - s.yerr[i]); consider(s.x[i], s.y[i] + s.yerr[i]); } }
  }
  for (const a of (res.annotations || [])) { if (a.type === "point" || a.type === "text") consider(a.x, a.y); if (a.type === "hline") consider(NaN, a.y); if (a.type === "vline") consider(a.x, NaN); }
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
  const body = $(".body", el), legend = $(".legend", el), panelsEl = $(".panels", el), box = $(".plotly-box", el);
  const cell = state.cells[cellIndex(el.dataset.id)];
  const view = body._view || (body._view = { hidden: new Set() });
  const panels = res.ok ? (res.subplots || [res]).filter(p => p.series && p.series.length) : [];
  if (!panels.length) { panelsEl.innerHTML = ""; box.hidden = true; legend.innerHTML = ""; return; }
  if (res.renderer === "plotly") { panelsEl.innerHTML = ""; drawPlotly(el, cell, res, legend); return; }
  box.hidden = true;
  while (panelsEl.children.length > panels.length) panelsEl.lastChild.remove();
  while (panelsEl.children.length < panels.length) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "plot"); svg.setAttribute("viewBox", "0 0 700 340");
    panelsEl.appendChild(svg);
  }
  const all = [];
  panels.forEach((p, i) => {
    const pres = Object.assign({}, p, { id: `${res.id}-${i}`, annotations: i === 0 ? res.annotations : null });
    drawSvg(panelsEl.children[i], el, cell, pres, view, all.length);
    all.push(...p.series);
  });
  legend.innerHTML = all.map((s, k) => s.label === "" ? "" : `<span class="lg ${view.hidden.has(s.label_plain) ? "off" : ""}" data-label="${esc(s.label_plain)}" title="Click to hide or show"><span class="swatch" style="background:${PALETTE[k % PALETTE.length]}"></span>${katexHtml(s.label)}</span>`).join("");
  legend.onclick = ev => { const lg = ev.target.closest(".lg"); if (!lg) return; const key = lg.dataset.label; if (view.hidden.has(key)) view.hidden.delete(key); else view.hidden.add(key); drawPlot(el, res); };
}

function drawSvg(svg, el, cell, res, view, offset) {
  const g0 = plotGeometry(cell, res, view);
  if (!g0) { svg.innerHTML = `<text x="350" y="170" text-anchor="middle" font-size="13" fill="#c0392b">Nothing to draw: all values are undefined on this range.</text>`; svg.onmousemove = null; return; }
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
  const baseY = (!logy && ylo < 0 && yhi > 0) ? sy(0) : (H - mb);
  // shaded regions and bands go under the curves
  for (const a of (res.annotations || [])) {
    if (a.type === "band") {
      g += `<rect x="${sx(a.x0).toFixed(1)}" y="${mt}" width="${(sx(a.x1) - sx(a.x0)).toFixed(1)}" height="${H - mt - mb}" fill="#8a97ab" fill-opacity="0.14" ${clip}/>`;
      if (a.label) g += `<text x="${((sx(a.x0) + sx(a.x1)) / 2).toFixed(1)}" y="${mt + 14}" font-size="11.5" text-anchor="middle" fill="#444">${esc(a.label)}</text>`;
    } else if (a.type === "shade") {
      const s = res.series[a.series]; if (!s) continue;
      const ya = interpSeries(s, a.x0), yb = interpSeries(s, a.x1);
      let d = ya === null ? "" : `M${sx(a.x0).toFixed(1)} ${sy(ya).toFixed(1)} `;
      s.x.forEach((x, i) => { const y = s.y[i]; if (x === null || y === null || x < a.x0 || x > a.x1) return; d += `${d ? "L" : "M"}${sx(x).toFixed(1)} ${sy(y).toFixed(1)} `; });
      if (yb !== null) d += `L${sx(a.x1).toFixed(1)} ${sy(yb).toFixed(1)} `;
      if (!d) continue;
      d += `L${sx(a.x1).toFixed(1)} ${baseY.toFixed(1)} L${sx(a.x0).toFixed(1)} ${baseY.toFixed(1)} Z`;
      g += `<path d="${d}" fill="${PALETTE[(a.series + offset) % PALETTE.length]}" fill-opacity="0.18" stroke="none" ${clip}/>`;
      if (a.label) g += `<text x="${((sx(a.x0) + sx(a.x1)) / 2).toFixed(1)}" y="${(baseY - 6).toFixed(1)}" font-size="11.5" text-anchor="middle" fill="#444">${esc(a.label)}</text>`;
    }
  }
  res.series.forEach((s, k) => {
    if (!SVG2D.has(s.type) || view.hidden.has(s.label_plain)) return;
    const color = PALETTE[(k + offset) % PALETTE.length];
    if (s.type === "points") {
      let h = `<g ${clip}>`;
      const r = s.size || 3.5;
      s.x.forEach((x, i) => {
        const y = s.y[i]; if (x === null || y === null || (logx && x <= 0) || (logy && y <= 0)) return;
        const px = sx(x), py = sy(y);
        if (s.yerr && s.yerr[i]) {
          const y1 = sy(y - s.yerr[i]), y2 = sy(y + s.yerr[i]);
          h += `<line x1="${px.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${px.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${color}" stroke-width="1.2"/>`;
          h += `<line x1="${(px - 4).toFixed(1)}" y1="${y1.toFixed(1)}" x2="${(px + 4).toFixed(1)}" y2="${y1.toFixed(1)}" stroke="${color}" stroke-width="1.2"/><line x1="${(px - 4).toFixed(1)}" y1="${y2.toFixed(1)}" x2="${(px + 4).toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${color}" stroke-width="1.2"/>`;
        }
        if (s.marker === "x") h += `<path d="M${(px - 4.5).toFixed(1)} ${(py - 4.5).toFixed(1)}L${(px + 4.5).toFixed(1)} ${(py + 4.5).toFixed(1)}M${(px - 4.5).toFixed(1)} ${(py + 4.5).toFixed(1)}L${(px + 4.5).toFixed(1)} ${(py - 4.5).toFixed(1)}" stroke="${color}" stroke-width="2" fill="none"/>`;
        else if (s.marker === "o") h += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5" fill="#fff" stroke="${color}" stroke-width="2"/>`;
        else h += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="${r}" fill="${color}"/>`;
        if (s.labels && s.labels[i]) h += `<text x="${(px + 7).toFixed(1)}" y="${(py - 7).toFixed(1)}" font-size="12.5" fill="${color}">${esc(s.labels[i])}</text>`;
      });
      g += h + "</g>";
    } else if (s.type === "segments") {
      let h = `<g ${clip} stroke="${color}" stroke-width="1.4" fill="${color}">`;
      for (const q of s.segments) {
        const x1 = sx(q[0]), y1 = sy(q[1]), x2 = sx(q[2]), y2 = sy(q[3]);
        if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
        h += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`;
        if (s.arrows) {
          const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, ax = x2 - ux * 5, ay = y2 - uy * 5;
          h += `<polygon points="${x2.toFixed(1)},${y2.toFixed(1)} ${(ax - uy * 2.5).toFixed(1)},${(ay + ux * 2.5).toFixed(1)} ${(ax + uy * 2.5).toFixed(1)},${(ay - ux * 2.5).toFixed(1)}" stroke="none"/>`;
        }
      }
      g += h + "</g>";
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
  for (const a of (res.annotations || [])) {
    if (a.type === "hline") {
      const py = sy(a.y); if (!Number.isFinite(py)) continue;
      g += `<line x1="${ml}" y1="${py.toFixed(1)}" x2="${W - mr}" y2="${py.toFixed(1)}" stroke="#555" stroke-dasharray="5 4" ${clip}/>`;
      if (a.label) g += `<text x="${W - mr - 4}" y="${(py - 5).toFixed(1)}" font-size="11.5" text-anchor="end" fill="#444">${esc(a.label)}</text>`;
    } else if (a.type === "vline") {
      const px = sx(a.x); if (!Number.isFinite(px)) continue;
      g += `<line x1="${px.toFixed(1)}" y1="${mt}" x2="${px.toFixed(1)}" y2="${H - mb}" stroke="#555" stroke-dasharray="5 4" ${clip}/>`;
      if (a.label) g += `<text x="${(px + 5).toFixed(1)}" y="${mt + 14}" font-size="11.5" fill="#444">${esc(a.label)}</text>`;
    } else if (a.type === "point") {
      const px = sx(a.x), py = sy(a.y); if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      g += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="4.5" fill="#fff" stroke="#333" stroke-width="1.8"/>`;
      g += `<text x="${(px + 8).toFixed(1)}" y="${(py - 8).toFixed(1)}" font-size="12.5" fill="#222">${esc(a.label || "")}</text>`;
    } else if (a.type === "text") {
      const px = sx(a.x), py = sy(a.y); if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      g += `<text x="${px.toFixed(1)}" y="${py.toFixed(1)}" font-size="12.5" text-anchor="middle" fill="#222">${esc(a.label || "")}</text>`;
    }
  }
  g += `<text x="${(ml + W - mr) / 2}" y="${H - 8}" font-size="12" text-anchor="middle" fill="#444">${esc(res.xlabel || "")}</text>`;
  if (res.ylabel) g += `<text x="14" y="${(mt + H - mb) / 2}" font-size="12" text-anchor="middle" fill="#444" transform="rotate(-90 14 ${(mt + H - mb) / 2})">${esc(res.ylabel)}</text>`;
  g += `<g class="cursor" hidden><line class="vx" y1="${mt}" y2="${H - mb}" stroke="#888" stroke-dasharray="3 3"/><g class="dots"></g></g>`;
  svg.innerHTML = `<defs><clipPath id="clip-${res.id}"><rect x="${ml}" y="${mt}" width="${W - ml - mr}" height="${H - mt - mb}"/></clipPath></defs>${g}`;
  attachPlotInteraction(svg, el, cell, res, g0, view, offset);
}

function attachPlotInteraction(svg, el, cell, res, geo, view, offset) {
  const tip = $(".tip", el);
  const toSvg = ev => { const r = svg.getBoundingClientRect(); return { px: (ev.clientX - r.left) * geo.W / r.width, py: (ev.clientY - r.top) * geo.H / r.height }; };
  const inside = p => p.px >= geo.ml && p.px <= geo.W - geo.mr && p.py >= geo.mt && p.py <= geo.H - geo.mb;
  let drag = null;
  svg.onmousemove = ev => {
    const p = toSvg(ev);
    if (drag) { if (drag.moved || Math.abs(p.px - drag.px) > 3) drag.moved = true; return; }
    const cursor = $(".cursor", svg);
    if (!inside(p) || !res.series.some(s => s.type === "line")) { cursor.hidden = true; tip.hidden = true; return; }
    const x = geo.ix(p.px);
    const rows = [];
    let dots = "";
    res.series.forEach((s, k) => {
      if (s.type !== "line" || view.hidden.has(s.label_plain)) return;
      let best = -1, bd = Infinity;
      for (let i = 0; i < s.x.length; i++) { if (s.x[i] === null || s.y[i] === null) continue; const d = Math.abs(s.x[i] - x); if (d < bd) { bd = d; best = i; } }
      if (best < 0) return;
      const y = s.y[best], color = PALETTE[(k + offset) % PALETTE.length];
      rows.push(`<span class="swatch" style="background:${color}"></span>${esc(s.label_plain)} = <b>${fmtVal(y)}</b>`);
      dots += `<circle cx="${geo.sx(s.x[best])}" cy="${geo.sy(y)}" r="4" fill="${color}" stroke="#fff"/>`;
    });
    if (!rows.length) { cursor.hidden = true; tip.hidden = true; return; }
    cursor.hidden = false;
    $(".vx", cursor).setAttribute("x1", p.px); $(".vx", cursor).setAttribute("x2", p.px);
    $(".dots", cursor).innerHTML = dots;
    tip.hidden = false;
    tip.innerHTML = `<div>${esc(res.var || "x")} = <b>${fmtVal(x)}</b></div>` + rows.map(r => `<div>${r}</div>`).join("");
    const wrap = $(".plot-wrap", el).getBoundingClientRect();
    const left = ev.clientX - wrap.left + 14, top = ev.clientY - wrap.top + 10;
    tip.style.left = Math.min(left, wrap.width - tip.offsetWidth - 8) + "px"; tip.style.top = top + "px";
  };
  svg.onmouseleave = () => { const c = $(".cursor", svg); if (c) c.hidden = true; tip.hidden = true; drag = null; };
  const setRange = (lo, hi) => {
    if (!(PLOT_KINDS[cell.kind] || {}).range) return;
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

// ---------- Plotting (Plotly, loaded on first use: contour, heatmap, surfaces, 3D) ----------
let plotlyLoading = null;
function ensurePlotly() {
  if (window.Plotly) return Promise.resolve();
  if (!plotlyLoading) plotlyLoading = new Promise((resolve, reject) => {
    const sc = document.createElement("script");
    sc.src = "/vendor/plotly.min.js";
    sc.onload = resolve; sc.onerror = () => { plotlyLoading = null; reject(new Error("load failed")); };
    document.head.appendChild(sc);
  });
  return plotlyLoading;
}
function sphereTraces() {
  const x = [], y = [], z = [];
  for (let j = 0; j <= 18; j++) {
    const v = j / 18 * Math.PI, rx = [], ry = [], rz = [];
    for (let i = 0; i <= 36; i++) { const u = i / 36 * 2 * Math.PI; rx.push(Math.sin(v) * Math.cos(u)); ry.push(Math.sin(v) * Math.sin(u)); rz.push(Math.cos(v)); }
    x.push(rx); y.push(ry); z.push(rz);
  }
  const traces = [{ type: "surface", x, y, z, opacity: 0.16, showscale: false, colorscale: [[0, "#9db4d6"], [1, "#9db4d6"]], hoverinfo: "skip",
                    contours: { x: { show: false }, y: { show: false }, z: { show: false } } }];
  const circle = (fx, fy, fz) => { const cx = [], cy = [], cz = []; for (let i = 0; i <= 72; i++) { const t = i / 72 * 2 * Math.PI; cx.push(fx(t)); cy.push(fy(t)); cz.push(fz(t)); } return { type: "scatter3d", mode: "lines", x: cx, y: cy, z: cz, line: { color: "#8a97ab", width: 1.5 }, hoverinfo: "skip" }; };
  traces.push(circle(Math.cos, Math.sin, () => 0), circle(Math.cos, () => 0, Math.sin), circle(() => 0, Math.cos, Math.sin));
  for (const [ax, labels] of [["x", ["|+⟩", "|−⟩"]], ["y", ["|+i⟩", "|−i⟩"]], ["z", ["|0⟩", "|1⟩"]]]) {
    const line = { x: [0, 0], y: [0, 0], z: [0, 0] }; line[ax] = [-1.15, 1.15];
    traces.push({ type: "scatter3d", mode: "lines", ...line, line: { color: "#666", width: 2 }, hoverinfo: "skip" });
    const txt = { x: [0, 0], y: [0, 0], z: [0, 0] }; txt[ax] = [1.3, -1.3];
    traces.push({ type: "scatter3d", mode: "text", ...txt, text: labels, textfont: { size: 13, color: "#333" }, hoverinfo: "skip" });
  }
  return traces;
}
async function drawPlotly(el, cell, res, legend) {
  const box = $(".plotly-box", el); box.hidden = false;
  try { await ensurePlotly(); } catch (e) { $(".err", el).textContent = "The contour/3D renderer (Plotly) could not be loaded."; return; }
  if (state.results.get(cell.id) !== res) return; // superseded while loading
  const traces = []; let three = !!res.three;
  const xl = res.xlabel || "x", yl = res.ylabel || "y";
  const colorbar = { thickness: 12, len: 0.75, title: { text: res.zlabel || "" } };
  res.series.forEach((s, k) => {
    const color = PALETTE[k % PALETTE.length], name = s.label_plain || "";
    if (s.type === "grid") {
      const base = { x: s.x, y: s.y, z: s.z, colorscale: "Viridis", colorbar, name, hovertemplate: `${esc(xl)} = %{x:.4g}<br>${esc(yl)} = %{y:.4g}<br>z = %{z:.4g}<extra></extra>` };
      if (s.style === "surface") { three = true; traces.push({ type: "surface", ...base }); }
      else if (s.style === "heatmap") traces.push({ type: "heatmap", ...base, zsmooth: "best" });
      else {
        const t = { type: "contour", ...base, contours: { coloring: "heatmap", showlabels: true, labelfont: { size: 10, color: "#222" } }, line: { width: 1 } };
        if (Array.isArray(res.levels) && res.levels.length > 1) { const lv = [...res.levels].sort((a, b) => a - b); t.autocontour = false; t.contours.start = lv[0]; t.contours.end = lv[lv.length - 1]; t.contours.size = (lv[lv.length - 1] - lv[0]) / (lv.length - 1); }
        else if (res.levels) { t.autocontour = false; const zs = s.z.flat().filter(v => v !== null); const lo = Math.min(...zs), hi = Math.max(...zs); t.contours.start = lo; t.contours.end = hi; t.contours.size = (hi - lo) / res.levels || 1; }
        traces.push(t);
      }
    } else if (s.type === "line3d") { three = true; traces.push({ type: "scatter3d", mode: "lines", x: s.x, y: s.y, z: s.z, line: { color, width: 4 }, name }); }
    else if (s.type === "points3d") { three = true; traces.push({ type: "scatter3d", mode: "markers", x: s.x, y: s.y, z: s.z, marker: { color, size: 3 }, name }); }
    else if (s.type === "sphere") { three = true; traces.push(...sphereTraces()); }
    else if (s.type === "vector3d") { three = true; traces.push({ type: "scatter3d", mode: "lines+markers", x: [0, s.x], y: [0, s.y], z: [0, s.z], line: { color, width: 7 }, marker: { size: [1, 7], color }, name, hovertemplate: `${esc(name)}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>` }); }
    else if (s.type === "line") traces.push({ type: "scatter", mode: "lines", x: s.x, y: s.y, line: { color, width: 2 }, name });
    else if (s.type === "points") traces.push({ type: "scatter", mode: "markers", x: s.x, y: s.y, marker: { color, size: 6 }, name });
  });
  const layout = { margin: { l: 55, r: 20, t: 10, b: 45 }, height: three ? 460 : 380, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
                   font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif", size: 12 }, showlegend: false, hovermode: "closest" };
  if (three) {
    layout.margin = { l: 0, r: 0, t: 0, b: 0 };
    layout.scene = { xaxis: { title: { text: xl } }, yaxis: { title: { text: yl } }, zaxis: { title: { text: res.zlabel || "z" } }, aspectmode: res.equal ? "cube" : "auto" };
    if (box._camera) layout.scene.camera = box._camera;
  } else {
    layout.xaxis = { title: { text: res.xlabel || "" }, range: res.xrange || undefined, zeroline: false };
    layout.yaxis = { title: { text: res.ylabel || "" }, range: res.yrange || undefined, zeroline: false };
    if (res.equal) layout.yaxis.scaleanchor = "x";
  }
  await Plotly.react(box, traces, layout, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"] });
  if (!box._wired) { box._wired = true; box.on("plotly_relayout", ev => { if (ev["scene.camera"]) box._camera = ev["scene.camera"]; }); }
  legend.innerHTML = res.series.map((s, k) => ({ s, k })).filter(({ s }) => s.label && s.type !== "sphere")
    .map(({ s, k }) => `<span class="lg">${s.type === "grid" ? "" : `<span class="swatch" style="background:${PALETTE[k % PALETTE.length]}"></span>`}${katexHtml(s.label)}</span>`).join("");
  legend.onclick = null;
}

// ---------- Reference panel ----------
async function loadCatalog(reload = false) {
  const r = await fetch(reload ? "/api/reload" : "/api/catalog", { method: reload ? "POST" : "GET" });
  state.catalog = await r.json();
  applyPlotKinds(state.catalog.plot_kinds);
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
  // Fixed order up to Units, then every other section alphabetically.
  const PINNED = ["Syntax", "Interactive", "Constants", "Physical constants", "Units"];
  const ordered = [...PINNED.filter(c => groups.has(c)), ...[...groups.keys()].filter(c => !PINNED.includes(c)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }))];
  let h = "";
  for (const c of ordered) {
    const items = groups.get(c);
    const open = q ? true : !state.refCollapsed.has(c);
    h += `<div class="ref-cat ${open ? "open" : ""}" data-cat="${esc(c)}"><span class="caret">${open ? "▾" : "▸"}</span>${esc(c)} <span class="count">${items.length}</span></div>`;
    if (!open) continue;
    h += `<div class="ref-items">`;
    for (const e of items) {
      const insert = e.example || (e.kind === "function" ? e.signature : e.name);
      const shown = e.kind === "function" || e.kind === "syntax" ? e.signature : e.name;
      h += `<div class="ref-item" data-insert="${esc(insert)}" ${e.kind === "plot" ? `data-plot="${esc(e.example)}"` : ""} title="${e.kind === "plot" ? "Click to add a plot cell of this kind" : "Click to insert"}">
        <span class="sig">${esc(shown)}${e.module !== "core" ? ` <span class="mod">${esc(e.module)}</span>` : ""}</span>
        <span class="doc">${esc(e.doc)}</span></div>`;
    }
    h += `</div>`;
  }
  $("#ref-list").innerHTML = h || `<div class="ref-cat">No matches</div>`;
}

function addPlotCell(kind) {
  const c = newCell("plot"); c.kind = kind;
  const K = PLOT_KINDS[kind];
  if (K) { c.exprs = K.ph1 || ""; c.expr2 = K.ph2 || ""; c.expr3 = K.ph3 || ""; }
  let at = state.cells.length;
  const active = state.activeInput && document.body.contains(state.activeInput) ? state.activeInput.closest(".cell") : null;
  if (active) at = cellIndex(active.dataset.id) + 1;
  state.cells.splice(at, 0, c);
  mountCells(); touch();
  const el = state.els.get(c.id); if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
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

// ---------- Document: title block, export, print, history ----------
function download(name, content, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function docFileName() { return (state.fileName || state.title || "worksheet").replace(/[^\w-]+/g, "_"); }
function today() { const d = new Date(); return d.toISOString().slice(0, 10); }
async function plotImages(cell) {
  const el = state.els.get(cell.id); if (!el) return [];
  const box = $(".plotly-box", el);
  if (box && !box.hidden && window.Plotly) {
    try { return [{ url: await Plotly.toImage(box, { format: "png", width: 900, height: 520 }), png: true }]; } catch (e) { return []; }
  }
  return [...el.querySelectorAll(".panels svg")].map(svg => {
    const src = svg.outerHTML.replace("<svg ", `<svg xmlns="http://www.w3.org/2000/svg" `);
    return { url: "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(src))), svg: src };
  });
}
const EXPORT_CSS = `body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#1f2328;margin:0;background:#fff}
article{max-width:820px;margin:0 auto;padding:40px 24px 80px;line-height:1.55}h1.title{font-size:30px;margin:0 0 4px}.meta{color:#6b7280;font-size:14px;margin-bottom:28px}
.text h1{font-size:24px}.text h2{font-size:20px}.text h3{font-size:16.5px}.text table{border-collapse:collapse;margin:8px 0}.text th,.text td{border:1px solid #e3e1db;padding:4px 10px}
.text blockquote{border-left:3px solid #e3e1db;margin:8px 0;padding:2px 12px;color:#4b5563}.text img{max-width:100%}.text code{font-family:ui-monospace,Menlo,monospace;background:#f0efea;padding:1px 4px;border-radius:4px}
.footnotes{font-size:13px;color:#4b5563;border-top:1px solid #e3e1db;margin-top:12px}.dmath{margin:8px 0}
.math{margin:10px 0 14px;padding-left:12px;border-left:3px solid #2f6fed}.math pre.src{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;margin:0 0 4px;white-space:pre-wrap;color:#374151}
.out{font-size:17px}.out-line{margin:2px 0 6px;display:flex;flex-wrap:wrap;align-items:baseline;gap:14px}.approx{color:#6b7280;font-size:.9em}.note{color:#b7791f;font-size:12px}.err{color:#c0392b;font-size:13px}
.reading{color:#6b7280;font-size:14px}.reading .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-right:6px}
.steps{margin:4px 0 8px;padding:6px 10px;border-left:2px solid #e3e1db;font-size:14px}.step{display:flex;flex-wrap:wrap;gap:8px;margin:3px 0}.step-n{color:#6b7280;font-size:12px;min-width:18px}
.plot{margin:10px 0 14px;padding-left:12px;border-left:3px solid #1f9d6f}.plot img,.plot svg{max-width:100%;width:700px;border:1px solid #e3e1db;border-radius:6px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:14px}.legend .swatch{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
.val{padding:0 2px}.val.err{color:#c0392b}footer{color:#9ca3af;font-size:12px;margin-top:40px}`;
async function exportHtml() {
  const parts = [];
  for (const c of state.cells) {
    const el = state.els.get(c.id), res = state.results.get(c.id) || {};
    if (c.type === "text") parts.push(`<section class="text">${markdown(c.source || "", { cellId: c.id, values: res.values || null })}</section>`);
    else if (c.type === "math") {
      let out = "";
      if (el) { const clone = $(".out", el).cloneNode(true); clone.querySelectorAll(".slider-row").forEach(n => n.remove()); out = clone.innerHTML; }
      parts.push(`<section class="math"><pre class="src">${esc(c.source || "")}</pre><div class="out">${out}</div>${res.error ? `<div class="err">${esc(res.error)}</div>` : ""}</section>`);
    } else {
      const imgs = await plotImages(c);
      const legend = el ? $(".legend", el).innerHTML : "";
      parts.push(`<section class="plot">${imgs.map(i => i.svg || `<img src="${i.url}" alt="plot">`).join("")}<div class="legend">${legend}</div></section>`);
    }
  }
  const meta = [state.author, state.savedAt || today()].filter(Boolean).join(" · ");
  const html = `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(state.title)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"><style>${EXPORT_CSS}</style></head>
<body><article><h1 class="title">${esc(state.title)}</h1><div class="meta">${esc(meta)}</div>\n${parts.join("\n")}\n<footer>Made with Quire</footer></article></body></html>`;
  download(docFileName() + ".html", html, "text/html");
}
async function exportMarkdown() {
  const meta = [state.author, state.savedAt || today()].filter(Boolean).join(" · ");
  const lines = [`# ${state.title}`, meta ? `*${meta}*` : "", ""];
  for (const c of state.cells) {
    const res = state.results.get(c.id) || {};
    if (c.type === "text") {
      let k = 0;
      lines.push((c.source || "").replace(/\{\{[^}]*\}\}/g, m => { const v = (res.values || [])[k++]; return v && !v.error ? `$${v.latex}$` : m; }), "");
    } else if (c.type === "math") {
      lines.push("```quire", c.source || "", "```");
      for (const o of res.outputs || []) if (o.latex) lines.push(`$$${o.head ? o.head + " = " : ""}${o.latex}${o.approx ? " \\approx " + o.approx : ""}$$`);
      if (res.error) lines.push(`> Error: ${res.error}`);
      lines.push("");
    } else {
      const imgs = await plotImages(c);
      const K = PLOT_KINDS[c.kind] || {};
      const alt = `${K.label || c.kind}: ${c.exprs || ""}`;
      lines.push(...(imgs.length ? imgs.map(i => `![${alt}](${i.url})`) : [`*(plot, ${alt})*`]), "");
    }
  }
  download(docFileName() + ".md", lines.join("\n"), "text/markdown");
}
function cellText(c) {
  if (c.type !== "plot") return c.source || "";
  const bits = [`${c.kind}: ${c.exprs || ""}`];
  if (c.expr2) bits.push(c.expr2); if (c.expr3) bits.push(c.expr3);
  if (c.var) bits.push(`var ${c.var}`);
  if (c.xmin || c.xmax) bits.push(`${c.xmin || ""}..${c.xmax || ""}`);
  if (c.ymin || c.ymax) bits.push(`y ${c.ymin || ""}..${c.ymax || ""}`);
  if (c.annot) bits.push(`annotate ${c.annot}`);
  return bits.join("\n");
}
function lineDiff(a, b) {
  const A = a.split("\n"), B = b.split("\n"), n = A.length, m = B.length;
  const L = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push([" ", A[i]]); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) { out.push(["-", A[i]]); i++; }
    else { out.push(["+", B[j]]); j++; }
  }
  while (i < n) out.push(["-", A[i++]]);
  while (j < m) out.push(["+", B[j++]]);
  return out;
}
function diffHtml(oldDoc, newDoc) {
  const oldMap = new Map((oldDoc.cells || []).map(c => [c.id, c])), newIds = new Set((newDoc.cells || []).map(c => c.id));
  const rows = [];
  for (const c of newDoc.cells || []) {
    const o = oldMap.get(c.id);
    if (!o) rows.push({ st: "added", text: lineDiff("", cellText(c)), type: c.type });
    else if (cellText(o) !== cellText(c)) rows.push({ st: "changed", text: lineDiff(cellText(o), cellText(c)), type: c.type });
  }
  for (const c of oldDoc.cells || []) if (!newIds.has(c.id)) rows.push({ st: "removed", text: lineDiff(cellText(c), ""), type: c.type });
  if (oldDoc.title !== newDoc.title) rows.unshift({ st: "changed", text: lineDiff(`title: ${oldDoc.title || ""}`, `title: ${newDoc.title || ""}`), type: "document" });
  if (!rows.length) return `<div class="hint">No differences in the cells.</div>`;
  return rows.map(r => `<div class="dcell ${r.st}"><div class="dhead">${r.type} cell · ${r.st}</div><pre class="diff">${r.text.map(([k, l]) => `<span class="${k === "+" ? "add" : k === "-" ? "del" : "same"}">${k} ${esc(l)}</span>`).join("")}</pre></div>`).join("");
}
async function documentDialog() {
  modal("Document", `
    <label class="row"><span>Author</span><input id="doc-author" class="name" value="${esc(state.author || "")}" placeholder="shown in the title block and exports"></label>
    <div class="hint">${state.fileName ? `Saved as <b>${esc(state.fileName)}</b>${state.savedAt ? " on " + esc(state.savedAt) : ""}` : "Not saved yet: save it to start a version history."}</div>
    <div class="doc-actions">
      <button data-do="html" title="A single HTML file with the math rendered and the plots as pictures">Export HTML</button>
      <button data-do="md" title="Markdown with LaTeX math; plots embedded as images">Export Markdown</button>
      <button data-do="print" title="Print, or save as PDF from the print dialog">Print / PDF</button>
    </div>
    <h4>History</h4><div id="doc-history" class="hint">${state.fileName ? "Loading…" : ""}</div><div id="doc-diff"></div>`);
  const author = $("#doc-author");
  author.addEventListener("input", () => { state.author = author.value; setDirty(true); renderTitleBlock(); });
  $("#modal-body").onclick = async ev => {
    const b = ev.target.closest("button"); if (!b) return;
    if (b.dataset.do === "html") exportHtml();
    if (b.dataset.do === "md") exportMarkdown();
    if (b.dataset.do === "print") { closeModal(); setTimeout(() => window.print(), 150); }
    if (b.dataset.diff) {
      try {
        const { doc } = await api("/api/version", { name: state.fileName, stamp: b.dataset.diff });
        $("#doc-diff").innerHTML = `<div class="hint">Changes since ${esc(b.dataset.time)} (− then, + now):</div>` + diffHtml(doc, serialize());
      } catch (e) { alert(e.message); }
    }
    if (b.dataset.restore) {
      if (!confirm(`Load the version saved ${b.dataset.time}? The current cells are replaced (nothing is written until you save).`)) return;
      try {
        const { doc } = await api("/api/version", { name: state.fileName, stamp: b.dataset.restore });
        const name = state.fileName;
        closeModal(); loadDoc(doc, name); setDirty(true);
      } catch (e) { alert(e.message); }
    }
  };
  if (!state.fileName) return;
  try {
    const r = await fetch("/api/history?name=" + encodeURIComponent(state.fileName)); const h = await r.json();
    const list = $("#doc-history"); if (!list) return;
    list.innerHTML = h.versions && h.versions.length
      ? h.versions.map(v => `<div class="ver"><span>${esc(v.time)}</span><span class="hint">${v.cells == null ? "" : v.cells + " cells"}</span><button data-diff="${esc(v.stamp)}" data-time="${esc(v.time)}">compare</button><button data-restore="${esc(v.stamp)}" data-time="${esc(v.time)}">restore</button></div>`).join("")
      : "No earlier versions yet: each save that changes the worksheet keeps the previous one here.";
  } catch (e) { /* ignore */ }
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
  $(".modal-box").classList.toggle("wide", title === "Document");
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
    state.fileName = r.saved; state.savedAt = r.saved_at || state.savedAt; setDirty(false); renderTitleBlock();
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

// ---------- Palette ----------
const PALETTE_ITEMS = [
  ["∫", "∫_{a}^{b} f dx", "definite integral"], ["∫ dx", "∫ f dx", "antiderivative"], ["Σ", "Σ_{k=1}^{n} a_k", "sum"], ["∏", "∏_{k=1}^{n} a_k", "product"],
  ["d/dx", "d/dx (f)", "derivative"], ["d²/dx²", "d²/dx² (f)", "second derivative"], ["√", "√(x)", "square root"], ["x²", "^2", "square"], ["xⁿ", "^(n)", "power"],
  ["a/b", "(a)/(b)", "fraction"], ["lim", "limit(f, x, 0)", "limit"], ["series", "series(f, x, 0, 6)", "Taylor series"],
  ["[ ]", "matrix([[a, b], [c, d]])", "matrix"], ["→", " -> ", "unit conversion"], ["==", " == ", "equation"], ["π", "π", ""], ["∞", "∞", ""], ["≤", " ≤ ", ""], ["≥", " ≥ ", ""], ["≠", " ≠ ", ""], ["×", " × ", ""], ["°", " deg", "degrees"],
];
const GREEK_LETTERS = "α β γ δ ε ζ η θ κ λ μ ν ξ ρ σ τ φ χ ψ ω Γ Δ Θ Λ Σ Φ Ψ Ω".split(" ");
function buildPalette() {
  const el = $("#palette");
  el.innerHTML = PALETTE_ITEMS.map(([label, text, title]) => `<button data-insert="${esc(text)}" title="${esc(title || label)}">${esc(label)}</button>`).join("")
    + `<span class="sep"></span>` + GREEK_LETTERS.map(g => `<button class="greek" data-insert="${g}">${g}</button>`).join("");
  el.addEventListener("click", ev => { const b = ev.target.closest("button[data-insert]"); if (b) insertTemplate(b.dataset.insert); });
}
function insertTemplate(text) {
  let inp = state.activeInput;
  if (!inp || !document.body.contains(inp)) { const c = insertCell("math", state.cells.length, true); inp = $("textarea", state.els.get(c.id)); }
  const a = inp.selectionStart ?? inp.value.length, b = inp.selectionEnd ?? a;
  inp.value = inp.value.slice(0, a) + text + inp.value.slice(b);
  // select the first placeholder so typing replaces it
  const m = /\b(f|a_k|x|n|a|b|c|d)\b/.exec(text);
  if (m && !/^[^A-Za-z]*$/.test(text)) inp.setSelectionRange(a + m.index, a + m.index + m[0].length);
  else inp.setSelectionRange(a + text.length, a + text.length);
  inp.dispatchEvent(new Event("input"));
  inp.focus();
}

// ---------- Autocomplete ----------
let acBox = null, acItems = [], acIndex = 0, acInput = null, acRange = null;
function acClose() { if (acBox) { acBox.remove(); acBox = null; } acItems = []; acInput = null; }
function acCandidates(prefix, afterArrow) {
  if (!state.catalog) return [];
  const p = prefix.toLowerCase();
  const pool = state.catalog.entries.filter(e => afterArrow ? e.kind === "unit" : (e.kind !== "syntax" && e.kind !== "plot"));
  const starts = pool.filter(e => e.name.toLowerCase().startsWith(p));
  const contains = pool.filter(e => !e.name.toLowerCase().startsWith(p) && (e.name.toLowerCase().includes(p) || e.doc.toLowerCase().includes(p)));
  return [...starts, ...contains].slice(0, 8);
}
function acShow(inp) {
  const pos = inp.selectionStart, before = inp.value.slice(0, pos);
  const arrow = /->\s*([A-Za-z_][\w/^ ]*)$/.exec(before);
  const word = /([A-Za-z_][A-Za-z0-9_]*)$/.exec(before);
  const prefix = arrow ? arrow[1].trim() : (word ? word[1] : "");
  if (prefix.length < 2 || (!arrow && /^\d/.test(prefix))) { acClose(); return; }
  const items = acCandidates(prefix, !!arrow);
  if (!items.length) { acClose(); return; }
  acItems = items; acIndex = 0; acInput = inp; acRange = [pos - prefix.length, pos];
  if (!acBox) { acBox = document.createElement("div"); acBox.className = "ac"; document.body.appendChild(acBox); }
  acRender();
  const r = inp.getBoundingClientRect();
  acBox.style.left = (r.left + window.scrollX + 12) + "px";
  acBox.style.top = (r.bottom + window.scrollY + 2) + "px";
}
function acRender() {
  acBox.innerHTML = acItems.map((e, i) => `<div class="${i === acIndex ? "sel" : ""}" data-i="${i}"><span class="sig">${esc(e.kind === "function" ? e.signature : e.name)}</span><span class="doc">${esc(e.doc)}</span></div>`).join("");
  acBox.onmousedown = ev => { const d = ev.target.closest("[data-i]"); if (d) { ev.preventDefault(); acAccept(+d.dataset.i); } };
}
function acAccept(i) {
  const e = acItems[i]; if (!e || !acInput) return;
  let text = e.name;
  if (e.kind === "function") { const m = /\((.*)\)/.exec(e.signature); text = e.name + "(" + (m ? m[1] : "") + ")"; }
  const inp = acInput, [a, b] = acRange;
  inp.value = inp.value.slice(0, a) + text + inp.value.slice(b);
  const m2 = e.kind === "function" ? /\(/.exec(text) : null;
  const caret = m2 ? a + m2.index + 1 : a + text.length;
  const argEnd = m2 ? a + text.length - 1 : caret;
  inp.setSelectionRange(caret, argEnd);
  inp.dispatchEvent(new Event("input"));
  acClose();
  inp.focus();
}
function acKey(ev) {
  if (!acBox || acInput !== ev.target) return false;
  if (ev.key === "ArrowDown") { acIndex = (acIndex + 1) % acItems.length; acRender(); ev.preventDefault(); return true; }
  if (ev.key === "ArrowUp") { acIndex = (acIndex - 1 + acItems.length) % acItems.length; acRender(); ev.preventDefault(); return true; }
  if (ev.key === "Tab" || (ev.key === "Enter" && !ev.shiftKey)) { acAccept(acIndex); ev.preventDefault(); return true; }
  if (ev.key === "Escape") { acClose(); ev.preventDefault(); return true; }
  return false;
}

// ---------- Wiring ----------
function init() {
  $("#title").addEventListener("input", ev => { state.title = ev.target.value; setDirty(true); renderTitleBlock(); });
  $("#btn-doc").onclick = documentDialog;
  $("#titleblock").addEventListener("click", documentDialog);
  $("#btn-new").onclick = () => { if (!state.dirty || confirm("Discard unsaved changes?")) loadDoc({ title: "Untitled", cells: [{ type: "math", source: "" }] }); };
  $("#btn-open").onclick = () => openDialog(false);
  $("#btn-examples").onclick = () => openDialog(true);
  $("#btn-save").onclick = () => save(false);
  $("#btn-saveas").onclick = () => save(true);
  $("#btn-data").onclick = () => $("#data-file").click();
  $("#data-file").addEventListener("change", async ev => {
    const file = ev.target.files[0]; if (!file) return;
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf); let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    try {
      const r = await api("/api/upload", { filename: file.name, content: btoa(bin) });
      if (r.image) {
        const snippet = `![${r.name}](${r.file})`;
        const inp = state.activeInput;
        if (inp && document.body.contains(inp) && inp.closest(".cell.text")) insertAtCursor(snippet);
        status(`Uploaded image <b>${esc(r.file)}</b>: write <code>${esc(snippet)}</code> in a text cell`);
      } else {
        status(`Uploaded as data file <b>${esc(r.name)}</b>: use read_csv(${esc(r.name)}) or column(${esc(r.name)}, header)`);
        evaluateNow();
      }
    } catch (e) { alert(e.message); }
    ev.target.value = "";
  });
  $("#btn-ref").onclick = () => { $("#ref").classList.toggle("hidden"); $("#btn-ref").classList.toggle("on"); };
  buildPalette();
  $("#btn-palette").onclick = () => { $("#palette").classList.toggle("hidden"); $("#btn-palette").classList.toggle("on"); };
  $("#btn-reload").onclick = () => loadCatalog(true);
  $("#ref-search").addEventListener("input", renderCatalog);
  $("#ref-list").addEventListener("click", ev => {
    const head = ev.target.closest(".ref-cat[data-cat]");
    if (head) {
      const c = head.dataset.cat;
      if (state.refCollapsed.has(c)) state.refCollapsed.delete(c); else state.refCollapsed.add(c);
      saveCollapsed(state.refCollapsed); renderCatalog(); return;
    }
    const it = ev.target.closest(".ref-item"); if (!it) return;
    if (it.dataset.plot) addPlotCell(it.dataset.plot); else insertAtCursor(it.dataset.insert);
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
