/* Visual Forensic AI — shared front-end code.
   Every number on every page comes from /api/* (the trained model and measurements on the
   uploaded pixels). Nothing here computes a verdict, and nothing is hard-coded. */
const VFA = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  /* ------------------------------------------------------------------ theme */
  const THEME_KEY = 'vfa-theme';
  const systemTheme = () => (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const applyTheme = (t) => {
    document.documentElement.dataset.theme = t;
    $$('[data-theme-btn]').forEach((b) => {
      b.textContent = t === 'dark' ? '☀' : '☾';
      b.setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
      b.title = `${t === 'dark' ? 'light' : 'dark'} theme (stored as ${THEME_KEY})`;
    });
  };
  let themeWired = false;
  const initTheme = () => {
    if (themeWired) return;                       // idempotent: boot() and DOMContentLoaded both call it
    themeWired = true;
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch (e) { /* private mode */ }
    applyTheme(stored === 'dark' || stored === 'light' ? stored : systemTheme());
    $$('[data-theme-btn]').forEach((b) => b.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
    }));
    if (window.matchMedia) {
      matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        let s = null; try { s = localStorage.getItem(THEME_KEY); } catch (err) { /* ignore */ }
        if (s !== 'dark' && s !== 'light') applyTheme(e.matches ? 'dark' : 'light');
      });
    }
  };

  /* ------------------------------------------------------------------ helpers */
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (v, n = 3) => (v == null || v === '' || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(n));
  const fmt2 = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(3))
    : typeof v === 'boolean' ? (v ? 'yes' : 'no') : esc(String(v ?? '—')));
  const pct = (v, n = 1) => (v == null || Number.isNaN(Number(v)) ? '—' : `${(Number(v) * 100).toFixed(n)}%`);
  const bytes = (n) => (n == null ? '—' : n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`);

  async function api(path, opts) {
    const r = await fetch(path, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      const err = new Error((body && (body.detail || body.error)) || `HTTP ${r.status}`);
      err.status = r.status; err.body = body;
      throw err;
    }
    return body;
  }

  /* --------------------------------------------------------------- rendering */
  const pill = (label, kind = '', value) =>
    `<span class="pill ${kind}">${label}${value != null ? ` <b>${esc(value)}</b>` : ''}</span>`;

  /** Table builder. `opts.num` lists the right-aligned numeric column indexes. A cell may be
      a string/number (HTML-escaped here, so no report or response can inject markup),
      {raw} for markup this file built itself, {n, d} for a number with d decimals, or null. */
  function cell(c, isNum) {
    if (c === null || c === undefined) return '<span class="hint">n/a</span>';
    if (typeof c === 'object' && !Array.isArray(c)) {
      if (c.raw !== undefined) return c.raw;
      if (c.n !== undefined) return fmt(c.n, c.d == null ? 3 : c.d);
      if (c.text !== undefined) return esc(String(c.text));
    }
    if (typeof c === 'number') return Number.isInteger(c) ? String(c) : fmt(c, isNum ? 4 : 3);
    if (typeof c === 'boolean') return c ? 'yes' : 'no';
    if (Array.isArray(c)) return esc(c.join(' × '));
    return esc(String(c));
  }

  function table(headers, rows, opts = {}) {
    const align = opts.num || [];
    const body = rows || [];
    if (!body.length) return `<p class="hint">${esc(opts.empty || 'nothing to show')}</p>`;
    const head = headers.map((h, i) => `<th class="${align.includes(i) ? 'num' : ''}">${esc(h)}</th>`).join('');
    const tr = body.map((r) => {
      const cells = (r.__row || r).map((c, i) =>
        `<td class="${align.includes(i) ? 'num' : ''}">${cell(c, align.includes(i))}</td>`).join('');
      const cls = r.__class || '';
      return `<tr${cls ? ` class="${esc(cls)}"` : ''}>${cells}</tr>`;
    }).join('');
    const cap = opts.caption ? `<caption>${esc(opts.caption)}</caption>` : '';
    return `<div class="tblwrap"><table class="tbl">${cap}<thead><tr>${head}</tr></thead><tbody>${tr}</tbody></table></div>`;
  }

  function kvTable(obj, skip = []) {
    const rows = Object.entries(obj || {}).filter(([k]) => !skip.includes(k));
    if (!rows.length) return '<p class="hint">no data</p>';
    return `<table class="kv"><tbody>${rows.map(([k, v]) => {
      let cell;
      if (v == null) cell = '<span class="hint">n/a</span>';
      else if (Array.isArray(v)) cell = v.length > 8 ? `${v.length} values` : v.map((x) => typeof x === 'object' ? esc(JSON.stringify(x)) : esc(String(x))).join(' × ');
      else if (typeof v === 'object') cell = esc(JSON.stringify(v));
      else if (typeof v === 'number') cell = `<span class="num">${fmt2(v)}</span>`;
      else if (typeof v === 'boolean') cell = v ? 'yes' : 'no';
      else cell = esc(String(v));
      return `<tr><th>${esc(k)}</th><td>${cell}</td></tr>`;
    }).join('')}</tbody></table>`;
  }

  const details = (summary, inner, open = false) =>
    `<details class="raw"${open ? ' open' : ''}><summary>${esc(summary)}</summary>${inner}</details>`;

  /* Minimal, injection-safe markdown renderer for the report pages: text is HTML-escaped
     first and only the tags below are added, so a report cannot inject markup. Figures are
     fetched through /api/report/<basename>, which the server resolves inside reports/figures/. */
  function mdLite(src) {
    const lines = String(src || '').replace(/\r/g, '').split('\n');
    const out = [];
    let para = [], list = [], table_ = null, code = null;
    const inline = (t) => esc(t)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
      .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (m, alt, href) => {
        const f = String(href).split('/').pop();
        return `<img alt="${alt}" loading="lazy" src="/api/report/${encodeURIComponent(f)}" title="${href}">`;
      })
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|\/[^)\s]*)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; } };
    const flushList = () => { if (list.length) { out.push(`<ul>${list.map((i) => `<li>${inline(i)}</li>`).join('')}</ul>`); list = []; } };
    const flushTable = () => {
      if (!table_) return;
      const head = table_[0], body = table_.slice(1);
      out.push('<div class="tblwrap"><table class="tbl"><thead><tr>' + head.map((c) => `<th>${inline(c)}</th>`).join('') +
        '</tr></thead><tbody>' + body.map((row) => `<tr>${row.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('') +
        '</tbody></table></div>');
      table_ = null;
    };
    const cells = (l) => l.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());
    for (const raw of lines) {
      const l = raw.replace(/\s+$/, '');
      if (code !== null) {
        if (/^```/.test(l)) { out.push(`<pre><code>${esc(code.join('\n'))}</code></pre>`); code = null; } else code.push(l);
        continue;
      }
      if (/^```/.test(l)) { flushPara(); flushList(); flushTable(); code = []; continue; }
      if (/^\s*\|.*\|\s*$/.test(l)) {
        if (/^\s*\|[\s:|-]+\|\s*$/.test(l)) continue;
        flushPara(); flushList();
        if (!table_) table_ = [];
        table_.push(cells(l));
        continue;
      }
      flushTable();
      if (!l.trim()) { flushPara(); flushList(); continue; }
      const h = /^(#{1,6})\s+(.*)$/.exec(l);
      if (h) { flushPara(); flushList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
      if (/^(-{3,}|\*{3,})$/.test(l.trim())) { flushPara(); flushList(); out.push('<hr>'); continue; }
      const li = /^\s*[-*]\s+(.*)$/.exec(l);
      if (li) { flushPara(); list.push(li[1]); continue; }
      if (/^>\s?/.test(l)) { flushPara(); flushList(); out.push(`<blockquote>${inline(l.replace(/^>\s?/, ''))}</blockquote>`); continue; }
      para.push(l);
    }
    if (code !== null) out.push(`<pre><code>${esc(code.join('\n'))}</code></pre>`);
    flushPara(); flushList(); flushTable();
    return out.join('');
  }

  /* ---------------------------------------------------------------- shared state */
  let cache = null;
  async function load() {
    if (cache) return cache;
    const out = { health: null, model: null, error: null };
    try { out.health = await api('/api/health'); } catch (e) { out.error = String(e.message || e); }
    try { out.model = await api('/api/model'); } catch (e) {
      out.error = e.status === 503 ? e.message : (out.error || String(e.message || e));
    }
    cache = out;
    return out;
  }

  /** Model status pills, shown on every page so the state of the deployment is never hidden. */
  function statusPills(s) {
    if (!s.model) return pill('no trained model loaded', 'bad') + ' ' + pill(s.error || 'run `python train.py` once', 'warn');
    const m = s.model.model || {};
    const met = m.metrics || {};
    const g = (s.model.generator_classifier || {}).task || {};
    return [
      pill('model', 'accent', m.architecture || '?'),
      pill('test F1', '', fmt(met.f1, 4)),
      pill('test ROC-AUC', '', fmt(met.roc_auc, 4)),
      pill('threshold', '', fmt(met.threshold ?? s.model.threshold, 3)),
      pill('speed', '', `${fmt(m.inference_ms_per_image, 1)} ms · CPU`),
      g.trained ? pill('generator ID', 'ok', `${(g.classes || []).length} classes`)
        : pill('generator ID', '', 'not trained (no labels)'),
    ].join(' ');
  }

  function decorate(s) {
    $$('[data-status-pills]').forEach((el) => { el.innerHTML = statusPills(s); });
    $$('[data-model-name]').forEach((el) => {
      const m = (s.model && s.model.model) || {};
      el.textContent = m.model_name ? `${m.model_name} · v${m.model_version || '?'} · trained ${m.training_date || '?'}` : 'no model loaded';
    });
    const badge = $('#headStatus');
    if (badge) badge.innerHTML = s.model ? '<span class="pill ok">model ready</span>' : '<span class="pill bad">model missing</span>';
  }

  async function boot(render) {
    initTheme();
    const nav = $('.nav');
    if (nav) {
      const here = location.pathname.replace(/\/$/, '') || '/';
      $$('a', nav).forEach((a) => { if (a.getAttribute('href') === here) a.setAttribute('aria-current', 'page'); });
    }
    const s = await load();
    decorate(s);
    await render(s);
    return s;
  }

  document.addEventListener('DOMContentLoaded', () => initTheme());
  return { $, $$, esc, fmt, fmt2, pct, bytes, api, pill, table, kvTable, details, mdLite, load, decorate, statusPills, boot, applyTheme, THEME_KEY };
})();
