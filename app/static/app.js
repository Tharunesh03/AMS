/* AMS AI Image Detector - front end.
   All numbers shown here come from /api/* (trained model + measured signals). Nothing is
   computed client-side except layout maths, and no verdict is ever produced in the browser. */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = { file: null, dataUrl: null, model: null, health: null };

/* ---------------------------------------------------------------- boot */
async function boot() {
  try {
    state.health = await (await fetch('/api/health')).json();
  } catch (e) { state.health = { status: 'unreachable' }; }
  try {
    const r = await fetch('/api/model');
    if (r.ok) state.model = await r.json();
    else { const e = await r.json().catch(() => ({})); renderNoModel(e.detail || 'model unavailable'); }
  } catch (e) { renderNoModel(String(e)); }
  if (state.model) { renderBadges(); renderModel(); renderSamples(); renderDataNote(); }
  wire();
}

function renderNoModel(msg) {
  $('#modelBadges').innerHTML = `<span class="badge bad">no trained model</span>`;
  $('#empty').innerHTML = `<h2>Model not trained yet</h2><p>${esc(msg)}</p>
    <p class="hint">Run <code>python train.py</code> once (it inspects the dataset, trains and compares the supervised
    candidates, then stores the selected model in <code>models/</code>). This service only loads that artifact -
    it never retrains at request time.</p>`;
}

function renderBadges() {
  const m = state.model.model || {}, sel = state.model.selection || {}, cmp = state.model.comparison || [];
  const met = m.metrics || {};
  const badges = [
    `<span class="badge ok"><b>${esc(m.architecture || '?')}</b> · selected</span>`,
    met.f1 != null ? `<span class="badge">test F1 <b>${fmt(met.f1, 3)}</b></span>` : '',
    met.roc_auc != null ? `<span class="badge">ROC-AUC <b>${fmt(met.roc_auc, 3)}</b></span>` : '',
    met.accuracy != null ? `<span class="badge">accuracy <b>${fmt(met.accuracy, 3)}</b></span>` : '',
    `<span class="badge">${(state.model.comparison || []).length} candidates compared</span>`,
    `<span class="badge">CPU · no external AI API</span>`,
    (state.model.generator_classifier || {}).task && (state.model.generator_classifier || {}).task.trained
      ? `<span class="badge ok">generator classifier ready (${(state.model.generator_classifier.task.classes || []).length} classes)</span>`
      : `<span class="badge">generator ID: n/a (no labels in the dataset)</span>`,
    (m.transfer_learning
      ? ((m.transfer_learning.weights_applied_for || []).length
          ? `<span class="badge ok">ImageNet transfer on ${m.transfer_learning.weights_applied_for.length}/${m.transfer_learning.deep_candidates} deep candidates</span>`
          : `<span class="badge">transfer requested · weights unreachable · trained from scratch</span>`)
      : (m.pretrained_backbone === false
          ? `<span class="badge">trained from scratch (weights unreachable)</span>`
          : m.pretrained_backbone ? `<span class="badge ok">ImageNet transfer</span>` : '')),
  ];
  $('#modelBadges').innerHTML = badges.filter(Boolean).join('');
  $('#footerModel').textContent = `${m.model_name || ''} · v${m.model_version || ''} · trained ${m.training_date || '?'}`;
  if (state.health && state.health.max_upload_bytes) {
    $('#maxSize').textContent = Math.round(state.health.max_upload_bytes / 1024 / 1024);
  }
}

/* ---------------------------------------------------------------- wiring */
function wire() {
  const drop = $('#drop'), input = $('#file');
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) setFile(f); });
  input.addEventListener('change', () => { if (input.files[0]) setFile(input.files[0]); });
  window.addEventListener('paste', (e) => {
    const it = Array.from(e.clipboardData?.items || []).find((i) => i.type.startsWith('image/'));
    if (it) { const f = it.getAsFile(); if (f) setFile(f, 'pasted image'); }
  });
  $('#run').addEventListener('click', run);
  $('#clear').addEventListener('click', () => { setFile(null); $('#result').hidden = true; $('#empty').hidden = false; });
  $$('.tabs').forEach((nav) => nav.addEventListener('click', (e) => {
    const b = e.target.closest('.tab'); if (!b) return;
    const group = b.parentElement;
    Array.from(group.children).forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    const host = group.parentElement;
    const key = b.dataset.tab || b.dataset.tab2;
    Array.from(host.querySelectorAll('.tabpane')).forEach((p) => {
      p.classList.toggle('active', p.id === (b.dataset.tab ? 'tab-' : 'pane-') + key);
    });
  }));
  $('#honesty').innerHTML = (state.model?.honesty || ['Prediction is probabilistic and is not proof of image origin.'])
    .map((h) => `<li>${esc(h)}</li>`).join('');
  renderReports();
}

function setFile(f, label) {
  state.file = f;
  $('#run').disabled = !f;
  $('#clear').disabled = !f;
  if (!f) { $('#previewWrap').hidden = true; $('#drop').querySelector('.drop-inner').hidden = false; $('#status').textContent = ''; return; }
  const url = URL.createObjectURL(f);
  state.dataUrl = url;
  $('#preview').src = url;
  $('#previewWrap').hidden = false;
  $('#drop').querySelector('.drop-inner').hidden = true;
  $('#status').textContent = label || `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
  $('#status').classList.remove('err');
}

async function run() {
  if (!state.file) return;
  $('#run').disabled = true;
  $('#status').textContent = 'analysing on CPU…';
  $('#status').classList.remove('err');
  const fd = new FormData();
  fd.append('file', state.file, state.file.name || 'upload.png');
  let res, body;
  const t0 = performance.now();
  try {
    res = await fetch('/api/detect', { method: 'POST', body: fd });
    body = await res.json();
  } catch (e) {
    $('#status').textContent = 'request failed: ' + e; $('#status').classList.add('err'); $('#run').disabled = false; return;
  }
  $('#run').disabled = false;
  if (!res.ok) { $('#status').textContent = `error ${res.status}: ${(body && body.detail) || 'failed'}`; $('#status').classList.add('err'); return; }
  const serverMs = body.timing_ms ?? '?';
  const totalMs = Math.round(performance.now() - t0);
  $('#status').textContent = `done · ${serverMs} ms inference (round trip ${totalMs} ms)`;
  renderResult(body);
}

/* ---------------------------------------------------------------- results */
function renderResult(r) {
  $('#empty').hidden = true;
  $('#result').hidden = false;
  const synth = r.predicted_class_id === 1;
  const v = $('#verdict');
  v.className = 'verdict ' + (synth ? 'synthetic' : 'authentic');
  v.innerHTML = `${esc(r.verdict)}<small>class label <code>${esc(r.predicted_class)}</code> · probability of being generated
     <b>${fmt(r.probability_synthetic, 3)}</b> vs threshold ${fmt(r.threshold, 3)}</small>`;
  const p = Math.max(0, Math.min(1, r.probability_synthetic));
  $('#probFill').style.left = (p * 100).toFixed(2) + '%';
  $('#thrMark').style.left = (Math.max(0, Math.min(1, r.threshold)) * 100).toFixed(2) + '%';
  $('#probTxt').textContent = `P(synthetic) = ${fmt(p, 3)}`;
  $('#probRaw').textContent = `probability ${fmt(p, 4)}`;
  $('#confRaw').textContent = `confidence ${fmt(r.confidence, 3)}`;
  $('#timeRaw').textContent = `${r.timing_ms} ms · ${r.working_size?.[0]}×${r.working_size?.[1]} from ${r.source_format || '?'}`;
  const tl = (state.model.model || {}).transfer_learning || {};
  const warns = (r.warnings || []).concat(
    tl.requested && !(tl.weights_applied_for || []).length
      ? [`deep candidates were trained from scratch: ${tl.note || 'pretrained weights were not reachable in the training environment'}`]
      : (r.model && r.model.pretrained_backbone === false
          ? ['deep candidates here were trained without ImageNet weights (download blocked in the training environment)']
          : []));
  $('#warnFlags').innerHTML = warns.map((w) => `<span class="flag">${esc(w)}</span>`).join('');

  /* evidence */
  const ev = r.explanation || {};
  const head = (ev.headline_evidence || []).map((e) => `<li><span class="chip ${e.strength}">${e.strength}</span>
      <span><b>${esc(e.label)}</b> — ${esc(e.meaning)}</span></li>`).join('');
  $('#headline').innerHTML = head || `<li><span class="chip LOW">LOW</span><span>no signal departs strongly from the authentic reference</span></li>`;
  $('#simple').textContent = ev.simple_explanation || 'explanation unavailable (run `python -m ams reference` to enable the evidence engine)';
  $('#technical').textContent = ev.technical_explanation || '';

  const rows = (ev.evidence || []).map((e) => `<tr>
      <td>${esc(e.label)}</td><td class="num">${fmt(e.value, 4)}</td><td class="num">${fmt(e.authentic_median, 4)}</td>
      <td class="num">${fmt(e.z, 2)}</td><td><span class="chip ${e.strength}">${e.strength}</span></td></tr>`).join('');
  $('#evTable').innerHTML = rows
    ? `<thead><tr><th>measured signal</th><th class="num">this image</th><th class="num">authentic median</th><th class="num">z (SD)</th><th>strength</th></tr></thead><tbody>${rows}</tbody>`
    : '<tbody><tr><td>no reference statistics available</td></tr></tbody>';

  /* saliency */
  const s = r.saliency || {};
  const camImgs = $('#camImgs'), camNote = $('#camNote');
  camImgs.innerHTML = '';
  if (s.available && s.overlay_png_b64) {
    camNote.innerHTML = `Grad-CAM from the <b>actual</b> production model, layer <code>${esc(s.target_layer || '')}</code>, on the
      ${s.model_input_size?.[0]}×${s.model_input_size?.[1]} model input grid. ${esc(s.upscaled || '')}`;
    camImgs.innerHTML = `
      <figure><img alt="model input" src="data:image/png;base64,${s.input_png_b64}"><figcaption>model input (preprocessed)</figcaption></figure>
      <figure><img alt="grad-cam overlay" src="data:image/png;base64,${s.overlay_png_b64}"><figcaption>Grad-CAM overlay · positive-class saliency</figcaption></figure>`;
    $('#camTable').innerHTML = `<thead><tr><th>property</th><th class="num">value</th></tr></thead><tbody>
        <tr><td>peak heatmap value</td><td class="num">${fmt(s.peak?.[0], 3)}</td></tr>
        <tr><td>spatial spread (SD of map)</td><td class="num">${fmt(s.spread, 3)}</td></tr>
        <tr><td>mass in top quartile</td><td class="num">${fmt(s.mass_in_top_quartile, 3)}</td></tr>
        <tr><td>pixels above 0.6 activation</td><td class="num">${s.positive_regions ?? '-'}</td></tr>
      </tbody>`;
  } else {
    camNote.innerHTML = `<b>No heatmap.</b> ${esc(s.reason || 'the selected model does not expose activations for Grad-CAM')}`;
    $('#camTable').innerHTML = '';
  }

  /* face */
  const fa = r.face_analysis || {};
  $('#faceDisclaimer').innerHTML = fa.disclaimer
    ? `<b>${esc(fa.kind || 'analytical module')}</b> · engine <code>${esc(fa.engine || '?')}</code> ·
       ${esc(fa.disclaimer)} <i>Trained classifier: ${fa.is_trained_classifier ? 'yes' : 'no'}.</i>`
    : 'face module not available';
  const sm = fa.summary || {};
  $('#faceSummary').innerHTML = Object.keys(sm).length
    ? `<thead><tr><th>whole-image / face-geometry measure</th><th class="num">value</th></tr></thead><tbody>` +
      Object.entries(sm).filter(([k]) => k !== 'flags')
        .map(([k, val]) => `<tr><td>${esc(k)}</td><td class="num">${Array.isArray(val) ? val.join('×') : typeof val === 'object' ? esc(JSON.stringify(val)) : esc(String(val))}</td></tr>`).join('') +
      `</tbody>` : '';
  const faces = fa.faces || [];
  $('#faceList').innerHTML = faces.length
    ? faces.map((f, i) => `<h3 style="margin-top:.8rem">face ${i + 1} · box ${f.box.join(',')} · ${f.geometry?.aspect_ratio ?? '?'} aspect</h3>
        <table class="tbl"><tbody>
        ${Object.entries({ ...(f.geometry || {}), ...(f.consistency || {}), ...(f.landmarks || {}) })
          .map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${typeof v === 'boolean' ? (v ? 'ok' : 'flagged') : fmt2(v)}</td></tr>`).join('')}
        </tbody></table>
        ${(f.notes || []).length ? `<ul class="hint">${f.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : '<p class="hint">no consistency flags raised on this face</p>'}`)
        .join('')
    : `<p class="hint">${esc(sm.note || 'no faces analysed')}</p>`;

  /* generator identification */
  const gi = r.generator_identification || {};
  const gc = (state.model && state.model.generator_classifier) || {};
  $('#genStatus').innerHTML = gi.available
    ? `<p>${gi.generator === 'UNKNOWN'
        ? '<b>Generator not named: the evidence is below the acceptance rule.</b>'
        : `Closest known generator: <b>${esc(gi.generator)}</b>`}</p>
       <table><thead><tr><th>ranked generator</th><th class="num">probability</th></tr></thead><tbody>
         ${(gi.ranking || []).map((c, i) => `<tr><td>${i + 1}. ${esc(c.generator)}</td>
            <td class="num">${Number(c.probability).toFixed(4)}</td></tr>`).join('')}
       </tbody></table>
       <p class="hint">${esc(gi.reason)}</p>
       <p class="hint">Acceptance rule (fitted on the validation split, not tuned on test): accept a name
        only when the top probability &ge; <b>${fmt((gi.acceptance_rule || {}).min_probability, 2)}</b>
        and its margin over the runner-up &ge; <b>${fmt((gi.acceptance_rule || {}).min_margin, 2)}</b>;
        validation precision on accepted predictions
        <b>${fmt((gi.acceptance_rule || {}).validation_precision_on_accepted, 3)}</b>.
        Model: <code>${esc(gi.model || '?')}</code>${gi.inference_ms != null ? `, ${fmt2(gi.inference_ms)} ms/image` : ''}.</p>
       ${(gi.measured_metrics && Object.keys(gi.measured_metrics).length)
        ? `<p class="hint">Held-out test metrics of this fingerprint model:
            macro F1 <b>${fmt(gi.measured_metrics.macro_f1, 4)}</b>,
            accuracy <b>${fmt(gi.measured_metrics.accuracy, 4)}</b>,
            top-3 <b>${gi.measured_metrics.top3_accuracy == null ? 'n/a (too few classes)' : fmt(gi.measured_metrics.top3_accuracy, 4)}</b>
            on ${fmt(gi.measured_metrics.n, 0)} images.</p>` : ''}
       <p class="warn-inline">${esc(gi.limitation || '')}</p>`
    : `<div class="warn-inline"><b>Generator identification is not offered by this deployment.</b><br>
       ${esc(gi.reason || gc.reason || 'no per-generator labels in the training data')}</div>
       <p class="hint">The model was trained on exactly the labels the dataset provides
       (<code>${esc((state.model.classes || []).join(', '))}</code>). Rather than guess a generator from a binary
       output, the app reports "insufficient evidence" — which is the truthful answer here.
       Add per-generator labelled folders under <code>data/</code> and retrain to enable it.</p>`;

  /* raw signals */
  const sig = ev.signals || {};
  $('#signalTable').innerHTML = Object.keys(sig).length
    ? `<thead><tr><th>signal</th><th class="num">value</th></tr></thead><tbody>` +
      Object.entries(sig).map(([k, v]) => `<tr><td><code>${esc(k)}</code></td><td class="num">${fmt(v, 5)}</td></tr>`).join('') + '</tbody>'
    : '';
  $('#sigGrid').textContent = r.model?.input_size ?? '?';
}
function fmt2(v) { return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : esc(String(v)); }

/* ---------------------------------------------------------------- model panes */
function renderModel() {
  const m = state.model.model || {}, sel = state.model.selection || {};
  const met = m.metrics || {};
  const kv = (k, v, full) => `<div class="kv${full ? ' full' : ''}"><dt>${esc(k)}</dt><dd>${v}</dd></div>`;
  $('#card').innerHTML = [
    kv('task', esc(m.task || 'binary AI/synthetic detection')),
    kv('architecture', `<code>${esc(m.architecture)}</code> (${esc(m.framework)})`),
    kv('parameters', m.params ? Number(m.params).toLocaleString() : 'n/a'),
    kv('input', `${m.image_size}px${m.image_size !== 224 ? ' <span class="hint">(sources are 32px crops)</span>' : ''}`),
    kv('classes', (m.classes || []).map((c) => `<code>${esc(c)}</code>`).join(' → ')),
    kv('threshold', fmt(m.threshold, 4)),
    kv('train / val / test', `${m.training_images} / ${m.validation_images} / ${m.test_images}`),
    kv('artifact size', `${fmt(m.model_size_mb, 2)} MB`),
    kv('inference', `${fmt(m.inference_ms_per_image, 2)} ms/image (CPU)`),
    kv('trained', `${esc(m.training_date || '?')} · v${esc(m.model_version || '')}`),
    kv('test metrics', ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'specificity', 'false_positive_rate', 'eer']
      .map((k) => `${k} <b>${fmt(met[k], 3)}</b>`).join(' · ')),
    kv('selection', esc(sel.reason || ''), true),
    kv('dataset', datasetSummary(m.dataset || {}), true),
    kv('known limitations', `<ul style="margin:.2rem 0 0;padding-left:1.05rem">${(m.known_limitations || []).map((l) => `<li>${esc(l)}</li>`).join('')}</ul>`, true),
    kv('face model status', esc(JSON.stringify((state.model.face_model || {}).reason || '')), true),
  ].join('');

  const cmp = state.model.comparison || [];
  const rows = cmp.map((c) => `<tr class="${sel.selected === c.name ? 'winner' : ''}">
      <td>${esc(c.name)}${sel.selected === c.name ? ' ★' : ''}
          ${c.tie_note ? `<div class="hint">tie-break: ${esc(c.tie_note)}</div>` : ''}</td>
      <td class="num">${c.rank || '-'}</td>
      <td>${esc(c.family)}</td>
      <td class="num">${fmt(c.accuracy, 3)}</td><td class="num">${fmt(c.f1, 3)}</td>
      <td class="num">${fmt(c.roc_auc, 3)}</td><td class="num">${fmt(c.recall, 3)}</td>
      <td class="num">${fmt(c.precision, 3)}</td>
      <td class="num">${fmt(c.inference_ms, 1)}</td><td class="num">${fmt(c.size_mb, 2)}</td>
      <td class="num"><b>${fmt(c.final_score, 3)}</b></td></tr>`).join('');
  $('#compareTable').innerHTML = `<thead><tr><th>model</th><th class="num">rank</th><th>family</th><th class="num">acc</th><th class="num">F1</th>
      <th class="num">AUC</th><th class="num">recall</th><th class="num">prec</th><th class="num">ms</th>
      <th class="num">MB</th><th class="num">score</th></tr></thead><tbody>${rows}</tbody>`;
  $('#selReason').textContent = `selected: ${sel.selected} — ${sel.reason}` +
    (sel.tie_break_applied ? ' (selected model is not rank 1 by score: the performance tie-break applied)' : '');

  const tm = state.model.reported_metrics || met || {};
  $('#evalTable').innerHTML = `<thead><tr><th>metric (test split, one evaluation)</th><th class="num">value</th></tr></thead><tbody>` +
    Object.entries(tm).filter(([k]) => !['confusion_matrix', 'threshold'].includes(k))
      .map(([k, v]) => `<tr><td><code>${esc(k)}</code></td><td class="num">${typeof v === 'number' ? fmt(v, 4) : esc(String(v))}</td></tr>`).join('') +
    `</tbody>`;
}

async function datasetSummary(d) {
  const insp = d.inspection || d;
  const bits = [];
  const push = (k, v) => { if (v !== undefined && v !== null) bits.push(`${k} <b>${esc(String(v))}</b>`); };
  push('images', insp.n_valid !== undefined ? `${insp.n_valid} usable of ${insp.n_images} found` : insp.n_images);
  push('classes', Object.entries(insp.classes_found || {}).map(([k, v]) => `${k}=${v}`).join(', '));
  push('capture groups', insp.n_groups);
  push('corrupt/rejected', insp.n_corrupt !== undefined ? insp.n_corrupt : insp.corrupt);
  push('near-duplicates', insp.n_duplicate_near !== undefined ? insp.n_duplicate_near : insp.duplicates_near);
  push('cross-source duplicates', insp.n_duplicate_perceptual_cross_source);
  push('formats', Object.entries(insp.formats || {}).map(([k, v]) => `${k}:${v}`).join(' ') || d.formats);
  push('sizes', Object.entries(insp.sizes || {}).slice(0, 3).map(([k, v]) => `${k}:${v}`).join(' ') || d.sizes);
  push('generator labels', (insp.generator_labels_available && insp.generator_labels_available.length)
    ? insp.generator_labels_available.join(', ') : 'none present');
  const notes = (insp.notes || d.notes || d.warnings || []);
  const warn = notes.length ? `<div class="hint">${notes.map((n) => `⚠ ${esc(String(n))}`).join('<br>')}</div>` : '';
  return `<span class="hint">${bits.join(' · ')}</span>${warn}`;
}

function renderDataNote() {
  const ds = (state.model && state.model.dataset) || {};
  const insp = ds.inspection || {};
  const n = insp.n_valid ?? ds.n_images;
  const cls = Object.entries(insp.classes_found || {}).map(([k, v]) => `${k} ${v}`).join(' · ');
  const size = Object.keys(insp.sizes || {})[0];
  const bits = [];
  if (n) bits.push(`trained on ${n.toLocaleString()} images (${cls || 'two classes'})`);
  if (size) bits.push(`${size} px crops`);
  if (insp.n_corrupt !== undefined) bits.push(`${insp.n_corrupt} corrupt files rejected`);
  bits.push('so other content, resolutions or generators are outside what it learned');
  $('#dataNote').textContent = 'Scope: this model is ' + bits.join(', ') + '.';
}

function renderReports() {
  const names = (state.health?.reports_available) || [];
  $('#reportLinks').innerHTML = names.length
    ? names.map((n) => `<button class="report-btn" data-r="${esc(n)}">${esc(n)}</button>`).join('')
    : '<span class="hint">no reports found in reports/</span>';
  $('#reportLinks').addEventListener('click', async (e) => {
    const b = e.target.closest('.report-btn'); if (!b) return;
    $$('#reportLinks .report-btn').forEach((x) => x.classList.remove('primary'));
    b.classList.add('primary');
    const r = await fetch('/api/report/' + encodeURIComponent(b.dataset.r));
    if (!r.ok) { $('#reportView').textContent = 'report unavailable'; return; }
    const j = await r.json();
    $('#reportView').innerHTML = j.markdown ? mdLite(j.markdown) : esc(JSON.stringify(j, null, 2));
  });
}

/* Minimal, injection-safe markdown renderer for the report panes. The text is HTML-escaped
   first and only this function's own tags are added, so nothing from a report can execute.
   Figures are rewritten to /api/report/<basename>, which the server also resolves inside
   reports/figures/. */
function mdLite(src) {
  const lines = String(src || '').replace(/\r/g, '').split('\n');
  const out = [];
  let para = [], list = [], table = null, code = null;
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
    if (!table) return;
    const head = table[0], body = table.slice(1);
    out.push(`<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>` +
      body.map((row) => `<tr>${row.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('') + '</tbody></table>');
    table = null;
  };
  const cells = (l) => l.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());
  for (const raw of lines) {
    const l = raw.replace(/\s+$/, '');
    if (code !== null) {
      if (/^```/.test(l)) { out.push(`<pre><code>${esc(code.join('\n'))}</code></pre>`); code = null; }
      else code.push(l);
      continue;
    }
    if (/^```/.test(l)) { flushPara(); flushList(); flushTable(); code = []; continue; }
    if (/^\s*\|.*\|\s*$/.test(l)) {
      if (/^\s*\|[\s:|-]+\|\s*$/.test(l)) continue;          // separator row
      flushPara(); flushList();
      if (!table) table = [];
      table.push(cells(l));
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

/* ---------------------------------------------------------------- samples */
async function renderSamples() {
  let data = { samples: [] };
  try { data = await (await fetch('/api/samples')).json(); } catch (e) {}
  $('#samplesNote').textContent = data.note || '';
  if (!data.samples.length) { $('#samples').innerHTML = '<span class="hint">no samples/ directory (created by train.py)</span>'; return; }
  $('#samples').innerHTML = data.samples.map((s) =>
    `<div class="sample" data-f="${esc(s.file)}" data-true="${esc(s.class)}" title="true class: ${esc(s.class)}">
       <img alt="sample ${esc(s.file)}" src="/api/samples/${encodeURIComponent(s.file)}"><span>${esc(s.class)}</span></div>`).join('');
  $$('#samples .sample').forEach((el) => el.addEventListener('click', async () => {
    const r = await fetch('/api/samples/' + encodeURIComponent(el.dataset.f));
    const blob = await r.blob();
    setFile(new File([blob], el.dataset.f, { type: 'image/png' }), `sample · true class ${el.dataset.true}`);
    run();
  }));
}

/* ---------------------------------------------------------------- utils */
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, n = 3) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(n));

boot();
