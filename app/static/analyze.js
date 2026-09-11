/* Analyze page: pick an image (file, drag-drop, paste, or a held-out sample), send it to
   /api/detect, and render exactly what came back. No verdict, probability or metric is computed
   here; the only client-side maths is positioning the gauge. */
(async function () {
  const { $, $$, esc, fmt, bytes, api, table } = VFA;
  const S = { file: null, label: '', busy: false, model: null, hasModel: false, maxBytes: 8 * 1024 * 1024 };

  await VFA.boot(afterBoot);

  function afterBoot(s) {
    const h = s.health || {};
    S.model = s.model;
    S.hasModel = !!h.model_loaded && !!s.model;
    S.maxBytes = h.max_upload_bytes || S.maxBytes;
    const ms = $('#maxSize');
    if (ms) ms.textContent = Math.round(S.maxBytes / 1048576);

    const drop = $('#drop'), fileIn = $('#file'), run = $('#run'), clear = $('#clear');
    if (!drop) return;
    run.disabled = true; clear.disabled = true;

    if (!S.hasModel) {
      const box = $('#emptyState');
      if (box) {
        box.innerHTML = `<h2>Nothing to score with yet</h2>
          <p class="prose">${esc(h.load_error || s.error || 'no trained artifact in models/')}</p>
          <p class="hint">Train once and reload: <code>python train.py</code> inspects the dataset, compares the supervised
          candidates and stores the selected model. This service only loads that artifact — it never retrains at request
          time, and it never shows a guess in place of a measurement.</p>`;
      }
    }

    const choose = (f, label) => {
      if (!f) return;
      if (f.size > S.maxBytes) { status(`that file is ${bytes(f.size)} — the limit is ${bytes(S.maxBytes)}`, true); return; }
      S.file = f;
      S.label = label || `${f.name} · ${bytes(f.size)}`;
      $('#dropInner').hidden = true;
      $('#preview').src = URL.createObjectURL(f);
      $('#previewWrap').hidden = false;
      run.disabled = !S.hasModel;
      clear.disabled = false;
      status(S.label);
    };
    const clearAll = () => {
      S.file = null; S.label = ''; fileIn.value = '';
      $('#previewWrap').hidden = true; $('#dropInner').hidden = false;
      $('#result').hidden = true; $('#emptyState').hidden = false;
      run.disabled = true; clear.disabled = true; status('');
    };

    drop.addEventListener('click', () => { if (S.hasModel) fileIn.click(); });
    drop.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (S.hasModel) fileIn.click(); }
    });
    fileIn.addEventListener('change', () => { if (fileIn.files[0]) choose(fileIn.files[0]); });
    ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) choose(f); });
    window.addEventListener('paste', (e) => {
      const it = Array.from(((e.clipboardData || {}).items || [])).find((i) => i.type.indexOf('image/') === 0);
      if (!it) return;
      const f = it.getAsFile();
      if (f) choose(f, 'pasted image · ' + bytes(f.size));
    });
    run.addEventListener('click', () => analyze(run));
    clear.addEventListener('click', clearAll);

    samples(choose);
    scope();
  }

  function status(text, bad) {
    const el = $('#status');
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('err', !!bad);
  }

  async function analyze(btn) {
    if (!S.hasModel) { status('no trained model — run `python train.py` first', true); return; }
    if (!S.file) return;
    const fd = new FormData();
    fd.append('file', S.file, S.file.name || 'image.png');
    S.busy = true; btn.disabled = true; btn.textContent = 'Analyzing…';
    status('scoring on CPU…');
    const t0 = performance.now();
    try {
      const r = await api('/api/detect', { method: 'POST', body: fd });
      status(`done · ${fmt(r.timing_ms, 1)} ms in the model, ${Math.round(performance.now() - t0)} ms round trip`);
      render(r);
    } catch (err) {
      status('error: ' + err.message, true);
    } finally {
      S.busy = false; btn.disabled = false; btn.textContent = 'Analyze';
    }
  }

  /* ---------------------------------------------------------------- verdict */
  function render(r) {
    $('#emptyState').hidden = true;
    $('#result').hidden = false;

    const p = Number(r.probability_synthetic);
    const synthetic = p >= Number(r.threshold);
    const v = $('#verdict');
    v.className = 'verdict ' + (synthetic ? 'synthetic' : 'authentic');
    const dim = (a) => (Array.isArray(a) ? a.join('×') : esc(String(a == null ? '?' : a)));
    v.innerHTML = `<strong>${esc(r.verdict)}</strong>
      <small>${esc(S.label || r.filename || 'image')} · original ${dim(r.original_size)} px, ${esc(r.source_mode || '?')} ·
      read at ${dim(r.working_size)} px, ${esc(r.source_format || '?')} · class label <code>${esc(r.predicted_class)}</code> ·
      model <code>${esc((r.model || {}).name || '?')}</code> (${esc((r.model || {}).family || '?')}, input ${(r.model || {}).input_size || '?'} px)</small>`;

    $('#probDot').style.left = (100 * Math.max(0, Math.min(1, p))).toFixed(1) + '%';
    $('#thrMark').style.left = (100 * Math.max(0, Math.min(1, Number(r.threshold)))).toFixed(1) + '%';
    $('#probVal').textContent = fmt(p, 4);
    $('#thrVal').textContent = fmt(r.threshold, 4);
    $('#marginVal').textContent = fmt(r.margin_over_threshold, 4);
    $('#timeVal').textContent = `${fmt(r.confidence, 3)} confidence in the called class · ${fmt(r.timing_ms, 1)} ms · ${bytes(r.upload_bytes)}`;

    const tl = ((S.model || {}).model || {}).transfer_learning || {};
    const flags = (r.warnings || []).slice();
    if (tl.requested && !(tl.weights_applied_for || []).length) {
      flags.push('deep candidates were trained from scratch — pretrained weights were not reachable offline');
    }
    $('#warnFlags').innerHTML = flags.length
      ? flags.map((w) => `<span class="pill warn">${esc(w)}</span>`).join(' ')
      : '<span class="pill ok">no warnings for this image</span>';

    explanation(r);
    cam(r.saliency);
    faces(r.face_analysis);
    generator(r.generator_identification);

    const sig = ((r.explanation || {}).signals) || {};
    $('#signalTable').innerHTML = table(['signal', 'value on the model input grid'],
      Object.keys(sig).map((k) => [{ raw: `<code>${esc(k)}</code>` }, { n: sig[k], d: 5 }]),
      { num: [1], empty: 'no signals returned' });
    const grid = $('#sigGrid');
    if (grid) grid.textContent = String((r.model || {}).input_size != null ? r.model.input_size : '?');
    $('#rawJson').textContent = JSON.stringify(r, null, 1);
  }

  /* ---------------------------------------------------------------- evidence */
  function explanation(r) {
    const e = r.explanation || {};
    $('#simple').textContent = e.simple_explanation
      || 'explanation unavailable — run `python -m vfa reference` to estimate the authentic reference statistics';
    const refs = (e.reference || {});
    const meta = [
      e.technical_explanation || '',
      refs.grid ? `Signals come from ${esc(String(refs.grid))} px; reference medians ${esc(String(refs.estimated_on || 'n/a'))}` : '',
      refs.classes ? `reference classes: ${esc(JSON.stringify(refs.classes))}` : '',
    ].filter(Boolean).join('\n\n');
    $('#technical').textContent = meta;

    const head = (e.headline_evidence || []);
    $('#headline').innerHTML = head.length
      ? head.map((ev) => `<li><span class="chip ${esc(ev.strength || 'LOW')}">${esc(ev.strength || '')}</span>
          <span><b>${esc(ev.label)}</b> — ${esc(ev.meaning || '')}
          <span class="hint">(measured ${fmt(ev.value, 4)}, authentic median ${fmt(ev.authentic_median, 4)}, ${esc(ev.direction || '')})</span></span></li>`).join('')
      : '<li><span class="chip LOW">LOW</span><span>no measured signal departs strongly from the authentic reference</span></li>';

    $('#evTable').innerHTML = table(
      ['measured signal', 'this image', 'authentic median', 'deviation z (SD)', 'strength'],
      (e.evidence || []).map((ev) => [ev.label, { n: ev.value, d: 4 }, { n: ev.authentic_median, d: 4 },
        { n: ev.z, d: 2 }, { raw: `<span class="chip ${esc(ev.strength || 'LOW')}">${esc(ev.strength || '')}</span>` }]),
      { num: [1, 2, 3], empty: 'no reference statistics available — the explanation falls back to the raw signals' });
  }

  /* ---------------------------------------------------------------- Grad-CAM */
  function cam(s) {
    const note = $('#camNote'), imgs = $('#camImgs'), tbl = $('#camTable');
    imgs.innerHTML = ''; tbl.innerHTML = '';
    if (!s || !s.available) {
      note.innerHTML = `<b>No heatmap for this deployment.</b> ${esc((s || {}).reason
        || 'the selected model does not expose activations that Grad-CAM can attribute')}`;
      return;
    }
    const size = Array.isArray(s.model_input_size) ? s.model_input_size.join('×') : (s.model_input_size || '?');
    note.innerHTML = `Grad-CAM from the <b>actual</b> production model — layer <code>${esc(s.target_layer || '')}</code>
      on the ${esc(String(size))} px input grid. ${esc(s.upscaled || '')}`;
    imgs.innerHTML = `<figure><img alt="model input" loading="lazy" src="data:image/png;base64,${s.input_png_b64}" />
        <figcaption>model input (preprocessed)</figcaption></figure>
      <figure><img alt="Grad-CAM overlay" loading="lazy" src="data:image/png;base64,${s.overlay_png_b64}" />
        <figcaption>Grad-CAM overlay · positive-class saliency</figcaption></figure>`;
    tbl.innerHTML = table(['property of the map', 'value'], [
      ['peak heatmap value', { n: (s.peak || [])[0], d: 3 }],
      ['spatial spread (SD of the map)', { n: s.spread, d: 3 }],
      ['mass in the top quartile', { n: s.mass_in_top_quartile, d: 3 }],
      ['pixels above 0.6 activation', { n: s.positive_regions, d: 0 }],
    ], { num: [1], empty: '' });
  }

  /* ---------------------------------------------------------------- faces */
  function faces(fa) {
    const dis = $('#faceDisclaimer'), sum = $('#faceSummary'), list = $('#faceList');
    sum.innerHTML = ''; list.innerHTML = '';
    if (!fa) { dis.innerHTML = '<span class="pill">face module not configured in this build</span>'; return; }
    if (fa.error) {
      dis.innerHTML = `<span class="pill bad">face module did not run</span> ${esc(fa.error)}`;
      return;
    }
    dis.innerHTML = `<b>${esc(fa.kind || 'analytical module')}</b> · engine <code>${esc(fa.engine || '?')}</code> ·
      detector available: ${fa.face_detector_available ? 'yes' : 'no'} · trained classifier:
      ${fa.is_trained_classifier ? 'yes' : '<b>no</b>'}.<br>${esc(fa.disclaimer || '')}`;

    const sm = fa.summary || {};
    const keys = Object.keys(sm).filter((k) => !Array.isArray(sm[k]));
    const flagList = sm.flags || [];
    sum.innerHTML = (keys.length
      ? table(['whole-image and region measure', 'value'], keys.map((k) => [k, { raw: cell(sm[k]) }]), { empty: '' })
      : '') + (flagList.length
        ? `<p class="hint">flags: ${flagList.map((f) => esc(String(f))).join(' · ')}</p>`
        : '<p class="hint">no consistency flags raised</p>');

    const fs = fa.faces || [];
    list.innerHTML = fs.length
      ? fs.map((f, i) => `<h3 style="margin-top:.9rem">face ${i + 1} · box ${esc((f.box || []).join(', '))} px ·
          detector score ${fmt(f.detector_score, 3)} · region ${esc(f.region || '?')}${f.detected ? '' : ' (assumed, not located)'}</h3>
          ${table(['measure', 'value'], Object.entries({ ...(f.geometry || {}), ...(f.consistency || {}), ...(f.landmarks || {}) })
            .map(([k, v]) => [k, { raw: cell(v) }]), { empty: 'no per-face measurements'})}
          ${(f.notes || []).length ? `<p class="hint">${f.notes.map((n) => '⚠ ' + esc(String(n))).join('<br>')}</p>` : ''}`)
        .join('')
      : `<p class="hint">${esc(sm.note || 'no face region analysed')}</p>`;
  }
  function cell(v) {
    if (v === null || v === undefined) return '<span class="hint">n/a</span>';
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    if (typeof v === 'number') return fmt(v, Number.isInteger(v) ? 0 : 3);
    if (Array.isArray(v)) return esc(v.map((x) => (typeof x === 'number' ? fmt(x, 2) : String(x))).join(' × '));
    if (typeof v === 'object') return esc(JSON.stringify(v));
    return esc(String(v));
  }

  /* ---------------------------------------------------------------- generator */
  function generator(gi) {
    const el = $('#genStatus');
    gi = gi || {};
    const gc = (S.model || {}).generator_classifier || {};   // artifact record, used when the API omits a reason
    if (!gi.available) {
      const labels = gi.dataset_labels || gc.classes || {};
      el.innerHTML = `<div class="pillrow"><span class="pill bad">not offered by this deployment</span></div>
        <p class="prose">${esc(gi.reason || gc.reason || 'no per-generator labels exist in the training data')}</p>
        <p class="hint">This model was trained on exactly the labels the dataset provides
          (<code>${esc(Object.keys(labels).join(', ') || 'real / fake')}</code>)${(gi.known_generators || []).length
            ? `; generators known to the code: ${esc(gi.known_generators.join(', '))}` : ''}.
          Rather than guess a generator from a binary output, the page says the evidence is insufficient — which is the
          truthful answer for this corpus.</p>
        <details class="raw"><summary>what would make it available</summary><div class="prose">
          <p>Labelled material — one folder (or filename token) per generator — then <code>python train.py</code>.
          The classifier, its abstention gate and the acceptance rule already exist and are tested in
          <code>vfa/generator.py</code>; only the labels are missing here, and inventing them is not an option.
          See the <a href="/model">model card</a>.</p></div></details>`;
      return;
    }
    const rule = gi.acceptance_rule || {};
    const mm = gi.measured_metrics || {};
    el.innerHTML = `
      <div class="pillrow">${gi.generator === 'UNKNOWN'
        ? '<span class="pill warn">abstained — no generator named (UNKNOWN)</span>'
        : `<span class="pill ok">closest known generator: ${esc(gi.generator)}</span>`}</div>
      <p class="prose">${esc(gi.reason || '')}</p>
      ${table(['ranked generator', 'probability'], (gi.ranking || []).map((c, i) => [`${i + 1}. ${c.generator}`, { n: c.probability, d: 4 }]),
        { num: [1], empty: 'no ranking returned' })}
      <p class="hint">Acceptance rule (fitted on validation, never tuned on test): a name is accepted only when the top
        probability ≥ <b>${fmt(rule.min_probability, 2)}</b> and its margin over the runner-up ≥ <b>${fmt(rule.min_margin, 2)}</b>;
        precision on accepted validation predictions <b>${fmt(rule.validation_precision_on_accepted, 3)}</b>.
        Model <code>${esc(gi.model || '?')}</code>${gi.inference_ms != null ? `, ${fmt(gi.inference_ms, 1)} ms/image` : ''};
        the test split was evaluated ${gi.test_evaluations != null ? gi.test_evaluations : 'once'}.</p>
      ${Object.keys(mm).length ? `<p class="hint">Held-out metrics of this fingerprint model: macro F1
        <b>${fmt(mm.macro_f1, 4)}</b>, accuracy <b>${fmt(mm.accuracy, 4)}</b>, top-3
        <b>${mm.top3_accuracy == null ? 'n/a (too few classes)' : fmt(mm.top3_accuracy, 4)}</b> on ${fmt(mm.n, 0)} images.</p>` : ''}
      ${gi.limitation ? `<p class="caution">${esc(gi.limitation)}</p>` : ''}`;
  }

  /* ---------------------------------------------------------------- samples */
  async function samples(choose) {
    const el = $('#samples'), note = $('#samplesNote');
    let data = { samples: [] };
    try { data = await api('/api/samples'); } catch (err) { el.innerHTML = '<span class="hint">no sample list</span>'; return; }
    if (note) note.textContent = data.note || '';
    const list = data.samples || [];
    if (!list.length) {
      el.innerHTML = '<span class="hint">no samples/ yet — <code>train.py</code> writes a few from the test split</span>';
      return;
    }
    el.innerHTML = list.map((s) => {
      const truth = s.class === 'real' ? 'authentic' : 'generated';
      return `<div class="sample" data-f="${esc(s.file)}" title="${esc(s.class)} · ${esc(s.source_path || '')} · ${esc(s.note || '')}">
        <img alt="sample ${esc(s.file)}" loading="lazy" src="/api/samples/${encodeURIComponent(s.file)}" />
        <span>${esc(truth)}</span></div>`;
    }).join('');
    $$('#samples .sample').forEach((box) => box.addEventListener('click', async () => {
      if (!S.hasModel) { status('no trained model — run `python train.py` first', true); return; }
      status('loading sample…');
      try {
        const r = await fetch('/api/samples/' + encodeURIComponent(box.dataset.f));
        if (!r.ok) throw new Error('sample not found');
        const blob = await r.blob();
        const truth = box.querySelector('span').textContent;
        choose(new File([blob], box.dataset.f, { type: blob.type || 'image/png' }), `test-split sample · truth: ${truth}`);
        if (S.file) analyze($('#run'));
      } catch (err) { status('sample error: ' + err.message, true); }
    }));
  }

  /* ---------------------------------------------------------------- scope + honesty */
  function scope() {
    const note = $('#dataNote'), ul = $('#honesty');
    const m = S.model || {};
    const info = m.model || {};
    const ds = m.dataset || {};
    const insp = ds.inspection || {};
    const split = m.split || {};
    const bits = [];
    if (insp.n_valid) bits.push(`${Number(insp.n_valid).toLocaleString()} images`);
    if (insp.classes_found) bits.push(Object.entries(insp.classes_found).map(([k, v]) => `${k} ${Number(v).toLocaleString()}`).join(' + '));
    if (insp.sizes) bits.push(`${Object.keys(insp.sizes)[0]} px sources`);
    if ((split.sizes || {}).test) bits.push(`tested on ${Number(split.sizes.test).toLocaleString()} held-out images, once per model`);
    if (note) {
      note.textContent = (bits.length ? bits.join(', ') + '. ' : '')
        + 'Anything else — other content, resolutions, pipelines or generators — is outside what it learned, '
        + 'so uncertainty grows off-distribution.';
    }
    if (ul) {
      ul.innerHTML = ((m.honesty || info.honesty || []).length ? (m.honesty || info.honesty)
        : ['Prediction is probabilistic and is not proof of image origin.'])
        .map((t) => `<li>${esc(t)}</li>`).join('');
    }
  }
})();
