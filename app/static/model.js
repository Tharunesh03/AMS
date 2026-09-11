/* Model card page. Everything below is read from /api/model, i.e. from models/model_info.json,
   models/metrics.json, models/preprocessing.json and models/ai_detector/classes.json. */
(async function () {
  const { $, $$, esc, fmt, api, table, kvTable, load } = VFA;
  const s = await load();
  VFA.decorate(s);

  const m = s.model;
  const slots = ['#deployed', '#preprocess', '#selection', '#dataset', '#secondary'];
  if (!m) {
    slots.forEach((sel) => {
      const el = $(sel);
      if (el) el.innerHTML = '<p class="hint">no artifact yet — this page reads <code>models/</code>, and '
        + '<code>python train.py</code> writes it. Nothing here is filled in by hand.</p>';
    });
    const ul = $('#honesty'), lim = $('#limitations');
    if (ul) ul.innerHTML = `<li>${esc(s.error || 'model unavailable')}</li>`;
    if (lim) lim.innerHTML = '';
    return;
  }

  const info = m.model || {};
  const cls = m.classes || {};
  const pre = m.preprocessing || {};
  const sel = m.selection || {};
  const tl = m.transfer_learning || info.transfer_learning || {};
  const gen = m.generator_classifier || {};
  const face = m.face_model || {};
  const ds = m.dataset || {};
  const insp = ds.inspection || {};
  const split = m.split || {};
  const met = info.metrics || {};

  /* ---------------------------------------------------------------- deployed */
  const meaning = (cls.labels_meaning || {});
  $('#deployed').innerHTML = kvTable({
    artifact: info.model_name,
    version: `${info.model_version || '?'} · written ${info.training_date || '?'}`,
    task: info.task,
    architecture: `${info.architecture || '?'} (${info.framework || '?'}) · family ${(sel.selected_family) || info.family || '?'}`,
    descriptor: `${info.descriptor || 'n/a'} · ${info.feature_dim || '?'} features`,
    'selected from': `${(info.selected_from || []).length} candidates on identical splits`,
    classes: (cls.labels || info.classes || [])
      .map((c, i) => `${c}${meaning[String(i)] ? ` = ${String(meaning[String(i)]).replace(/\s*\/\s*/g, ', ')}` : ''}`).join(' · '),
    threshold: `${fmt(info.threshold, 4)} — ${info.threshold_policy || ''}`,
    'test accuracy': fmt(met.accuracy, 4),
    'test F1': fmt(met.f1, 4),
    'test ROC-AUC': fmt(met.roc_auc, 4),
    'artifact size': `${fmt(info.model_size_mb, 3)} MB`,
    inference: `${fmt(info.inference_ms_per_image, 2)} ms/image on 2 CPU threads`,
    protocol: info.eval_protocol,
    'test evaluations': info.test_evaluations,
    'load with': info.load_with,
  });

  /* -------------------------------------------------------------- preprocessing */
  $('#preprocess').innerHTML = kvTable({
    'input grid': `${pre.resize_to ?? info.image_size ?? '?'} px (cache at ${pre.cache_size ?? '?'} px)`,
    channels: 'RGB (colour is dropped for the forensic descriptor, see photometric below)',
    resize: 'largest-side, no cropping; anything bigger is downscaled, smaller is upscaled and that is reported',
    normalisation: `mean [${(pre.mean || []).map((x) => fmt(x, 3)).join(', ')}] · std [${(pre.std || []).map((x) => fmt(x, 3)).join(', ')}]`,
    'photometric for features': pre.photometric_for_features || 'n/a',
    'feature grid': pre.feature_size,
    augmentation: pre.augmented_fields || 'none at inference',
  }) + '<p class="hint">The same code path is used by the CLI, the API and the training run, so a number measured '
     + 'in one place means the same in the other.</p>';

  /* ---------------------------------------------------------------- selection */
  const w = sel.weights || {};
  const wrow = { f1: 'macro F1 on test', roc_auc: 'ROC-AUC on test', accuracy: 'accuracy on test',
                 speed: 'latency term, min(1, 3.33 / ms)', efficiency: 'size + pipeline term' };
  $('#selection').innerHTML = table(['component', 'weight', 'what it measures'],
    Object.keys(w).map((k) => [k, { n: w[k], d: 2 }, wrow[k] || '']), { num: [1] })
    + kvTable({
      selected: `${sel.selected} (${sel.selected_family || '?'})`,
      rank: `rank ${sel.selected_rank} of ${(m.comparison || []).length} by the same score`,
      'gap to runner-up': fmt(sel.score_gap_to_runner_up, 4),
      'tie-break applied': sel.tie_break_applied ? 'yes — equal performance, cheaper model chosen' : 'no',
      rationale: sel.reason,
    });

  /* -------------------------------------------------------------- limitations */
  $('#limitations').innerHTML = (m.limitations || []).map((l) => `<li>${esc(l)}</li>`).join('')
    || '<li class="hint">the artifact carries no limitations list</li>';
  $('#honesty').innerHTML = (m.honesty || []).map((l) => `<li>${esc(l)}</li>`).join('');

  /* ---------------------------------------------------------------- dataset */
  $('#dataset').innerHTML = kvTable({
    classes: Object.entries(insp.classes_found || {}).map(([k, v]) => `${k} ${Number(v).toLocaleString()}`).join(' / ') || 'n/a',
    'images found / usable': `${Number(insp.n_images ?? 0).toLocaleString()} / ${Number(insp.n_valid ?? 0).toLocaleString()}`,
    'unreadable, rejected': insp.n_corrupt,
    'capture groups': Number(insp.n_groups ?? 0).toLocaleString(),
    'exact duplicate sets': insp.n_duplicate_exact,
    'near duplicates': insp.n_duplicate_near,
    'cross-source perceptual duplicates': insp.n_duplicate_perceptual_cross_source,
    formats: Object.entries(insp.formats || {}).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(' / ') || 'n/a',
    sizes: Object.entries(insp.sizes || {}).map(([k, v]) => `${k} (${v.toLocaleString()})`).join(' / ') || 'n/a',
    'class balance': Object.entries(insp.class_balance || {}).map(([k, v]) => `${k} ${fmt(v, 3)}`).join(' / ') || 'n/a',
    'generator labels found': (insp.generator_labels_available || []).length
      ? insp.generator_labels_available.join(', ') : 'none — one label dimension only',
    split: `${split.strategy || 'n/a'} · train ${Number((split.sizes || {}).train ?? 0).toLocaleString()} / `
      + `val ${Number((split.sizes || {}).val ?? 0).toLocaleString()} / test ${Number((split.sizes || {}).test ?? 0).toLocaleString()}`,
    seed: split.seed,
    notes: (insp.notes || []).join(' · ') || 'none recorded',
  });

  /* ------------------------------------------------- second tasks, stated honestly */
  $('#secondary').innerHTML = [
    block('Generator fingerprinting', gen.available === false
      ? `<span class="pill bad">not trainable on this dataset</span>`
      : `<span class="pill ok">${(gen.classes && Object.keys(gen.classes).length) || 0} classes</span>`,
      gen.reason || gen.note || 'no record'),
    block('Face model', face.trained
      ? '<span class="pill ok">trained</span>' : '<span class="pill warn">analytical CV module, not a classifier</span>',
      face.reason || ''),
    block('Transfer learning', tl.requested
      ? ((tl.weights_applied_for || []).length
          ? `<span class="pill ok">weights applied for ${tl.weights_applied_for.length} of ${tl.deep_candidates || '?'} deep candidates</span>`
          : '<span class="pill warn">requested · weights unreachable · trained from scratch</span>')
      : '<span class="pill">not requested in this run</span>',
      `${tl.note || ''} Deep candidates: ${(tl.trained_from_scratch || []).join(', ') || 'none'}`),
  ].join('');

  function block(title, pillHtml, text) {
    return `<h3 style="margin-top:.7rem">${esc(title)}</h3><div class="pillrow">${pillHtml}</div>
      <p class="hint">${esc(String(text || ''))}</p>`;
  }

  /* ------------------------------------------------------- the card, verbatim */
  const btn = $('#loadCard'), view = $('#cardView'), state = $('#cardState');
  if (btn) {
    btn.addEventListener('click', async () => {
      btn.disabled = true; state.textContent = 'reading models/MODEL_CARD.md…';
      try {
        const j = await api('/api/report/MODEL_CARD.md');
        view.innerHTML = VFA.mdLite(j.markdown);
        view.hidden = false;
        state.textContent = `rendered from ${j.name}${j.written_at ? ` · written ${j.written_at}` : ''}`;
        btn.textContent = 'Reload the card';
      } catch (err) {
        state.textContent = `not available: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  }
})();
