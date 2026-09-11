/* Metrics page. Every number here is read from models/metrics.json through /api/metrics -
   the very document the selection was made from. Nothing on this page is recomputed or edited. */
(async function () {
  const { $, esc, fmt, api, table, kvTable, load } = VFA;
  const s = await load();
  VFA.decorate(s);

  const slots = ['#compare', '#selNote', '#selected', '#confusion', '#reliability',
                 '#ablations', '#secondary', '#repro', '#figs1', '#curveFigs', '#confFig', '#ablationNotes'];
  const noArtifact = (why) => slots.forEach((q) => {
    const el = $(q);
    if (el) el.innerHTML = `<p class="hint">no metrics to show${why ? `: ${esc(why)}` : ''}. This page renders
      <code>models/metrics.json</code>, which <code>python train.py</code> writes.</p>`;
  });

  let mm = null;
  try { mm = await api('/api/metrics'); } catch (err) { noArtifact(err.message); return; }
  if (!mm || !mm.tasks) { noArtifact('the file is empty or partial'); return; }

  const task = mm.tasks.ai_detector || {};
  const cmp = mm.comparison || [];
  const sel = mm.selection || {};
  const met = task.metrics || {};
  const labels = task.classes || ['real', 'fake'];

  /* ---------------------------------------------------------------- all candidates */
  const rows = cmp.map((c) => {
    const cm = c.metrics || {};
    const r = [
      c.rank,
      { raw: `<b>${esc(c.name)}</b>${c.selected ? ' <span class="pill ok">selected</span>' : ''}
         ${c.tie_note ? `<div class="hint">tie-break: ${esc(c.tie_note)}</div>` : ''}` },
      c.family,
      { n: c.accuracy, d: 4 }, { n: c.f1, d: 4 }, { n: c.roc_auc, d: 4 },
      { n: cm.recall, d: 4 }, { n: cm.precision, d: 4 }, { n: cm.false_positive_rate, d: 4 },
      { n: cm.log_loss, d: 4 },
      { n: c.inference_ms, d: 2 }, { n: c.size_mb, d: 3 },
      { raw: `<b>${fmt(c.final_score, 4)}</b>` },
    ];
    if (c.selected) r.__class = 'selected';
    return r;
  });
  $('#compare').innerHTML = table(
    ['rank', 'candidate', 'family', 'accuracy', 'F1', 'ROC-AUC', 'recall', 'precision', 'FPR', 'log loss', 'ms/img', 'MB', 'score'],
    rows,
    { num: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], empty: 'no candidates recorded',
      caption: `${cmp.length} candidates · identical splits · the test split was evaluated once per model` });

  const w = sel.weights || {};
  $('#selNote').innerHTML = `
    <p class="prose"><b>${esc(sel.selected)}</b> (${esc(sel.selected_family || '?')}) was selected — ${esc(sel.reason || '')}</p>
    <p class="hint">weights: ${Object.keys(w).map((k) => `${esc(k)} ${fmt(w[k], 2)}`).join(' · ')}
      · selected rank ${sel.selected_rank} of ${cmp.length} · gap to runner-up ${fmt(sel.score_gap_to_runner_up, 4)}
      · tie-break applied: ${sel.tie_break_applied ? 'yes' : 'no'}</p>
    <p class="hint">Speed = min(1, 3.33 / ms-per-image); Efficiency = 0.5·(1 − log₁₀(max(MB, 1))/3) + 0.5·pipeline term.
      The formula is published so the choice can be re-checked — it was not tuned after seeing the test split.</p>`;

  /* ---------------------------------------------------------------- selected detail */
  $('#selected').innerHTML = table(['metric (test split)', 'value'],
    Object.keys(met).filter((k) => k !== 'confusion_matrix').map((k) => [
      { raw: `<code>${esc(k)}</code>` },
      typeof met[k] === 'number' ? { n: met[k], d: Number.isInteger(met[k]) ? 0 : 4 } : { raw: esc(String(met[k])) },
    ]), { num: [1], empty: 'no metrics recorded for the selected model' });

  const cm = met.confusion_matrix;
  if (Array.isArray(cm) && cm.length === 2) {
    const fp = cm[0][1], fn = cm[1][0], tn = cm[0][0], tp = cm[1][1];
    $('#confusion').innerHTML = table(['true \\ predicted', labels[0], labels[1]], [
      [`actual ${labels[0]}`, { n: tn, d: 0 }, { n: fp, d: 0 }],
      [`actual ${labels[1]}`, { n: fn, d: 0 }, { n: tp, d: 0 }],
    ], { num: [1, 2], caption: 'counts on the held-out test split' })
      + `<p class="hint">An authentic image is wrongly called generated ${fmt(fp / (tn + fp), 3)} of the time;
         a generated image is missed ${fmt(fn / (fn + tp), 3)} of the time. Those two numbers are the real cost of
         a probabilistic decision, and they depend on the threshold — which was chosen on validation data.</p>`;
    $('#confFig').innerHTML = fig('confusion_selected.png', 'confusion matrix of the selected model');
  }

  /* ---------------------------------------------------------------- calibration */
  const rel = task.reliability || [];
  $('#reliability').innerHTML = table(['predicted-probability bin', 'images', 'mean predicted', 'observed rate', 'gap'],
    rel.map((b) => [b.bin, { n: b.n, d: 0 }, { n: b.predicted, d: 3 }, { n: b.observed, d: 3 },
      { n: Math.abs(Number(b.predicted || 0) - Number(b.observed || 0)), d: 3 }]),
    { num: [1, 2, 3, 4], empty: 'no reliability table recorded' })
    + `<p class="hint">Expected calibration error <b>${fmt(task.calibration_ece, 4)}</b> — the mean absolute gap in the
       last column. Small means the probability can be read as a probability; it says nothing about how often the
       class is right, which is what accuracy measures.</p>`;
  $('#figs1').innerHTML = fig('roc_test.png', 'ROC curve on the test split')
    + fig('reliability_production.png', 'reliability diagram of the deployed model');

  const deepNames = cmp.filter((c) => c.family === 'deep').map((c) => c.name);
  $('#curveFigs').innerHTML = deepNames.length
    ? `<h3 style="margin-top:.7rem">training curves · deep candidates</h3><div class="figs">${
        deepNames.map((n) => fig(`curves_${n}.png`, `training curves · ${n}`)).join('')}</div>
       <p class="hint">loss and metric per epoch, as recorded during training.</p>`
    : '';

  /* ---------------------------------------------------------------- ablations */
  const ab = mm.ablations || [];
  $('#ablations').innerHTML = table(['arm', 'accuracy', 'F1', 'ROC-AUC', 'recall', 'ms/img', 'MB'],
    ab.map((a) => [{ raw: `<code>${esc(a.id)}</code>` }, { n: a.accuracy, d: 4 }, { n: a.f1, d: 4 },
      { n: a.roc_auc, d: 4 }, { n: a.recall, d: 4 }, { n: a.inference_ms, d: 2 }, { n: a.size_mb, d: 3 }]),
    { num: [1, 2, 3, 4, 5, 6], empty: 'no ablation arms recorded in this run' });
  $('#ablationNotes').innerHTML = ab.map((a) => `<p class="hint"><code>${esc(a.id)}</code> — ${esc(a.note || 'no note recorded')}
     ${a.error ? `<b>error:</b> ${esc(String(a.error))}` : ''}</p>`).join('');

  /* ---------------------------------------------------------------- second tasks */
  const gen = mm.generator_classifier || {};
  const face = mm.face_model || {};
  const tl = mm.transfer_learning || {};
  $('#secondary').innerHTML = [
    ['Generator fingerprinting', gen.available === false ? 'not trainable on this dataset' : 'available',
      `${gen.reason || 'no record'}${(gen.classes && Object.keys(gen.classes).length) ? ` Classes: ${Object.keys(gen.classes).join(', ')}.` : ''}`],
    ['Face model', face.trained ? 'trained' : 'analytical CV module, not a trained classifier',
      `${face.reason || ''}${face.analytical_cv_module ? ` Implementation: ${face.analytical_cv_module}.` : ''
       }${face.identity_model_available ? ' Identity model available.' : ' No identity or biometric model.'}`],
    ['Transfer learning', tl.requested
      ? ((tl.weights_applied_for || []).length ? `weights applied for ${tl.weights_applied_for.length} deep candidates` : 'weights unreachable in the training environment')
      : 'not requested',
      `${tl.note || ''}${(tl.trained_from_scratch || []).length ? ` Trained from scratch: ${tl.trained_from_scratch.join(', ')}.` : ''}`],
  ].map(([t, state, why]) => `<h3 style="margin-top:.7rem">${esc(t)}</h3>
      <div class="pillrow"><span class="pill warn">${esc(state)}</span></div>
      <p class="hint">${esc(why)}</p>`).join('')
    + '<p class="hint"><a href="/model">provenance and limitations on the model card →</a></p>';

  /* ---------------------------------------------------------------- reproducibility */
  const rp = mm.reproducibility || {};
  const cfg = rp.config || {};
  const flat = {};
  Object.keys(rp).filter((k) => k !== 'config').forEach((k) => { flat[k] = rp[k]; });
  ['train_frac', 'val_frac', 'epochs', 'batch_size', 'lr', 'patience', 'pretrained', 'deep', 'classical',
   'max_images_per_class', 'run_name', 'ablations']
    .filter((k) => cfg[k] !== undefined && flat[k] === undefined)
    .forEach((k) => { flat[k] = Array.isArray(cfg[k]) ? `${cfg[k].length} arms recorded` : cfg[k]; });
  $('#repro').innerHTML = kvTable(flat)
    + `<p class="hint">Written by the run itself (schema ${esc(String(mm.schema_version || '?'))}, generated
       ${esc(String(mm.generated_at || '?'))}). Re-verify the deployed artifact with
       <code>python -m vfa evaluate</code>: it re-scores the whole held-out split from <code>models/</code> and writes
       <code>reports/production_evaluation.md</code> — view that <a href="/reports">on the reports page</a>.</p>`;

  function fig(name, alt) {
    return `<figure><img src="/api/report/${encodeURIComponent(name)}" alt="${esc(alt)}" loading="lazy"
      onerror="this.closest('figure').hidden = true" /><figcaption>${esc(alt)}</figcaption></figure>`;
  }
})();
