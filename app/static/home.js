/* Overview page. Every fact shown here is read from /api/model + /api/health, i.e. from the
   artifacts that `python train.py` wrote - the page adds no interpretation of its own. */
(async function () {
  const { $, esc, load } = VFA;
  const s = await load();
  VFA.decorate(s);

  const h = s.health || {};
  const m = s.model || {};
  const info = m.model || {};
  const ds = m.dataset || {};
  const insp = ds.inspection || {};
  const split = m.split || {};

  const line = $('#datasetLine');
  if (line) {
    const sizes = Object.entries(insp.sizes || {}).map(([k, v]) => `${k} px (${v.toLocaleString()} files)`);
    const bits = [];
    if (insp.n_valid) bits.push(`${Number(insp.n_valid).toLocaleString()} valid images in ${Object.keys(insp.classes_found || {}).length} classes: `
      + Object.entries(insp.classes_found || {}).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(', '));
    if (sizes.length && sizes.length < 4) bits.push(sizes.join(', '));
    if (insp.n_groups) bits.push(`${Number(insp.n_groups).toLocaleString()} capture groups (frames of the same source never split across train and test)`);
    if (insp.n_corrupt != null) bits.push(`${insp.n_corrupt} unreadable files rejected during inspection`);
    if ((insp.notes || []).length) bits.push(...insp.notes);
    line.innerHTML = bits.length
      ? bits.map((b) => esc(String(b))).join('. ') + '.'
      : `No inspection record found in ${esc(h.models_dir || 'models/')}. Run <code>python train.py</code> once — this service only loads what that run wrote.`;
  }

  const line2 = $('#dataLine2');
  if (line2) {
    const sz = split.sizes || {};
    const bits = [];
    if (split.strategy) bits.push(`splits: ${split.strategy}`);
    if (sz.train) bits.push(`${Number(sz.train).toLocaleString()} train / ${Number(sz.val).toLocaleString()} validation / ${Number(sz.test).toLocaleString()} test`);
    if (split.seed != null) bits.push(`seed ${split.seed}`);
    bits.push(`${(m.comparison || []).length} candidate models compared on those identical splits`);
    if ((h.samples_available || []).length) bits.push(`${h.samples_available.length} test-split images ready to try on the Analyze page`);
    line2.innerHTML = bits.map((b) => esc(String(b))).join(' · ');
  }

  const bits = $('#modelBits');
  if (bits) {
    bits.innerHTML = info.architecture
      ? `Deployed: <code>${esc(info.model_name || info.architecture)}</code> · ${esc(info.framework || '')} · `
        + `${esc(String(info.descriptor || ''))} · ${esc(String(info.feature_dim || ''))} features · `
        + `${fmtMb(info.model_size_mb)} · ${esc(String(info.inference_ms_per_image || '?'))} ms/image`
      : 'No trained model in this build yet.';
  }

  const ul = $('#honesty');
  if (ul) {
    const items = (m.honesty || []).slice();
    if (!h.model_loaded) {
      items.push('No prediction is offered by this build: the service reports that state instead of guessing.');
    }
    ul.innerHTML = (items.length ? items : ['Prediction is probabilistic and is not proof of image origin.'])
      .map((t) => `<li>${esc(t)}</li>`).join('');
  }

  function fmtMb(v) {
    return v == null ? 'size n/a' : `${Number(v).toFixed(v < 1 ? 3 : 1)} MB`;
  }
})();
