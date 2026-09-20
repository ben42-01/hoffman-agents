/* How a conscious agent works: a step-by-step, exact walk through Hoffman &
 * Prakash's six-tuple (X, G, P, D, A, N).
 *
 * Everything on this page is computed, not animated for effect: the kernels are
 * the matrices shown, each step samples from them with a seeded RNG, and the
 * long-run section computes the joint kernel Q and its stationary distribution
 * exactly. The default agent is the one used in the library's tests and in
 * examples/12_ergodic_diagnostics, so the numbers match the library.
 */
(function () {
  'use strict';

  /* ────────────── math ────────────── */

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const X = ['calm', 'alert'], G = ['stay', 'move'], W = ['left', 'right'];
  const JOINT = [];
  for (const x of X) for (const w of W) JOINT.push(`${x}|${w}`);

  const BASE_P = { left: [[0.9, 0.1], [0.6, 0.4]], right: [[0.3, 0.7], [0.1, 0.9]] };

  function buildAgent(params) {
    const s = params.stickiness;
    const P = {};
    for (const w of W) P[w] = BASE_P[w].map((row, i) => row.map((v, j) => (1 - s) * v + (i === j ? s : 0)));
    const D = [[1 - params.moveCalm, params.moveCalm], [1 - params.moveAlert, params.moveAlert]];
    const r = params.reliability;
    const A = { stay: [[1, 0], [0, 1]], move: [[1 - r, r], [r, 1 - r]] };
    return { P, D, A };
  }

  // Q((x,w),(x',w')) = P[w](x,x') * sum_g D(x',g) A[g](w,w')
  function jointKernel(agent) {
    const Q = JOINT.map(() => new Array(JOINT.length).fill(0));
    X.forEach((x, i) => W.forEach((w, k) => {
      X.forEach((_, j) => W.forEach((__, l) => {
        let act = 0;
        G.forEach((g, gi) => { act += agent.D[j][gi] * agent.A[g][k][l]; });
        Q[i * 2 + k][j * 2 + l] = agent.P[w][i][j] * act;
      }));
    }));
    return Q;
  }

  function leftApply(mu, M) {
    return M[0].map((_, j) => mu.reduce((s, m, i) => s + m * M[i][j], 0));
  }

  function stationary(Q) {
    let pi = Q.map(() => 1 / Q.length);
    for (let it = 0; it < 20000; it++) {
      const stepped = leftApply(pi, Q);
      const next = pi.map((p, i) => 0.5 * (p + stepped[i]));
      const diff = next.reduce((s, v, i) => s + Math.abs(v - pi[i]), 0);
      pi = next;
      if (diff < 1e-14) break;
    }
    const total = pi.reduce((a, b) => a + b, 0);
    return pi.map(p => p / total);
  }

  // |lambda_2| by deflated power iteration (same method as the library).
  function lambda2(Q, pi) {
    const r = mulberry32(0x5EED);
    let x = Q.map(() => r() - 0.5);
    const deflate = v => { const s = v.reduce((a, b) => a + b, 0); return v.map((vi, i) => vi - s * pi[i]); };
    const norm = v => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    x = deflate(x);
    let n = norm(x);
    if (n < 1e-12) return 0;
    x = x.map(v => v / n);
    const logs = [];
    for (let it = 0; it < 400; it++) {
      x = deflate(leftApply(x, Q));
      n = norm(x);
      if (n < 1e-300) return 0;
      logs.push(Math.log(n));
      x = x.map(v => v / n);
    }
    const tail = logs.slice(-50);
    return Math.min(1, Math.exp(tail.reduce((a, b) => a + b, 0) / tail.length));
  }

  function kron(A, B) {
    const out = [];
    for (let i = 0; i < A.length; i++) for (let k = 0; k < B.length; k++) {
      const row = [];
      for (let j = 0; j < A[0].length; j++) for (let l = 0; l < B[0].length; l++) row.push(A[i][j] * B[k][l]);
      out.push(row);
    }
    return out;
  }

  const sample = (row, u) => {
    let acc = 0;
    for (let j = 0; j < row.length; j++) { acc += row[j]; if (u < acc) return j; }
    return row.length - 1;
  };

  /* ────────────── state ────────────── */

  const PHASES = [
    { key: 'perceive', kernel: 'P', title: 'Perceive', formula: "x′ ~ P[w](x, ·)",
      text: 'The world state w selects one perception kernel. Its row for the current experience x is a probability distribution over the next experience x′.' },
    { key: 'decide', kernel: 'D', title: 'Decide', formula: "g ~ D(x′, ·)",
      text: 'The decision kernel D turns the new experience x′ into a distribution over actions, and one action g is drawn from it.' },
    { key: 'act', kernel: 'A', title: 'Act', formula: "w′ ~ A[g](w, ·)",
      text: 'The chosen action selects an action kernel. Its row for the current world state w is a distribution over the next world state w′. The cycle is complete: N ← N + 1.' },
  ];
  const SPEEDS = { slow: 1600, normal: 900, fast: 300 };

  const S = {
    params: { stickiness: 0, moveCalm: 0.2, moveAlert: 0.8, reliability: 1 },
    seed: 7,
    rng: null,
    agent: null, Q: null, pi: null, lam: 0,
    x: 0, w: 0, xNext: null, g: null, wNext: null,
    phase: 0, N: 0,
    counts: [0, 0, 0, 0],
    tv: [],
    log: [],
    lastDraw: null, lastPhase: -1,
    playing: false, timer: null, speed: 'normal',
    initialised: false,
  };

  /* ────────────── helpers ────────────── */

  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 3) => v.toFixed(d);
  const el = (tag, attrs = {}, text) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };

  function recompute() {
    S.agent = buildAgent(S.params);
    S.Q = jointKernel(S.agent);
    S.pi = stationary(S.Q);
    S.lam = lambda2(S.Q, S.pi);
  }

  function resetRun() {
    stop();
    S.rng = mulberry32(S.seed);
    S.x = 0; S.w = 0; S.xNext = S.g = S.wNext = null;
    S.phase = 0; S.N = 0;
    S.counts = [0, 0, 0, 0];
    S.tv = [];
    S.log = [];
    S.lastDraw = null; S.lastPhase = -1;
    renderAll();
  }

  function tvDistance() {
    const total = S.counts.reduce((a, b) => a + b, 0);
    if (total === 0) return 1;
    return 0.5 * S.counts.reduce((s, c, i) => s + Math.abs(c / total - S.pi[i]), 0);
  }

  // Advance one phase (animated mode).
  function stepPhase() {
    const u = S.rng();
    S.lastPhase = S.phase;
    if (S.phase === 0) {
      S.g = null;
      const row = S.agent.P[W[S.w]][S.x];
      S.xNext = sample(row, u);
      S.lastDraw = { row, u, pick: S.xNext, labels: X, rowLabel: X[S.x], kernelLabel: `P[${W[S.w]}]` };
    } else if (S.phase === 1) {
      const row = S.agent.D[S.xNext];
      S.g = sample(row, u);
      S.lastDraw = { row, u, pick: S.g, labels: G, rowLabel: X[S.xNext], kernelLabel: 'D' };
    } else {
      const row = S.agent.A[G[S.g]][S.w];
      S.wNext = sample(row, u);
      S.lastDraw = { row, u, pick: S.wNext, labels: W, rowLabel: W[S.w], kernelLabel: `A[${G[S.g]}]` };
    }
    const done = S.phase === 2;
    renderStep();
    if (done) completeCycle(true);
    S.phase = done ? 0 : S.phase + 1;
    if (done) setTimeout(renderStage, 10);
  }

  function completeCycle(logIt) {
    const from = `${X[S.x]}|${W[S.w]}`;
    S.x = S.xNext; S.w = S.wNext;
    S.N++;
    S.counts[S.x * 2 + S.w]++;
    if (logIt) {
      S.log.unshift(`N=${S.N}: (${from}) → x′=${X[S.x]} → g=${G[S.g]} → w′=${W[S.w]}`);
      S.log.length = Math.min(S.log.length, 6);
    }
    recordTv();
    renderLog();
    renderLongRun();
  }

  function recordTv() {
    const n = S.N;
    const last = S.tv.length ? S.tv[S.tv.length - 1].n : 0;
    if (n < 50 || n - last >= Math.max(1, Math.floor(n / 150))) S.tv.push({ n, tv: tvDistance() });
  }

  // Fast mode: full cycles without animation.
  function runCycles(k) {
    stop();
    for (let i = 0; i < k; i++) {
      S.xNext = sample(S.agent.P[W[S.w]][S.x], S.rng());
      S.g = sample(S.agent.D[S.xNext], S.rng());
      S.wNext = sample(S.agent.A[G[S.g]][S.w], S.rng());
      const x = S.xNext, w = S.wNext;
      S.x = x; S.w = w; S.N++;
      S.counts[x * 2 + w]++;
      if (S.N < 50 || i % Math.max(1, Math.floor(S.N / 150)) === 0) S.tv.push({ n: S.N, tv: tvDistance() });
    }
    S.phase = 0; S.lastDraw = null; S.lastPhase = -1; S.g = null;
    S.log.unshift(`… ran ${k.toLocaleString()} cycles without animation`);
    S.log.length = Math.min(S.log.length, 6);
    renderAll();
  }

  function play() {
    if (S.playing) return;
    S.playing = true;
    $('howPlay').textContent = 'Pause';
    const tick = () => {
      if (!S.playing) return;
      stepPhase();
      S.timer = setTimeout(tick, SPEEDS[S.speed]);
    };
    tick();
  }

  function stop() {
    S.playing = false;
    clearTimeout(S.timer);
    const b = $('howPlay');
    if (b) b.textContent = 'Play';
  }

  /* ────────────── stage (the agent diagram) ────────────── */

  function renderStage() {
    const svg = $('howStage');
    if (!svg) return;
    svg.innerHTML = '';
    const cols = [
      { title: 'World  W', items: W, x: 90, active: S.w },
      { title: 'Experience  X', items: X, x: 310, active: S.xNext ?? S.x },
      { title: 'Action  G', items: G, x: 530, active: S.g },
    ];
    // arrow of the phase that just ran is highlighted
    const justRan = S.lastDraw ? S.lastPhase : -1;
    const arrows = [
      { from: 0, to: 1, label: 'P', phase: 0 },
      { from: 1, to: 2, label: 'D', phase: 1 },
      { from: 2, to: 0, label: 'A', phase: 2 },
    ];
    const defs = el('defs');
    for (const [id, fill] of [['howArrow', '#334155'], ['howArrowOn', '#60a5fa']]) {
      const marker = el('marker', { id, viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse' });
      marker.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill }));
      defs.appendChild(marker);
    }
    svg.appendChild(defs);

    for (const a of arrows) {
      const on = a.phase === justRan;
      const g = el('g', { class: `how-arrow${on ? ' on' : ''}` });
      let d;
      if (a.to > a.from) {
        d = `M${cols[a.from].x + 70},110 L${cols[a.to].x - 70},110`;
      } else {
        d = `M${cols[2].x},200 C${cols[2].x},262 ${cols[0].x},262 ${cols[0].x},200`;
      }
      g.appendChild(el('path', { d, 'marker-end': on ? 'url(#howArrowOn)' : 'url(#howArrow)' }));
      const lx = a.to > a.from ? (cols[a.from].x + cols[a.to].x) / 2 : (cols[0].x + cols[2].x) / 2;
      const ly = a.to > a.from ? 100 : 252;
      g.appendChild(el('text', { x: lx, y: ly, 'text-anchor': 'middle', class: 'how-arrow-label' }, a.label));
      svg.appendChild(g);
    }

    cols.forEach((c) => {
      svg.appendChild(el('text', { x: c.x, y: 36, 'text-anchor': 'middle', class: 'how-col-title' }, c.title));
      c.items.forEach((name, i) => {
        const cy = 90 + i * 70;
        const on = c.active === i;
        const g = el('g', { class: `how-node${on ? ' on' : ''}` });
        g.appendChild(el('rect', { x: c.x - 60, y: cy - 22, width: 120, height: 44, rx: 10 }));
        g.appendChild(el('text', { x: c.x, y: cy + 5, 'text-anchor': 'middle' }, name));
        svg.appendChild(g);
      });
    });
    svg.appendChild(el('text', { x: 310, y: 292, 'text-anchor': 'middle', class: 'how-counter' }, `N = ${S.N}  (completed perceive → decide → act cycles)`));
  }

  /* ────────────── math panel ────────────── */

  function matrixHtml(M, rowLabels, colLabels, highlightRow, highlightCol) {
    let h = '<table class="how-matrix"><thead><tr><th></th>';
    h += colLabels.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    M.forEach((row, i) => {
      h += `<tr class="${i === highlightRow ? 'hl' : ''}"><th>${rowLabels[i]}</th>`;
      h += row.map((v, j) => `<td class="${i === highlightRow && j === highlightCol ? 'pick' : ''}">${fmt(v, 2)}</td>`).join('');
      h += '</tr>';
    });
    return h + '</tbody></table>';
  }

  function renderStep() {
    const ranPhase = S.lastDraw ? S.phase : null;
    const shown = ranPhase ?? S.phase;
    const p = PHASES[shown];
    document.querySelectorAll('.how-phase').forEach((e, i) => e.classList.toggle('active', i === shown));
    $('howStepTitle').textContent = `Step ${shown + 1} · ${p.title}`;
    $('howFormula').textContent = p.formula;
    $('howStepText').textContent = p.text;

    let M, rows, cols, hr = null, hc = null;
    if (p.key === 'perceive') {
      M = S.agent.P[W[S.w]]; rows = X; cols = X; hr = S.x;
      $('howKernelName').textContent = `P[w = ${W[S.w]}]  (rows: current x, columns: next x′)`;
    } else if (p.key === 'decide') {
      M = S.agent.D; rows = X; cols = G; hr = S.xNext;
      $('howKernelName').textContent = 'D  (rows: experience x′, columns: action g)';
    } else {
      const g = S.g ?? 0;
      M = S.agent.A[G[g]]; rows = W; cols = W; hr = S.w;
      $('howKernelName').textContent = `A[g = ${G[g]}]  (rows: current w, columns: next w′)`;
    }
    if (S.lastDraw) hc = S.lastDraw.pick;
    $('howMatrix').innerHTML = matrixHtml(M, rows, cols, hr, hc);
    renderDraw();
    renderStage();
    renderLog();
  }

  function renderLog() {
    $('howLog').innerHTML = S.log.map(l => `<li>${l}</li>`).join('') || '<li class="muted">Press Step or Play to run the agent.</li>';
  }

  // The sampling bar: the chosen row laid out on [0, 1], with the uniform draw u.
  function renderDraw() {
    const box = $('howDraw');
    if (!S.lastDraw) {
      box.innerHTML = '<p class="muted">Each step draws a uniform number u in [0, 1) and picks the outcome whose segment of the row contains u.</p>';
      return;
    }
    const { row, u, pick, labels, rowLabel, kernelLabel } = S.lastDraw;
    let acc = 0;
    const segs = row.map((p, j) => {
      const s = `<div class="how-seg${j === pick ? ' pick' : ''}" style="width:${(p * 100).toFixed(3)}%" title="${labels[j]}: ${fmt(p, 3)}"><span>${p >= 0.12 ? `${labels[j]} ${fmt(p, 2)}` : ''}</span></div>`;
      acc += p;
      return s;
    }).join('');
    let lo = 0;
    for (let j = 0; j < pick; j++) lo += row[j];
    box.innerHTML = `
      <div class="how-bar">${segs}<div class="how-u" style="left:${(u * 100).toFixed(2)}%"></div></div>
      <p class="how-draw-text">Row <b>${kernelLabel}(${rowLabel}, ·)</b>. The draw <b>u = ${fmt(u, 3)}</b> falls in
      [${fmt(lo, 2)}, ${fmt(lo + row[pick], 2)}), so the outcome is <b>${labels[pick]}</b>.</p>`;
  }

  /* ────────────── long-run section ────────────── */

  function renderQ() {
    const cells = S.Q.map((row, i) => `<tr><th>${JOINT[i]}</th>${row.map(v =>
      `<td style="background: rgba(57,135,229,${(0.08 + 0.7 * v).toFixed(3)})">${fmt(v, 3)}</td>`).join('')}</tr>`).join('');
    $('howQ').innerHTML = `<table class="how-matrix how-q"><thead><tr><th>from \\ to</th>${JOINT.map(j => `<th>${j}</th>`).join('')}</tr></thead><tbody>${cells}</tbody></table>`;
    $('howLambda').textContent = `|λ₂| = ${fmt(S.lam)}: the chain forgets its starting state by a factor of about ${fmt(S.lam, 2)} per cycle.`;
  }

  function showTip(evt, html) {
    const tip = $('howTip');
    tip.innerHTML = html;
    tip.style.display = 'block';
    const r = tip.parentElement.getBoundingClientRect();
    tip.style.left = `${Math.min(evt.clientX - r.left + 12, r.width - tip.offsetWidth - 4)}px`;
    tip.style.top = `${evt.clientY - r.top + 12}px`;
  }
  const hideTip = () => { $('howTip').style.display = 'none'; };

  function renderBars() {
    const svg = $('howBars');
    svg.innerHTML = '';
    const Wd = 560, H = 240, m = { l: 44, r: 12, t: 16, b: 34 };
    const total = S.counts.reduce((a, b) => a + b, 0);
    const emp = S.counts.map(c => (total ? c / total : 0));
    const ymax = Math.max(0.5, Math.ceil(Math.max(...S.pi, ...emp) * 10) / 10);
    const y = v => m.t + (H - m.t - m.b) * (1 - v / ymax);
    for (let t = 0; t <= ymax + 1e-9; t += 0.1) {
      svg.appendChild(el('line', { x1: m.l, x2: Wd - m.r, y1: y(t), y2: y(t), class: 'how-gridline' }));
      svg.appendChild(el('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end', class: 'how-tick' }, t.toFixed(1)));
    }
    const band = (Wd - m.l - m.r) / JOINT.length;
    const bw = 22, gap = 2;
    const barPath = (x0, v) => {
      const top = y(v), base = y(0), r = Math.min(4, Math.max(0, base - top));
      return `M${x0},${base} L${x0},${top + r} Q${x0},${top} ${x0 + r},${top} L${x0 + bw - r},${top} Q${x0 + bw},${top} ${x0 + bw},${top + r} L${x0 + bw},${base} Z`;
    };
    JOINT.forEach((name, i) => {
      const cx = m.l + band * i + band / 2;
      const series = [
        { v: S.pi[i], cls: 'how-bar-exact', label: 'exact π' },
        { v: emp[i], cls: 'how-bar-sim', label: 'simulated' },
      ];
      series.forEach((sr, k) => {
        const x0 = cx - bw - gap / 2 + k * (bw + gap);
        const p = el('path', { d: barPath(x0, sr.v), class: sr.cls });
        const hit = el('rect', { x: x0 - 4, y: m.t, width: bw + 8, height: H - m.t - m.b, fill: 'transparent' });
        const tip = `<b>${name}</b><br>exact π: ${fmt(S.pi[i])}<br>simulated: ${fmt(emp[i])} (${S.counts[i].toLocaleString()} of ${total.toLocaleString()})`;
        hit.addEventListener('mousemove', e => showTip(e, tip));
        hit.addEventListener('mouseleave', hideTip);
        svg.appendChild(p);
        svg.appendChild(hit);
      });
      svg.appendChild(el('text', { x: cx, y: H - m.b + 18, 'text-anchor': 'middle', class: 'how-tick' }, name));
    });
    svg.appendChild(el('line', { x1: m.l, x2: Wd - m.r, y1: y(0), y2: y(0), class: 'how-axis' }));

    $('howPiTable').innerHTML = `<table class="how-matrix"><thead><tr><th>state (x|w)</th><th>exact π</th><th>simulated</th><th>count</th></tr></thead><tbody>${
      JOINT.map((n, i) => `<tr><th>${n}</th><td>${fmt(S.pi[i])}</td><td>${fmt(emp[i])}</td><td>${S.counts[i].toLocaleString()}</td></tr>`).join('')}</tbody></table>`;
  }

  function renderTv() {
    const svg = $('howTv');
    svg.innerHTML = '';
    const Wd = 560, H = 200, m = { l: 44, r: 16, t: 14, b: 34 };
    const pts = S.tv;
    const nmax = Math.max(10, S.N);
    const lx = n => m.l + (Wd - m.l - m.r) * (Math.log10(n) / Math.log10(nmax));
    const y = v => m.t + (H - m.t - m.b) * (1 - v);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg.appendChild(el('line', { x1: m.l, x2: Wd - m.r, y1: y(t), y2: y(t), class: 'how-gridline' }));
      svg.appendChild(el('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end', class: 'how-tick' }, t.toFixed(2)));
    }
    for (let d = 1; d <= nmax; d *= 10) {
      svg.appendChild(el('text', { x: lx(d), y: H - m.b + 18, 'text-anchor': 'middle', class: 'how-tick' }, d.toLocaleString()));
    }
    svg.appendChild(el('text', { x: Wd - m.r, y: H - 4, 'text-anchor': 'end', class: 'how-tick' }, 'cycles N (log scale)'));
    svg.appendChild(el('line', { x1: m.l, x2: Wd - m.r, y1: y(0), y2: y(0), class: 'how-axis' }));
    if (pts.length === 0) return;
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${lx(p.n).toFixed(1)},${y(Math.min(1, p.tv)).toFixed(1)}`).join(' ');
    svg.appendChild(el('path', { d, class: 'how-line' }));
    const last = pts[pts.length - 1];
    svg.appendChild(el('circle', { cx: lx(last.n), cy: y(Math.min(1, last.tv)), r: 4, class: 'how-dot' }));

    const cross = el('line', { y1: m.t, y2: H - m.b, class: 'how-cross', visibility: 'hidden' });
    svg.appendChild(cross);
    const hit = el('rect', { x: m.l, y: m.t, width: Wd - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' });
    hit.addEventListener('mousemove', (e) => {
      const r = svg.getBoundingClientRect();
      const px = (e.clientX - r.left) * (Wd / r.width);
      let best = pts[0];
      for (const p of pts) if (Math.abs(lx(p.n) - px) < Math.abs(lx(best.n) - px)) best = p;
      cross.setAttribute('x1', lx(best.n)); cross.setAttribute('x2', lx(best.n));
      cross.setAttribute('visibility', 'visible');
      showTip(e, `after <b>${best.n.toLocaleString()}</b> cycles<br>distance to π: ${fmt(best.tv)}`);
    });
    hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    svg.appendChild(hit);
    $('howTvNow').textContent = `Now: ${fmt(last.tv)} after ${S.N.toLocaleString()} cycles.`;
  }

  function renderLongRun() {
    renderBars();
    renderTv();
  }

  /* ────────────── combination ────────────── */

  function renderKron() {
    const P = S.agent.P.left;
    const K = kron(P, P);
    const labels = [];
    for (const a of X) for (const b of X) labels.push(`${a}·${b}`);
    const cells = K.map((row, i) => `<tr><th>${labels[i]}</th>${row.map((v, j) => {
      const block = (Math.floor(i / 2) + Math.floor(j / 2)) % 2 === 0;
      return `<td class="${block ? 'blk' : ''}">${fmt(v, 3)}</td>`;
    }).join('')}</tr>`).join('');
    $('howKron').innerHTML = `<table class="how-matrix how-kron"><thead><tr><th></th>${labels.map(l => `<th>${l}</th>`).join('')}</tr></thead><tbody>${cells}</tbody></table>`;
    $('howKronFactor').innerHTML = matrixHtml(P, X, X, null, null);
    $('howKronFactor2').innerHTML = matrixHtml(P, X, X, null, null);
  }

  /* ────────────── wiring ────────────── */

  function renderAll() {
    renderStep();
    renderQ();
    renderLongRun();
    renderKron();
  }

  function bindSlider(id, key, fmtLabel) {
    const input = $(id), out = $(`${id}Val`);
    const update = () => {
      S.params[key] = parseFloat(input.value);
      out.textContent = fmtLabel(S.params[key]);
      recompute();
      resetRun();
    };
    input.addEventListener('input', update);
    out.textContent = fmtLabel(S.params[key]);
  }

  function init() {
    if (S.initialised) { renderAll(); return; }
    S.initialised = true;
    recompute();
    $('howStep').addEventListener('click', () => { stop(); stepPhase(); });
    $('howPlay').addEventListener('click', () => (S.playing ? stop() : play()));
    $('howReset').addEventListener('click', resetRun);
    $('howRun1k').addEventListener('click', () => runCycles(1000));
    $('howRun10k').addEventListener('click', () => runCycles(10000));
    $('howSpeed').addEventListener('change', (e) => { S.speed = e.target.value; });
    bindSlider('howStick', 'stickiness', v => v.toFixed(2));
    bindSlider('howMoveCalm', 'moveCalm', v => v.toFixed(2));
    bindSlider('howMoveAlert', 'moveAlert', v => v.toFixed(2));
    bindSlider('howRel', 'reliability', v => v.toFixed(2));
    $('howDefaults').addEventListener('click', () => {
      Object.assign(S.params, { stickiness: 0, moveCalm: 0.2, moveAlert: 0.8, reliability: 1 });
      for (const [id, key] of [['howStick', 'stickiness'], ['howMoveCalm', 'moveCalm'], ['howMoveAlert', 'moveAlert'], ['howRel', 'reliability']]) {
        $(id).value = S.params[key];
        $(`${id}Val`).textContent = S.params[key].toFixed(2);
      }
      recompute();
      resetRun();
    });
    resetRun();
  }

  window.SixTuple = { init, stop };
})();
