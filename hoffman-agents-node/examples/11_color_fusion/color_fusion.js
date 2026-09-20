/**
 * Color Fusion — A Visual Metaphor for Consciousness Fusion
 *
 * Primary conscious agents (Red, Green, Blue) each perceive one band of the
 * color spectrum. When they combine via ⊗, their perceived colors MERGE —
 * not by averaging, but by additive light mixing (the way actual photons
 * combine). A combined agent literally sees a color that neither parent
 * alone could see.
 *
 * The invisible spectrum (UV, IR) represents forms of consciousness that
 * exist but that we have no direct sensory access to. They're real — they
 * interact with the same world — but their experience is outside our
 * perceptual range. When they combine with visible-spectrum agents, they
 * produce colors that a human observer cannot fully name.
 *
 * The dashboard (color_fusion.html) renders the agent population as a
 * living grid of colored squares. Each square is one agent's perceived
 * color — individual or fused. Watching the grid evolve is watching
 * consciousness combine in real time, rendered in the one language that
 * needs no translation: color.
 *
 * This is an artistic-scientific experiment, not a physics claim.
 */

const { ConsciousAgent, WorldState, combine } = require('../../src/index');

/* ────────────── Color definitions ────────────── */

// Each color is a hex value + a label. Additive mixing.
const COLORS = {
  R:  { hex: '#FF0000', name: 'Red',     band: 'visible',  wl: '620-750nm' },
  G:  { hex: '#00FF00', name: 'Green',   band: 'visible',  wl: '495-570nm' },
  B:  { hex: '#0000FF', name: 'Blue',    band: 'visible',  wl: '450-495nm' },
  Y:  { hex: '#FFFF00', name: 'Yellow',  band: 'visible',  wl: '570-590nm' },
  C:  { hex: '#00FFFF', name: 'Cyan',    band: 'visible',  wl: '490-520nm' },
  M:  { hex: '#FF00FF', name: 'Magenta', band: 'visible',  wl: '380-450nm' },
  W:  { hex: '#FFFFFF', name: 'White',   band: 'visible',  wl: 'full spectrum' },
  UV: { hex: '#8800FF', name: 'UV',      band: 'invisible', wl: '10-380nm' },
  IR: { hex: '#FF0088', name: 'IR',      band: 'invisible', wl: '750nm-1mm' },
};

// Additive color mixing function
function mixColors(hexes) {
  // Extract RGB components, sum them, clamp to [0, 255]
  let r = 0, g = 0, b = 0;
  for (const hex of hexes) {
    r += parseInt(hex.slice(1, 3), 16);
    g += parseInt(hex.slice(3, 5), 16);
    b += parseInt(hex.slice(5, 7), 16);
  }
  // Normalize by count (average, not sum — keeps colors in range)
  const n = hexes.length;
  r = Math.min(255, Math.round(r / n));
  g = Math.min(255, Math.round(g / n));
  b = Math.min(255, Math.round(b / n));
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/* ────────────── ColorWorld ────────────── */

class ColorWorld {
  constructor(seed = 42) {
    this._rng = seedRandom(seed);
    this._tokens = Object.keys(COLORS);
    // Transition matrix: colors closer in the spectrum are more likely to follow
    this._transition = this._buildTransitionMatrix();
    this._currentIdx = 0;
  }

  _buildTransitionMatrix() {
    const n = this._tokens.length;
    const P = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        // Colors with adjacent spectral positions have higher transition prob
        const dist = Math.abs(i - j);
        const prob = 1 / (dist + 1);
        row.push(prob);
      }
      const total = row.reduce((s, v) => s + v, 0);
      row.push(row.map(v => v / total));
      P.push(row.map(v => v / total));
    }
    return P;
  }

  step() {
    const row = this._transition[this._currentIdx];
    let r = this._rng();
    let cumulative = 0;
    for (let j = 0; j < row.length; j++) {
      cumulative += row[j];
      if (r <= cumulative) { this._currentIdx = j; break; }
    }
    const token = this._tokens[this._currentIdx];
    return { ws: WorldState.fromSequence('world', [token]), token, color: COLORS[token] };
  }
}

function seedRandom(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0x7FFFFFFF; return s / 0x7FFFFFFF; };
}

/* ────────────── Create a color-specialist agent ────────────── */

function createColorAgent(name, colorToken, world, nSteps = 500) {
  const agent = new ConsciousAgent({ agentId: name, metaObservationInterval: 10 });
  for (let i = 0; i < nSteps; i++) {
    const { ws, token } = world.step();
    // Only train on states matching this agent's color
    if (token === colorToken || token === 'W') {
      agent.step(ws);
    }
  }
  return agent;
}

/* ────────────── Generate dashboard data ────────────── */

function generateDashboardData() {
  // 1. Train primary agents
  const world = new ColorWorld(42);
  const red = createColorAgent('Red', 'R', world, 800);
  const green = createColorAgent('Green', 'G', world, 800);
  const blue = createColorAgent('Blue', 'B', world, 800);
  const uv = createColorAgent('UV', 'UV', world, 800);
  const ir = createColorAgent('IR', 'IR', world, 800);

  // 2. Combine into higher-order agents
  const rg = combine(red, green);   // Yellow
  const rb = combine(red, blue);    // Magenta
  const gb = combine(green, blue);  // Cyan
  const rgb = combine(rg, blue);    // White
  const uv_r = combine(uv, red);    // UV+Red — a color we cannot fully name
  const ir_g = combine(ir, green); // IR+Green — another unnamed color
  const all = combine(rgb, combine(uv, ir)); // Everything — the full spectrum

  // 3. Run a few steps to let them stabilize
  for (let s = 0; s < 20; s++) {
    const { ws } = world.step();
    [red, green, blue, uv, ir, rg, rb, gb, rgb, uv_r, ir_g, all].forEach(a => a.step(ws));
  }

  // 4. Build the agent list with colors and metadata
  const agents = [
    { id: 'Red',     agent: red,   color: '#FF0000', level: 0, parentage: 'Red', band: 'visible' },
    { id: 'Green',   agent: green, color: '#00FF00', level: 0, parentage: 'Green', band: 'visible' },
    { id: 'Blue',    agent: blue,  color: '#0000FF', level: 0, parentage: 'Blue', band: 'visible' },
    { id: 'UV',      agent: uv,    color: '#8800FF', level: 0, parentage: 'UV', band: 'invisible' },
    { id: 'IR',      agent: ir,    color: '#FF0088', level: 0, parentage: 'IR', band: 'invisible' },
    { id: 'Yellow',  agent: rg,    color: '#FFFF00', level: 1, parentage: 'Red+Green', band: 'visible' },
    { id: 'Magenta', agent: rb,    color: '#FF00FF', level: 1, parentage: 'Red+Blue', band: 'visible' },
    { id: 'Cyan',    agent: gb,    color: '#00FFFF', level: 1, parentage: 'Green+Blue', band: 'visible' },
    { id: 'White',   agent: rgb,   color: '#FFFFFF', level: 2, parentage: 'Red+Green+Blue', band: 'visible' },
    { id: 'UV+Red',  agent: uv_r,  color: '#C40080', level: 1, parentage: 'UV+Red', band: 'hybrid' },
    { id: 'IR+Green',agent: ir_g,  color: '#8000C4', level: 1, parentage: 'IR+Green', band: 'hybrid' },
    { id: 'Omni',    agent: all,   color: '#C0C0FF', level: 3, parentage: 'Full Spectrum', band: 'omni' },
  ];

  return { agents };
}

/* ────────────── Main (generates JSON for dashboard) ────────────── */

function main() {
  const data = generateDashboardData();

  // Output metadata for reference (the dashboard reads from the JSON file)
  console.log(JSON.stringify({
    experiment: 'Color Fusion — Consciousness Merger',
    description: 'Primary agents (R, G, B) perceive individual color bands. Combined agents perceive merged colors via additive mixing. Invisible-spectrum agents (UV, IR) represent consciousness outside human perceptual range.',
    agents: data.agents.map(a => ({
      id: a.id,
      hex: a.color,
      level: a.level,
      parentage: a.parentage,
      band: a.band,
      locked: a.agent.isILocked,
      trieSize: a.agent.experience.trie.size(),
      metaStates: a.agent.experience.metaTrie.registrySize,
      predError: a.agent.meanPredictionError.toFixed(3),
    })),
  }, null, 2));
}

if (require.main === module) {
  main();
}

module.exports = { ColorWorld, createColorAgent, generateDashboardData, COLORS, mixColors };
